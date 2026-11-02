import contextlib
import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

import libcst as cst

# from libcst.display import dump
from libcst.metadata import (
    PositionProvider,
    QualifiedNameProvider,
)
from lsprotocol import types
from lsprotocol.types import Diagnostic

from diagnostics.diagnostics import ErrorCode
from diagnostics.helper import make_diagnostic
from reader.aicon_files.extractor.action import ActionExtractor
from reader.aicon_files.extractor.connection import ConnectionExtractor
from reader.aicon_files.extractor.estimator import EstimatorExtractor
from reader.aicon_files.extractor.goal import GoalExtractor
from reader.aicon_files.extractor.sensor import SensorExtractor
from reader.aicon_files.extractor.util import extract_shapes_dict
from reader.aicon_files.packages import (
    package_name_and_import_root,
    resolve_quantity_definition,
)
from reader.aicon_files.reader_types import (
    ParsedComponent,
    ParsedConnection,
    ParsedGoal,
    ParsedQuantity,
)
from reader.aicon_files.util import get_node_position
from reader.spec.spec import AICON_SPEC
from validator.action_validator import ActionValidator
from validator.connection_validator import ConnectionValidator
from validator.estimator_validator import EstimatorValidator
from validator.goal_validator import GoalValidator
from validator.sensor_validator import SensorValidator

logger = logging.getLogger(__name__)

_FRAMEWORK_KIND_BY_BASE_CLASS = {
    "SensorComponent": "sensor",
    "EstimationComponent": "estimator",
    "ActionComponent": "action",
    "Goal": "goal",
    "ActiveInterconnection": "connection",
}


@dataclass(frozen=True)
class _BuildConnectionEntry:
    class_name: str | None
    name_argument: str | None
    position: cst.CSTNode


@dataclass(frozen=True)
class _GetBuildingFunctionsConnectionBuildersAssignment:
    uses_build_connections: bool
    position: cst.CSTNode


def _get_simple_string_value(node: cst.SimpleString) -> str | None:
    value = node.evaluated_value
    if isinstance(value, str):
        return value
    return None


def _extract_call_name_argument(call: cst.Call) -> str | None:
    if not call.args:
        return None
    first_argument = call.args[0]
    if first_argument.keyword is not None:
        return None
    if not isinstance(first_argument.value, cst.SimpleString):
        return None
    return _get_simple_string_value(first_argument.value)


class AICONComponentParser(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (QualifiedNameProvider, PositionProvider)

    components: dict[str, ParsedComponent]
    goals: dict[str, ParsedGoal]
    connections: dict[str, ParsedConnection]
    diagnostics: list[Diagnostic]
    document_uri: str
    has_build_connections_function: bool
    _class_definitions: dict[str, cst.ClassDef]
    _class_nodes: list[cst.ClassDef]

    def __init__(
        self, spec: AICON_SPEC, document_uri: str, project_root: Path | None = None
    ) -> None:
        self.components = {}
        self.goals = {}
        self.connections = {}
        self.spec = spec
        self.diagnostics = []
        self.document_uri = document_uri
        self._document_path = self._resolve_document_path(document_uri)
        self._project_root = (
            project_root.resolve()
            if project_root is not None
            else self._resolve_default_project_root()
        )
        self._document_package_name = self._resolve_document_package_name()
        self._class_depth = 0
        self._class_definitions = {}
        self._class_nodes = []
        self._build_connections_position = None
        self._build_connection_entries: dict[str, _BuildConnectionEntry] = {}
        self._get_building_functions_position = None
        self._connection_builders_assignment = None
        self.has_build_connections_function = False
        self._external_class_kind_cache: dict[str, str | None] = {}

    def _resolve_document_path(self, document_uri: str) -> Path | None:
        parsed_uri = urlparse(document_uri)
        if parsed_uri.scheme != "file":
            return None
        return Path(unquote(parsed_uri.path)).resolve()

    def _resolve_default_project_root(self) -> Path | None:
        if self._document_path is None:
            return None
        return self._document_path.parent

    def _resolve_document_package_name(self) -> str | None:
        # Inside a package, the full dotted name (walking up through the __init__.py files, also above the project
        # root), so that relative imports of parent packages (`from ..util import ...`) resolve.
        package_and_root = (
            package_name_and_import_root(self._document_path)
            if self._document_path is not None
            else None
        )
        if package_and_root is not None:
            self._package_import_root: Path | None = package_and_root[1]
            return package_and_root[0]
        self._package_import_root = None
        if self._document_path is None or self._project_root is None:
            return None

        try:
            relative_path = self._document_path.relative_to(self._project_root)
        except ValueError:
            return None

        module_path = relative_path.with_suffix("")
        package_parts = list(module_path.parts[:-1])
        if module_path.name == "__init__":
            package_parts = list(module_path.parts[:-1])

        if not package_parts:
            return None
        return ".".join(package_parts)

    def _resolve_missing_quantities(self, parsed_component: object) -> None:
        """For a component without a literal cls_quantity_names_and_shapes of its own: the quantities inherited from a
        base class, or, if the class or a base class sets them when it is built (they then depend e.g. on the
        instance name), those of the spec, as they cannot be read from the code."""
        component = cast(Any, parsed_component)
        if component.quantities_pos is not None or self._document_path is None:
            return
        resolved = resolve_quantity_definition(
            self._document_path,
            component.name,
            set(_FRAMEWORK_KIND_BY_BASE_CLASS) | {"BatchedEstimationComponent"},
        )
        if resolved is None:
            return
        kind, definition = resolved
        if kind == "static" and definition is not None:
            try:
                component.quantities = extract_shapes_dict(definition)
            except TypeError:
                return
        else:
            instances = self.spec.get_component_instances_by_class_name(component.name)
            if not instances:
                return
            component.quantities = {
                name: ParsedQuantity(shape=tuple(props.shape))
                for name, props in self.spec.components[instances[0]].quantities.items()
            }
        component.quantities_pos = component.position

    def _parse_sensor_component(self, node: cst.ClassDef) -> None:
        name_pos = get_node_position(self, node.name)
        component_extractor = SensorExtractor(
            self.document_uri, node.name.value, name_pos, metadata_owner=self
        )
        node.visit(component_extractor)
        parsed_component = component_extractor.get_parsed_component()
        self._resolve_missing_quantities(parsed_component)
        component_validator = SensorValidator(self.spec, parsed_component)
        self.diagnostics.extend(component_validator.diagnostics)
        self.components[node.name.value] = parsed_component

    def _parse_estimator_component(self, node: cst.ClassDef) -> None:
        name_pos = get_node_position(self, node.name)
        component_extractor = EstimatorExtractor(
            self.document_uri, node.name.value, name_pos, metadata_owner=self
        )
        node.visit(component_extractor)
        parsed_component = component_extractor.get_parsed_component()
        self._resolve_missing_quantities(parsed_component)
        component_validator = EstimatorValidator(self.spec, parsed_component)
        self.diagnostics.extend(component_validator.diagnostics)
        self.components[node.name.value] = parsed_component

    def _parse_action_component(self, node: cst.ClassDef) -> None:
        name_pos = get_node_position(self, node.name)
        component_extractor = ActionExtractor(
            self.document_uri, node.name.value, name_pos, metadata_owner=self
        )
        node.visit(component_extractor)
        parsed_component = component_extractor.get_parsed_component()
        self._resolve_missing_quantities(parsed_component)
        component_validator = ActionValidator(self.spec, parsed_component)
        self.diagnostics.extend(component_validator.diagnostics)
        self.components[node.name.value] = parsed_component

    def _parse_goal(self, node: cst.ClassDef) -> None:
        logger.debug("Found Goal: %s", node.name.value)
        goal_position = get_node_position(self, node.name)
        goal_extractor = GoalExtractor(
            self.document_uri,
            node.name.value,
            goal_position,
            metadata_owner=self,
        )
        node.visit(goal_extractor)
        parsed_goal = goal_extractor.get_parsed_goal()
        goal_validator = GoalValidator(
            self.spec,
            parsed_goal,
        )
        self.diagnostics.extend(goal_validator.diagnostics)
        self.goals[node.name.value] = parsed_goal

    def _parse_connection(self, node: cst.ClassDef) -> None:
        logger.debug("Found ActiveInterconnection: %s", node.name.value)
        connection_position = get_node_position(self, node.name)
        connection_extractor = ConnectionExtractor(
            self.document_uri,
            node.name.value,
            connection_position,
            metadata_owner=self,
        )
        node.visit(connection_extractor)
        parsed_connection = connection_extractor.get_parsed_connection()
        connection_validator = ConnectionValidator(self.spec, parsed_connection)
        self.diagnostics.extend(connection_validator.diagnostics)
        self.connections[node.name.value] = parsed_connection

    def _extract_dotted_name(self, expression: cst.BaseExpression) -> str | None:
        if isinstance(expression, cst.Name):
            return expression.value
        if isinstance(expression, cst.Subscript):
            return self._extract_dotted_name(expression.value)
        if isinstance(expression, cst.Attribute):
            left = self._extract_dotted_name(expression.value)
            if left is None:
                return None
            return f"{left}.{expression.attr.value}"
        return None

    def _extract_class_reference_names(
        self, expression: cst.BaseExpression
    ) -> list[str]:
        reference_names: list[str] = []
        qualified_names = self.get_metadata(
            QualifiedNameProvider, expression, default=set()
        )
        for qualified_name in sorted(qualified_names, key=lambda name: name.name):
            reference_names.append(qualified_name.name)

        dotted_name = self._extract_dotted_name(expression)
        if dotted_name is not None:
            reference_names.append(dotted_name)

        if isinstance(expression, cst.Attribute):
            reference_names.append(expression.attr.value)

        unique_names: list[str] = []
        for name in reference_names:
            if name not in unique_names:
                unique_names.append(name)
        return unique_names

    def _extract_base_class_names(self, node: cst.ClassDef) -> list[str]:
        base_names: list[str] = []
        for base in node.bases:
            for base_name in self._extract_class_reference_names(base.value):
                if base_name not in base_names:
                    base_names.append(base_name)
        return base_names

    def _resolve_framework_kind(self, class_name: str) -> str | None:
        return _FRAMEWORK_KIND_BY_BASE_CLASS.get(class_name.rsplit(".", maxsplit=1)[-1])

    @contextlib.contextmanager
    def _temporary_import_paths(self):
        path_candidates: list[Path] = []
        package_import_root: Path | None = getattr(self, "_package_import_root", None)
        if package_import_root is not None:
            path_candidates.append(package_import_root)
        if self._project_root is not None:
            path_candidates.append(self._project_root)
        if self._document_path is not None:
            path_candidates.append(self._document_path.parent)

        inserted_paths: list[str] = []
        for path in path_candidates:
            path_string = str(path)
            if path_string in inserted_paths or path_string in sys.path:
                continue
            sys.path.insert(0, path_string)
            inserted_paths.append(path_string)

        try:
            yield
        finally:
            for path_string in inserted_paths:
                if path_string in sys.path:
                    sys.path.remove(path_string)

    def _split_module_and_symbol_name(self, class_name: str) -> tuple[str | None, str]:
        if class_name.startswith("."):
            stripped_name = class_name.lstrip(".")
            if not stripped_name:
                return None, ""
            relative_prefix = "." * (len(class_name) - len(stripped_name))
            module_name, _, symbol_name = stripped_name.rpartition(".")
            if module_name:
                return f"{relative_prefix}{module_name}", symbol_name
            return relative_prefix, symbol_name

        module_name, _, symbol_name = class_name.rpartition(".")
        if not module_name:
            return None, symbol_name
        return module_name, symbol_name

    def _find_module_source_path(self, module_name: str) -> Path | None:
        try:
            with self._temporary_import_paths():
                if module_name.startswith("."):
                    if self._document_package_name is None:
                        return None
                    module_spec = importlib.util.find_spec(
                        module_name, package=self._document_package_name
                    )
                else:
                    module_spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError, TypeError):
            return None

        if module_spec is None or module_spec.origin is None:
            return None
        if module_spec.origin in {"built-in", "frozen"}:
            return None
        module_source = Path(module_spec.origin)
        if module_source.suffix != ".py":
            return None
        return module_source

    def _extract_module_import_name(self, node: cst.BaseExpression) -> str | None:
        return self._extract_dotted_name(node)

    def _extract_source_class_bases(
        self, module_source_path: Path, target_class_name: str
    ) -> tuple[dict[str, list[str]], list[str] | None]:
        source = module_source_path.read_text(encoding="utf-8")
        module = cst.parse_module(source)

        imported_names: dict[str, str] = {}
        imported_modules: dict[str, str] = {}
        class_bases: dict[str, list[str]] = {}

        for statement in module.body:
            if not isinstance(statement, cst.SimpleStatementLine):
                continue
            for small_statement in statement.body:
                if isinstance(small_statement, cst.ImportFrom):
                    imported_module_name = ""
                    if small_statement.module is not None:
                        extracted_name = self._extract_module_import_name(
                            small_statement.module
                        )
                        if extracted_name is not None:
                            imported_module_name = extracted_name
                    relative_prefix = "." * len(small_statement.relative)
                    if isinstance(small_statement.names, cst.ImportStar):
                        continue
                    for alias in small_statement.names:
                        imported_name = self._extract_module_import_name(alias.name)
                        if imported_name is None:
                            continue
                        resolved_alias: str
                        if alias.asname is not None:
                            extracted_alias = self._extract_module_import_name(
                                alias.asname.name
                            )
                            if extracted_alias is None:
                                continue
                            resolved_alias = extracted_alias
                        else:
                            resolved_alias = imported_name.rsplit(".", maxsplit=1)[-1]
                        module_prefix = f"{relative_prefix}{imported_module_name}"
                        if module_prefix:
                            imported_names[resolved_alias] = (
                                f"{module_prefix}.{imported_name}"
                            )
                        else:
                            imported_names[resolved_alias] = imported_name
                if isinstance(small_statement, cst.Import):
                    for alias in small_statement.names:
                        imported_module = self._extract_module_import_name(alias.name)
                        if imported_module is None:
                            continue
                        resolved_alias: str
                        if alias.asname is not None:
                            extracted_alias = self._extract_module_import_name(
                                alias.asname.name
                            )
                            if extracted_alias is None:
                                continue
                            resolved_alias = extracted_alias
                        else:
                            resolved_alias = imported_module.split(".", maxsplit=1)[0]
                        imported_modules[resolved_alias] = imported_module

        for statement in module.body:
            if not isinstance(statement, cst.ClassDef):
                continue
            base_names: list[str] = []
            for base in statement.bases:
                raw_name = self._extract_dotted_name(base.value)
                if raw_name is None:
                    continue
                if raw_name in imported_names:
                    base_names.append(imported_names[raw_name])
                    continue
                root_name, dot, remainder = raw_name.partition(".")
                imported_module = imported_modules.get(root_name)
                if imported_module is not None:
                    if dot:
                        base_names.append(f"{imported_module}.{remainder}")
                    else:
                        base_names.append(imported_module)
                    continue
                base_names.append(raw_name)
            class_bases[statement.name.value] = base_names

        return class_bases, class_bases.get(target_class_name)

    def _resolve_external_class_kind_from_source(
        self, class_name: str, module_name: str, symbol_name: str
    ) -> str | None:
        module_source_path = self._find_module_source_path(module_name)
        if module_source_path is None:
            return None

        try:
            class_bases_by_name, base_names = self._extract_source_class_bases(
                module_source_path,
                symbol_name,
            )
        except (OSError, cst.ParserSyntaxError, UnicodeDecodeError):
            return None

        if base_names is None:
            return None

        visited_source_classes: set[str] = set()

        def _resolve_from_source_base(base_name: str) -> str | None:
            direct_kind = self._resolve_framework_kind(base_name)
            if direct_kind is not None:
                return direct_kind

            if base_name in class_bases_by_name:
                if base_name in visited_source_classes:
                    return None
                visited_source_classes.add(base_name)
                for nested_base in class_bases_by_name[base_name]:
                    nested_kind = _resolve_from_source_base(nested_base)
                    if nested_kind is not None:
                        return nested_kind
                return None

            return self._resolve_class_kind(base_name, set())

        for base_name in base_names:
            resolved_kind = _resolve_from_source_base(base_name)
            if resolved_kind is not None:
                return resolved_kind
        return None

    def _resolve_external_class_kind(self, class_name: str) -> str | None:
        cached_kind = self._external_class_kind_cache.get(class_name)
        if cached_kind is not None or class_name in self._external_class_kind_cache:
            return cached_kind

        module_name, symbol_name = self._split_module_and_symbol_name(class_name)
        if module_name is None or not symbol_name:
            self._external_class_kind_cache[class_name] = None
            return None

        try:
            with self._temporary_import_paths():
                if module_name.startswith("."):
                    if self._document_package_name is None:
                        self._external_class_kind_cache[class_name] = None
                        return None
                    module = importlib.import_module(
                        module_name,
                        package=self._document_package_name,
                    )
                else:
                    module = importlib.import_module(module_name)
            imported_symbol = getattr(module, symbol_name)
        except ModuleNotFoundError:
            resolved_from_source = self._resolve_external_class_kind_from_source(
                class_name, module_name, symbol_name
            )
            self._external_class_kind_cache[class_name] = resolved_from_source
            return resolved_from_source
        except TypeError:
            resolved_from_source = self._resolve_external_class_kind_from_source(
                class_name, module_name, symbol_name
            )
            self._external_class_kind_cache[class_name] = resolved_from_source
            return resolved_from_source
        except ImportError:
            resolved_from_source = self._resolve_external_class_kind_from_source(
                class_name, module_name, symbol_name
            )
            self._external_class_kind_cache[class_name] = resolved_from_source
            return resolved_from_source
        except AttributeError:
            self._external_class_kind_cache[class_name] = None
            return None
        except Exception:  # noqa: BLE001 - importing ran user code that failed (e.g. a missing runtime dependency)
            resolved_from_source = self._resolve_external_class_kind_from_source(
                class_name, module_name, symbol_name
            )
            self._external_class_kind_cache[class_name] = resolved_from_source
            return resolved_from_source

        if not isinstance(imported_symbol, type):
            self._external_class_kind_cache[class_name] = None
            return None

        for inherited_class in imported_symbol.__mro__:
            kind = self._resolve_framework_kind(inherited_class.__name__)
            if kind is not None:
                self._external_class_kind_cache[class_name] = kind
                return kind

        self._external_class_kind_cache[class_name] = None
        return None

    def _resolve_class_kind(self, class_name: str, visited: set[str]) -> str | None:
        direct_kind = self._resolve_framework_kind(class_name)
        if direct_kind is not None:
            return direct_kind

        if class_name in visited:
            return None
        visited.add(class_name)

        class_node = self._class_definitions.get(class_name)
        if class_node is None:
            class_node = self._class_definitions.get(
                class_name.rsplit(".", maxsplit=1)[-1]
            )
        if class_node is None:
            return self._resolve_external_class_kind(class_name)

        for base_class_name in self._extract_base_class_names(class_node):
            resolved_kind = self._resolve_class_kind(base_class_name, visited)
            if resolved_kind is not None:
                return resolved_kind
        return None

    def _parse_classes(self) -> None:
        self._parse_all_classes()
        # A class that other classes of this file derive from is a building block (e.g. a common sensor base), not
        # a component of the graph: it is not expected in the spec.
        local_bases = {
            base_name.rsplit(".", maxsplit=1)[-1]
            for class_node in self._class_nodes
            for base_name in self._extract_base_class_names(class_node)
        }
        not_in_spec = {"W101", "W102", "W201"}
        self.diagnostics = [
            diagnostic
            for diagnostic in self.diagnostics
            if not (
                diagnostic.code in not_in_spec
                and any(
                    f"'{base}' not found in spec" in diagnostic.message
                    for base in local_bases
                )
            )
        ]

    def _parse_all_classes(self) -> None:
        for class_node in self._class_nodes:
            class_kind = self._resolve_class_kind(class_node.name.value, set())
            if class_kind == "sensor":
                logger.debug("Found SensorComponent: %s", class_node.name.value)
                self._parse_sensor_component(class_node)
            elif class_kind == "estimator":
                logger.debug("Found EstimationComponent: %s", class_node.name.value)
                self._parse_estimator_component(class_node)
            elif class_kind == "action":
                logger.debug("Found ActionComponent: %s", class_node.name.value)
                self._parse_action_component(class_node)
            elif class_kind == "goal":
                self._parse_goal(class_node)
            elif class_kind == "connection":
                self._parse_connection(class_node)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self._class_depth += 1
        self._class_definitions[node.name.value] = node
        self._class_nodes.append(node)
        return None

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        _ = original_node
        self._class_depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if self._class_depth != 0:
            return None
        if node.name.value != "build_connections":
            if node.name.value != "get_building_functions":
                return None
            self._get_building_functions_position = get_node_position(self, node.name)
            self._connection_builders_assignment = (
                self._extract_get_building_functions_connection_builders_assignment(
                    node
                )
            )
            return None

        self._build_connections_position = get_node_position(self, node.name)
        self._build_connection_entries = self._extract_build_connections_entries(node)
        self.has_build_connections_function = True
        return None

    def leave_Module(self, original_node: cst.Module) -> None:
        _ = original_node
        self._parse_classes()
        self._validate_build_connections()
        self._validate_get_building_functions()

    def _extract_build_connections_entries(
        self, function_node: cst.FunctionDef
    ) -> dict[str, _BuildConnectionEntry]:
        entries: dict[str, _BuildConnectionEntry] = {}
        for statement in function_node.body.body:
            if not isinstance(statement, cst.SimpleStatementLine):
                continue
            for small_statement in statement.body:
                # the builders as `connections = {...}` or returned directly as `return {...}`
                if isinstance(small_statement, cst.Return):
                    builders = small_statement.value
                else:
                    if not isinstance(small_statement, cst.Assign):
                        continue
                    if len(small_statement.targets) != 1:
                        continue
                    assign_target = small_statement.targets[0].target
                    if not isinstance(assign_target, cst.Name):
                        continue
                    if assign_target.value != "connections":
                        continue
                    builders = small_statement.value
                if not isinstance(builders, cst.Dict):
                    continue
                for element in builders.elements:
                    if not isinstance(element, cst.DictElement):
                        continue
                    if not isinstance(element.key, cst.SimpleString):
                        continue
                    connection_name = _get_simple_string_value(element.key)
                    if connection_name is None:
                        continue
                    value_expression = element.value
                    if not isinstance(value_expression, cst.Lambda):
                        entries[connection_name] = _BuildConnectionEntry(
                            class_name=None,
                            name_argument=None,
                            position=element.value,
                        )
                        continue
                    lambda_body = value_expression.body
                    if not isinstance(lambda_body, cst.Call):
                        entries[connection_name] = _BuildConnectionEntry(
                            class_name=None,
                            name_argument=None,
                            position=element.value,
                        )
                        continue
                    class_name: str | None = None
                    if isinstance(lambda_body.func, cst.Name):
                        class_name = lambda_body.func.value
                    elif isinstance(lambda_body.func, cst.Attribute):
                        class_name = lambda_body.func.attr.value
                    entries[connection_name] = _BuildConnectionEntry(
                        class_name=class_name,
                        name_argument=_extract_call_name_argument(lambda_body),
                        position=element.value,
                    )
                return entries
        return entries

    def _extract_get_building_functions_connection_builders_assignment(
        self, function_node: cst.FunctionDef
    ) -> _GetBuildingFunctionsConnectionBuildersAssignment | None:
        for statement in function_node.body.body:
            if not isinstance(statement, cst.SimpleStatementLine):
                continue
            for small_statement in statement.body:
                if not isinstance(small_statement, cst.Assign):
                    continue
                if len(small_statement.targets) != 1:
                    continue
                assign_target = small_statement.targets[0].target
                if not isinstance(assign_target, cst.Name):
                    continue
                if assign_target.value != "connection_builders":
                    continue
                value = small_statement.value
                if not isinstance(value, cst.Call):
                    return _GetBuildingFunctionsConnectionBuildersAssignment(
                        uses_build_connections=False,
                        position=small_statement.value,
                    )
                if value.args:
                    return _GetBuildingFunctionsConnectionBuildersAssignment(
                        uses_build_connections=False,
                        position=small_statement.value,
                    )
                if isinstance(value.func, cst.Name):
                    is_build_connections_call = value.func.value == "build_connections"
                elif isinstance(value.func, cst.Attribute):
                    is_build_connections_call = (
                        value.func.attr.value == "build_connections"
                    )
                else:
                    is_build_connections_call = False
                return _GetBuildingFunctionsConnectionBuildersAssignment(
                    uses_build_connections=is_build_connections_call,
                    position=small_statement.value,
                )
        return None

    def _validate_build_connections(self) -> None:
        if not self.spec.connections:
            return
        if self._build_connections_position is None:
            if not self.connections:
                return
            reference_range = (
                next(iter(self.connections.values())).position
                if self.connections
                else types.Range(
                    start=types.Position(line=0, character=0),
                    end=types.Position(line=0, character=0),
                )
            )
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.BUILD_CONNECTIONS_FUNCTION_MISSING,
                    range=reference_range,
                )
            )
            return

        for connection_name, connection_props in self.spec.connections.items():
            entry = self._build_connection_entries.get(connection_name)
            if entry is None:
                self.diagnostics.append(
                    make_diagnostic(
                        ErrorCode.BUILD_CONNECTIONS_ENTRY_MISSING,
                        range=self._build_connections_position,
                        connection_name=connection_name,
                    )
                )
                continue

            expected_class_name = connection_props.class_name
            if entry.class_name != expected_class_name:
                self.diagnostics.append(
                    make_diagnostic(
                        ErrorCode.BUILD_CONNECTIONS_CLASS_MISMATCH,
                        range=get_node_position(self, entry.position),
                        connection_name=connection_name,
                        actual_class_name=entry.class_name or "(unknown)",
                        expected_class_name=expected_class_name,
                    )
                )
                continue

            if entry.name_argument != connection_name:
                self.diagnostics.append(
                    make_diagnostic(
                        ErrorCode.BUILD_CONNECTIONS_NAME_ARGUMENT_MISMATCH,
                        range=get_node_position(self, entry.position),
                        connection_name=connection_name,
                        actual_name_argument=entry.name_argument or "(missing)",
                        expected_name_argument=connection_name,
                    )
                )

    def _validate_get_building_functions(self) -> None:
        if self._get_building_functions_position is None:
            return
        if self._connection_builders_assignment is None:
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.GET_BUILDING_FUNCTIONS_CONNECTION_BUILDERS_INVALID,
                    range=self._get_building_functions_position,
                )
            )
            return
        if not self._connection_builders_assignment.uses_build_connections:
            self.diagnostics.append(
                make_diagnostic(
                    ErrorCode.GET_BUILDING_FUNCTIONS_CONNECTION_BUILDERS_INVALID,
                    range=get_node_position(
                        self, self._connection_builders_assignment.position
                    ),
                )
            )
