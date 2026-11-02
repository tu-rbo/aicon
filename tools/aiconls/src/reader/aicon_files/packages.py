"""Python packages around a project: dotted module names of files, and classes a project imports from outside its
root (e.g. an example that reuses classes of a sibling example or of a shared package of the repository).

Paths are derived from the import statements only; no module is imported or executed."""

import importlib.util
from pathlib import Path

import libcst as cst

# How many re-exports (e.g. `from .sensors import Foo` in a package's __init__.py) are followed to find a class.
_MAX_REEXPORT_DEPTH = 4


def package_name_and_import_root(file_path: Path) -> tuple[str, Path] | None:
    """For a file inside a package (its directory has an __init__.py): the dotted name of that package and the
    directory containing the top-level package. None for a file outside a package."""
    directory = file_path.parent
    if not (directory / "__init__.py").exists():
        return None
    parts: list[str] = []
    while (directory / "__init__.py").exists():
        parts.insert(0, directory.name)
        directory = directory.parent
    return ".".join(parts), directory


def module_name_and_import_root(file_path: Path) -> tuple[str, Path] | None:
    """The dotted module name of a file inside a package and the directory containing the top-level package."""
    package = package_name_and_import_root(file_path)
    if package is None:
        return None
    package_name, import_root = package
    if file_path.stem == "__init__":
        return package_name, import_root
    return f"{package_name}.{file_path.stem}", import_root


def _module_file(import_root: Path, module_name: str) -> Path | None:
    module_path = import_root.joinpath(*module_name.split("."))
    if module_path.with_suffix(".py").is_file():
        return module_path.with_suffix(".py")
    if (module_path / "__init__.py").is_file():
        return module_path / "__init__.py"
    return None


def _dotted_name(node: cst.BaseExpression | None) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr.value}"
    return ""


class _ImportFromCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.statements: list[cst.ImportFrom] = []

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        self.statements.append(node)


def _from_imports(file_path: Path) -> list[tuple[str, dict[str, str]]]:
    """The `from ... import ...` statements of a file, also those inside functions (e.g. imports in a builder
    function): (absolute module name, {local name: imported name}). Relative imports are made absolute with the
    file's package; those of a file outside a package are left out."""
    try:
        module = cst.parse_module(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, cst.ParserSyntaxError):
        return []
    package = package_name_and_import_root(file_path)
    package_name = package[0] if package is not None else None
    if file_path.stem != "__init__" and package is not None:
        current_module = f"{package_name}.{file_path.stem}"
    else:
        current_module = package_name
    collector = _ImportFromCollector()
    module.visit(collector)
    imports: list[tuple[str, dict[str, str]]] = []
    for small_statement in collector.statements:
        if isinstance(small_statement.names, cst.ImportStar):
            continue
        module_name = _dotted_name(small_statement.module)
        level = len(small_statement.relative)
        if level:
            if current_module is None:
                continue
            try:
                anchor = (
                    package_name if file_path.stem != "__init__" else current_module
                )
                module_name = importlib.util.resolve_name(
                    "." * level + module_name, anchor
                )
            except (ImportError, ValueError):
                continue
        names: dict[str, str] = {}
        for alias in small_statement.names:
            imported_name = _dotted_name(alias.name)
            local_name = (
                _dotted_name(alias.asname.name)
                if alias.asname is not None
                else imported_name
            )
            names[local_name] = imported_name
        imports.append((module_name, names))
    return imports


def _defines_class(file_path: Path, class_name: str) -> bool:
    try:
        module = cst.parse_module(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, cst.ParserSyntaxError):
        return False
    return any(
        isinstance(statement, cst.ClassDef) and statement.name.value == class_name
        for statement in module.body
    )


def _find_class_file(
    module_file: Path, class_name: str, import_roots: list[Path], depth: int = 0
) -> Path | None:
    """The file defining `class_name`, starting at `module_file` and following its re-exports."""
    if _defines_class(module_file, class_name):
        return module_file
    if depth >= _MAX_REEXPORT_DEPTH:
        return None
    for module_name, names in _from_imports(module_file):
        if class_name not in names:
            continue
        for import_root in import_roots:
            next_file = _module_file(import_root, module_name)
            if next_file is not None:
                return _find_class_file(
                    next_file, names[class_name], import_roots, depth + 1
                )
    return None


def find_imported_class_files(
    project_files: list[Path], class_names: set[str], project_root: Path
) -> dict[Path, set[str]]:
    """Files outside `project_root` that define classes of `class_names` which the project files import (directly or
    through re-exports), as {file: class names}. Only modules below the project files' import roots are followed,
    i.e. the repository the project belongs to, not installed packages."""
    import_roots: list[Path] = []
    for project_file in project_files:
        package = package_name_and_import_root(project_file)
        if package is not None and package[1] not in import_roots:
            import_roots.append(package[1])
    if project_root not in import_roots:
        import_roots.append(project_root)

    found: dict[Path, set[str]] = {}
    for project_file in project_files:
        for module_name, names in _from_imports(project_file):
            for local_name, imported_name in names.items():
                if local_name not in class_names:
                    continue
                for import_root in import_roots:
                    module_file = _module_file(import_root, module_name)
                    if module_file is None:
                        continue
                    class_file = _find_class_file(
                        module_file, imported_name, import_roots
                    )
                    if (
                        class_file is not None
                        and project_root not in class_file.parents
                    ):
                        found.setdefault(class_file, set()).add(imported_name)
                    break
    return found


# How many levels of base classes are searched for a component's quantity definition.
_MAX_BASE_CLASS_DEPTH = 6


def _class_node(module: cst.Module, class_name: str) -> cst.ClassDef | None:
    for statement in module.body:
        if isinstance(statement, cst.ClassDef) and statement.name.value == class_name:
            return statement
    return None


class _RuntimeQuantityDefinition(cst.CSTVisitor):
    """Whether a class sets its quantities when it is built: it assigns self.cls_quantity_names_and_shapes or
    self.local_quantity_names_and_shapes, or overrides initial_definitions."""

    def __init__(self) -> None:
        self.found = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if node.name.value == "initial_definitions":
            self.found = True

    def visit_AssignTarget(self, node: cst.AssignTarget) -> None:
        target = node.target
        if (
            isinstance(target, cst.Attribute)
            and isinstance(target.value, cst.Name)
            and target.value.value == "self"
            and target.attr.value
            in ("cls_quantity_names_and_shapes", "local_quantity_names_and_shapes")
        ):
            self.found = True


def _static_quantity_definition(class_node: cst.ClassDef) -> cst.BaseExpression | None:
    for statement in class_node.body.body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small_statement in statement.body:
            if isinstance(small_statement, cst.Assign) and any(
                isinstance(t.target, cst.Name)
                and t.target.value == "cls_quantity_names_and_shapes"
                for t in small_statement.targets
            ):
                return small_statement.value
            if (
                isinstance(small_statement, cst.AnnAssign)
                and isinstance(small_statement.target, cst.Name)
                and small_statement.target.value == "cls_quantity_names_and_shapes"
                and small_statement.value is not None
            ):
                return small_statement.value
    return None


def resolve_quantity_definition(
    file_path: Path, class_name: str, framework_classes: set[str], depth: int = 0
) -> tuple[str, cst.BaseExpression | None] | None:
    """Where the quantities of `class_name` (defined in `file_path`) come from, for a class without a literal
    cls_quantity_names_and_shapes of its own: ("static", dict expression) inherited from a base class, ("runtime",
    None) if the class or a base class sets them when it is built, or None if neither is found. The search stops at
    `framework_classes` (e.g. SensorComponent), whose definitions are generic."""
    if depth > _MAX_BASE_CLASS_DEPTH:
        return None
    try:
        module = cst.parse_module(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, cst.ParserSyntaxError):
        return None
    class_node = _class_node(module, class_name)
    if class_node is None:
        return None
    if depth > 0:
        static_definition = _static_quantity_definition(class_node)
        if static_definition is not None:
            return "static", static_definition
    runtime = _RuntimeQuantityDefinition()
    class_node.visit(runtime)
    if runtime.found:
        return "runtime", None

    imported = {
        local: (module_name, name)
        for module_name, names in _from_imports(file_path)
        for local, name in names.items()
    }
    package = package_name_and_import_root(file_path)
    import_roots = [package[1]] if package is not None else [file_path.parent]
    for base in class_node.bases:
        base_name = _dotted_name(base.value).rsplit(".", maxsplit=1)[-1]
        if not base_name or base_name in framework_classes:
            continue
        if _class_node(module, base_name) is not None:
            resolved = resolve_quantity_definition(
                file_path, base_name, framework_classes, depth + 1
            )
        elif base_name in imported:
            module_name, imported_name = imported[base_name]
            resolved = None
            for import_root in import_roots:
                module_file = _module_file(import_root, module_name)
                if module_file is None:
                    continue
                class_file = _find_class_file(module_file, imported_name, import_roots)
                if class_file is not None:
                    resolved = resolve_quantity_definition(
                        class_file, imported_name, framework_classes, depth + 1
                    )
                break
        else:
            resolved = None
        if resolved is not None:
            return resolved
    return None
