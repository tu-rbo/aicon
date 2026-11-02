import libcst as cst
import libcst.matchers as m
from lsprotocol.types import Range

from reader.aicon_files.extractor.util import extract_shapes_dict
from reader.aicon_files.reader_types import ParsedConnection
from reader.aicon_files.util import get_node_position, get_simple_string


class ConnectionExtractor(cst.CSTVisitor):
    name: str
    name_pos: Range
    uri: str
    metadata_owner: cst.CSTVisitor
    quantities_pos: Range | None
    connection_func_params: dict[str, Range]
    quantity_positions: dict[str, Range]

    def __init__(
        self,
        uri: str,
        name: str,
        name_pos: Range,
        metadata_owner: cst.CSTVisitor,
    ) -> None:
        self.uri = uri
        self.name = name
        self.name_pos = name_pos
        self.metadata_owner = metadata_owner
        self.quantities = {}
        self.quantities_pos = None
        self.connection_func_params = {}
        self.quantity_positions = {}
        self.switching_quantities: set[str] = set()
        self._inside_define_connection_function = False

    def visit_Call(self, node: cst.Call) -> bool | None:
        if not m.matches(
            node.func,
            m.Attribute(
                value=m.Call(func=m.Name("super")),
                attr=m.Name("__init__"),
            ),
        ):
            return None

        if len(node.args) < 2:
            return None

        # required_signs_dict={"quantity": {"condition": sign, ...}, ...}: quantities used for switching gradients
        for argument in node.args:
            if (
                argument.keyword is None
                or argument.keyword.value != "required_signs_dict"
                or not isinstance(argument.value, cst.Dict)
            ):
                continue
            for element in argument.value.elements:
                if not isinstance(element, cst.DictElement):
                    continue
                if isinstance(element.key, cst.SimpleString):
                    self.switching_quantities.add(get_simple_string(element.key))
                if not isinstance(element.value, cst.Dict):
                    continue
                for condition in element.value.elements:
                    if isinstance(condition, cst.DictElement) and isinstance(
                        condition.key, cst.SimpleString
                    ):
                        self.switching_quantities.add(get_simple_string(condition.key))

        quantities_node = node.args[1].value
        if not m.matches(quantities_node, m.Dict()):
            return None

        try:
            self.quantities = extract_shapes_dict(quantities_node)
        except (AssertionError, TypeError):
            return None

        for element in cst.ensure_type(quantities_node, cst.Dict).elements:
            dict_element = cst.ensure_type(element, cst.DictElement)
            quantity_name = get_simple_string(dict_element.key)
            self.quantity_positions[quantity_name] = get_node_position(
                self.metadata_owner, dict_element.key
            )

        self.quantities_pos = get_node_position(self.metadata_owner, quantities_node)
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if node.name.value == "define_implicit_connection_function":
            self._inside_define_connection_function = True
            return None

        if (
            self._inside_define_connection_function
            and node.name.value == "connection_func"
        ):
            for parameter in (
                list(node.params.posonly_params)
                + list(node.params.params)
                + list(node.params.kwonly_params)
            ):
                self.connection_func_params[parameter.name.value] = get_node_position(
                    self.metadata_owner, parameter.name
                )
            return False
        return None

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        if original_node.name.value == "define_implicit_connection_function":
            self._inside_define_connection_function = False

    def get_parsed_connection(self) -> ParsedConnection:
        return ParsedConnection(
            name=self.name,
            position=self.name_pos,
            document_uri=self.uri,
            quantities=self.quantities,
            quantities_pos=self.quantities_pos,
            connection_func_params=self.connection_func_params,
            quantity_positions=self.quantity_positions,
            switching_quantities=self.switching_quantities,
        )
