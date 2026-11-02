from typing import Generic, TypeVar

import libcst as cst
import libcst.matchers as m
from lsprotocol.types import Range

from reader.aicon_files.extractor.util import extract_shapes_dict
from reader.aicon_files.reader_types import (
    ConnectionQuantityReference,
    ParsedAction,
    ParsedEstimator,
    ParsedSensor,
    QuantityReference,
)
from reader.aicon_files.util import get_node_position, get_simple_string

ParsedQuantifiedComponent = TypeVar(
    "ParsedQuantifiedComponent",
    ParsedSensor,
    ParsedAction,
    ParsedEstimator,
)


class ComponentExtractor(cst.CSTVisitor, Generic[ParsedQuantifiedComponent]):
    quantities_pos: Range | None

    uri: str
    name: str
    name_pos: Range

    def __init__(
        self,
        uri: str,
        name: str,
        name_pos: Range,
        parsed_component_type: type[ParsedQuantifiedComponent],
        metadata_owner: cst.CSTVisitor,
    ) -> None:
        self.quantities = {}
        self.name = name
        self.name_pos = name_pos
        self.parsed_component_type = parsed_component_type
        self.uri = uri
        self.metadata_owner = metadata_owner
        self.quantities_pos = None
        self.quantity_references: list[QuantityReference] = []
        self.connection_references: dict[str, Range] = {}
        self.connected_quantity_references: list[ConnectionQuantityReference] = []
        # local names bound to a connection, e.g. `c = self.connections["DirectMeasurement"]`
        self._connection_aliases: dict[str, str] = {}

    def visit_Assign(self, node: cst.Assign) -> bool | None:
        target = node.targets[0].target
        if (
            m.matches(target, m.Name())
            and isinstance(node.value, cst.Subscript)
            and self._is_self_connections_subscript(node.value)
        ):
            connection_key = self._extract_simple_string_subscript_key(node.value)
            if connection_key is not None:
                self._connection_aliases[cst.ensure_type(target, cst.Name).value] = (
                    connection_key[0]
                )
        if m.matches(target, m.Name()) and (
            cst.ensure_type(target, cst.Name).value == "cls_quantity_names_and_shapes"
        ):
            try:
                self.quantities = extract_shapes_dict(node.value)
            except TypeError:
                pass
            self.quantities_pos = get_node_position(self.metadata_owner, node)
            return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        # the annotated form: cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {...}
        if (
            m.matches(node.target, m.Name("cls_quantity_names_and_shapes"))
            and node.value is not None
        ):
            try:
                self.quantities = extract_shapes_dict(node.value)
            except TypeError:
                pass
            self.quantities_pos = get_node_position(self.metadata_owner, node)
            return False
        return None

    def _extract_simple_string_subscript_key(
        self, node: cst.Subscript
    ) -> tuple[str, cst.SimpleString] | None:
        if len(node.slice) != 1:
            return None

        slice_value = node.slice[0].slice
        if not isinstance(slice_value, cst.Index):
            return None
        if not isinstance(slice_value.value, cst.SimpleString):
            return None

        return get_simple_string(slice_value.value), slice_value.value

    def _is_self_connections_subscript(self, node: cst.Subscript) -> bool:
        return m.matches(
            node.value,
            m.Attribute(
                value=m.Name("self"),
                attr=m.Name("connections"),
            ),
        )

    def _is_self_quantities_subscript(self, node: cst.Subscript) -> bool:
        return m.matches(
            node.value,
            m.Attribute(
                value=m.Name("self"),
                attr=m.Name("quantities"),
            ),
        )

    def visit_Subscript(self, node: cst.Subscript) -> bool | None:
        if self._is_self_quantities_subscript(node):
            quantity_key = self._extract_simple_string_subscript_key(node)
            if quantity_key is None:
                return None
            quantity_name, quantity_name_node = quantity_key
            self.quantity_references.append(
                QuantityReference(
                    quantity_name=quantity_name,
                    position=get_node_position(self.metadata_owner, quantity_name_node),
                )
            )
            return None

        if self._is_self_connections_subscript(node):
            connection_key = self._extract_simple_string_subscript_key(node)
            if connection_key is None:
                return None
            connection_name, connection_name_node = connection_key
            if connection_name not in self.connection_references:
                self.connection_references[connection_name] = get_node_position(
                    self.metadata_owner, connection_name_node
                )
            return None

        if m.matches(
            node.value, m.Attribute(value=m.Name(), attr=m.Name("connected_quantities"))
        ):
            alias = cst.ensure_type(
                cst.ensure_type(node.value, cst.Attribute).value, cst.Name
            ).value
            aliased_key = self._extract_simple_string_subscript_key(node)
            if alias in self._connection_aliases and aliased_key is not None:
                self.connected_quantity_references.append(
                    ConnectionQuantityReference(
                        connection_name=self._connection_aliases[alias],
                        quantity_name=aliased_key[0],
                        position=get_node_position(self.metadata_owner, aliased_key[1]),
                    )
                )
            return None

        if not m.matches(
            node.value,
            m.Attribute(
                value=m.Subscript(),
                attr=m.Name("connected_quantities"),
            ),
        ):
            return None

        connected_quantities_key = self._extract_simple_string_subscript_key(node)
        if connected_quantities_key is None:
            return None
        quantity_name, quantity_name_node = connected_quantities_key

        connected_quantities_base = cst.ensure_type(node.value, cst.Attribute).value
        connection_key = self._extract_simple_string_subscript_key(
            cst.ensure_type(connected_quantities_base, cst.Subscript)
        )
        if connection_key is None:
            return None
        connection_name, connection_name_node = connection_key
        if not self._is_self_connections_subscript(
            cst.ensure_type(connected_quantities_base, cst.Subscript)
        ):
            return None

        if connection_name not in self.connection_references:
            self.connection_references[connection_name] = get_node_position(
                self.metadata_owner, connection_name_node
            )

        self.connected_quantity_references.append(
            ConnectionQuantityReference(
                connection_name=connection_name,
                quantity_name=quantity_name,
                position=get_node_position(self.metadata_owner, quantity_name_node),
            )
        )
        return None

    def get_parsed_component(self) -> ParsedQuantifiedComponent:
        return self.parsed_component_type(
            document_uri=self.uri,
            name=self.name,
            position=self.name_pos,
            quantities=self.quantities,
            goals=[],
            instance_name=self.name,
            class_name=self.name,
            quantities_pos=self.quantities_pos,
            quantity_references=self.quantity_references,
            connection_references=self.connection_references,
            connected_quantity_references=self.connected_quantity_references,
        )
