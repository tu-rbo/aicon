import libcst as cst
import libcst.matchers as m

from reader.aicon_files.extractor.util import build_shape_tuple, extract_shapes_dict
from reader.spec.spec_types import SpecQuantities


def _build_quantities_dict(quantities: SpecQuantities) -> cst.Dict:
    return cst.Dict(
        elements=[
            cst.DictElement(
                key=cst.SimpleString(value=repr(name)),
                value=build_shape_tuple(props.shape),
            )
            for name, props in quantities.items()
        ]
    )


class ConnectionUpdater(cst.CSTTransformer):
    connection_name: str
    spec_quantities: SpecQuantities
    _inside_target_connection: bool
    updated: bool

    def __init__(self, connection_name: str, spec_quantities: SpecQuantities):
        self.connection_name = connection_name
        self.spec_quantities = spec_quantities
        self._inside_target_connection = False
        self.updated = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        if node.name.value == self.connection_name:
            self._inside_target_connection = True
        return None

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.BaseStatement:
        if original_node.name.value == self.connection_name:
            self._inside_target_connection = False
        return updated_node

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        if not self._inside_target_connection:
            return updated_node

        if not m.matches(
            original_node.func,
            m.Attribute(
                value=m.Call(func=m.Name("super")),
                attr=m.Name("__init__"),
            ),
        ):
            return updated_node

        if len(original_node.args) < 2:
            return updated_node

        quantities_node = original_node.args[1].value
        if not m.matches(quantities_node, m.Dict()):
            return updated_node

        try:
            quantities = extract_shapes_dict(quantities_node)
        except TypeError:
            return updated_node

        if quantities == self.spec_quantities:
            return updated_node

        self.updated = True
        updated_args = list(updated_node.args)
        updated_args[1] = updated_args[1].with_changes(
            value=_build_quantities_dict(self.spec_quantities)
        )
        return updated_node.with_changes(args=tuple(updated_args))
