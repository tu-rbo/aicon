from collections.abc import Sequence

import libcst as cst
import libcst.matchers as m

from reader.aicon_files.extractor.util import build_shape_tuple
from reader.spec.spec_types import SpecComponentProps, SpecQuantities


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


class ComponentUpdater(cst.CSTTransformer):
    spec_component: SpecComponentProps
    component_name: str
    _inside_target_component: bool
    updated: bool

    def __init__(self, component_name: str, spec_component: SpecComponentProps):
        self.spec_component = spec_component
        self.component_name = component_name
        self._inside_target_component = False
        self.updated = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        if node.name.value == self.component_name:
            self._inside_target_component = True
        return None

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> (
        cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel
    ):
        if original_node.name.value == self.component_name:
            self._inside_target_component = False
            if self._has_cls_quantity_names_and_shapes(updated_node.body.body):
                return updated_node

            assignment = cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[
                            cst.AssignTarget(
                                target=cst.Name("cls_quantity_names_and_shapes")
                            )
                        ],
                        value=_build_quantities_dict(self.spec_component.quantities),
                    )
                ]
            )
            class_body = list(updated_node.body.body)

            if len(class_body) == 1 and m.matches(
                class_body[0], m.SimpleStatementLine(body=[m.Pass()])
            ):
                class_body = [assignment]
            else:
                insert_index = 0
                if class_body and self._is_docstring_statement(class_body[0]):
                    insert_index = 1
                class_body.insert(insert_index, assignment)

            self.updated = True
            return updated_node.with_changes(
                body=updated_node.body.with_changes(body=tuple(class_body))
            )
        return updated_node

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> (
        cst.BaseSmallStatement
        | cst.FlattenSentinel[cst.BaseSmallStatement]
        | cst.RemovalSentinel
    ):
        if not self._inside_target_component:
            return updated_node

        target = original_node.targets[0].target
        if not m.matches(target, m.Name()):
            return updated_node

        if cst.ensure_type(target, cst.Name).value != "cls_quantity_names_and_shapes":
            return updated_node

        self.updated = True
        return updated_node.with_changes(
            value=_build_quantities_dict(self.spec_component.quantities)
        )

    def leave_AnnAssign(
        self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> cst.BaseSmallStatement | cst.RemovalSentinel:
        # the annotated form: cls_quantity_names_and_shapes: Dict[str, Tuple[int]] = {...}
        if not self._inside_target_component:
            return updated_node
        if not m.matches(original_node.target, m.Name("cls_quantity_names_and_shapes")):
            return updated_node
        self.updated = True
        return updated_node.with_changes(
            value=_build_quantities_dict(self.spec_component.quantities)
        )

    def _has_cls_quantity_names_and_shapes(self, body: Sequence[cst.CSTNode]) -> bool:
        return any(
            m.matches(
                statement,
                m.SimpleStatementLine(
                    body=[
                        m.Assign(
                            targets=[
                                m.AssignTarget(
                                    target=m.Name("cls_quantity_names_and_shapes")
                                )
                            ]
                        )
                        | m.AnnAssign(target=m.Name("cls_quantity_names_and_shapes"))
                    ]
                ),
            )
            for statement in body
        )

    def _is_docstring_statement(self, statement: cst.CSTNode) -> bool:
        return m.matches(
            statement,
            m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())]),
        )
