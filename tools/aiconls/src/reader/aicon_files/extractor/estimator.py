import libcst as cst
from lsprotocol.types import Range

from reader.aicon_files.extractor.component import ComponentExtractor
from reader.aicon_files.reader_types import ParsedEstimator
from reader.aicon_files.util import get_node_position, get_simple_string


class EstimatorExtractor(ComponentExtractor[ParsedEstimator]):
    def __init__(
        self, uri: str, name: str, name_pos: Range, metadata_owner: cst.CSTVisitor
    ) -> None:
        super().__init__(
            uri=uri,
            name=name,
            name_pos=name_pos,
            parsed_component_type=ParsedEstimator,
            metadata_owner=metadata_owner,
        )
        self.f_func_params: dict[str, Range] = {}
        self.f_output_signature_diff_names: dict[str, Range] = {}
        self.f_output_signature_non_diff_names: dict[str, Range] = {}
        self._inside_define_estimation_function_f = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if node.name.value == "define_estimation_function_f":
            self._inside_define_estimation_function_f = True
            return None

        if self._inside_define_estimation_function_f and node.name.value == "f_func":
            for parameter in (
                list(node.params.posonly_params)
                + list(node.params.params)
                + list(node.params.kwonly_params)
            ):
                self.f_func_params[parameter.name.value] = get_node_position(
                    self.metadata_owner, parameter.name
                )
            return False
        return None

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        if original_node.name.value == "define_estimation_function_f":
            self._inside_define_estimation_function_f = False

    def _extract_output_signature_names(
        self, output_names_node: cst.BaseExpression
    ) -> dict[str, Range]:
        if isinstance(output_names_node, (cst.List, cst.Tuple)):
            output_names_elements = output_names_node.elements
        else:
            return {}

        output_signature_names: dict[str, Range] = {}
        for output_name_element in output_names_elements:
            if not isinstance(output_name_element, cst.Element):
                continue
            output_name_node = output_name_element.value
            if not isinstance(output_name_node, cst.SimpleString):
                continue
            output_name = get_simple_string(output_name_node)
            output_signature_names[output_name] = get_node_position(
                self.metadata_owner, output_name_node
            )
        return output_signature_names

    def visit_Return(self, node: cst.Return) -> bool | None:
        if not self._inside_define_estimation_function_f:
            return None
        if node.value is None or not isinstance(node.value, cst.Tuple):
            return None
        if len(node.value.elements) < 3:
            return None

        diff_names_element = node.value.elements[1]
        non_diff_names_element = node.value.elements[2]
        if not isinstance(diff_names_element, cst.Element) or not isinstance(
            non_diff_names_element, cst.Element
        ):
            return None

        self.f_output_signature_diff_names.update(
            self._extract_output_signature_names(diff_names_element.value)
        )
        self.f_output_signature_non_diff_names.update(
            self._extract_output_signature_names(non_diff_names_element.value)
        )
        return None

    def get_parsed_component(self) -> ParsedEstimator:
        parsed_estimator = super().get_parsed_component()
        parsed_estimator.f_func_params = self.f_func_params
        parsed_estimator.f_output_signature_diff_names = (
            self.f_output_signature_diff_names
        )
        parsed_estimator.f_output_signature_non_diff_names = (
            self.f_output_signature_non_diff_names
        )
        return parsed_estimator
