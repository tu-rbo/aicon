import libcst as cst
from lsprotocol.types import Range

from reader.aicon_files.reader_types import ParsedGoal
from reader.aicon_files.util import get_node_position


class GoalExtractor(cst.CSTVisitor):
    name: str
    name_pos: Range
    uri: str
    metadata_owner: cst.CSTVisitor
    goal_func_params: dict[str, Range]

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
        self.goal_func_params = {}
        self._inside_define_goal_cost_function = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if node.name.value == "define_goal_cost_function":
            self._inside_define_goal_cost_function = True
            return None

        if self._inside_define_goal_cost_function and node.name.value == "goal_func":
            for parameter in (
                list(node.params.posonly_params)
                + list(node.params.params)
                + list(node.params.kwonly_params)
            ):
                self.goal_func_params[parameter.name.value] = get_node_position(
                    self.metadata_owner, parameter.name
                )
            return False
        return None

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        if original_node.name.value == "define_goal_cost_function":
            self._inside_define_goal_cost_function = False

    def get_parsed_goal(self) -> ParsedGoal:
        return ParsedGoal(
            name=self.name,
            position=self.name_pos,
            document_uri=self.uri,
            goal_func_params=self.goal_func_params,
        )
