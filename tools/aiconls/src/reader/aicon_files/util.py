from typing import cast

import libcst as cst
import libcst.matchers as m
from libcst.metadata import CodeRange, PositionProvider
from lsprotocol import types


def code_range_to_range(location: CodeRange):
    out_range = types.Range(
        start=types.Position(
            line=location.start.line - 1,
            character=location.start.column,
        ),
        end=types.Position(
            line=location.end.line - 1,
            character=location.end.column,
        ),
    )

    return out_range


def get_node_position(parser: cst.CSTVisitor, node: cst.CSTNode):
    location = cast(CodeRange, parser.get_metadata(PositionProvider, node))
    return code_range_to_range(location)


def get_identifier_name(node: cst.CSTNode) -> str:
    if m.matches(node, m.Name()):
        assert isinstance(node, cst.Name)
        return node.value
    raise TypeError("Identifier is not a Name")


def get_simple_string(node: cst.CSTNode) -> str:
    assert isinstance(node, cst.SimpleString)
    return str(node.evaluated_value)
