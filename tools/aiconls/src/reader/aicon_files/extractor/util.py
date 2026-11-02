import libcst as cst
import libcst.matchers as m

from reader.aicon_files.reader_types import ParsedQuantities, ParsedQuantity
from reader.aicon_files.util import get_simple_string
from reader.spec.spec_types import DYNAMIC_DIMENSION, TensorShape


def extract_shapes_dict(node: cst.CSTNode) -> ParsedQuantities:
    if not m.matches(node, m.Dict()):
        raise TypeError("local quantities wrong")

    quantities: ParsedQuantities = {}

    for element in cst.ensure_type(node, cst.Dict).elements:
        dict_element = cst.ensure_type(element, cst.DictElement)
        if not isinstance(dict_element.value, cst.Tuple):
            raise TypeError("quantity shape is not a tuple")

        tuple_elements: list[int] = []
        for t_element in dict_element.value.elements:
            tuple_element = cst.ensure_type(t_element, cst.Element)
            value = tuple_element.value
            if (
                isinstance(value, cst.UnaryOperation)
                and isinstance(value.operator, cst.Minus)
                and isinstance(value.expression, cst.Integer)
            ):
                tuple_elements.append(-int(value.expression.value))
                continue
            if not isinstance(value, cst.Integer) or int(value.value) == 0:
                # e.g. a named constant, or 0 for a quantity that starts empty: the size is not known statically
                tuple_elements.append(DYNAMIC_DIMENSION)
                continue
            tuple_elements.append(int(value.value))

        quantities[get_simple_string(dict_element.key)] = ParsedQuantity(
            shape=tuple(tuple_elements)
        )

    return quantities


def build_shape_tuple(shape: TensorShape) -> cst.Tuple:
    elements: list[cst.Element] = []
    for value in shape:
        integer = cst.Integer(value=str(abs(value)))
        if value < 0:
            elements.append(
                cst.Element(
                    value=cst.UnaryOperation(operator=cst.Minus(), expression=integer)
                )
            )
        else:
            elements.append(cst.Element(value=integer))
    return cst.Tuple(elements=elements)
