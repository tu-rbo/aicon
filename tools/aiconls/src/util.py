import re

from lsprotocol import types

from reader.aicon_files.reader_types import ParsedQuantities
from reader.spec.spec_types import SpecQuantities, shapes_match


def match_to_range(m: re.Match[str], content: str, group: int = 0):
    start_idx, end_idx = m.start(group), m.end(group)
    return _indices_to_range(start_idx, end_idx, content)


def _indices_to_range(start_idx: int, end_idx: int, content: str):

    start_line = content.count("\n", 0, start_idx)
    end_line = content.count("\n", 0, end_idx)

    last_nl_before_start = content.rfind("\n", 0, start_idx)
    last_nl_before_end = content.rfind("\n", 0, end_idx)

    start_col = start_idx - (last_nl_before_start + 1)
    end_col = end_idx - (last_nl_before_end + 1)

    return types.Range(
        start=types.Position(line=start_line, character=start_col),
        end=types.Position(line=end_line, character=end_col),
    )


def _yaml_key_pattern(name: str) -> str:
    """Pattern for a YAML mapping key that may be quoted (e.g. "E[dist]"); group 1 is the bare name."""
    return rf"[\"']?({re.escape(name)})[\"']?"


def _entry_not_found_message(kind: str, name: str) -> str:
    message = f"{kind} entry not found in spec: {name}!"
    if "[" in name or "]" in name:
        message += f' Names containing brackets must be quoted, e.g. "{name}":'
    return message


def find_component_in_spec_in_connection(component_name: str, content: str):
    # find location
    pattern = re.compile(rf"\[[^\]]*({re.escape(component_name)})[^\]]*]")
    m = pattern.search(content)
    if m:
        return match_to_range(m, content, 1)

    else:
        raise RuntimeError(f"Component: {component_name} not found in spec! {content}")


def find_component_in_connection_components(
    connection_name: str, component_name: str, content: str
):
    return find_nth_component_in_connection_components(
        connection_name, component_name, 1, content
    )


def find_nth_component_in_connection_components(
    connection_name: str,
    component_name: str,
    occurrence: int,
    content: str,
):
    if occurrence < 1:
        raise RuntimeError("Occurrence index must be >= 1.")

    connections_block = re.compile(
        r"(?ms)^connections:(?:\s*(?:#.*)?)?$.*?(?=^[^ \t\n][^:\n]*:(?:\s*(?:#.*)?)?$|\Z)"
    ).search(content)
    if not connections_block:
        raise RuntimeError("Connections section not found in spec!")

    connection_header = re.compile(
        rf"(?m)^([ \t]+){_yaml_key_pattern(connection_name)}:(?:\s*(?:#.*)?)?$"
    ).search(content, connections_block.start(), connections_block.end())
    if not connection_header:
        raise RuntimeError(_entry_not_found_message("Connection", connection_name))

    connection_indent = connection_header.group(1)
    next_connection_header = re.compile(
        rf"(?m)^{re.escape(connection_indent)}[^ \t\n][^:\n]*:(?:\s*(?:#.*)?)?$"
    ).search(content, connection_header.end(), connections_block.end())
    connection_block_end = (
        next_connection_header.start()
        if next_connection_header
        else connections_block.end()
    )

    inline_components_match = re.compile(r"components:\s*\[([^\]]*)\]").search(
        content, connection_header.start(), connection_block_end
    )
    if inline_components_match:
        components_content_start = inline_components_match.start(1)
        occurrences = list(
            re.finditer(
                rf"\b{re.escape(component_name)}\b", inline_components_match.group(1)
            )
        )
        if len(occurrences) >= occurrence:
            match = occurrences[occurrence - 1]
            start_idx = components_content_start + match.start()
            end_idx = components_content_start + match.end()
            return _indices_to_range(start_idx, end_idx, content)

    components_header_match = re.compile(
        r"(?m)^([ \t]+)components:\s*(?:#.*)?$"
    ).search(content, connection_header.start(), connection_block_end)
    if not components_header_match:
        raise RuntimeError(
            f"Components list not found in connection '{connection_name}'."
        )

    components_indent = components_header_match.group(1)
    next_property_match = re.compile(
        rf"(?m)^{re.escape(components_indent)}[^ \t\n-][^:\n]*:(?:\s*(?:#.*)?)?$"
    ).search(content, components_header_match.end(), connection_block_end)
    components_block_end = (
        next_property_match.start() if next_property_match else connection_block_end
    )
    list_item_matches = list(
        re.compile(
            rf"(?m)^{re.escape(components_indent)}[ \t]+-\s*([^\n#]+?)\s*(?:#.*)?$"
        ).finditer(
            content,
            components_header_match.end(),
            components_block_end,
        )
    )
    occurrences = [
        m for m in list_item_matches if m.group(1).strip("'\"") == component_name
    ]
    if len(occurrences) >= occurrence:
        match = occurrences[occurrence - 1]
        return _indices_to_range(match.start(1), match.end(1), content)

    raise RuntimeError(
        f"Component '{component_name}' occurrence {occurrence} not found in "
        f"connection '{connection_name}' components."
    )


def find_goal_in_spec_in_component(goal_name: str, content: str):
    pattern = re.compile(rf"\[.*({re.escape(goal_name)}).*]")
    m = pattern.search(content)
    if m:
        return match_to_range(m, content, 1)

    raise RuntimeError("Goal not found in component goals in spec!")


def find_name_in_spec_section(section_name: str, name: str, content: str):
    block_pattern = re.compile(
        rf"(?ms)^{re.escape(section_name)}:(?:\s*(?:#.*)?)?$.*?(?=^[^ \t\n][^:\n]*:(?:\s*(?:#.*)?)?$|\Z)"
    )
    block_match = block_pattern.search(content)
    if not block_match:
        raise RuntimeError(f"{section_name.title()} section not found in spec!")

    pattern = re.compile(rf"(?m)^[ \t]+{_yaml_key_pattern(name)}:(?:\s*(?:#.*)?)?$")
    m = pattern.search(
        content,
        block_match.start(),
        block_match.end(),
    )
    if m:
        return match_to_range(m, content, 1)

    raise RuntimeError(_entry_not_found_message(section_name.title(), name))


def _find_spec_section_block(section_name: str, content: str) -> re.Match[str]:
    block_pattern = re.compile(
        rf"(?ms)^{re.escape(section_name)}:(?:\s*(?:#.*)?)?$.*?(?=^[^ \t\n][^:\n]*:(?:\s*(?:#.*)?)?$|\Z)"
    )
    block_match = block_pattern.search(content)
    if not block_match:
        raise RuntimeError(f"{section_name.title()} section not found in spec!")
    return block_match


def _find_entry_block(
    section_name: str, entry_name: str, content: str
) -> tuple[int, int]:
    block_match = _find_spec_section_block(section_name, content)
    header_match = re.compile(
        rf"(?m)^([ \t]+){_yaml_key_pattern(entry_name)}:(?:\s*(?:#.*)?)?$"
    ).search(content, block_match.start(), block_match.end())
    if not header_match:
        raise RuntimeError(_entry_not_found_message(section_name.title(), entry_name))

    indent = header_match.group(1)
    next_header_match = re.compile(
        rf"(?m)^{re.escape(indent)}[^ \t\n][^:\n]*:(?:\s*(?:#.*)?)?$"
    ).search(content, header_match.end(), block_match.end())
    end_index = next_header_match.start() if next_header_match else block_match.end()
    return header_match.start(), end_index


def _find_entry_property_value_range(
    section_name: str,
    entry_name: str,
    property_name: str,
    content: str,
):
    entry_start, entry_end = _find_entry_block(section_name, entry_name, content)
    match = re.compile(rf"(?m)^[ \t]+{re.escape(property_name)}:\s*([^\n#]+)").search(
        content, entry_start, entry_end
    )
    if match:
        value = match.group(1).strip()
        if value:
            leading_ws = len(match.group(1)) - len(match.group(1).lstrip())
            value_start = match.start(1) + leading_ws
            value_end = value_start + len(value)
            return _indices_to_range(value_start, value_end, content)
    return None


def find_connection_component_name_in_spec(component_name: str, content: str):
    return find_name_in_spec_section("components", component_name, content)


def find_goal_name_in_spec(goal_name: str, content: str):
    return find_name_in_spec_section("goals", goal_name, content)


def find_connection_name_in_spec(connection_name: str, content: str):
    return find_name_in_spec_section("connections", connection_name, content)


def find_component_definition_in_spec(
    instance_name: str, _class_name: str, content: str
):
    class_name_range = _find_entry_property_value_range(
        "components",
        instance_name,
        "class_name",
        content,
    )
    if class_name_range is not None:
        return class_name_range
    return find_connection_component_name_in_spec(instance_name, content)


def find_goal_definition_in_spec(instance_name: str, _class_name: str, content: str):
    class_name_range = _find_entry_property_value_range(
        "goals",
        instance_name,
        "class_name",
        content,
    )
    if class_name_range is not None:
        return class_name_range
    return find_goal_name_in_spec(instance_name, content)


def find_connection_definition_in_spec(
    instance_name: str, class_name: str, content: str
):
    class_name_range = _find_entry_property_value_range(
        "connections",
        instance_name,
        "class_name",
        content,
    )
    if class_name_range is not None:
        return class_name_range
    return find_connection_name_in_spec(instance_name, content)


def is_subset_quantities(a: ParsedQuantities, b: SpecQuantities):
    """Check if quantities a are a subset of b"""
    if len(b) < len(a):
        return False

    for a_name, a_quant in a.items():
        if a_name not in b:
            return False

        if not shapes_match(a_quant.shape, b[a_name].shape):
            return False

    return True


def deep_diff_quantities(
    parsed_quantities: ParsedQuantities, spec_quantities: SpecQuantities
) -> bool:

    if len(parsed_quantities) != len(spec_quantities):
        return False

    for a_name, a_quant in parsed_quantities.items():
        if a_name not in spec_quantities:
            return False

        if not shapes_match(a_quant.shape, spec_quantities[a_name].shape):
            return False

    return True
