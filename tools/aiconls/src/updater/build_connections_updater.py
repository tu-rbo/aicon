import libcst as cst

from reader.spec.spec_types import SpecConnectionProps


def _render_build_connections_function(
    connections: dict[str, SpecConnectionProps],
) -> cst.FunctionDef:
    content = "def build_connections():\n    connections = {\n"
    for connection_name, connection_props in connections.items():
        content += (
            f'        "{connection_name}": lambda device, dtype, mockbuild: '
            f'{connection_props.class_name}("{connection_name}", '
            "device=device, dtype=dtype, mockbuild=mockbuild),\n"
        )
    content += "    }\n    return connections\n"
    return cst.ensure_type(cst.parse_statement(content), cst.FunctionDef)


class BuildConnectionsUpdater(cst.CSTTransformer):
    def __init__(self, connections: dict[str, SpecConnectionProps]):
        self.connections = connections

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        _ = original_node
        generated_function = _render_build_connections_function(self.connections)
        updated_body: list[cst.BaseStatement] = []
        replaced = False

        for statement in updated_node.body:
            if isinstance(statement, cst.FunctionDef) and statement.name.value == (
                "build_connections"
            ):
                if not replaced:
                    updated_body.append(generated_function)
                    replaced = True
                continue
            updated_body.append(statement)

        if not replaced:
            insert_index = next(
                (
                    index
                    for index, statement in enumerate(updated_body)
                    if isinstance(statement, cst.ClassDef)
                ),
                len(updated_body),
            )
            updated_body.insert(insert_index, generated_function)

        return updated_node.with_changes(body=tuple(updated_body))
