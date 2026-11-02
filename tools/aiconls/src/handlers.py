from lsprotocol import types

from diagnostics.diagnostics import ErrorCode
from lsp_utils import (
    get_connection_name_from_connected_quantities_completion,
    get_connection_name_from_member_completion,
    get_diagnostic_from_Code_Action,
    get_document_content,
    is_connected_quantities_completion_line,
    is_connection_completion_line,
    is_connection_member_completion_line,
    is_python_file,
    is_quantity_completion_line,
    is_self_connected_quantities_completion_line,
    is_spec_yaml,
    log_lsp_exceptions,
)
from server import AICONLanguageServer
from util import (
    find_component_definition_in_spec,
    find_connection_definition_in_spec,
    find_goal_definition_in_spec,
)


def _extract_target_uri(args: tuple[object, ...]) -> str | None:
    if not args:
        return None

    target = args[0]
    if isinstance(target, list):
        if target and isinstance(target[0], str):
            return target[0]
        return None

    if isinstance(target, str):
        return target

    return None


def _resolve_component_name_for_generation(
    ls: AICONLanguageServer, selected_name: str
) -> str:
    if ls.spec is None:
        return selected_name
    matching_instances = ls.spec.get_component_instances_by_class_name(selected_name)
    if matching_instances:
        return matching_instances[0]
    return selected_name


def _resolve_goal_name_for_generation(
    ls: AICONLanguageServer, selected_name: str
) -> str:
    if ls.spec is None:
        return selected_name
    matching_instances = ls.spec.get_goal_instances_by_class_name(selected_name)
    if matching_instances:
        return matching_instances[0]
    return selected_name


def _resolve_connection_name_for_generation(
    ls: AICONLanguageServer, selected_name: str
) -> str:
    if ls.spec is None:
        return selected_name
    matching_instances = ls.spec.get_connection_instances_by_class_name(selected_name)
    if matching_instances:
        return matching_instances[0]
    return selected_name


def _resolve_component_instance_name_for_completion(
    ls: AICONLanguageServer, component_name: str
) -> str:
    if ls.spec is None:
        return component_name
    matching_instances = ls.spec.get_component_instances_by_class_name(component_name)
    if matching_instances:
        return matching_instances[0]
    return component_name


def did_open(ls: AICONLanguageServer, params: types.DidOpenTextDocumentParams):
    """Parse each document when it is opened"""
    doc = ls.workspace.get_text_document(params.text_document.uri)
    if is_spec_yaml(params.text_document.uri):
        ls.parse(doc, validate_spec_after=False)
        ls.parse_all_project_python_files()
        ls.validate_spec(doc)
        return
    ls.parse(doc)


def did_change(ls: AICONLanguageServer, params: types.DidOpenTextDocumentParams):
    """Parse each document when it is changed"""
    ls.clear_dynamic_diagnostics()
    doc = ls.workspace.get_text_document(params.text_document.uri)
    ls.parse(doc)


def goto_implementation(ls: AICONLanguageServer, params: types.ImplementationParams):
    """Jump to an object's implementation."""
    if is_spec_yaml(params.text_document.uri):
        doc = ls.workspace.get_text_document(params.text_document.uri)
        word = doc.word_at_position(params.position)
        if ls.spec is None:
            return None

        component_class_name = ls.spec.get_component_class_name_by_instance_name(word)
        if component_class_name is not None and component_class_name in ls.components:
            return types.Location(
                uri=ls.components[component_class_name].document_uri,
                range=ls.components[component_class_name].position,
            )
        if word in ls.components:
            return types.Location(
                uri=ls.components[word].document_uri,
                range=ls.components[word].position,
            )

        goal_class_name = ls.spec.get_goal_class_name_by_instance_name(word)
        if goal_class_name is not None and goal_class_name in ls.goals:
            return types.Location(
                uri=ls.goals[goal_class_name].document_uri,
                range=ls.goals[goal_class_name].position,
            )
        if word in ls.goals:
            return types.Location(
                uri=ls.goals[word].document_uri,
                range=ls.goals[word].position,
            )

        connection_class_name = ls.spec.get_connection_class_name_by_instance_name(word)
        if (
            connection_class_name is not None
            and connection_class_name in ls.connections
        ):
            return types.Location(
                uri=ls.connections[connection_class_name].document_uri,
                range=ls.connections[connection_class_name].position,
            )
        if word in ls.connections:
            return types.Location(
                uri=ls.connections[word].document_uri,
                range=ls.connections[word].position,
            )

    return None


def goto_definition(ls: AICONLanguageServer, params: types.DefinitionParams):
    """Jump to an object's implementation."""
    if is_python_file(params.text_document.uri):
        doc = ls.workspace.get_text_document(params.text_document.uri)
        ls.refresh_document_if_stale(doc)
        word = doc.word_at_position(params.position)

        if ls.spec:
            spec = ls.workspace.get_text_document(ls.spec_uri)
            spec_content = get_document_content(spec)
            component_instances = ls.spec.get_component_instances_by_class_name(word)
            if component_instances:
                try:
                    position = find_component_definition_in_spec(
                        component_instances[0], word, spec_content
                    )
                except RuntimeError:
                    return None
                return types.Location(uri=ls.spec_uri, range=position)

            goal_instances = ls.spec.get_goal_instances_by_class_name(word)
            if goal_instances:
                try:
                    position = find_goal_definition_in_spec(
                        goal_instances[0], word, spec_content
                    )
                except RuntimeError:
                    return None
                return types.Location(uri=ls.spec_uri, range=position)

            connection_instances = ls.spec.get_connection_instances_by_class_name(word)
            if connection_instances:
                try:
                    position = find_connection_definition_in_spec(
                        connection_instances[0], word, spec_content
                    )
                except RuntimeError:
                    return None
                return types.Location(uri=ls.spec_uri, range=position)

    return None


def completions(ls: AICONLanguageServer, params: types.CompletionParams):
    document = ls.workspace.get_text_document(params.text_document.uri)
    if is_python_file(params.text_document.uri):
        ls.refresh_document_if_stale(document)
    lines = get_document_content(document).splitlines()

    if params.position.line >= len(lines):
        return None

    line = lines[params.position.line]
    is_quantity_line = is_quantity_completion_line(line, params.position.character)
    is_connection_line = is_connection_completion_line(line, params.position.character)
    is_connection_member_line = is_connection_member_completion_line(
        line, params.position.character
    )
    is_connected_quantities_line = is_connected_quantities_completion_line(
        line, params.position.character
    )
    is_self_connected_quantities_line = is_self_connected_quantities_completion_line(
        line, params.position.character
    )
    if (
        not is_quantity_line
        and not is_connection_line
        and not is_connection_member_line
        and not is_connected_quantities_line
        and not is_self_connected_quantities_line
    ):
        return None

    if is_self_connected_quantities_line:
        connection = ls.get_connection_for_position(document, params.position)
        if connection is None:
            ls.parse(document)
            connection = ls.get_connection_for_position(document, params.position)
        if connection is None:
            return None
        return [
            types.CompletionItem(
                label=name,
                kind=types.CompletionItemKind.Reference,
            )
            for name in connection.quantities
        ]

    component = ls.get_component_for_position(document, params.position)
    if component is None:
        ls.parse(document)
        component = ls.get_component_for_position(document, params.position)

    if component is None:
        return None

    component_name = next(
        (
            name
            for name, parsed_component in ls.components.items()
            if parsed_component is component
        ),
        None,
    )
    if component_name is None:
        return None
    component_instance_name = _resolve_component_instance_name_for_completion(
        ls, component_name
    )

    if is_quantity_line:
        completion_labels = list(component.quantities.keys())

        return [
            types.CompletionItem(
                label=name,
                kind=types.CompletionItemKind.Reference,
            )
            for name in completion_labels
        ]

    if is_connection_line:
        completion_labels = ls.get_connection_names_for_component(
            component_instance_name
        )
        return [
            types.CompletionItem(
                label=name,
                kind=types.CompletionItemKind.Reference,
            )
            for name in completion_labels
        ]

    if is_connection_member_line:
        connection_name = get_connection_name_from_member_completion(
            line, params.position.character
        )
        if connection_name is None:
            return None
        if connection_name not in ls.get_connection_names_for_component(
            component_instance_name
        ):
            return None
        return [
            types.CompletionItem(
                label="connected_quantities[]",
                insert_text='connected_quantities[""]',
                kind=types.CompletionItemKind.Property,
            )
        ]

    connection_name = get_connection_name_from_connected_quantities_completion(
        line, params.position.character
    )
    if connection_name is None:
        return None
    if connection_name not in ls.get_connection_names_for_component(
        component_instance_name
    ):
        return None

    completion_labels = ls.get_connection_quantities_for_component(
        component_instance_name, connection_name
    )
    return [
        types.CompletionItem(
            label=name,
            kind=types.CompletionItemKind.Reference,
        )
        for name in completion_labels
    ]


async def generate_files(ls: AICONLanguageServer):
    await ls.generate_files()


def generate_experiment_specification(ls: AICONLanguageServer):
    ls.generate_experiment_specification()


def revalidate_generated_component(ls: AICONLanguageServer, *args: object):
    target_uri = _extract_target_uri(args)

    if isinstance(target_uri, str):
        target_document = ls.workspace.get_text_document(target_uri)
        ls.parse(target_document)

    ls.validate_spec()


def fix_all_spec_mismatches(ls: AICONLanguageServer, *args: object):
    target_uri = _extract_target_uri(args)

    if isinstance(target_uri, str):
        ls.fix_all_spec_mismatches(target_uri)
        return

    ls.fix_all_spec_mismatches()


def run_dynamic_check(ls: AICONLanguageServer):
    ls.run_dynamic_check()


def rename_quantity(ls: AICONLanguageServer, params: types.RenameParams):
    if is_python_file(params.text_document.uri):
        document = ls.workspace.get_text_document(params.text_document.uri)
        ls.refresh_document_if_stale(document)

    return ls.build_quantity_rename_edit(
        params.text_document.uri,
        params.position,
        params.new_name,
    )


def initialize(ls: AICONLanguageServer, _: types.InitializedParams):
    if not ls.is_project_root_available():
        ls.notify_project_root_unavailable()
        return
    ls.update_spec()
    if ls.spec is None:
        return
    # Scan the whole project so that the spec consistency check (W011-W013) does not
    # report entities as missing that are implemented in files not opened yet.
    ls.parse_all_project_python_files()
    ls.validate_spec()


def _build_ignore_warning_action(
    document_uri: str,
    source: str,
    diagnostic: types.Diagnostic,
) -> types.CodeAction | None:
    line_index = diagnostic.range.start.line
    lines = source.splitlines()
    if line_index < 0 or line_index >= len(lines):
        return None

    line = lines[line_index]
    code = str(diagnostic.code) if diagnostic.code is not None else None
    code_suffix = f" {code}" if code else ""
    range_start = types.Position(line=line_index, character=len(line))
    edit = types.TextEdit(
        range=types.Range(start=range_start, end=range_start),
        new_text=f" # aiconls: ignore-warning{code_suffix}",
    )

    title_suffix = f" ({code})" if code else ""
    return types.CodeAction(
        title=f"Ignore this warning{title_suffix}",
        kind=types.CodeActionKind.QuickFix,
        diagnostics=[diagnostic],
        edit=types.WorkspaceEdit(changes={document_uri: [edit]}),
    )


def code_action_provider(ls: AICONLanguageServer, params: types.CodeActionParams):
    if is_python_file(params.text_document.uri):
        document = ls.workspace.get_text_document(params.text_document.uri)
        ls.refresh_document_if_stale(document)

    code_actions: list[types.CodeAction] = []
    document = ls.workspace.get_text_document(params.text_document.uri)
    source = get_document_content(document)

    diagnostic = get_diagnostic_from_Code_Action(
        params, ErrorCode.COMPONENT_NOT_IMPLEMENTED.value.code
    )
    if diagnostic is not None:
        component_name = document.source[
            document.offset_at_position(
                diagnostic.range.start
            ) : document.offset_at_position(diagnostic.range.end)
        ]
        component_name = _resolve_component_name_for_generation(ls, component_name)
        code_action = ls.generate_component(component_name, diagnostic)
        if code_action is not None:
            code_actions.append(code_action)

    diagnostic = get_diagnostic_from_Code_Action(
        params, ErrorCode.GOAL_NOT_IMPLEMENTED.value.code
    )
    if diagnostic is not None:
        goal_name = document.source[
            document.offset_at_position(
                diagnostic.range.start
            ) : document.offset_at_position(diagnostic.range.end)
        ]
        goal_name = _resolve_goal_name_for_generation(ls, goal_name)
        code_action = ls.generate_goal(goal_name, diagnostic)
        if code_action is not None:
            code_actions.append(code_action)

    diagnostic = get_diagnostic_from_Code_Action(
        params, ErrorCode.CONNECTION_NOT_IMPLEMENTED.value.code
    )
    if diagnostic is not None:
        connection_name = document.source[
            document.offset_at_position(
                diagnostic.range.start
            ) : document.offset_at_position(diagnostic.range.end)
        ]
        connection_name = _resolve_connection_name_for_generation(ls, connection_name)
        code_action = ls.generate_connection(connection_name, diagnostic)
        if code_action is not None:
            code_actions.append(code_action)

    missing_quantities_diagnostic = get_diagnostic_from_Code_Action(
        params, ErrorCode.QUANTITIES_NOT_IMPLEMENTED.value.code
    )
    if missing_quantities_diagnostic is not None:
        code_action = ls.generate_component_update(
            params.text_document.uri, missing_quantities_diagnostic
        )
        if code_action is not None:
            code_actions.append(code_action)

    mismatch_diagnostic = get_diagnostic_from_Code_Action(
        params, ErrorCode.SHAPE_SPEC_MISMATCH.value.code
    )
    if mismatch_diagnostic is not None:
        code_action = ls.generate_component_update(
            params.text_document.uri, mismatch_diagnostic
        )
        if code_action is not None:
            code_actions.append(code_action)
        spec_only_action = ls.generate_spec_from_component_update(
            params.text_document.uri, mismatch_diagnostic
        )
        if spec_only_action is not None:
            code_actions.append(spec_only_action)
        propagate_action = ls.generate_spec_from_component_update_and_propagate(
            params.text_document.uri, mismatch_diagnostic
        )
        if propagate_action is not None:
            code_actions.append(propagate_action)

    connection_mismatch_diagnostic = get_diagnostic_from_Code_Action(
        params, ErrorCode.QUANTITY_NOT_AVAILABLE.value.code
    )
    if connection_mismatch_diagnostic is not None:
        code_action = ls.generate_connection_update(
            params.text_document.uri, connection_mismatch_diagnostic
        )
        if code_action is not None:
            code_actions.append(code_action)

    build_connections_codes = {
        ErrorCode.BUILD_CONNECTIONS_FUNCTION_MISSING.value.code,
        ErrorCode.BUILD_CONNECTIONS_ENTRY_MISSING.value.code,
        ErrorCode.BUILD_CONNECTIONS_CLASS_MISMATCH.value.code,
        ErrorCode.BUILD_CONNECTIONS_NAME_ARGUMENT_MISMATCH.value.code,
    }
    build_connections_diagnostic = next(
        (
            diagnostic
            for diagnostic in params.context.diagnostics
            if str(diagnostic.code) in build_connections_codes
        ),
        None,
    )
    if build_connections_diagnostic is not None:
        code_action = ls.generate_build_connections_update(
            params.text_document.uri, build_connections_diagnostic
        )
        if code_action is not None:
            code_actions.append(code_action)

    if mismatch_diagnostic is not None or connection_mismatch_diagnostic is not None:
        code_actions.append(
            types.CodeAction(
                title="Fix all spec mismatches",
                kind=types.CodeActionKind.SourceFixAll,
                command=types.Command(
                    title="Fix all spec mismatches",
                    command=AICONLanguageServer.FIX_ALL_SPEC_MISMATCHES_COMMAND,
                    arguments=[params.text_document.uri],
                ),
            )
        )

    for diagnostic in params.context.diagnostics:
        if diagnostic.severity != types.DiagnosticSeverity.Warning:
            continue
        ignore_action = _build_ignore_warning_action(
            document_uri=params.text_document.uri,
            source=source,
            diagnostic=diagnostic,
        )
        if ignore_action is not None:
            code_actions.append(ignore_action)

    if len(code_actions) == 0:
        return None

    return code_actions


def register_handlers(server: AICONLanguageServer) -> None:
    server.feature(types.TEXT_DOCUMENT_DID_OPEN)(log_lsp_exceptions(did_open))
    server.feature(types.INITIALIZED)(log_lsp_exceptions(initialize))
    server.feature(types.TEXT_DOCUMENT_DID_CHANGE)(log_lsp_exceptions(did_change))
    server.feature(types.TEXT_DOCUMENT_IMPLEMENTATION)(
        log_lsp_exceptions(goto_implementation)
    )
    server.feature(types.TEXT_DOCUMENT_DEFINITION)(log_lsp_exceptions(goto_definition))
    server.feature(types.TEXT_DOCUMENT_RENAME)(log_lsp_exceptions(rename_quantity))
    server.feature(
        types.TEXT_DOCUMENT_COMPLETION,
        types.CompletionOptions(trigger_characters=['"', "."]),
    )(log_lsp_exceptions(completions))
    server.command(AICONLanguageServer.GENERATE_FILES_COMMAND)(
        log_lsp_exceptions(generate_files)
    )
    server.command(AICONLanguageServer.GENERATE_EXPERIMENT_SPECIFICATION_COMMAND)(
        log_lsp_exceptions(generate_experiment_specification)
    )
    server.command(AICONLanguageServer.REVALIDATE_GENERATED_COMPONENT_COMMAND)(
        log_lsp_exceptions(revalidate_generated_component)
    )
    server.command(AICONLanguageServer.FIX_ALL_SPEC_MISMATCHES_COMMAND)(
        log_lsp_exceptions(fix_all_spec_mismatches)
    )
    server.command(AICONLanguageServer.RUN_DYNAMIC_CHECK_COMMAND)(
        log_lsp_exceptions(run_dynamic_check)
    )
    server.feature(
        types.TEXT_DOCUMENT_CODE_ACTION,
        types.CodeActionOptions(
            code_action_kinds=[
                types.CodeActionKind.QuickFix,
                types.CodeActionKind.SourceFixAll,
            ],
        ),
    )(code_action_provider)
