import pathlib
from typing import Any

from lsprotocol import types

from generator.action import gen_action_head
from generator.estimator import gen_estimator_head
from generator.goal import gen_goal_head
from generator.sensor import gen_sensor_head
from lsp_utils import DiagnosticSink, get_document_content
from reader.spec.spec_types import SpecComponentTypes
from updater.action_updater import ActionUpdater
from updater.estimator_updater import EstimatorUpdater
from updater.sensor_updater import SensorUpdater


def spec_has_errors(diagnostic_sink: DiagnosticSink, spec_uri: str) -> bool:
    return any(
        diagnostic.severity == types.DiagnosticSeverity.Error
        for diagnostic in diagnostic_sink.diagnostics.get(spec_uri, (0, []))[1]
    )


def notify_spec_malformed(ls: Any) -> None:
    ls.window_log_message(
        types.LogMessageParams(message="Spec malformed", type=types.MessageType.Error)
    )
    ls.window_show_message(
        types.ShowMessageParams(message="Spec malformed", type=types.MessageType.Error)
    )


def get_component_header(component_type: SpecComponentTypes) -> str:
    if component_type == SpecComponentTypes.Sensor:
        return gen_sensor_head()
    if component_type == SpecComponentTypes.Estimator:
        return gen_estimator_head()
    if component_type == SpecComponentTypes.Action:
        return gen_action_head()
    if component_type == SpecComponentTypes.Goal:
        return gen_goal_head()
    raise RuntimeError(f"No header generator for component type: {component_type}")


def build_generation_workspace_edit(
    ls: Any, target_file: pathlib.Path, generated_content: str, header: str
) -> tuple[types.WorkspaceEdit, str]:
    # sometimes the document is not saved to disk but open in the editor
    workspace_document = ls.workspace.get_text_document(target_file.as_uri())
    try:
        get_document_content(workspace_document)
    except FileNotFoundError:
        pass

    target_file_exists = target_file.exists()
    target_file_uri = target_file.as_uri()

    if target_file_exists:
        target_file_content = target_file.read_text()
        target_line = target_file_content.count("\n")
        target_position = types.Position(line=target_line, character=0)
        new_text = generated_content
    else:
        target_position = types.Position(line=0, character=0)
        new_text = header + generated_content

    text_edit = types.TextEdit(
        range=types.Range(start=target_position, end=target_position),
        new_text=new_text,
    )

    if target_file_exists:
        workspace_edit = types.WorkspaceEdit(changes={target_file_uri: [text_edit]})
    else:
        workspace_edit = types.WorkspaceEdit(
            document_changes=[
                types.CreateFile(uri=target_file_uri),
                types.TextDocumentEdit(
                    text_document=types.OptionalVersionedTextDocumentIdentifier(
                        uri=target_file_uri,
                        version=None,
                    ),
                    edits=[text_edit],
                ),
            ]
        )

    return workspace_edit, target_file_uri


def build_full_document_range(source: str) -> types.Range:
    lines = source.split("\n")
    return types.Range(
        start=types.Position(line=0, character=0),
        end=types.Position(line=len(lines) - 1, character=len(lines[-1])),
    )


def create_component_updater(
    component_type: SpecComponentTypes, component_name: str, spec_component: Any
) -> SensorUpdater | ActionUpdater | EstimatorUpdater | None:
    if component_type == SpecComponentTypes.Sensor:
        return SensorUpdater(component_name, spec_component)
    if component_type == SpecComponentTypes.Action:
        return ActionUpdater(component_name, spec_component)
    if component_type == SpecComponentTypes.Estimator:
        return EstimatorUpdater(component_name, spec_component)
    return None
