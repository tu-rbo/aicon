from types import SimpleNamespace
from typing import cast

from lsprotocol import types

from handlers import goto_definition, goto_implementation
from reader.spec.spec import AICON_SPEC
from reader.spec.spec_types import (
    SpecComponentProps,
    SpecComponentTypes,
    SpecConnectionProps,
    SpecFormat,
    SpecGoalProps,
)
from server import AICONLanguageServer


class _DummyDocument:
    def __init__(self, uri: str, source: str, word: str | None = None):
        self.uri = uri
        self.source = source
        self._word = word

    def word_at_position(self, _position: types.Position) -> str:
        if self._word is None:
            raise RuntimeError("No word configured for document")
        return self._word


class _DummyWorkspace:
    def __init__(self, documents: dict[str, _DummyDocument]):
        self.documents = documents

    def get_text_document(self, uri: str) -> _DummyDocument:
        return self.documents[uri]


class _DummyLanguageServer:
    def __init__(self, python_doc: _DummyDocument, spec_doc: _DummyDocument):
        self.workspace = _DummyWorkspace(
            {
                python_doc.uri: python_doc,
                spec_doc.uri: spec_doc,
            }
        )
        self.spec_uri = spec_doc.uri
        self.spec = AICON_SPEC(
            SpecFormat(
                components={
                    "ee_pose_sensor": SpecComponentProps(
                        component_type=SpecComponentTypes.Sensor,
                        quantities={},
                        goals=[],
                        instance_name="ee_pose_sensor",
                        class_name="EEPoseSensor",
                    )
                },
                goals={
                    "drawer_goal": SpecGoalProps(
                        name="DrawerGoal",
                        instance_name="drawer_goal",
                    )
                },
                connections={
                    "control_loop": SpecConnectionProps(
                        components=["ee_pose_sensor"],
                        instance_name="control_loop",
                        class_name="ControlLoop",
                    )
                },
            )
        )
        self.components = {
            "EEPoseSensor": SimpleNamespace(
                document_uri=python_doc.uri,
                position=types.Range(
                    start=types.Position(line=2, character=0),
                    end=types.Position(line=2, character=12),
                ),
            )
        }
        self.goals = {
            "DrawerGoal": SimpleNamespace(
                document_uri=python_doc.uri,
                position=types.Range(
                    start=types.Position(line=5, character=0),
                    end=types.Position(line=5, character=10),
                ),
            )
        }
        self.connections = {
            "ControlLoop": SimpleNamespace(
                document_uri=python_doc.uri,
                position=types.Range(
                    start=types.Position(line=8, character=0),
                    end=types.Position(line=8, character=11),
                ),
            )
        }

    def refresh_document_if_stale(self, _document: _DummyDocument) -> None:
        return None


def _definition_params(uri: str) -> types.DefinitionParams:
    return types.DefinitionParams(
        text_document=types.TextDocumentIdentifier(uri=uri),
        position=types.Position(line=0, character=0),
    )


def test_goto_definition_resolves_component_in_components_section() -> None:
    python_uri = "file:///project/component.py"
    spec_uri = "file:///project/spec.yaml"
    python_doc = _DummyDocument(python_uri, "EEPoseSensor", word="EEPoseSensor")
    spec_doc = _DummyDocument(
        spec_uri,
        """
components:
  ee_pose_sensor:
    class_name: EEPoseSensor
    component_type: Sensor
""".strip(),
    )
    ls = _DummyLanguageServer(python_doc, spec_doc)

    location = goto_definition(
        cast(AICONLanguageServer, ls),
        _definition_params(python_uri),
    )

    assert location is not None
    assert location.uri == spec_uri
    assert location.range.start.line == 2
    assert location.range.start.character == 16


def test_goto_definition_returns_none_when_component_cannot_be_located_in_spec() -> (
    None
):
    python_uri = "file:///project/component.py"
    spec_uri = "file:///project/spec.yaml"
    python_doc = _DummyDocument(python_uri, "EEPoseSensor", word="EEPoseSensor")
    spec_doc = _DummyDocument(
        spec_uri,
        """
connections:
  DistGraspHandConnection:
    components: [GripperAction, EEForceSensor]
""".strip(),
    )
    ls = _DummyLanguageServer(python_doc, spec_doc)

    location = goto_definition(
        cast(AICONLanguageServer, ls),
        _definition_params(python_uri),
    )

    assert location is None


def test_goto_implementation_resolves_component_instance_from_spec() -> None:
    python_uri = "file:///project/component.py"
    spec_uri = "file:///project/spec.yaml"
    python_doc = _DummyDocument(python_uri, "EEPoseSensor", word="EEPoseSensor")
    spec_doc = _DummyDocument(
        spec_uri,
        "components:\n  ee_pose_sensor:\n    class_name: EEPoseSensor",
        word="ee_pose_sensor",
    )
    ls = _DummyLanguageServer(python_doc, spec_doc)

    location = goto_implementation(
        cast(AICONLanguageServer, ls),
        types.ImplementationParams(
            text_document=types.TextDocumentIdentifier(uri=spec_uri),
            position=types.Position(line=1, character=4),
        ),
    )

    assert location is not None
    assert location.uri == python_uri
    assert location.range.start.line == 2
