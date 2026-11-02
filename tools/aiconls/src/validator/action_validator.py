from lsprotocol.types import Diagnostic

from reader.aicon_files.reader_types import ParsedAction
from reader.spec.spec import AICON_SPEC
from validator.component_validator import ComponentValidator


class ActionValidator(ComponentValidator):
    diagnostics: list[Diagnostic]

    def __init__(self, spec: AICON_SPEC, parsed_action: ParsedAction) -> None:
        super().__init__(spec, parsed_action)
