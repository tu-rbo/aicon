from lsprotocol.types import Diagnostic

from reader.aicon_files.reader_types import ParsedSensor
from reader.spec.spec import AICON_SPEC
from validator.component_validator import ComponentValidator


class SensorValidator(ComponentValidator):
    diagnostics: list[Diagnostic]

    def __init__(self, spec: AICON_SPEC, parsed_sensor: ParsedSensor) -> None:
        super().__init__(spec, parsed_sensor)
