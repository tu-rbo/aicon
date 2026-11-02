import libcst as cst
from lsprotocol.types import Range

from reader.aicon_files.extractor.component import ComponentExtractor
from reader.aicon_files.reader_types import ParsedSensor


class SensorExtractor(ComponentExtractor[ParsedSensor]):
    def __init__(
        self,
        uri: str,
        name: str,
        name_pos: Range,
        metadata_owner: cst.CSTVisitor,
    ) -> None:
        super().__init__(
            uri=uri,
            name=name,
            name_pos=name_pos,
            parsed_component_type=ParsedSensor,
            metadata_owner=metadata_owner,
        )
