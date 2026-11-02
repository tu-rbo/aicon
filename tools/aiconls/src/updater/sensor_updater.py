from reader.spec.spec_types import SpecComponentProps
from updater.component_updater import ComponentUpdater


class SensorUpdater(ComponentUpdater):
    def __init__(self, component_name: str, spec_component: SpecComponentProps):
        super().__init__(component_name=component_name, spec_component=spec_component)
