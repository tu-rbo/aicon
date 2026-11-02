from reader.spec.spec_types import SpecComponentTypes


def get_file_name_from_component_type(component_type: SpecComponentTypes):
    if component_type == SpecComponentTypes.Sensor:
        return "sensors.py"
    elif component_type == SpecComponentTypes.Action:
        return "actions.py"
    elif component_type == SpecComponentTypes.Estimator:
        return "estimators.py"
    elif component_type == SpecComponentTypes.Goal:
        return "goals.py"

    raise RuntimeError("Component type doesnt exist %s", component_type)
