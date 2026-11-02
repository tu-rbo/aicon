from generator.action import gen_action
from generator.estimator import gen_estimator
from generator.goal import gen_goal
from generator.sensor import gen_sensor
from reader.spec.spec_types import SpecComponentProps, SpecComponentTypes


def generate_general_component(
    component_name: str, component_props: SpecComponentProps
):
    if component_props.component_type == SpecComponentTypes.Sensor:
        return gen_sensor(component_name, component_props)
    elif component_props.component_type == SpecComponentTypes.Estimator:
        return gen_estimator(component_name, component_props)
    elif component_props.component_type == SpecComponentTypes.Goal:
        return gen_goal(component_name)
    elif component_props.component_type == SpecComponentTypes.Action:
        return gen_action(component_name, component_props)

    raise RuntimeError(
        f"No generator for {component_props.component_type} - {SpecComponentTypes.Estimator.value}"
    )
