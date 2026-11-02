from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias


class SpecComponentTypes(Enum):
    Sensor = 1
    Action = 2
    Estimator = 3
    Goal = 4


class SpecQuantityTypes(Enum):
    Generic = 1
    Gaussian = 2
    Categorical = 3
    Particle = 4
    Reference = 5


TensorShape = tuple[int, ...]

# marks a dimension whose size is only known at runtime, e.g. a variable number of points
DYNAMIC_DIMENSION = -1


def is_dynamic_shape(shape: TensorShape) -> bool:
    return DYNAMIC_DIMENSION in shape


def shapes_match(actual: TensorShape, expected: TensorShape) -> bool:
    """Shapes containing a dynamic dimension are excluded from shape checks."""
    if is_dynamic_shape(actual) or is_dynamic_shape(expected):
        return True
    return actual == expected


@dataclass
class SpecComponentQuantityGeneric:
    shape: TensorShape
    q_type: Literal[SpecQuantityTypes.Generic] = SpecQuantityTypes.Generic


@dataclass
class SpecComponentQuantityGaussian:
    dim: int
    shape: TensorShape
    uncertainty_name: str
    q_type: Literal[SpecQuantityTypes.Gaussian] = SpecQuantityTypes.Gaussian


@dataclass
class SpecComponentQuantityCategorical:
    shape: TensorShape
    dim: int
    q_type: Literal[SpecQuantityTypes.Categorical] = SpecQuantityTypes.Categorical


@dataclass
class SpecComponentQuantityParticle:
    dim: int
    shape: TensorShape
    particle_count: int
    particle_weights_name: str
    q_type: Literal[SpecQuantityTypes.Particle] = SpecQuantityTypes.Particle


@dataclass
class SpecComponentQuantityReference:
    shape: TensorShape
    referenced_quantity_name: str
    q_type: Literal[SpecQuantityTypes.Reference] = SpecQuantityTypes.Reference


SpecComponentQuantity = (
    SpecComponentQuantityGeneric
    | SpecComponentQuantityGaussian
    | SpecComponentQuantityParticle
    | SpecComponentQuantityCategorical
    | SpecComponentQuantityReference
)

SpecComponentQuantityType: TypeAlias = SpecComponentQuantity


SpecQuantities = dict[str, SpecComponentQuantity]


@dataclass
class SpecComponentProps:
    component_type: SpecComponentTypes
    quantities: SpecQuantities
    goals: list[str]
    instance_name: str
    class_name: str
    # normalized torch dtype name, e.g. "float64"; None uses the framework default
    dtype: str | None = None


@dataclass
class SpecConnectionProps:
    components: list[str]
    instance_name: str = ""
    class_name: str = ""


@dataclass
class SpecGoalProps:
    name: str
    instance_name: str = ""

    @property
    def class_name(self) -> str:
        return self.name


@dataclass
class SpecFormat:
    components: dict[str, SpecComponentProps]
    goals: dict[str, SpecGoalProps]
    connections: dict[str, SpecConnectionProps]


SpecComponents = dict[str, SpecComponentProps]
SpecGoals = dict[str, SpecGoalProps]
SpecConnections = dict[str, SpecConnectionProps]
