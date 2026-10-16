"""
Graph Specifications (spec.yaml)

Writes the graph of an AICON system - its components with their quantities, the connections between them and the
goals - as a `spec.yaml`, the project specification that the AICON linter (aiconls) checks the code against. The
graph is read from the component builders of an experiment specification, built in mock mode, so no hardware or
simulation is needed.

Quantities are written as `Generic` (a tensor of a fixed shape), except for Gaussian estimates: a vector of shape (n,)
is a `Gaussian` with its covariance when the component has exactly one quantity of shape (n, n) with "uncertainty" in
its name and no other vector of shape (n,). The covariance is then not listed itself; the linter adds it.

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import json
import re
from typing import Callable, Dict, Iterable, List, Tuple

from aicon_base.components import ActionComponent, EstimationComponent, SensorComponent

COMPONENT_TYPES = ((SensorComponent, "Sensor"), (ActionComponent, "Action"), (EstimationComponent, "Estimator"))


def graph_spec(component_builders: Dict[str, Callable], goal_variants: Iterable[Dict[str, Callable]] = ()) -> Dict:
    """The graph built by `component_builders` as the content of a spec.yaml: components, goals and connections, each
    in the order in which they first appear. `goal_variants` are component builders of the same graph with other
    goals (e.g. one per goal an experiment can be run with); their goals are listed too, but not attached. Components
    that are neither sensors, actions nor estimators (e.g. visualizers) have no type in a spec; they are only named
    under "omitted"."""
    components, goals, connections, omitted = {}, {}, {}, {}
    for name, build in component_builders.items():
        component = build(True)
        component_type = next((t for cls, t in COMPONENT_TYPES if isinstance(component, cls)), None)
        if component_type is None:
            omitted[name] = type(component).__name__
            continue
        components[name] = {
            "component_type": component_type,
            "class_name": type(component).__name__,
            "dtype": str(component.dtype).replace("torch.", ""),
            "goals": list(component.goals),
            "quantities": _quantity_specs(dict(component.local_quantity_names_and_shapes)),
        }
        for goal_name, goal in component.goals.items():
            goals[goal_name] = type(goal).__name__
        for connection_name, connection in component.connections.items():
            entry = connections.setdefault(connection_name, {"class_name": type(connection).__name__, "components": []})
            entry["components"].append(name)
    for builders in goal_variants:
        for build in builders.values():
            for goal_name, goal in build(True).goals.items():
                goals.setdefault(goal_name, type(goal).__name__)
    return {"components": components, "goals": goals, "connections": connections, "omitted": omitted}


def _quantity_specs(shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Dict]:
    """The quantities of one component as spec entries ({"shape": ...} or {"q_type": "Gaussian", ...}), see above."""
    shapes = {q: tuple(int(s) for s in shape) for q, shape in shapes.items()}
    covariances = {}
    for q, shape in shapes.items():
        if len(shape) != 1:
            continue
        vectors = [v for v, s in shapes.items() if s == shape]
        matrices = [m for m, s in shapes.items() if "uncertainty" in m and s == shape * 2]
        if len(vectors) == 1 and len(matrices) == 1:
            covariances[q] = matrices[0]
    specs = {}
    for q, shape in shapes.items():
        if q in covariances.values():
            continue
        if q in covariances:
            specs[q] = {"q_type": "Gaussian", "dim": shape[0], "uncertainty_name": covariances[q]}
        else:
            specs[q] = {"shape": shape}
    return specs


def _name(name: str) -> str:
    """A key or list item as YAML: plain if it is an identifier, quoted otherwise (e.g. "E[dist]")."""
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else json.dumps(name)


def _shape(shape: Tuple[int, ...]) -> str:
    shape = tuple(int(s) for s in shape)
    return "(" + ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "") + ")"


def spec_yaml(spec: Dict, header: Iterable[str] = ()) -> str:
    """`spec` (from `graph_spec`) as spec.yaml text, preceded by `header` as comment lines."""
    lines: List[str] = ["# " + h if h else "#" for h in header]
    if spec.get("omitted"):
        lines.append(
            "# Not listed, as a spec has no component type for them: "
            + ", ".join(f"{n} ({c})" for n, c in spec["omitted"].items())
            + "."
        )
    lines.append("components:")
    for name, c in spec["components"].items():
        lines += [
            f"  {_name(name)}:",
            f"    component_type: {c['component_type']}",
            f"    class_name: {c['class_name']}",
            f"    dtype: {c['dtype']}",
        ]
        if c["goals"]:
            lines.append("    goals: [" + ", ".join(_name(g) for g in c["goals"]) + "]")
        if c["quantities"]:
            lines.append("    quantities:")
            for q, q_spec in c["quantities"].items():
                lines.append(f"      {_name(q)}:")
                if "shape" in q_spec:
                    lines.append(f"        shape: {_shape(q_spec['shape'])}")
                else:
                    lines += [
                        f"        q_type: {q_spec['q_type']}",
                        f"        dim: {q_spec['dim']}",
                        f"        uncertainty_name: {_name(q_spec['uncertainty_name'])}",
                    ]
    if spec["goals"]:
        lines += ["", "goals:"]
        for name, class_name in spec["goals"].items():
            lines += [f"  {_name(name)}:", f"    class_name: {class_name}"]
    if spec["connections"]:
        lines += ["", "connections:"]
        for name, c in spec["connections"].items():
            lines += [
                f"  {_name(name)}:",
                f"    class_name: {c['class_name']}",
                "    components: [" + ", ".join(_name(n) for n in c["components"]) + "]",
            ]
    return "\n".join(lines) + "\n"
