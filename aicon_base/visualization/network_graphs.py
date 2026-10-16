"""
Network Graph Visualization Functions for AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import colorsys
from typing import Callable, Dict, List, Tuple, Union, Iterable

import matplotlib.axis
import matplotlib.legend_handler
import matplotlib.patches
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from aicon_base.components.base_components import Component

from aicon_base.components.action_components import ActionComponent
from aicon_base.components.sensor_components import SensorComponent
from aicon_base.components.estimation_components import EstimationComponent
from aicon_base.connections.connections import ActiveInterconnection

# Trace marker of a nullspace escape
CR_MARKER = "ConflictResolution"

types_and_colors = {
    "Real-World Sensors and Actuators": "#e5c890",
    "Recursive Estimator": "#8caaee",
    "subnode": "#DAF3FB",
    "Action": "red",
    "Sensor": "yellow",
    "Estimator": "blue",
    "Active Interconnection": "#a6d189",
    "Specified Goal": "#e78284",
}

# Structural (non-gradient) edges between boxes
EDGE_COLOR = "#dce0e8"
EDGE_WIDTH = 2.0

# Label colour per box type; dark by default because all box colours are light pastels
DEFAULT_FONT_COLOR = "#232634"
types_and_font_colors = {}


def font_color_for(node_type: str) -> str:
    return types_and_font_colors.get(node_type, DEFAULT_FONT_COLOR)


def get_basic_network_graph(
    component_builders: Dict[str, Callable], connection_builders: Dict[str, Callable]
) -> Tuple[nx.Graph, np.array, List[str]]:
    action_quantity_names, connections, goals, node_types, related_quantities = obtain_code_based_structure(
        component_builders, connection_builders
    )

    G = nx.Graph()
    for k, v in related_quantities.items():
        for n in v:
            if node_types[k] is not None:
                G.add_node(n, type=node_types[k])
        for n1 in v:
            for n2 in v:
                if n1 != n2:
                    G.add_edge(n1, n2, visible=False)
        for g in goals[k]:
            G.add_node(g, type="Specified Goal")
            for n in v:
                G.add_edge(g, n, visible=True)
    for connection, nodes in connections.items():
        if node_types[connection] is not None:
            G.add_node(connection, type=node_types[connection])
        for n in nodes:
            G.add_edge(connection, n, visible=True)

    pos = nx.kamada_kawai_layout(G)

    return G, pos, action_quantity_names


def get_fancy_network_graph(
    component_builders: Dict[str, Callable],
    connection_builders: Dict[str, Callable],
    name_remapping: Dict[str, str],
    simple_combinations: Dict[str, str],
    recursive_estimators: Dict[str, str],
    type_combinations=Dict[str, str],
    summarized_recursive_estimators=List[str],
    cut_edges=List[Tuple[str]],
    cut_nodes=List[str],
    node_name_to_layer: Union[Dict[str, str], None] = None,
) -> Tuple[nx.Graph, np.array, List[str], Dict[str, str]]:
    action_quantity_names, connections, goals, node_types, related_quantities = obtain_code_based_structure(
        component_builders, connection_builders
    )

    G = nx.Graph()
    later_remappings = dict()
    for k, v in related_quantities.items():
        for n in v:
            display_type = node_types[k]
            G, later_remappings = add_node_with_remappings(
                n,
                G,
                display_type,
                later_remappings,
                name_remapping,
                recursive_estimators,
                simple_combinations,
                type_combinations,
                node_name_to_layer,
                summarized_recursive_estimators,
            )
        for n1 in v:
            for n2 in v:
                if n1 != n2:
                    n1_remapped = n1
                    n2_remapped = n2
                    if n1 in later_remappings:
                        n1_remapped = later_remappings[n1]
                    if n2 in later_remappings:
                        n2_remapped = later_remappings[n2]
                    G.add_edge(n1_remapped, n2_remapped, visible=False)
        for g in goals[k]:
            display_type = "Specified Goal"
            G, later_remappings = add_node_with_remappings(
                g,
                G,
                display_type,
                later_remappings,
                name_remapping,
                recursive_estimators,
                simple_combinations,
                type_combinations,
                node_name_to_layer,
                summarized_recursive_estimators,
            )
            g_remapped = g
            if g in later_remappings:
                g_remapped = later_remappings[g]
            for n in v:
                n_remapped = n
                if n in later_remappings:
                    n_remapped = later_remappings[n]
                G.add_edge(g_remapped, n_remapped, visible=True)
    for connection, nodes in connections.items():
        display_type = node_types[connection]
        remapped_nodes = []
        for n in nodes:
            n_remapped = n
            if n in later_remappings:
                n_remapped = later_remappings[n]
            if n_remapped in G.nodes:
                remapped_nodes.append(n_remapped)
        if len(remapped_nodes) == 0:
            continue
        G, later_remappings = add_node_with_remappings(
            connection,
            G,
            display_type,
            later_remappings,
            name_remapping,
            recursive_estimators,
            simple_combinations,
            type_combinations,
            node_name_to_layer,
            summarized_recursive_estimators,
        )
        c_remapped = connection
        if connection in later_remappings:
            c_remapped = later_remappings[c_remapped]
        for n in remapped_nodes:
            G.add_edge(c_remapped, n, visible=True)

    for e in cut_edges:
        G.remove_edge(*e)
    edges = G.edges()
    for n in cut_nodes:
        for e in edges:
            if e[0] == n or e[1] == n:
                G.remove_edge(*e)
        G.remove_node(n)
    edges = G.edges()
    for e in edges:
        if e[0] == e[1]:
            G.remove_edge(*e)

    subnodes = [n for n, d in G.nodes(data=True) if d["type"] == "subnode"]
    added_subnodes = {}

    # layout all nodes except subnodes, then overlay subnodes on recursive nodes
    pos = nx.kamada_kawai_layout(G, scale=1.0, weight=None)

    subnode_dist = 0.075
    for n, d in G.nodes(data=True):
        if n in subnodes:
            p = d["parent_name"]
            x = pos[p][0]
            y = pos[p][1] + d["n_brothers"] * subnode_dist + 0.0
            if d["n_brothers"] == 1 and d["display_name"] == "Uncertainty":
                y -= subnode_dist * 2
            elif d["n_brothers"] == 1 and d["display_name"] == "State":
                y -= subnode_dist
            else:
                if p not in added_subnodes:
                    added_subnodes[p] = 0
                else:
                    added_subnodes[p] += 1
                y -= added_subnodes[p] * subnode_dist * 1 + 0.9 * subnode_dist
            pos[n][0] = x
            pos[n][1] = y
    return G, pos, action_quantity_names, later_remappings


def obtain_code_based_structure(component_builders, connection_builders):
    related_quantities = {}
    node_type = {}
    goals = {}
    action_quantity_names = []
    used_connections = set()
    for name, c in component_builders.items():
        comp: Component = c(mockbuild=True)
        if isinstance(comp, ActionComponent):
            node_type[name] = "Action"
            action_quantity_names.extend(list(comp.quantities.keys()))
            used_connections.update(list(comp.connections.keys()))
        elif isinstance(comp, SensorComponent):
            node_type[name] = "Sensor"
            used_connections.update(list(comp.connections.keys()))
        elif isinstance(comp, EstimationComponent):
            node_type[name] = "Estimator"
            used_connections.update(list(comp.connections.keys()))
        else:
            node_type[name] = "Other"  # the visualizer itself is such a component, it is not getting drawn
        related_quantities[name] = [k for k in comp.quantities.keys()]
        goals[name] = []
        for g in comp.goals.values():
            if g.is_active:
                goals[name].append(g.name)
    connections = {}
    for name, c in connection_builders.items():
        if name in used_connections:
            con: ActiveInterconnection = c(
                device=torch.get_default_device(), dtype=torch.get_default_dtype(), mockbuild=True
            )
            connections[name] = [k for k in con.connected_quantities.keys()]
            node_type[name] = "Active Interconnection"
    return action_quantity_names, connections, goals, node_type, related_quantities


def add_node_with_remappings(
    name,
    G,
    display_type,
    later_remappings,
    name_remapping,
    recursive_estimators,
    simple_combinations,
    type_combinations,
    node_name_to_layer,
    summarized_recursive_estimators,
):
    internal_name = name
    display_name = name
    n_subnodes = 0
    if display_type in type_combinations:
        display_type = type_combinations[display_type]
    if name in name_remapping:
        display_name = name_remapping[name]
    if name in simple_combinations:
        internal_name = simple_combinations[name]
        display_name = internal_name
        later_remappings[name] = internal_name
    if name in recursive_estimators:
        internal_name = recursive_estimators[name]
        display_name = internal_name
        later_remappings[name] = internal_name
        display_type = "Recursive Estimator"
        if internal_name not in summarized_recursive_estimators:
            n_subnodes = list(recursive_estimators.values()).count(internal_name)
            if n_subnodes > 1:
                subnode_name = name
                if name in name_remapping:
                    subnode_name = name_remapping[name]
                G.add_node(
                    name,
                    type="subnode",
                    display_name=subnode_name,
                    n_subnodes=0,
                    parent_name=internal_name,
                    n_brothers=n_subnodes - 1,
                    layer=-1,
                )
    if internal_name not in G.nodes:
        if node_name_to_layer is not None:
            layer = node_name_to_layer[display_name]
        else:
            layer = -1
        while len(display_name) < 18:
            display_name = " " + display_name + " "
        if n_subnodes < 2:
            n_subnodes = 0
        else:
            for i in range(n_subnodes):
                display_name += "\n"
            display_name += "\n\n"
        if display_type is not None:
            G.add_node(internal_name, type=display_type, display_name=display_name, n_subnodes=n_subnodes, layer=layer)
    return G, later_remappings


def measure_label_boxes(
    ax: matplotlib.axis.Axis, labels: Dict[str, str], font_size: int = 12, pad: float = 0.4
) -> Dict[str, Tuple[float, float]]:
    """(width, height) of each label's drawn box, in DATA coordinates.

    Every size here is a function of the current axis limits, because the text itself is sized
    in points and only the axes decide how many data units a point is worth. Callers must
    therefore fix the limits before measuring and keep them fixed while using the result --
    `tidy_layout` does, and re-applies them on every redraw.

    All labels are laid out in a single canvas draw: measuring them one at a time costs a full
    draw each and dominates the runtime for a graph of any size.
    """
    texts = {}
    for key, label in labels.items():
        texts[key] = ax.text(
            0.0,
            0.0,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            bbox={"boxstyle": "round,pad=%f" % pad, "fc": "white", "ec": "none", "lw": 0},
        )
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    sizes = {}
    for key, t in texts.items():
        bb = t.get_bbox_patch().get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        sizes[key] = (abs(x1 - x0), abs(y1 - y0))
        t.remove()
    return sizes


def _node_labels(G: nx.Graph) -> Dict[str, str]:
    return {n: d["display_name"] for n, d in G.nodes(data=True)}


def _pad_sides(label: str, n: int) -> str:
    """Widen a label by n spaces on both sides, keeping it centred and its line count intact.

    Padding around the whole block instead would put the trailing spaces after the last blank
    line, which turns them into an extra line and makes the box taller on every call.
    """
    return "\n".join(" " * n + line + " " * n for line in label.split("\n"))


def _groups_of(G: nx.Graph) -> Dict[str, List[str]]:
    """Nodes that must move together: a recursive estimator and the sub-quantities inside it."""
    groups = {}
    for n, d in G.nodes(data=True):
        head = d["parent_name"] if d["type"] == "subnode" else n
        groups.setdefault(head, []).append(n)
    return groups


def fit_parents_around_subnodes(
    G: nx.Graph, pos: Dict[str, Tuple[float, float]], ax: matplotlib.axis.Axis, font_size: int = 12, gap: float = 0.01
) -> None:
    """Grow each estimator box (by padding its label with blank lines) until its sub-quantity boxes fit inside it,
    and stack them centred in it, using the rendered text sizes.

    Mutates `display_name` and `pos` in place.
    """
    groups = {head: members for head, members in _groups_of(G).items() if len(members) > 1}
    if not groups:
        return

    # always rebuild from the builder's own label: this runs several times over the same graph,
    # and re-padding an already padded label grows the box a little further on every pass
    for head in groups:
        d = G.nodes[head]
        if "base_display_name" not in d:
            d["base_display_name"] = d["display_name"].rstrip("\n")
        d["display_name"] = d["base_display_name"]

    labels = _node_labels(G)
    probe = dict(labels)
    for head in groups:
        probe[(head, "line")] = labels[head] + "\n"  # what one blank line costs
        probe[(head, "space")] = _pad_sides(labels[head], 1)  # what one space each side costs
    sizes = measure_label_boxes(ax, probe, font_size=font_size)

    line_h_of = {}
    for head, members in groups.items():
        subs = [n for n in members if n != head]
        line_h = sizes[(head, "line")][1] - sizes[head][1]
        space_w = sizes[(head, "space")][0] - sizes[head][0]
        if line_h <= 0:
            continue
        line_h_of[head] = line_h
        stack_h = sum(sizes[n][1] for n in subs) + (len(subs) - 1) * gap
        # the blank lines have to open up room for the whole stack plus a margin above and below
        n_lines = int(np.ceil((stack_h + 2 * gap) / line_h))
        needed_w = max(sizes[n][0] for n in subs) + 2 * gap - sizes[head][0]
        label = labels[head]
        if needed_w > 0 and space_w > 0:
            label = _pad_sides(label, int(np.ceil(needed_w / space_w)))
        G.nodes[head]["display_name"] = label + "\n" * n_lines

    sizes = measure_label_boxes(ax, _node_labels(G), font_size=font_size)
    for head, members in groups.items():
        if head not in line_h_of:
            continue
        subs = sorted((n for n in members if n != head), key=lambda n: -pos[n][1])
        stack_h = sum(sizes[n][1] for n in subs) + (len(subs) - 1) * gap
        # the title keeps the top line of the block and the blank lines below it are the free
        # space, so that space is centred half a line below the node's own centre
        y = pos[head][1] - line_h_of[head] / 2.0 + stack_h / 2.0
        for n in subs:
            h = sizes[n][1]
            pos[n] = np.asarray([pos[head][0], y - h / 2.0])
            y -= h + gap


def separate_overlapping_boxes(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    ax: matplotlib.axis.Axis,
    font_size: int = 12,
    gap: float = 0.02,
    iterations: int = 600,
) -> None:
    """Push node boxes apart until none of them overlap, moving each estimator group as one.

    kamada_kawai places points, not boxes, so it happily puts two long labels on top of each
    other (Object 2D Keypoints and its static transformation). This is the usual box-separation
    sweep: on each pass, every overlapping pair is pushed apart along whichever axis it overlaps
    least, which keeps the displacement small and so keeps the layout the algorithm produced.
    """
    sizes = measure_label_boxes(ax, _node_labels(G), font_size=font_size)
    groups = _groups_of(G)

    centres, half, offsets = {}, {}, {}
    for head, members in groups.items():
        x0 = min(pos[n][0] - sizes[n][0] / 2.0 for n in members)
        x1 = max(pos[n][0] + sizes[n][0] / 2.0 for n in members)
        y0 = min(pos[n][1] - sizes[n][1] / 2.0 for n in members)
        y1 = max(pos[n][1] + sizes[n][1] / 2.0 for n in members)
        centres[head] = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        half[head] = np.asarray([(x1 - x0) / 2.0, (y1 - y0) / 2.0])
        offsets[head] = {n: np.asarray(pos[n], dtype=float) - centres[head] for n in members}

    heads = list(groups.keys())
    for _ in range(iterations):
        moved = False
        for i, a in enumerate(heads):
            for b in heads[i + 1 :]:
                d = centres[b] - centres[a]
                overlap = half[a] + half[b] + gap - np.abs(d)
                if overlap[0] <= 0 or overlap[1] <= 0:
                    continue
                axis = 0 if overlap[0] < overlap[1] else 1
                # a zero component would leave the pair stuck exactly on top of each other
                direction = 1.0 if d[axis] >= 0 else -1.0
                shift = overlap[axis] / 2.0 * direction
                centres[a][axis] -= shift
                centres[b][axis] += shift
                moved = True
        if not moved:
            break

    for head, members in groups.items():
        for n in members:
            pos[n] = centres[head] + offsets[head][n]


def tidy_layout(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    ax: matplotlib.axis.Axis,
    font_size: int = 12,
    gap: float = 0.02,
    passes: int = 3,
    margin: float = 0.04,
):
    """Make the graph drawable without overlaps, and return (pos, xlim, ylim) to draw it at.

    Box sizes are fixed in points but the layout is in data units, so the limits decide how big
    every box is in the space the separation works in. Each pass therefore fits the geometry at
    one fixed pair of limits and only then proposes tighter ones for the next pass.

    The returned limits are those of the last pass, widened where the content needs it (widening only shrinks the
    boxes in data units, so it cannot create overlaps). Re-apply them after each `ax.clear()`.
    """
    pos = {n: np.asarray(p, dtype=float) for n, p in pos.items()}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    xlim = (min(xs) - 0.25 * span, max(xs) + 0.25 * span)
    ylim = (min(ys) - 0.25 * span, max(ys) + 0.25 * span)

    for i in range(passes):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        fit_parents_around_subnodes(G, pos, ax, font_size=font_size, gap=gap / 2.0)
        separate_overlapping_boxes(G, pos, ax, font_size=font_size, gap=gap)

        sizes = measure_label_boxes(ax, _node_labels(G), font_size=font_size)
        x0 = min(pos[n][0] - sizes[n][0] / 2.0 for n in G.nodes)
        x1 = max(pos[n][0] + sizes[n][0] / 2.0 for n in G.nodes)
        y0 = min(pos[n][1] - sizes[n][1] / 2.0 for n in G.nodes)
        y1 = max(pos[n][1] + sizes[n][1] / 2.0 for n in G.nodes)
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        if i < passes - 1:
            xlim, ylim = (x0 - mx, x1 + mx), (y0 - my, y1 + my)
        else:
            xlim = (min(xlim[0], x0 - mx), max(xlim[1], x1 + mx))
            ylim = (min(ylim[0], y0 - my), max(ylim[1], y1 + my))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return pos, xlim, ylim


def draw_basic_graph(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    ax: matplotlib.axis.Axis,
    remappings: Dict[str, str],
    legend: bool = True,
    multigrad_legend: Union[Dict[str, str], None] = None,
    legend_fontsize: int = 14,
    label_zorder: Union[float, None] = None,
    legend_loc: str = "lower right",
):
    """label_zorder lifts the node boxes above the gradient arrows, which are drawn at zorder 5
    and up and otherwise cover the label of any node a thick arrow passes over or ends on.
    Left at None the labels keep matplotlib's default and the arrows stay on top, as before."""
    legend_patches = []
    for t, color in types_and_colors.items():
        labels = {n: d["display_name"] for n, d in G.nodes(data=True) if d["type"] == t}
        if len(labels) > 0:
            texts = nx.draw_networkx_labels(
                G,
                pos,
                labels=labels,
                ax=ax,
                font_color=font_color_for(t),
                bbox={"boxstyle": "round,pad=0.4", "fc": color, "ec": None, "lw": 0},
            )
            if label_zorder is not None:
                for text in texts.values():
                    text.set_zorder(label_zorder)
            if t == "subnode":
                t = "Estimated Sub-Quantity"
            legend_patches.append(matplotlib.patches.Patch(color=color, label=t))
    nx.draw_networkx_edges(
        G,
        pos,
        [(u, v) for u, v, d in G.edges(data=True) if d["visible"]],
        ax=ax,
        edge_color=EDGE_COLOR,
        width=EDGE_WIDTH,
        arrows=False,
    )
    handler_map = {}
    if multigrad_legend is None:
        ramp_key = matplotlib.lines.Line2D([0], [0], color=GRADIENT_BASE_COLOR, label=GRAD_LEGEND_LABEL)
        legend_patches.append(ramp_key)
        # keyed on the handle itself so the other (Patch) entries keep matplotlib's default handler
        handler_map[ramp_key] = GradientRampHandler()
    else:
        for k, v in multigrad_legend.items():
            legend_patches.append(matplotlib.lines.Line2D([0], [0], color=v, label=k))
    if legend:
        # the ramp key stacks GRAD_LEGEND_LEVELS arrows, so its row is that many times taller
        ax.legend(
            handles=legend_patches,
            loc=legend_loc,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            fontsize=legend_fontsize,
            handler_map=handler_map or None,
            handlelength=2.6,
            handleheight=GRAD_LEGEND_LEVELS * 0.72,
            labelspacing=0.45,
        )
    return ax, legend_patches


# Gradient overlay: level 0 is the descended path, deeper levels are mixed into it and drawn thinner, paler
# and more transparent. Ramps are (level 0, deepest level).
GRAD_LINEWIDTH = (14.0, 9.0)
GRAD_ARROWSIZE = (30.0, 22.0)
GRAD_ALPHA = (0.8, 0.7)  # below 1 so labels under an arrow stay readable
GRAD_MAX_DESATURATION = 0.18  # fraction of the base colour's saturation, deepest level
GRAD_MAX_LIGHTENING = 0.12  # added lightness, deepest level
GRADIENT_BASE_COLOR = "#e78284"
# Tag on the node where a deeper-level gradient joins the one above it
MIXIN_LABEL = "Combination\nthrough Projection"
# Tag on the node a nullspace escape acts on
CONFLICT_LABEL = "Conflict Resolution\nthrough Exploration"
# Legend key: number of ramp levels shown and scaling of the line widths to fit a legend handle
GRAD_LEGEND_LEVELS = 3
GRAD_LEGEND_LW_SCALE = 0.36
GRAD_LEGEND_LABEL = "Currently Utilized Gradients\n(thinner and paler = lower priority)"
# Tags: opacity, font size, gap to the node box (data units) and padding
SPECIAL_LABEL_ALPHA = 1.0
GRAD_BADGE_FONTSIZE = 10
GRAD_BADGE_GAP = 0.010
GRAD_BADGE_PAD = 0.25
# Tag placement costs (in units of overlap area): leaving the axes, and sliding away from the arrow's tail
GRAD_BADGE_OUTSIDE_PENALTY = 4.0
GRAD_BADGE_WALK_PENALTY = 0.25
# Tags stay above arrows even when arrows are drawn over the boxes
GRAD_BADGE_ZORDER = 30.0


def _rect_overlap(a, b):
    """Area shared by two (cx, cy, w, h) rectangles."""
    dx = min(a[0] + a[2] / 2.0, b[0] + b[2] / 2.0) - max(a[0] - a[2] / 2.0, b[0] - b[2] / 2.0)
    dy = min(a[1] + a[3] / 2.0, b[1] + b[3] / 2.0) - max(a[1] - a[3] / 2.0, b[1] - b[3] / 2.0)
    return max(dx, 0.0) * max(dy, 0.0)


def _ray_exit(origin, direction, box):
    """How far along `direction` from `origin` the ray leaves the (cx, cy, w, h) box.

    The origin is not the box centre in general: a gradient leaving a sub-quantity starts at the
    sub-quantity, which sits somewhere inside its estimator's box, and it is the estimator's
    outline the arrow has to clear.
    """
    cx, cy, w, h = box
    best = float("inf")
    for p, d, lo, hi in (
        (origin[0], direction[0], cx - w / 2.0, cx + w / 2.0),
        (origin[1], direction[1], cy - h / 2.0, cy + h / 2.0),
    ):
        if abs(d) < 1e-12:
            continue  # parallel to this pair of sides, they cannot be the exit
        t = ((hi if d > 0 else lo) - p) / d
        if t > 0.0:
            best = min(best, t)
    return 0.0 if best == float("inf") else best


def _half_extent(direction, w, h):
    """Distance from a (w, h) box's centre to its outline, along `direction`."""
    best = float("inf")
    for d, half in ((direction[0], w / 2.0), (direction[1], h / 2.0)):
        if abs(d) > 1e-12:
            best = min(best, half / abs(d))
    return 0.0 if best == float("inf") else best


def draw_gradient_badges(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    ax: matplotlib.axis.Axis,
    badges: List[Tuple[str, str, str]],
    face,
    zorder: Union[float, None] = None,
    node_font_size: int = 12,
    font_size: int = GRAD_BADGE_FONTSIZE,
    gap: float = GRAD_BADGE_GAP,
):
    """Draw the conflict / mix-in tags next to their nodes, on the tail of the node's outgoing gradient arrow (or above
    or below the node if it has none), avoiding overlaps.

    `badges` is a list of (node, text, edge_colour, target_or_None).
    """
    if not badges:
        return
    node_sizes = measure_label_boxes(ax, _node_labels(G), font_size=node_font_size)
    badge_sizes = measure_label_boxes(
        ax, {i: b[1] for i, b in enumerate(badges)}, font_size=font_size, pad=GRAD_BADGE_PAD
    )
    occupied = [(pos[n][0], pos[n][1], node_sizes[n][0], node_sizes[n][1]) for n in G]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # sub-quantities are drawn inside their estimator's box, so tags are placed outside the whole group
    anchor = {}
    for head, members in _groups_of(G).items():
        x0 = min(pos[n][0] - node_sizes[n][0] / 2.0 for n in members)
        x1 = max(pos[n][0] + node_sizes[n][0] / 2.0 for n in members)
        y0 = min(pos[n][1] - node_sizes[n][1] / 2.0 for n in members)
        y1 = max(pos[n][1] + node_sizes[n][1] / 2.0 for n in members)
        for n in members:
            anchor[n] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)

    def score(bx, by, bw, bh):
        rect = (bx, by, bw, bh)
        outside = (
            max(by + bh / 2.0 - max(ylim), 0.0)
            + max(min(ylim) - (by - bh / 2.0), 0.0)
            + max(bx + bw / 2.0 - max(xlim), 0.0)
            + max(min(xlim) - (bx - bw / 2.0), 0.0)
        )
        return sum(_rect_overlap(rect, o) for o in occupied) + GRAD_BADGE_OUTSIDE_PENALTY * outside * bw

    for i, badge in enumerate(badges):
        node, text, edge = badge[0], badge[1], badge[2]
        target = badge[3] if len(badge) > 3 else None
        bw, bh = badge_sizes[i]
        ax_c, ay_c, aw, ah = anchor[node]
        best = None

        if target is not None and target in pos:
            origin = (float(pos[node][0]), float(pos[node][1]))
            span = (float(pos[target][0]) - origin[0], float(pos[target][1]) - origin[1])
            length = (span[0] ** 2 + span[1] ** 2) ** 0.5
            if length > 1e-9:
                d = (span[0] / length, span[1] / length)
                # just clear of the box the arrow leaves, so the tag sits on the arrow's tail
                start = _ray_exit(origin, d, (ax_c, ay_c, aw, ah)) + gap + _half_extent(d, bw, bh)
                # if the tail is blocked, slide along the same arrow at a small cost
                for step in (0.0, 0.1, 0.2, 0.35, 0.5):
                    t_along = start + step * length
                    if t_along + _half_extent(d, bw, bh) > length:
                        break  # past the far end, that is the target's tag position now
                    bx = origin[0] + d[0] * t_along
                    by = origin[1] + d[1] * t_along
                    cost = score(bx, by, bw, bh) + GRAD_BADGE_WALK_PENALTY * step * bw * bh
                    if best is None or cost < best[0] - 1e-9:
                        best = (cost, bx, by)

        if best is None:
            # no outgoing arrow (end of a path): place the tag above or below the box, whichever is emptier
            reach = ah / 2.0 + gap + bh / 2.0
            for sign in (1.0, -1.0):  # above first, so ties go upwards
                by = float(ay_c) + sign * reach
                for slide in (0.0, 0.35, -0.35, 0.7, -0.7):
                    bx = float(ax_c) + slide * bw
                    if abs(bx - ax_c) > aw / 2.0:
                        continue
                    cost = score(bx, by, bw, bh)
                    if best is None or cost < best[0] - 1e-9:
                        best = (cost, bx, by)

        _, x, y = best
        t = ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=font_size,
            color=font_color_for("Specified Goal"),
            bbox={
                "boxstyle": "round,pad=%f" % GRAD_BADGE_PAD,
                "fc": face,
                "ec": edge,
                "lw": 2.0,
                "alpha": SPECIAL_LABEL_ALPHA,
            },
        )
        if zorder is not None:
            t.set_zorder(zorder)
        # later tags avoid the ones already placed, so two annotated neighbours do not stack
        occupied.append((x, y, bw, bh))


class GradientRampHandler(matplotlib.legend_handler.HandlerBase):
    """Legend key for the gradient overlay: a few arrows sampled from the actual width/colour ramp, so the key shows
    that thinner and paler means lower priority."""

    def __init__(self, n_levels: int = GRAD_LEGEND_LEVELS, **kwargs):
        super().__init__(**kwargs)
        self.n_levels = max(int(n_levels), 1)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        base = matplotlib.colors.to_rgb(GRADIENT_BASE_COLOR)
        lws = level_ramp(GRAD_LINEWIDTH[0], GRAD_LINEWIDTH[1], self.n_levels)
        heads = level_ramp(GRAD_ARROWSIZE[0], GRAD_ARROWSIZE[1], self.n_levels)
        alphas = level_ramp(GRAD_ALPHA[0], GRAD_ALPHA[1], self.n_levels)
        arrows = []
        for i in range(self.n_levels):
            # rungs run top to bottom, each a little shorter than the one above, so the ramp
            # reads as an ordering even in a black-and-white print of the figure
            y = height * (self.n_levels - i - 0.5) / self.n_levels - ydescent
            span = width * (1.0 - 0.18 * i / max(self.n_levels - 1.0, 1.0))
            arrow = matplotlib.patches.FancyArrowPatch(
                (-xdescent, y),
                (-xdescent + span, y),
                arrowstyle="->",
                mutation_scale=heads[i] * GRAD_LEGEND_LW_SCALE,
                linewidth=lws[i] * GRAD_LEGEND_LW_SCALE,
                color=reduce_saturation(base, i, self.n_levels),
                alpha=alphas[i],
                shrinkA=0.0,
                shrinkB=0.0,
            )
            arrow.set_transform(trans)
            arrows.append(arrow)
        return arrows


def level_ramp(top, bottom, n_levels):
    """Evenly spaced values from top (level 0) down to bottom (the deepest level).

    With one level there is nothing to distinguish, so it gets the top value.
    """
    if n_levels <= 1:
        return [top]
    return [top + (bottom - top) * i / (n_levels - 1.0) for i in range(n_levels)]


def reduce_saturation(
    color, this_level, n_levels, max_desaturation=GRAD_MAX_DESATURATION, max_lightening=GRAD_MAX_LIGHTENING
):
    """Pale the colour towards the deepest level, by a bounded amount.

    The shift is keyed on the level's rank within the stack rather than on this_level/n_levels,
    so the deepest level always lands on exactly the configured washout whatever the depth.
    """
    if n_levels <= 1:
        return color
    f = this_level / (n_levels - 1.0)
    h, l, s = colorsys.rgb_to_hls(*color)
    new_s = s * (1.0 - max_desaturation * f)
    new_l = min(1.0, l + max_lightening * f)
    return colorsys.hls_to_rgb(h, new_l, new_s)


def parse_gradient_trace(trace, level):
    if trace[0][0] == "[" and trace[-1][-1] == "]":
        return dict()
    this_grad = []
    grads = {}
    depth = 0  # unclosed delimiters of the sub-trace we are currently collecting
    open_delim = None
    recursive_parsing_goal = []

    def merge_lower_levels(goal):
        if len(goal) == 0:
            return
        for k, v in parse_gradient_trace(goal, level=level + 1).items():
            if k in grads:
                grads[k].extend(v)
            else:
                grads[k] = v

    for entry in trace:
        # a single entry can open and/or close several groups, e.g. "pose_ee))"
        opened = entry.count("(") + entry.count("[")
        closed = entry.count(")") + entry.count("]")

        if depth > 0:
            # inside a sub-trace: hand the entry down verbatim, only the delimiter
            # closing our own group is stripped (the recursion handles nested ones)
            depth += opened - closed
            if depth <= 0:
                # bracketed (projection) groups keep their delimiters, the recursion drops them
                recursive_parsing_goal.append(entry if open_delim == "[" else entry[:-1])
                merge_lower_levels(recursive_parsing_goal)
                recursive_parsing_goal = []
                depth = 0
                open_delim = None
            else:
                recursive_parsing_goal.append(entry)
            continue

        if entry[: len(CR_MARKER)] == CR_MARKER:
            this_grad.append(CR_MARKER)
            if len(entry) == len(CR_MARKER):
                continue
            open_delim = "("
            content = entry[len(CR_MARKER) + 1 :]
        elif entry[0] == "(":
            open_delim = "("
            content = entry[1:]
        elif entry[0] == "[":
            open_delim = "["
            content = entry
        elif "," in entry:
            split_entry = entry.split(",")
            this_grad.append(split_entry[0])
            if level in grads:
                grads[level].extend(
                    [
                        this_grad,
                    ]
                )
            else:
                grads[level] = [
                    this_grad,
                ]
            level += 1
            this_grad = [split_entry[1]]
            continue
        else:
            this_grad.append(entry)
            continue

        # we just opened a sub-trace with this entry
        depth = opened - closed
        if depth <= 0:
            # sub-trace consists of this single entry, e.g. "ConflictResolution(pose_ee)"
            merge_lower_levels([content if open_delim == "[" else content[:-1]])
            depth = 0
            open_delim = None
        else:
            recursive_parsing_goal.append(content)

    # malformed trace with an unclosed group: keep what we collected instead of dropping it
    merge_lower_levels(recursive_parsing_goal)

    if level in grads:
        grads[level].extend(
            [
                this_grad,
            ]
        )
    else:
        grads[level] = [
            this_grad,
        ]
    return grads


def draw_current_active_gradients(
    gradients: List[List[str]],
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    ax: matplotlib.axis.Axis,
    remappings: Union[Dict[str, str], None] = None,
    colors: Union[List[str], None] = None,
    lw: Union[List[float], float, None] = None,
    adaptive_linewidths: bool = False,
    adaptive_arcs: bool = False,
    remap_gradients: bool = False,
    label_zorder: Union[float, None] = None,
):
    if remappings is None:
        remappings = dict()

    grad_level_dict = {}
    for g in gradients:
        g = "-".join(g)
        d = parse_gradient_trace(g.split("-"), level=0)
        for k, v in d.items():
            # v is a list of traces for that level, so extend rather than nest it
            if k in grad_level_dict:
                grad_level_dict[k].extend(v)
            else:
                grad_level_dict[k] = list(v)

    # levels can be sparse, so index them by rank instead of by their raw value
    grad_level_dict = {i: grad_level_dict[k] for i, k in enumerate(sorted(grad_level_dict.keys()))}
    return draw_gradient_levels(
        grad_level_dict,
        G,
        pos,
        ax,
        remappings=remappings,
        adaptive_linewidths=adaptive_linewidths,
        remap_gradients=remap_gradients,
        label_zorder=label_zorder,
    )


def draw_gradient_levels(
    grad_level_dict: Dict[int, List[List[str]]],
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    ax: matplotlib.axis.Axis,
    remappings: Union[Dict[str, str], None] = None,
    adaptive_linewidths: bool = False,
    remap_gradients: bool = False,
    label_zorder: Union[float, None] = None,
):
    """Draw an already-levelled set of gradient paths.

    Split out of draw_current_active_gradients so a caller that has the level structure in hand
    can reach the drawing without going through the trace strings. That round trip is lossy for
    this graph: it joins a trace on "-" and splits on it again, which tears any node whose name
    contains a hyphen ("Target\\nBearing-Only\\nMeasurement", "RGB-D Camera") in half.
    """
    if remappings is None:
        remappings = dict()
    num_levels = len(grad_level_dict.keys())

    base_color = matplotlib.colors.to_rgb(GRADIENT_BASE_COLOR)

    lw = level_ramp(GRAD_LINEWIDTH[0], GRAD_LINEWIDTH[1], num_levels)
    arrow_sizes = level_ramp(GRAD_ARROWSIZE[0], GRAD_ARROWSIZE[1], num_levels)
    alphas = level_ramp(GRAD_ALPHA[0], GRAD_ALPHA[1], num_levels)
    styles = ["solid"] * num_levels
    colors = [reduce_saturation(base_color, i, num_levels) for i in range(num_levels)]

    all_conflict_nodes = []
    all_mixin_nodes = []
    # Arrow leaving each node, so a tag can sit on its tail; level 0 is filled last and wins.
    outgoing = {}
    lw_local = 3.0

    for lvl in reversed(range(num_levels)):
        gradients = grad_level_dict[lvl]
        each_gradient_edges, each_gradient_nodes, conflict_nodes = get_each_gradient_path(
            G, gradients, remap_gradients, remappings
        )
        all_conflict_nodes.extend([n for n in conflict_nodes if n in G])
        if lvl > 0:
            # the last entry of a deeper-level trace is where it joins the level above
            for path in each_gradient_nodes:
                junction = next((n for n in reversed(path) if n in G), None)
                if junction is not None:
                    all_mixin_nodes.append(junction)
        for i in reversed(range(len(gradients))):
            # a trace entry we cannot map onto the graph is skipped rather than
            # taking down the whole visualization
            edges = [e for e in each_gradient_edges[i] if e[0] in G and e[1] in G]
            nodes = [n for n in each_gradient_nodes[i] if n in G]
            if len(edges) == 0 and len(nodes) == 0:
                continue
            for u, v in edges:
                outgoing[u] = v
            collection = nx.draw_networkx_edges(
                G,
                pos,
                edges,
                edge_color=[colors[lvl]] * len(edges),
                ax=ax,
                width=lw[lvl],
                alpha=alphas[lvl],
                min_source_margin=23.0,
                min_target_margin=20.0,
                arrowsize=arrow_sizes[lvl],
                arrows=True,
                style=styles[lvl],
                arrowstyle="->",
            )
            if isinstance(collection, Iterable):
                [c.set_zorder(5 + lvl) for c in collection]
            else:
                collection.set_zorder(5 + lvl)

            node_types = {n: d["type"] for n, d in G.nodes(data=True) if n in nodes}
            possible_types = set(node_types.values())
            for t in possible_types:
                this_type_nodes = []
                for node in nodes:
                    if t == node_types[node]:
                        this_type_nodes.append(node)
                for node in this_type_nodes:
                    if not adaptive_linewidths:
                        lw_local = 3.0
                    texts = nx.draw_networkx_labels(
                        G,
                        pos,
                        labels={n: d["display_name"] for n, d in G.nodes(data=True) if n in [node]},
                        ax=ax,
                        font_color=font_color_for(t),
                        bbox={
                            "boxstyle": "round,pad=0.4",
                            "fc": types_and_colors[t],
                            "ec": colors[lvl],
                            "lw": lw_local,
                        },
                    )
                    if label_zorder is not None:
                        for text in texts.values():
                            text.set_zorder(label_zorder)

    # tags follow the node labels' zorder rule and are drawn last
    special_zorder = GRAD_BADGE_ZORDER if label_zorder is None else max(label_zorder + 1, GRAD_BADGE_ZORDER)

    # a node that is also a mix-in junction keeps the conflict tag
    badges = [
        (node, MIXIN_LABEL, colors[0], outgoing.get(node))
        for node in dict.fromkeys(n for n in all_mixin_nodes if n not in all_conflict_nodes)
    ]
    badges += [(node, CONFLICT_LABEL, "black", outgoing.get(node)) for node in dict.fromkeys(all_conflict_nodes)]
    if badges:
        draw_gradient_badges(G, pos, ax, badges, face=colors[0], zorder=special_zorder)
    return ax


def get_each_gradient_path(G, gradients, remap_gradients, remappings):
    subnodes = [n for n, d in G.nodes(data=True) if d["type"] == "subnode"]
    each_gradient_edges = []
    each_gradient_nodes = []
    conflict_nodes = []
    for gradient_trace in gradients:
        edges = []
        nodes = []
        current_node = gradient_trace[0]
        last_node = None
        if current_node in remappings and current_node not in subnodes:
            current_node = remappings[current_node]
        next_node = None
        for idx in range(len(gradient_trace) - 1):
            # only take remapped node for node coloring if the original is not a subnode of a rec. estimator node
            if gradient_trace[idx] in subnodes:
                nodes.append(gradient_trace[idx])
            else:
                nodes.append(current_node)
            next_node = gradient_trace[idx + 1]
            if next_node in remappings and next_node not in subnodes:
                next_node = remappings[next_node]
            if current_node != next_node:
                if next_node == CR_MARKER:
                    pass
                elif current_node == CR_MARKER:
                    if last_node is not None:
                        edges.append((last_node, next_node))
                else:
                    edges.append((current_node, next_node))  # TODO currently is not going via the connections
            last_node = current_node
            current_node = next_node
        if gradient_trace[-1] in subnodes:
            nodes.append(gradient_trace[-1])
        else:
            nodes.append(next_node)
        while CR_MARKER in nodes:
            idx = nodes.index(CR_MARKER)
            nodes.remove(CR_MARKER)
            # mark the node the resolution acts on: the following one, or the
            # preceding one if the marker sits at the end of the trace
            if idx < len(nodes):
                conflict_nodes.append(nodes[idx])
            elif len(nodes) > 0:
                conflict_nodes.append(nodes[-1])
        each_gradient_edges.append(edges)
        each_gradient_nodes.append(nodes)
    if remap_gradients:
        remapped_gradient_edges = []
        remapped_gradient_nodes = []
        for i in range(len(gradients)):
            nodes = each_gradient_nodes[i].copy()
            remapped_edges = []
            for e in each_gradient_edges[i]:
                if e[0] not in G or e[1] not in G:
                    continue
                found_replacements = []
                for p in nx.all_simple_paths(G, e[0], e[1], cutoff=2):
                    if len(p) == 3 and G.nodes[p[1]]["type"] == "Active Interconnection":
                        found_replacements.append((p[0], p[1]))
                        found_replacements.append((p[1], p[2]))
                        nodes.append(p[1])
                if len(found_replacements) > 0:
                    remapped_edges.extend(found_replacements)
                else:
                    remapped_edges.append(e)
            remapped_gradient_edges.append(list(set(remapped_edges)))
            remapped_gradient_nodes.append(list(set(nodes)))
        each_gradient_edges = remapped_gradient_edges
        each_gradient_nodes = remapped_gradient_nodes
    return each_gradient_edges, each_gradient_nodes, conflict_nodes


def get_initial_animatable_figure_for_graph(
    G: nx.Graph, pos: Dict[str, Tuple[float, float]], remappings: Union[Dict[str, str], None] = None
):
    ax, fig = get_initial_figure_for_graph(G, pos, remappings)
    plt.ion()
    plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)
    background = fig.canvas.copy_from_bbox(fig.bbox)
    fig.canvas.blit(fig.bbox)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)
    return fig, ax, background


def get_initial_figure_for_graph(
    G: nx.Graph, pos: Dict[str, Tuple[float, float]], remappings: Union[Dict[str, str], None] = None
):
    fig, ax = plt.subplots(1, 1)
    # maximise the window where the backend allows it (Tk)
    window = getattr(plt.get_current_fig_manager(), "window", None)
    if hasattr(window, "attributes"):
        window.attributes("-zoomed", True)
    ax, _ = draw_basic_graph(G, pos, ax, remappings=remappings, legend=False)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    fig.tight_layout()
    return ax, fig


def update_fig_with_gradients(
    gradients: List[List[str]],
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    fig,
    ax,
    background,
    remappings: Union[Dict[str, str], None] = None,
):
    ax, _ = draw_basic_graph(G, pos, ax, remappings=remappings, legend=False)
    ax = draw_current_active_gradients(gradients, G, pos, ax, remappings=remappings, remap_gradients=False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    ax.clear()


# TODO second mode: gradient intensitiy for all available options
