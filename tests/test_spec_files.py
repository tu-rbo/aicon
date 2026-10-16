"""Each example's spec.yaml describes its current graph (regenerate them with `python -m examples.spec_files`)."""

import os

import pytest

from examples.spec_files import GRAPHS, REPOSITORY_ROOT, unavailable_reason


@pytest.mark.parametrize("graph", GRAPHS, ids=[g.spec_file for g in GRAPHS])
def test_spec_file_is_current(graph, request):
    reason = unavailable_reason(graph)
    if reason:
        pytest.skip(reason)
    if graph.needs_ros:
        request.getfixturevalue("ros_context")
    with open(os.path.join(REPOSITORY_ROOT, graph.spec_file)) as f:
        assert f.read() == graph.text(), f"{graph.spec_file} is outdated, run `python -m examples.spec_files`"
