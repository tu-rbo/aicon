import torch
from lsprotocol.types import Diagnostic

from dynamic.diff_checker import check_differentials
from dynamic.loader import spec_torch_dtype
from reader.aicon_files.reader_types import (
    ParsedComponents,
    ParsedConnections,
    ParsedGoals,
)
from reader.spec.spec import AICON_SPEC


def dynamic_analysis(
    spec: AICON_SPEC,
    parsed_components: ParsedComponents,
    parsed_goals: ParsedGoals,
    parsed_connections: ParsedConnections,
) -> dict[str, list[Diagnostic]]:
    diagnostics_by_uri: dict[str, list[Diagnostic]] = {}

    # The program usually sets the dtype of its components as the default (e.g. for tensors created inside them), so
    # the check runs with it as well.
    previous_default_dtype = torch.get_default_dtype()
    spec_dtype = spec_torch_dtype(spec, list(spec.components))
    if spec_dtype is not None and spec_dtype.is_floating_point:
        torch.set_default_dtype(spec_dtype)
    try:
        diagnostics_by_uri.update(
            check_differentials(
                spec, parsed_components, parsed_goals, parsed_connections
            )
        )
    finally:
        torch.set_default_dtype(previous_default_dtype)

    return diagnostics_by_uri
