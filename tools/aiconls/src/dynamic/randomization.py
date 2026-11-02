from typing import Any

import torch


def randomize_tensor_mapping(all_inputs: Any) -> Any:
    randomized_inputs = all_inputs.clone()
    for quantity_name, quantity_value in all_inputs.items():
        if not isinstance(quantity_value, torch.Tensor):
            continue
        randomized_inputs[quantity_name] = sample_allowed_tensor_values(
            quantity_name=quantity_name,
            tensor=quantity_value,
        )
    return randomized_inputs


def sample_allowed_tensor_values(
    *, quantity_name: str, tensor: torch.Tensor
) -> torch.Tensor:
    normalized_name = quantity_name.lower()

    if tensor.dtype == torch.bool:
        return torch.rand(tensor.shape, device=tensor.device) > 0.5

    if normalized_name == "dt":
        return torch.full(
            tensor.shape,
            0.05,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    if tensor.is_floating_point():
        if (
            "uncertainty" in normalized_name
            and tensor.ndim >= 2
            and tensor.shape[-1] == tensor.shape[-2]
        ):
            gaussian = torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            identity = torch.eye(
                tensor.shape[-1],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for _ in range(max(0, tensor.ndim - 2)):
                identity = identity.unsqueeze(0)
            return gaussian @ gaussian.transpose(-1, -2) + identity * 1e-3

        if (
            "likelihood" in normalized_name
            or "prob" in normalized_name
            or "activation" in normalized_name
        ):
            return torch.rand(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            ).clamp(min=1e-6, max=1.0 - 1e-6)

        return torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    if tensor.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return torch.randint(
            low=0,
            high=2,
            size=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    return tensor
