from __future__ import annotations

import torch
from torch import nn

from .model import FastWeightState, ModelState, TitansOutput


def _resolve_output_device(output_device: int | str | torch.device) -> torch.device:
    if isinstance(output_device, torch.device):
        return output_device
    if isinstance(output_device, str):
        return torch.device(output_device)
    if isinstance(output_device, int):
        if output_device < 0:
            return torch.device("cpu")
        return torch.device("cuda", output_device)
    raise TypeError(f"Unsupported output device: {output_device!r}")


def _gather_tensor(
    outputs: list[torch.Tensor],
    output_device: int | str | torch.device,
    *,
    dim: int,
) -> torch.Tensor:
    target_device = _resolve_output_device(output_device)
    tensors = [
        tensor if tensor.device == target_device else tensor.to(target_device)
        for tensor in outputs
    ]
    if tensors[0].ndim == 0:
        return torch.stack(tensors, dim=0)
    return torch.cat(tensors, dim=dim)


def _require_matching(values: list[object], *, field_name: str) -> object:
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise ValueError(f"Mismatched {field_name} across DataParallel replicas.")
    return first


def gather_parallel_output(
    outputs: list[object],
    output_device: int | str | torch.device,
    *,
    dim: int = 0,
) -> object:
    if not outputs:
        raise ValueError("Cannot gather an empty list of outputs.")

    out = outputs[0]
    if isinstance(out, torch.Tensor):
        return _gather_tensor(outputs, output_device, dim=dim)
    if out is None:
        return None
    if isinstance(out, TitansOutput):
        return TitansOutput(
            logits=gather_parallel_output([item.logits for item in outputs], output_device, dim=dim),
            loss=gather_parallel_output([item.loss for item in outputs], output_device, dim=dim),
            state=gather_parallel_output([item.state for item in outputs], output_device, dim=dim),
        )
    if isinstance(out, ModelState):
        return ModelState(
            layer_states=gather_parallel_output(
                [item.layer_states for item in outputs],
                output_device,
                dim=dim,
            ),
            tokens_seen=_require_matching(
                [item.tokens_seen for item in outputs],
                field_name="tokens_seen",
            ),
        )
    if isinstance(out, FastWeightState):
        return FastWeightState(
            weights=gather_parallel_output([item.weights for item in outputs], output_device, dim=dim),
            biases=gather_parallel_output([item.biases for item in outputs], output_device, dim=dim),
            momentum_weights=gather_parallel_output(
                [item.momentum_weights for item in outputs],
                output_device,
                dim=dim,
            ),
            momentum_biases=gather_parallel_output(
                [item.momentum_biases for item in outputs],
                output_device,
                dim=dim,
            ),
        )
    if isinstance(out, list):
        expected_len = len(out)
        if any(len(item) != expected_len for item in outputs[1:]):
            raise ValueError("Mismatched list lengths across DataParallel replicas.")
        return [
            gather_parallel_output([item[index] for item in outputs], output_device, dim=dim)
            for index in range(expected_len)
        ]
    if isinstance(out, tuple):
        expected_len = len(out)
        if any(len(item) != expected_len for item in outputs[1:]):
            raise ValueError("Mismatched tuple lengths across DataParallel replicas.")
        return tuple(
            gather_parallel_output([item[index] for item in outputs], output_device, dim=dim)
            for index in range(expected_len)
        )
    if isinstance(out, dict):
        expected_keys = set(out)
        if any(set(item) != expected_keys for item in outputs[1:]):
            raise ValueError("Mismatched dict keys across DataParallel replicas.")
        return {
            key: gather_parallel_output([item[key] for item in outputs], output_device, dim=dim)
            for key in out
        }
    if isinstance(out, (bool, int, float, str)):
        return _require_matching(outputs, field_name=type(out).__name__)
    raise TypeError(f"Unsupported output type for DataParallel gather: {type(out)!r}")


class TitansDataParallel(nn.DataParallel):
    def gather(self, outputs, output_device):
        return gather_parallel_output(outputs, output_device, dim=self.dim)
