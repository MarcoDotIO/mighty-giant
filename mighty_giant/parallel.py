from __future__ import annotations

import torch
from torch import nn

from .model import MightyGiantOutput
from .state import LowRankState, SSMState, SequenceState


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
    target = _resolve_output_device(output_device)
    tensors = [t if t.device == target else t.to(target) for t in outputs]
    if tensors[0].ndim == 0:
        return torch.stack(tensors, dim=0)
    return torch.cat(tensors, dim=dim)


def _require_matching(values: list[object], *, field_name: str) -> object:
    first = values[0]
    for v in values[1:]:
        if v != first:
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
    if isinstance(out, MightyGiantOutput):
        return MightyGiantOutput(
            logits=gather_parallel_output([o.logits for o in outputs], output_device, dim=dim),
            loss=gather_parallel_output([o.loss for o in outputs], output_device, dim=dim),
            state=gather_parallel_output([o.state for o in outputs], output_device, dim=dim),
        )
    if isinstance(out, SequenceState):
        return SequenceState(
            ssm_states=gather_parallel_output(
                [o.ssm_states for o in outputs], output_device, dim=dim
            ),
            memory_state=gather_parallel_output(
                [o.memory_state for o in outputs], output_device, dim=dim
            ),
            tokens_seen=_require_matching(
                [o.tokens_seen for o in outputs], field_name="tokens_seen"
            ),
        )
    if isinstance(out, SSMState):
        return SSMState(
            h=gather_parallel_output([o.h for o in outputs], output_device, dim=dim),
        )
    if isinstance(out, LowRankState):
        return LowRankState(
            down_weights=gather_parallel_output([o.down_weights for o in outputs], output_device, dim=dim),
            up_weights=gather_parallel_output([o.up_weights for o in outputs], output_device, dim=dim),
            down_biases=gather_parallel_output([o.down_biases for o in outputs], output_device, dim=dim),
            momentum_down_w=gather_parallel_output([o.momentum_down_w for o in outputs], output_device, dim=dim),
            momentum_up_w=gather_parallel_output([o.momentum_up_w for o in outputs], output_device, dim=dim),
            momentum_down_b=gather_parallel_output([o.momentum_down_b for o in outputs], output_device, dim=dim),
        )
    if isinstance(out, list):
        expected_len = len(out)
        if any(len(o) != expected_len for o in outputs[1:]):
            raise ValueError("Mismatched list lengths across DataParallel replicas.")
        return [
            gather_parallel_output([o[i] for o in outputs], output_device, dim=dim)
            for i in range(expected_len)
        ]
    if isinstance(out, tuple):
        expected_len = len(out)
        if any(len(o) != expected_len for o in outputs[1:]):
            raise ValueError("Mismatched tuple lengths across DataParallel replicas.")
        return tuple(
            gather_parallel_output([o[i] for o in outputs], output_device, dim=dim)
            for i in range(expected_len)
        )
    if isinstance(out, dict):
        expected_keys = set(out)
        if any(set(o) != expected_keys for o in outputs[1:]):
            raise ValueError("Mismatched dict keys across DataParallel replicas.")
        return {
            k: gather_parallel_output([o[k] for o in outputs], output_device, dim=dim)
            for k in out
        }
    if isinstance(out, (bool, int, float, str)):
        return _require_matching(outputs, field_name=type(out).__name__)
    raise TypeError(f"Unsupported output type for DataParallel gather: {type(out)!r}")


class MightyGiantDataParallel(nn.DataParallel):
    def gather(self, outputs, output_device):
        return gather_parallel_output(outputs, output_device, dim=self.dim)
