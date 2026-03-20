from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HybridConfig
from ..state import LowRankState


class LowRankFastWeightMemory(nn.Module):
    """Low-rank fast-weight memory module.

    A static shared trunk (MLP) plus per-sequence low-rank adapters that are
    updated at test time via surprise-based gradient descent.

    Forward: M(q) = trunk(q) + sum_layer( up_i @ silu(down_i @ q + bias_i) )

    At d_model=1024, rank=16 this stores ~65K mutable params per batch element
    vs ~2M for a full-width MLP -- a 30x reduction.
    """

    def __init__(self, d_model: int, depth: int, rank: int):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.rank = rank

        trunk_layers = []
        for i in range(depth):
            trunk_layers.append(nn.Linear(d_model, d_model))
            if i < depth - 1:
                trunk_layers.append(nn.SiLU())
        self.trunk = nn.Sequential(*trunk_layers)

        self.init_down = nn.ParameterList(
            [nn.Parameter(torch.randn(rank, d_model) * 0.01) for _ in range(depth)]
        )
        self.init_up = nn.ParameterList(
            [nn.Parameter(torch.randn(d_model, rank) * 0.01) for _ in range(depth)]
        )
        self.init_down_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(rank)) for _ in range(depth)]
        )

    def init_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> LowRankState:
        down_weights = []
        up_weights = []
        down_biases = []
        momentum_down_w = []
        momentum_up_w = []
        momentum_down_b = []

        for i in range(self.depth):
            dw = self.init_down[i].to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
            uw = self.init_up[i].to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
            db = self.init_down_bias[i].to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1).clone()

            down_weights.append(dw)
            up_weights.append(uw)
            down_biases.append(db)
            momentum_down_w.append(torch.zeros_like(dw))
            momentum_up_w.append(torch.zeros_like(uw))
            momentum_down_b.append(torch.zeros_like(db))

        return LowRankState(
            down_weights=down_weights,
            up_weights=up_weights,
            down_biases=down_biases,
            momentum_down_w=momentum_down_w,
            momentum_up_w=momentum_up_w,
            momentum_down_b=momentum_down_b,
        )

    def forward_with_state(
        self,
        q: torch.Tensor,
        state: LowRankState,
    ) -> torch.Tensor:
        """Query the memory.

        Args:
            q: [batch, seq_len, d_model] or [batch, d_model] query.
            state: current low-rank fast-weight state.

        Returns:
            [batch, seq_len, d_model] or [batch, d_model] retrieved values.
        """
        squeezed = q.ndim == 2
        if squeezed:
            q = q.unsqueeze(1)

        trunk_out = self.trunk(q)

        adapter_out = torch.zeros_like(trunk_out)
        for i in range(self.depth):
            down = state.down_weights[i]
            up = state.up_weights[i]
            bias = state.down_biases[i]

            hidden = torch.einsum("bti,bri->btr", q, down) + bias.unsqueeze(1)
            if i < self.depth - 1:
                hidden = F.silu(hidden)
            adapter_out = adapter_out + torch.einsum("btr,bdr->btd", hidden, up)

        result = trunk_out + adapter_out
        if squeezed:
            result = result.squeeze(1)
        return result

    def forward(
        self,
        q: torch.Tensor,
        state: LowRankState,
    ) -> torch.Tensor:
        """Convenience alias for forward_with_state."""
        return self.forward_with_state(q, state)
