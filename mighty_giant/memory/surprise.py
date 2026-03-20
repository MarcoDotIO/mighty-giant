from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..state import LowRankState
from .fast_weights import LowRankFastWeightMemory


class MemoryUpdater(nn.Module):
    """Surprise-based memory update at segment granularity.

    Adapts the Titans update rule for low-rank fast-weight state:
        S_i = eta * S_{i-1} - theta * grad(loss)
        M_i = (1 - alpha) * M_{i-1} + S_i

    where loss = ||M(k) - v||^2 (associative memory loss) and
    alpha, eta, theta are segment-level scalars derived from the
    mean-pooled segment representation.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.alpha_head = nn.Linear(d_model, 1)
        self.eta_head = nn.Linear(d_model, 1)
        self.theta_head = nn.Linear(d_model, 1)
        nn.init.constant_(self.theta_head.bias, -3.0)

    def update(
        self,
        segment_hidden: torch.Tensor,
        memory: LowRankFastWeightMemory,
        state: LowRankState,
    ) -> LowRankState:
        """Update memory state using surprise from the current segment.

        Args:
            segment_hidden: [batch, seg_len, d_model] post-backbone hidden states.
            memory: the memory module (for forward pass).
            state: current low-rank state (must have grad enabled on weights).

        Returns:
            Updated LowRankState.
        """
        pooled = segment_hidden.mean(dim=1)
        alpha = torch.sigmoid(self.alpha_head(pooled))
        eta = torch.sigmoid(self.eta_head(pooled))
        theta = F.softplus(self.theta_head(pooled)) + 1e-6

        keys = F.normalize(self.key_proj(F.silu(segment_hidden)), dim=-1, eps=1e-6)
        values = self.value_proj(F.silu(segment_hidden))

        predicted = memory.forward_with_state(keys, state)
        associative_loss = F.mse_loss(predicted, values, reduction="mean")

        fast_params = state.down_weights + state.up_weights + state.down_biases
        grads = torch.autograd.grad(
            associative_loss,
            fast_params,
            create_graph=self.training,
            retain_graph=self.training,
        )

        total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
        clip_coef = (1.0 / (total_norm + 1e-6)).clamp(max=1.0)
        grads = tuple(g * clip_coef for g in grads)

        n_down = len(state.down_weights)
        n_up = len(state.up_weights)

        grad_down_w = grads[:n_down]
        grad_up_w = grads[n_down : n_down + n_up]
        grad_down_b = grads[n_down + n_up :]

        new_down_w = []
        new_up_w = []
        new_down_b = []
        new_mom_down_w = []
        new_mom_up_w = []
        new_mom_down_b = []

        alpha_w = alpha.unsqueeze(-1)
        eta_w = eta.unsqueeze(-1)
        theta_w = theta.unsqueeze(-1)

        for i in range(n_down):
            mom_dw = eta_w * state.momentum_down_w[i] - theta_w * grad_down_w[i]
            mom_uw = eta_w * state.momentum_up_w[i] - theta_w * grad_up_w[i]
            mom_db = eta * state.momentum_down_b[i] - theta * grad_down_b[i]

            new_dw = (1.0 - alpha_w) * state.down_weights[i] + mom_dw
            new_uw = (1.0 - alpha_w) * state.up_weights[i] + mom_uw
            new_db = (1.0 - alpha) * state.down_biases[i] + mom_db

            new_down_w.append(new_dw)
            new_up_w.append(new_uw)
            new_down_b.append(new_db)
            new_mom_down_w.append(mom_dw)
            new_mom_up_w.append(mom_uw)
            new_mom_down_b.append(mom_db)

        return LowRankState(
            down_weights=new_down_w,
            up_weights=new_up_w,
            down_biases=new_down_b,
            momentum_down_w=new_mom_down_w,
            momentum_up_w=new_mom_up_w,
            momentum_down_b=new_mom_down_b,
        )
