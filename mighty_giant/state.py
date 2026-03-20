from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LowRankState:
    """Per-sequence mutable state for the low-rank fast-weight memory.

    Each list has one entry per memory layer.  Shapes per entry:
        down_weights:  [batch, rank, in_dim]
        up_weights:    [batch, out_dim, rank]
        down_biases:   [batch, rank]
        momentum_down_w: same as down_weights
        momentum_up_w:   same as up_weights
        momentum_down_b: same as down_biases
    """

    down_weights: list[torch.Tensor]
    up_weights: list[torch.Tensor]
    down_biases: list[torch.Tensor]
    momentum_down_w: list[torch.Tensor]
    momentum_up_w: list[torch.Tensor]
    momentum_down_b: list[torch.Tensor]

    def _map(self, fn) -> LowRankState:
        return LowRankState(
            down_weights=[fn(t) for t in self.down_weights],
            up_weights=[fn(t) for t in self.up_weights],
            down_biases=[fn(t) for t in self.down_biases],
            momentum_down_w=[fn(t) for t in self.momentum_down_w],
            momentum_up_w=[fn(t) for t in self.momentum_up_w],
            momentum_down_b=[fn(t) for t in self.momentum_down_b],
        )

    def detach(self) -> LowRankState:
        return self._map(lambda t: t.detach())

    def enable_grad(self) -> LowRankState:
        def _enable(t: torch.Tensor) -> torch.Tensor:
            return t.detach().requires_grad_(True)

        return LowRankState(
            down_weights=[_enable(t) for t in self.down_weights],
            up_weights=[_enable(t) for t in self.up_weights],
            down_biases=[_enable(t) for t in self.down_biases],
            momentum_down_w=[t.detach() for t in self.momentum_down_w],
            momentum_up_w=[t.detach() for t in self.momentum_up_w],
            momentum_down_b=[t.detach() for t in self.momentum_down_b],
        )


@dataclass
class SSMState:
    """Per-layer SSM hidden state."""

    h: torch.Tensor  # [batch, n_heads, mimo_rank, d_state]

    def detach(self) -> SSMState:
        return SSMState(h=self.h.detach())


@dataclass
class SequenceState:
    """Full model state carried across segments and batches."""

    ssm_states: list[SSMState]
    memory_state: LowRankState
    tokens_seen: int = 0

    def detach(self) -> SequenceState:
        return SequenceState(
            ssm_states=[s.detach() for s in self.ssm_states],
            memory_state=self.memory_state.detach(),
            tokens_seen=self.tokens_seen,
        )

    def enable_grad(self) -> SequenceState:
        return SequenceState(
            ssm_states=self.ssm_states,
            memory_state=self.memory_state.enable_grad(),
            tokens_seen=self.tokens_seen,
        )
