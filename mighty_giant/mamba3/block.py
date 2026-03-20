from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HybridConfig
from .ssm import MultiHeadSSM, parallel_scan


class Mamba3Block(nn.Module):
    """Active Mamba-3-style block used by Mighty Giant.

    Architecture:
        pre-RMSNorm -> gate+value split -> MultiHeadSSM -> gated output projection
    """

    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_state = min(config.d_state, self.head_dim)
        self.mimo_rank = config.mimo_rank if config.is_mimo_layer(layer_idx) else 1

        self.norm = nn.RMSNorm(config.d_model)
        self.in_proj = nn.Linear(config.d_model, config.d_model * 2, bias=False)
        self.ssm = MultiHeadSSM(
            config.d_model,
            self.d_state,
            self.n_heads,
            config,
            mimo_rank=self.mimo_rank,
        )
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.ssm.init_state(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        ssm_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            ssm_state: [batch, n_heads, mimo_rank, d_state] or None

        Returns:
            output: [batch, seq_len, d_model]
            new_ssm_state: [batch, n_heads, mimo_rank, d_state]
        """
        residual = x
        x = self.norm(x)

        gate_and_value = self.in_proj(x)
        gate, value = gate_and_value.chunk(2, dim=-1)

        y, new_state = self.ssm(x, value, ssm_state)
        out = self.out_proj(F.silu(gate) * y)
        return residual + out, new_state
