from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HybridConfig
from .rope import DataDependentRoPE


def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Associative prefix scan for h_t = a_t * h_{t-1} + b_t.

    The first two dimensions of `a` and `b` must match `[batch, time]`.
    Trailing dimensions may differ as long as they are broadcast-compatible.
    """
    if a.shape[:2] != b.shape[:2]:
        raise ValueError("parallel_scan expects matching [batch, time] dimensions.")

    batch, seq_len = a.shape[:2]
    a_rest = a.shape[2:]
    b_rest = b.shape[2:]

    log_t = 0
    while (1 << log_t) < seq_len:
        log_t += 1
    padded_len = 1 << log_t

    if padded_len > seq_len:
        a_pad = torch.ones(
            (batch, padded_len - seq_len, *a_rest),
            device=a.device,
            dtype=a.dtype,
        )
        b_pad = torch.zeros(
            (batch, padded_len - seq_len, *b_rest),
            device=b.device,
            dtype=b.dtype,
        )
        a = torch.cat([a, a_pad], dim=1)
        b = torch.cat([b, b_pad], dim=1)

    aa, bb = a, b
    for depth in range(log_t):
        stride = 1 << depth
        a_left = torch.cat(
            [
                torch.ones((batch, stride, *a_rest), device=a.device, dtype=a.dtype),
                aa[:, :-stride],
            ],
            dim=1,
        )
        b_left = torch.cat(
            [
                torch.zeros((batch, stride, *b_rest), device=b.device, dtype=b.dtype),
                bb[:, :-stride],
            ],
            dim=1,
        )
        new_aa = aa * a_left
        new_bb = aa * b_left + bb
        aa, bb = new_aa, new_bb

    return bb[:, :seq_len]


class BCNorm(nn.Module):
    """RMS normalization applied to B/C projections (Mamba-3 §3.4)."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * x / (rms + 1e-6)


class MultiHeadSSM(nn.Module):
    """Multi-head SSM core with trapezoidal update and optional MIMO expansion.

    The recurrent state is stored as rank channels per head with shape
    `[mimo_rank, d_state]`. This keeps the richer trapezoidal / BC-bias / RoPE
    backbone features while avoiding the prohibitive `d_state x head_dim` state
    tensor that is too expensive at the large presets.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_heads: int,
        config: HybridConfig,
        *,
        mimo_rank: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.mimo_rank = mimo_rank

        self.dt_proj = nn.Linear(d_model, n_heads, bias=True)
        self.A_log = nn.Parameter(torch.randn(n_heads, d_state) * 0.1)
        self.lambda_proj = nn.Linear(d_model, n_heads, bias=True)

        self.B_proj = nn.Linear(d_model, n_heads * d_state * mimo_rank, bias=False)
        self.C_proj = nn.Linear(d_model, n_heads * d_state * mimo_rank, bias=False)

        self.bc_norm_B = BCNorm(d_state) if config.bc_norm else nn.Identity()
        self.bc_norm_C = BCNorm(d_state) if config.bc_norm else nn.Identity()

        if config.bc_bias:
            self.B_bias = nn.Parameter(torch.zeros(n_heads, mimo_rank, d_state))
            self.C_bias = nn.Parameter(torch.zeros(n_heads, mimo_rank, d_state))
        else:
            self.B_bias = None
            self.C_bias = None

        self.rope = DataDependentRoPE(d_model, d_state, n_heads=n_heads)
        self.D = nn.Parameter(torch.ones(n_heads))

        # Keep the recurrent state in rank channels rather than head_dim channels.
        # This preserves upper-layer MIMO behavior without materializing
        # [batch, heads, time, d_state, head_dim] activations.
        self.mimo_in_proj = nn.Linear(self.head_dim, mimo_rank, bias=False)
        self.mimo_out_proj = nn.Linear(mimo_rank, self.head_dim, bias=False)

        self.dt_min = config.dt_min
        self.dt_max = config.dt_max

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.n_heads,
            self.mimo_rank,
            self.d_state,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        value: torch.Tensor,
        prev_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model] normalized hidden states for SSM params.
            value: [batch, seq_len, d_model] value stream used for state input.
            prev_h: [batch, n_heads, mimo_rank, d_state] or None

        Returns:
            y: [batch, seq_len, d_model]
            final_h: [batch, n_heads, mimo_rank, d_state]
        """
        batch, seq_len, _ = x.shape
        H = self.n_heads
        N = self.d_state
        R = self.mimo_rank

        dt = F.softplus(self.dt_proj(x)).clamp(self.dt_min, self.dt_max)
        dt = dt.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # [B, H, T, 1, 1]

        A = -F.softplus(self.A_log).unsqueeze(0).unsqueeze(2)  # [1, H, 1, N]
        alpha = torch.exp(dt.squeeze(-1) * A)  # [B, H, T, N]
        lam = torch.sigmoid(self.lambda_proj(x)).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        B = self.B_proj(x).view(batch, seq_len, H, R, N).permute(0, 2, 1, 3, 4)
        C = self.C_proj(x).view(batch, seq_len, H, R, N).permute(0, 2, 1, 3, 4)

        B = self.bc_norm_B(B.reshape(-1, N)).view(batch, H, seq_len, R, N)
        C = self.bc_norm_C(C.reshape(-1, N)).view(batch, H, seq_len, R, N)

        if self.B_bias is not None:
            B = B + self.B_bias[None, :, None, :, :]
        if self.C_bias is not None:
            C = C + self.C_bias[None, :, None, :, :]

        B, C = self.rope(x, B, C)

        value_heads = value.view(batch, seq_len, H, self.head_dim).permute(0, 2, 1, 3)
        x_rank = self.mimo_in_proj(value_heads)  # [B, H, T, R]
        Bx = B * x_rank.unsqueeze(-1) * dt  # [B, H, T, R, N]

        if prev_h is None:
            prev_h = torch.zeros(
                batch, H, R, N, device=x.device, dtype=x.dtype
            )
        h = prev_h
        prev_Bx = torch.zeros_like(Bx[:, :, 0])
        outputs = []
        for t in range(seq_len):
            alpha_t = alpha[:, :, t].unsqueeze(2)  # [B, H, 1, N]
            lam_t = lam[:, :, t]  # [B, H, 1, 1]
            h = alpha_t * h + (1.0 - lam_t) * alpha_t * prev_Bx + lam_t * Bx[:, :, t]
            y_rank_t = (C[:, :, t] * h).sum(dim=-1)  # [B, H, R]
            outputs.append(y_rank_t)
            prev_Bx = Bx[:, :, t]

        final_h = h.contiguous()
        y_rank = torch.stack(outputs, dim=2)  # [B, H, T, R]
        y = self.mimo_out_proj(y_rank)
        y = y + self.D[None, :, None, None] * value_heads
        y = y.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, self.d_model)

        return y, final_h
