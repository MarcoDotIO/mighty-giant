from __future__ import annotations

import torch
import torch.nn as nn


def _build_freqs(d_state: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Base frequencies for RoPE-style rotation, one per pair of state dims."""
    n_pairs = d_state // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, n_pairs, device=device, dtype=dtype) / n_pairs))
    return freqs


def build_rotation_matrices(
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables from angle tensor for block-diagonal rotation."""
    return theta.cos(), theta.sin()


def apply_rotation(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply block-diagonal 2x2 rotation to paired dimensions."""
    paired_dim = x.shape[-1] // 2
    if paired_dim == 0:
        return x

    x1 = x[..., :paired_dim]
    x2 = x[..., paired_dim : 2 * paired_dim]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    rotated = torch.cat([out1, out2], dim=-1)
    if 2 * paired_dim == x.shape[-1]:
        return rotated
    return torch.cat([rotated, x[..., 2 * paired_dim :]], dim=-1)


class DataDependentRoPE(nn.Module):
    """Data-dependent RoPE on B/C projections for the live Mamba backbone."""

    def __init__(self, d_model: int, d_state: int, n_heads: int = 1):
        super().__init__()
        self.d_state = d_state
        self.n_heads = n_heads
        self.n_pairs = d_state // 2
        self.theta_proj = nn.Linear(d_model, n_heads * self.n_pairs)
        base_freqs = _build_freqs(d_state, torch.device("cpu"), torch.float32)
        self.register_buffer("base_freqs", base_freqs, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            B: [batch, heads, seq_len, d_state] or [batch, heads, seq_len, rank, d_state]
            C: [batch, heads, seq_len, d_state] or [batch, heads, seq_len, rank, d_state]
        """
        batch, seq_len, _ = x.shape
        theta_data = self.theta_proj(x).view(batch, seq_len, self.n_heads, self.n_pairs)
        theta_data = theta_data.permute(0, 2, 1, 3)

        base = self.base_freqs.to(device=x.device, dtype=x.dtype)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        positional_theta = positions[:, None] * base[None, :]
        theta = positional_theta[None, None, :, :] + theta_data
        cos, sin = build_rotation_matrices(theta)

        if B.ndim == 5:
            cos = cos.unsqueeze(3)
            sin = sin.unsqueeze(3)

        return apply_rotation(B, cos, sin), apply_rotation(C, cos, sin)
