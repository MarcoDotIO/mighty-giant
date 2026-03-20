from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HybridConfig


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq_len, d = x.shape
    head_dim = d // n_heads
    return x.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, n_heads * head_dim)


class MACFusionBlock(nn.Module):
    """Sparse MAC fusion block -- expensive cross-attention over persistent
    and retrieved memory slots, placed every ``fusion_block_every`` layers.

    Attention pattern:
        Q: current segment hidden states
        KV: [persistent_slots || retrieved_memory_slots || segment_hidden]
        Causal mask on the segment portion, full attention to the prefix.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.pre_norm = nn.RMSNorm(config.d_model)

        self.persistent_memory = nn.Parameter(
            torch.randn(config.persistent_slots, config.d_model) * 0.02
        )

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        retrieved_slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: [batch, seg_len, d_model] current segment hidden states.
            retrieved_slots: [batch, n_retrieved, d_model] memory retrievals.

        Returns:
            [batch, seg_len, d_model] output with residual.
        """
        residual = h
        h_normed = self.pre_norm(h)

        batch, seg_len, _ = h_normed.shape
        n_persistent = self.config.persistent_slots
        n_retrieved = retrieved_slots.size(1)
        prefix_len = n_persistent + n_retrieved

        persistent = self.persistent_memory.unsqueeze(0).expand(batch, -1, -1)
        persistent = persistent.to(dtype=h_normed.dtype)

        context = torch.cat([persistent, retrieved_slots, h_normed], dim=1)

        q = _split_heads(self.q_proj(h_normed), self.n_heads)
        k = _split_heads(self.k_proj(context), self.n_heads)
        v = _split_heads(self.v_proj(context), self.n_heads)

        total_kv_len = prefix_len + seg_len
        mask = torch.zeros(seg_len, total_kv_len, device=h.device, dtype=h.dtype)
        causal_part = torch.triu(
            torch.full((seg_len, seg_len), float("-inf"), device=h.device, dtype=h.dtype),
            diagonal=1,
        )
        mask[:, prefix_len:] = causal_part

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask.unsqueeze(0).unsqueeze(0),
            dropout_p=0.0,
        )
        attn = _merge_heads(attn)

        gated = attn * torch.sigmoid(self.gate(attn))
        return residual + self.out_proj(gated)
