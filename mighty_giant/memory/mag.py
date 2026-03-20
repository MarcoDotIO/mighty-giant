from __future__ import annotations

import torch
import torch.nn as nn


class MAGInjection(nn.Module):
    """Memory As Gate -- cheap gated residual injection applied at every layer.

    Given current hidden states h and a memory readout r (computed once per
    segment), produces:
        gate = sigmoid(W_g [h ; r_broadcast])
        output = h + gate * W_r(r_broadcast)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        memory_readout: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, d_model] current hidden states.
            memory_readout: [batch, d_model] mean-pooled memory retrieval.

        Returns:
            [batch, seq_len, d_model] gated output.
        """
        r = memory_readout.unsqueeze(1).expand_as(h)
        gate = torch.sigmoid(self.gate_proj(torch.cat([h, r], dim=-1)))
        return h + gate * self.value_proj(r)
