from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block with pre-RMSNorm (Llama pattern).

    Alternates with Mamba3Blocks in the backbone.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))
