from __future__ import annotations

import math
from dataclasses import asdict, dataclass


MODEL_PRESETS = {
    "tiny_test": {
        "max_seq_len": 128,
        "d_model": 64,
        "n_layers": 4,
        "d_state": 16,
        "mimo_rank": 2,
        "ffn_hidden": 128,
        "segment_len": 32,
        "memory_depth": 1,
        "fast_weight_rank": 8,
        "persistent_slots": 4,
        "retrieved_memory_slots": 2,
        "fusion_block_every": 4,
        "tie_embeddings": True,
    },
    "1_3b": {
        "max_seq_len": 4096,
        "d_model": 2048,
        "n_layers": 24,
        "d_state": 128,
        "mimo_rank": 4,
        "ffn_hidden": 5504,
        "segment_len": 256,
        "memory_depth": 1,
        "fast_weight_rank": 32,
        "persistent_slots": 4,
        "retrieved_memory_slots": 2,
        "fusion_block_every": 12,
        "tie_embeddings": True,
    },
    "4_5b": {
        # ~4.6B params — fits B200 192GB with batch_size=4, seq_len=4096
        "max_seq_len": 4096,
        "d_model": 3456,
        "n_layers": 28,
        "d_state": 128,
        "mimo_rank": 4,
        "ffn_hidden": 9216,  # ~2.67x d_model
        "segment_len": 256,
        "memory_depth": 1,
        "fast_weight_rank": 48,
        "persistent_slots": 4,
        "retrieved_memory_slots": 2,
        "fusion_block_every": 14,
        "tie_embeddings": True,
    },
}


@dataclass
class HybridConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int

    # Mamba-3 SSM
    d_state: int = 64
    mimo_rank: int = 4
    mimo_start_fraction: float = 0.75
    bc_norm: bool = True
    bc_bias: bool = True
    dt_min: float = 0.001
    dt_max: float = 0.1

    # SwiGLU FFN
    ffn_hidden: int = 0

    # Titans Memory
    segment_len: int = 256
    memory_depth: int = 1
    fast_weight_rank: int = 16
    persistent_slots: int = 4
    retrieved_memory_slots: int = 2
    fusion_block_every: int = 12

    tie_embeddings: bool = True

    @property
    def ffn_hidden_resolved(self) -> int:
        if self.ffn_hidden > 0:
            return self.ffn_hidden
        raw = int(self.d_model * 8 / 3)
        return ((raw + 255) // 256) * 256

    @property
    def head_dim(self) -> int:
        return 64

    @property
    def n_heads(self) -> int:
        return self.d_model // self.head_dim

    def is_mimo_layer(self, layer_idx: int) -> bool:
        return layer_idx >= int(self.n_layers * self.mimo_start_fraction)

    def is_fusion_layer(self, layer_idx: int) -> bool:
        if self.fusion_block_every <= 0:
            return False
        return layer_idx > 0 and layer_idx % self.fusion_block_every == 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> HybridConfig:
        return cls(**raw)


def build_model_config(
    preset: str,
    vocab_size: int,
    *,
    seq_len_override: int | None = None,
    segment_len_override: int | None = None,
) -> HybridConfig:
    if preset not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Expected one of: {', '.join(sorted(MODEL_PRESETS))}"
        )
    config = dict(MODEL_PRESETS[preset])
    config["vocab_size"] = vocab_size
    if seq_len_override is not None:
        config["max_seq_len"] = seq_len_override
    if segment_len_override is not None:
        config["segment_len"] = segment_len_override
    return HybridConfig(**config)
