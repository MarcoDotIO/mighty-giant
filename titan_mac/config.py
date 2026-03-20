from __future__ import annotations

from dataclasses import asdict, dataclass


MODEL_PRESETS = {
    "tiny_test": {
        "max_seq_len": 128,
        "d_model": 64,
        "n_heads": 4,
        "ffn_hidden": 128,
        "n_layers": 2,
        "segment_len": 32,
        "memory_tokens": 8,
        "persistent_tokens": 4,
        "memory_hidden": 128,
        "memory_depth": 2,
        "conv_kernel_size": 3,
        "tie_embeddings": True,
    },
    "paper_170m": {
        "max_seq_len": 4096,
        "d_model": 768,
        "n_heads": 12,
        "ffn_hidden": 3072,
        "n_layers": 12,
        "segment_len": 256,
        "memory_tokens": 64,
        "persistent_tokens": 8,
        "memory_hidden": 1536,
        "memory_depth": 2,
        "conv_kernel_size": 3,
        "tie_embeddings": True,
    },
}


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_heads: int
    ffn_hidden: int
    n_layers: int
    segment_len: int
    memory_tokens: int
    persistent_tokens: int
    memory_hidden: int
    memory_depth: int
    conv_kernel_size: int = 3
    tie_embeddings: bool = True

    @property
    def prefix_tokens(self) -> int:
        return self.memory_tokens + self.persistent_tokens

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        return cls(**raw)


def build_model_config(
    preset: str,
    vocab_size: int,
    *,
    seq_len_override: int | None = None,
    segment_len_override: int | None = None,
) -> ModelConfig:
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
    return ModelConfig(**config)
