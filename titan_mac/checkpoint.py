from __future__ import annotations

from pathlib import Path

import torch

from .config import ModelConfig


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    model_config: ModelConfig,
    train_args: dict,
    tokenizer_ref: str,
    step: int,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    base_model = unwrap_model(model)
    payload = {
        "model_state": _to_cpu(base_model.state_dict()),
        "optimizer_state": _to_cpu(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": _to_cpu(scaler.state_dict()) if scaler is not None else None,
        "model_config": model_config.to_dict(),
        "train_args": train_args,
        "tokenizer_ref": tokenizer_ref,
        "step": step,
    }
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(checkpoint_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return checkpoint_path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    return torch.load(Path(path), map_location=device)


def model_config_from_checkpoint(checkpoint: dict) -> ModelConfig:
    return ModelConfig.from_dict(checkpoint["model_config"])
