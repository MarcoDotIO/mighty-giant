from __future__ import annotations

from contextlib import nullcontext

import torch


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if requested not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{requested}'.")
    return DTYPE_MAP[requested]


def use_grad_scaler(device: torch.device, dtype: torch.dtype) -> bool:
    return device.type == "cuda" and dtype == torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu" and dtype == torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=dtype)
    return nullcontext()
