from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer


def load_tokenizer(tokenizer_ref: str):
    if not tokenizer_ref:
        raise ValueError("A tokenizer path or model id is required.")

    path = Path(tokenizer_ref).expanduser()
    looks_like_local_path = path.is_absolute() or tokenizer_ref.startswith(".") or tokenizer_ref.startswith("~")
    if looks_like_local_path and not path.exists():
        raise FileNotFoundError(
            f"Tokenizer path does not exist: {path}. "
            "Pass a real local tokenizer directory or a Hugging Face model id "
            "such as 'meta-llama/Llama-2-7b-hf'."
        )

    resolved_ref = str(path) if path.exists() else tokenizer_ref
    tokenizer = AutoTokenizer.from_pretrained(resolved_ref, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
