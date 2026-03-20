#!/usr/bin/env python3
"""Mighty Giant training script.

Supports four training stages:
  Stage 1: Pretrain pure Mamba-3 backbone (memory disabled) on 2K-4K context
  Stage 2: Attach Titans memory, differential LR, same context
  Stage 3: Context curriculum (4K -> 16K -> 64K)
  Stage 4: Evaluation only

Usage:
  python train.py --stage 1 --preset prototype_400m --dataset fineweb --dataset-path /data/fineweb
  python train.py --stage 2 --resume checkpoints/stage1_final.pt
  python train.py --stage 3 --resume checkpoints/stage2_final.pt --context-curriculum 4096,16384,65536
  python train.py --stage 4 --resume checkpoints/stage3_final.pt --eval-only
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

# Reuse titan_mac utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from titan_mac.checkpoint import load_checkpoint, save_checkpoint
from titan_mac.data import load_datasets
from titan_mac.device import autocast_context, resolve_device, resolve_dtype, use_grad_scaler
from titan_mac.tokenization import load_tokenizer

from mighty_giant.config import HybridConfig, build_model_config
from mighty_giant.model import MightyGiantLM


def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def maybe_init_wandb(args, *, model, config, param_count):
    if not args.wandb:
        return None

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(
            "--wandb was requested but WANDB_API_KEY is not set. Add it to .env or the shell environment."
        )

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. pip install wandb") from exc

    project = args.wandb_project or os.getenv("WANDB_PROJECT", "mighty-giant")
    entity = os.getenv("WANDB_ENTITY") or None

    run_config = config.to_dict()
    run_config.update({
        "stage": args.stage,
        "preset": args.preset,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "lr_schedule": args.lr_schedule,
        "grad_clip": args.grad_clip,
        "param_count": param_count,
    })

    run = wandb.init(
        project=project,
        entity=entity,
        name=args.wandb_run_name or f"stage{args.stage}-{args.preset}",
        config=run_config,
        resume="allow",
    )
    # Skip wandb.watch for large models — it doubles memory usage for gradient logging
    if param_count < 1e9:
        wandb.watch(_unwrap_model(model), log="gradients", log_freq=100)
    return run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mighty Giant training")

    # Stage
    p.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--eval-only", action="store_true")

    # Model
    p.add_argument("--preset", type=str, default="4_5b")
    p.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

    # Data
    p.add_argument("--dataset", type=str, default="fineweb")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-sequences", type=int, default=None)

    # Training (defaults from experiment sweep: exp 97 best config)
    p.add_argument("--lr", type=float, default=4.8e-4)
    p.add_argument("--memory-lr-scale", type=float, default=1.0,
                    help="LR multiplier for memory params in stage 2+")
    p.add_argument("--backbone-lr-scale", type=float, default=0.1,
                    help="LR multiplier for backbone params in stage 2+")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--lr-schedule", type=str, default="cosine",
                    choices=["cosine", "flat"],
                    help="LR schedule after warmup: 'flat' holds LR constant, 'cosine' decays")
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)

    # Context curriculum (stage 3)
    p.add_argument("--context-curriculum", type=str, default="4096,16384,65536",
                    help="Comma-separated seq lengths for stage 3")
    p.add_argument("--curriculum-steps", type=int, default=20000,
                    help="Steps per curriculum stage")

    # Checkpoint
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # Device
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")

    # Seq len override
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--segment-len", type=int, default=None)

    # WandB
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", default=None, help="Override WANDB_PROJECT for this run.")
    p.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")

    return p.parse_args()


def build_optimizer(
    model: MightyGiantLM,
    args: argparse.Namespace,
) -> torch.optim.AdamW:
    """Build AdamW with differential learning rates for staged training."""
    betas = (args.beta1, args.beta2)
    fused = torch.cuda.is_available()

    if args.stage == 1:
        return torch.optim.AdamW(
            [{"params": list(model.parameters()), "_lr_scale": 1.0}],
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
            fused=fused,
        )

    # Stages 2-3: differential LR
    backbone_params = []
    memory_params = []
    memory_module_names = {"memory", "memory_updater", "memory_query_proj", "mag_injections", "fusion_blocks"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top_module = name.split(".")[0]
        if top_module in memory_module_names:
            memory_params.append(param)
        else:
            backbone_params.append(param)

    return torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": args.lr * args.backbone_lr_scale,
                "_lr_scale": args.backbone_lr_scale,
            },
            {
                "params": memory_params,
                "lr": args.lr * args.memory_lr_scale,
                "_lr_scale": args.memory_lr_scale,
            },
        ],
        betas=betas,
        weight_decay=args.weight_decay,
        fused=fused,
    )


def lr_schedule(step: int, warmup: int, total: int, lr: float, mode: str = "flat") -> float:
    """LR schedule with linear warmup, then flat or cosine decay."""
    if step < warmup:
        # Start from 1/warmup instead of 0 so step 0 isn't wasted
        return lr * max(step, 1) / max(warmup, 1)
    if mode == "flat":
        return lr
    # cosine
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(
    model: MightyGiantLM,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 50,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    state = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            input_ids = batch.to(device)
            # Reset state if batch size changed (e.g. last batch is smaller)
            if state is not None and input_ids.size(0) != state.ssm_states[0].h.size(0):
                state = None
            with autocast_context(device, dtype):
                output = model(input_ids, labels=input_ids, state=state)
            total_loss += output.loss.item()
            state = output.state.detach()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def _fmt_num(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


# ── ANSI helpers ──────────────────────────────────────────────────
_B = "\033[1m"
_D = "\033[2m"
_R = "\033[0m"
_BLUE = "\033[38;5;75m"
_GREEN = "\033[38;5;114m"
_YELLOW = "\033[38;5;221m"
_RED = "\033[38;5;203m"
_CYAN = "\033[38;5;117m"
_MAGENTA = "\033[38;5;183m"
_GRAY = "\033[38;5;245m"


def _bar(progress: float, width: int = 30) -> str:
    filled = int(width * min(progress, 1.0))
    empty = width - filled
    bar = f"{_BLUE}{'━' * filled}{_GRAY}{'─' * empty}{_R}"
    pct = f"{progress * 100:5.1f}%"
    return f"{bar} {_D}{pct}{_R}"


def _print_banner(stage: int, preset: str, memory_enabled: bool) -> None:
    mem_status = f"{_GREEN}ON{_R}" if memory_enabled else f"{_GRAY}OFF{_R}"
    print(f"""
{_B}{_BLUE}  ╔══════════════════════════════════════════╗
  ║         Mighty Giant  {_R}{_B}{_MAGENTA}Training{_R}{_B}{_BLUE}            ║
  ╚══════════════════════════════════════════╝{_R}
""")
    print(f"  {_CYAN}❯{_R} {_B}Stage {stage}{_R}  {_D}({preset}){_R}  memory={mem_status}")


def _print_model_card(config, param_count: int, trainable_count: int, device, dtype) -> None:
    print(f"""
  {_GRAY}┌─────────────────────────────────────────┐{_R}
  {_GRAY}│{_R} {_B}Model{_R}                                     {_GRAY}│{_R}
  {_GRAY}├─────────────────────────────────────────┤{_R}
  {_GRAY}│{_R}  Parameters    {_B}{_fmt_num(param_count):>8s}{_R} ({_fmt_num(trainable_count)} trainable) {_GRAY}│{_R}
  {_GRAY}│{_R}  Layers        {_B}{config.n_layers:>8d}{_R}                    {_GRAY}│{_R}
  {_GRAY}│{_R}  d_model       {_B}{config.d_model:>8d}{_R}                    {_GRAY}│{_R}
  {_GRAY}│{_R}  d_state       {_B}{config.d_state:>8d}{_R}                    {_GRAY}│{_R}
  {_GRAY}│{_R}  Segment len   {_B}{config.segment_len:>8d}{_R}                    {_GRAY}│{_R}
  {_GRAY}│{_R}  Device        {_B}{str(device):>8s}{_R}  dtype={_D}{dtype}{_R}    {_GRAY}│{_R}
  {_GRAY}└─────────────────────────────────────────┘{_R}
""")


def train(args: argparse.Namespace) -> None:
    # Performance settings (from experiment sweep)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer)

    config = build_model_config(
        args.preset,
        vocab_size,
        seq_len_override=args.seq_len,
        segment_len_override=args.segment_len,
    )

    memory_enabled = args.stage >= 3
    model = MightyGiantLM(config, memory_enabled=memory_enabled)

    _print_banner(args.stage, args.preset, memory_enabled)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"  {_YELLOW}↻{_R} Loading checkpoint: {_D}{args.resume}{_R}")
        ckpt = load_checkpoint(args.resume, device="cpu")
        model_state = ckpt["model_state"]

        # For stage 2, load backbone weights from stage 1 (memory weights init fresh)
        if args.stage == 2:
            current_keys = set(model.state_dict().keys())
            filtered = {k: v for k, v in model_state.items() if k in current_keys}
            model.load_state_dict(filtered, strict=False)
            print(f"    Loaded {len(filtered)}/{len(current_keys)} keys from checkpoint")
        else:
            model.load_state_dict(model_state)
            start_step = ckpt.get("step", 0)
            print(f"    Resumed from step {start_step}")

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _print_model_card(config, param_count, trainable_count, device, dtype)

    model = model.to(device=device, dtype=dtype)

    # WandB
    wandb_run = maybe_init_wandb(args, model=model, config=config, param_count=param_count)

    if args.eval_only:
        seq_len = args.seq_len or config.max_seq_len
        _, val_dataset = load_datasets(
            args.dataset, args.dataset_path, tokenizer,
            seq_len=seq_len, max_docs=args.max_docs, max_sequences=args.max_sequences,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        val_loss = evaluate(model, val_loader, device, dtype)
        print(f"Validation loss: {val_loss:.4f}, perplexity: {math.exp(val_loss):.2f}")
        return

    optimizer = build_optimizer(model, args)

    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler(device, dtype) else None

    # Context curriculum for stage 3
    if args.stage == 3:
        curriculum = [int(x) for x in args.context_curriculum.split(",")]
    else:
        curriculum = [args.seq_len or config.max_seq_len]

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = start_step
    _step_start = time.time()
    _tokens_total = 0

    for curriculum_idx, seq_len in enumerate(curriculum):
        print(f"\n  {_CYAN}❯{_R} {_B}Curriculum {curriculum_idx + 1}/{len(curriculum)}{_R}  seq_len={_B}{seq_len}{_R}")
        print(f"    {_D}Loading dataset...{_R}", flush=True)

        t_data = time.time()
        train_dataset, val_dataset = load_datasets(
            args.dataset, args.dataset_path, tokenizer,
            seq_len=seq_len, max_docs=args.max_docs, max_sequences=args.max_sequences,
        )
        is_iterable = isinstance(train_dataset, IterableDataset)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=not is_iterable, drop_last=True,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"    {_GREEN}✔{_R} Dataset loaded in {time.time() - t_data:.1f}s  "
              f"({len(train_dataset) if hasattr(train_dataset, '__len__') else '?'} train sequences)", flush=True)

        tok_per_step = args.batch_size * seq_len
        stage_steps = args.curriculum_steps if args.stage == 3 else args.max_steps
        stage_end = global_step + stage_steps

        print(f"    {_D}Tokens/step: {_fmt_num(tok_per_step)}  "
              f"Target: {_fmt_num(stage_steps)} steps{_R}")
        print(f"\n    {_D}Starting first step (may take a few minutes for large models)...{_R}", flush=True)

        # Snapshot a few weights to verify they change
        _diag_params = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                _diag_params[name] = p.data.clone().flatten()[:8]
                if len(_diag_params) >= 3:
                    break

        model.train()
        state = None
        epoch = 0

        while global_step < stage_end:
            epoch += 1
            for batch in train_loader:
                if global_step >= stage_end:
                    break

                step_t0 = time.time()

                input_ids = batch.to(device)

                # LR schedule
                current_lr = lr_schedule(
                    global_step - start_step,
                    args.warmup_steps,
                    args.max_steps,
                    args.lr,
                    mode=args.lr_schedule,
                )
                for pg in optimizer.param_groups:
                    base_scale = pg.get("_lr_scale", 1.0)
                    pg["lr"] = current_lr * base_scale

                optimizer.zero_grad(set_to_none=True)

                # Forward
                fwd_t0 = time.time()
                with autocast_context(device, dtype):
                    output = model(input_ids, labels=input_ids, state=state)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                fwd_ms = (time.time() - fwd_t0) * 1000

                loss = output.loss
                state = output.state.detach()

                # Backward
                bwd_t0 = time.time()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                bwd_ms = (time.time() - bwd_t0) * 1000

                step_ms = (time.time() - step_t0) * 1000
                global_step += 1
                _tokens_total += input_ids.numel()

                # GC: freeze after first step to avoid mid-training pauses
                if global_step == start_step + 1:
                    gc.collect()
                    gc.freeze()
                    gc.disable()
                    _step_start = time.time()
                    _tokens_total = input_ids.numel()

                    # Diagnostic: verify gradients flow and weights change
                    print(f"\n    {_YELLOW}⚡ Step 1 diagnostics:{_R}", flush=True)
                    grad_norms = {}
                    zero_grads = 0
                    total_params = 0
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            total_params += 1
                            if p.grad is not None:
                                gn = p.grad.norm().item()
                                if gn == 0:
                                    zero_grads += 1
                                if len(grad_norms) < 5:
                                    grad_norms[name] = gn
                            else:
                                zero_grads += 1
                                if len(grad_norms) < 5:
                                    grad_norms[name] = "None"
                    print(f"      Grad norms (sample): ", flush=True)
                    for name, gn in grad_norms.items():
                        print(f"        {name}: {gn}", flush=True)
                    print(f"      Zero/None grads: {zero_grads}/{total_params}", flush=True)

                    # Check if weights actually changed
                    weight_changes = {}
                    for name, p in model.named_parameters():
                        if name in _diag_params:
                            new_vals = p.data.flatten()[:8]
                            old_vals = _diag_params[name].to(new_vals.device)
                            diff = (new_vals - old_vals).abs().max().item()
                            weight_changes[name] = diff
                    print(f"      Weight changes: {weight_changes}", flush=True)

                    # Check logit distribution
                    with torch.no_grad():
                        sample_out = model(input_ids[:1, :32], labels=None, state=None, reset_state=True)
                        logits = sample_out.logits[0, -1]  # last token
                        probs = torch.softmax(logits.float(), dim=-1)
                        top5 = torch.topk(probs, 5)
                        print(f"      Logit stats: min={logits.min().item():.4f} max={logits.max().item():.4f} std={logits.std().item():.4f}", flush=True)
                        print(f"      Top-5 probs: {top5.values.tolist()}", flush=True)
                        print(f"      Expected uniform prob: {1/config.vocab_size:.6f}", flush=True)
                        entropy = -(probs * probs.log()).sum().item()
                        print(f"      Entropy: {entropy:.4f} (uniform={math.log(config.vocab_size):.4f})", flush=True)
                    print(flush=True)

                # Log first 5 steps individually, then every log_every
                steps_done = global_step - start_step
                should_log = steps_done <= 5 or global_step % args.log_every == 0

                if should_log:
                    elapsed = time.time() - _step_start
                    tok_per_sec = _tokens_total / max(elapsed, 1e-6)
                    gpu_mem = ""
                    if device.type == "cuda":
                        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                        gpu_mem = f"  {_GRAY}mem={mem_gb:.1f}GB{_R}"
                    progress = (global_step - start_step) / max(stage_end - start_step, 1)
                    bar = _bar(progress)
                    loss_color = _GREEN if loss.item() < 4.0 else _YELLOW if loss.item() < 8.0 else _RED
                    print(
                        f"  {bar}  "
                        f"step {_B}{global_step}{_R}  "
                        f"{loss_color}loss={loss.item():.4f}{_R}  "
                        f"ppl={math.exp(min(loss.item(), 20)):.1f}  "
                        f"lr={current_lr:.1e}  "
                        f"{_CYAN}{_fmt_num(tok_per_sec)} tok/s{_R}"
                        f"{gpu_mem}  "
                        f"{_D}fwd={fwd_ms:.0f}ms bwd={bwd_ms:.0f}ms total={step_ms:.0f}ms{_R}",
                        flush=True,
                    )
                else:
                    # Heartbeat: dot every step so you know it's alive
                    print(f"    {_D}.{_R}", end="", flush=True)
                    if steps_done % args.log_every == args.log_every - 1:
                        print(flush=True)  # newline before next full log

                # WandB train metrics (always log, not just on printed steps)
                if wandb_run is not None and should_log:
                    wandb_metrics = {
                        "train/loss": loss.item(),
                        "train/perplexity": math.exp(min(loss.item(), 20)),
                        "train/lr": current_lr,
                        "train/tokens_per_sec": tok_per_sec,
                        "train/step_ms": step_ms,
                        "train/fwd_ms": fwd_ms,
                        "train/bwd_ms": bwd_ms,
                    }
                    if device.type == "cuda":
                        wandb_metrics["train/gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                    wandb_run.log(wandb_metrics, step=global_step)

                if global_step % args.eval_every == 0:
                    eval_start = time.time()
                    val_loss = evaluate(model, val_loader, device, dtype)
                    eval_time = time.time() - eval_start
                    print(
                        f"    {_MAGENTA}⟐ eval{_R}  "
                        f"val_loss={_B}{val_loss:.4f}{_R}  "
                        f"val_ppl={_B}{math.exp(min(val_loss, 20)):.1f}{_R}  "
                        f"{_D}({eval_time:.1f}s){_R}"
                    )
                    model.train()

                    # WandB eval metrics
                    if wandb_run is not None:
                        eval_metrics = {
                            "eval/loss": val_loss,
                            "eval/perplexity": math.exp(min(val_loss, 20)),
                        }
                        wandb_run.log(eval_metrics, step=global_step)

                if global_step % args.save_every == 0:
                    ckpt_path = checkpoint_dir / f"stage{args.stage}_step{global_step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=None,
                        scaler=scaler,
                        model_config=config,
                        train_args=vars(args),
                        tokenizer_ref=args.tokenizer,
                        step=global_step,
                    )
                    print(f"    {_GREEN}💾 saved{_R}  {_D}{ckpt_path}{_R}")

    # Final checkpoint
    final_path = checkpoint_dir / f"stage{args.stage}_final.pt"
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        model_config=config,
        train_args=vars(args),
        tokenizer_ref=args.tokenizer,
        step=global_step,
    )
    print(f"\n  {_GREEN}✔{_R} {_B}Training complete.{_R}")
    print(f"    Final checkpoint: {_D}{final_path}{_R}\n")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
