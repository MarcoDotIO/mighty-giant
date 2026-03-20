# Mighty Giant

A 5.9B parameter hybrid language model combining Mamba-3 SSM backbone with Titans long-term memory architecture.

## Architecture

- **Backbone**: Mamba-3 with interleaved SwiGLU FFN layers
- **Memory**: Low-rank fast-weight memory with MAG injection and sparse MAC fusion
- **Size**: 5.9B parameters (28 layers, d_model=3456, d_state=128)
- **Context**: Segment length 256, expandable via memory system

## Training Stages

### Stage 1: Backbone Pretraining
Pure Mamba-3 backbone on language modeling (memory disabled).

```bash
python train.py \
  --stage 1 \
  --preset 4_5b \
  --dataset dolma \
  --dataset-path /path/to/dolma \
  --lr 4.8e-4 \
  --lr-schedule cosine \
  --warmup-steps 200 \
  --max-steps 100000 \
  --batch-size 32 \
  --seq-len 256
```

### Stage 2: Instruction Tuning
Fine-tune on code instructions for downstream tasks.

```bash
python train.py \
  --stage 2 \
  --preset 4_5b \
  --dataset code_feedback \
  --checkpoint checkpoints/stage1_step5000.pt \
  --lr 3e-5 \
  --lr-schedule cosine \
  --warmup-steps 100 \
  --max-steps 2000 \
  --batch-size 16 \
  --seq-len 512
```

### Stage 3: Memory Training
Enable memory system with context curriculum (not yet implemented).

## Setup

### Local Development

```bash
# Install dependencies
pip install torch transformers datasets wandb

# Run training
python train.py --stage 1 --preset 4_5b --dataset dolma --dataset-path /data/dolma
```

### RunPod Deployment

```bash
# Deploy to RunPod B200
./deploy.sh --setup  # First time setup
./deploy.sh --train  # Sync code and start training

# Check status
./deploy.sh --status

# Stop pod
./deploy.sh --stop
```

## Model Checkpoints

Checkpoints available at [huggingface.co/MarcoDotIO/mighty-giant-checkpoints](https://huggingface.co/MarcoDotIO/mighty-giant-checkpoints)

Download:
```bash
huggingface-cli download MarcoDotIO/mighty-giant-checkpoints stage1_step5000.pt --local-dir ./checkpoints
```

## Requirements

- **Stage 1**: NVIDIA B200 (192GB) for batch_size=32, seq_len=256
- **Stage 2**: NVIDIA H100 (80GB) for batch_size=16, seq_len=512
- Python 3.11+
- PyTorch 2.4+
- CUDA 12.4+

## Configuration

Model presets in `mighty_giant/config.py`:
- `tiny_test`: 64M params (for testing)
- `1_3b`: 1.3B params
- `4_5b`: 5.9B params (default)

## Project Structure

```
mighty_giant/
├── config.py          # Model configurations
├── model.py           # MightyGiantLM main model
├── mamba3/            # Mamba-3 SSM implementation
├── memory/            # Titans memory components
└── state.py           # State management

titan_mac/
├── data.py            # Dataset loaders
├── tokenization.py    # Tokenizer utilities
└── checkpoint.py      # Checkpoint save/load

train.py               # Training script
deploy.py              # RunPod deployment
```

## Citation

Based on:
- Mamba-3: [arXiv:2501.xxxxx](https://arxiv.org)
- Titans: [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)

## License

MIT
