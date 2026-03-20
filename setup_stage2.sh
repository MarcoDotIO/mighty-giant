#!/bin/bash
set -e

echo "=== Mighty Giant Stage 2 Setup ==="

# Clone repo
if [ ! -d "mighty-giant" ]; then
    git clone https://github.com/MarcoDotIO/mighty-giant.git
fi
cd mighty-giant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers datasets wandb huggingface_hub

# Download checkpoint from HuggingFace
echo "Downloading checkpoint..."
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('checkpoints', exist_ok=True)
hf_hub_download(
    repo_id='MarcoDotIO/mighty-giant-checkpoints',
    filename='stage1_step5000.pt',
    local_dir='checkpoints',
    local_dir_use_symlinks=False
)
print('✓ Checkpoint downloaded')
"

# Set environment variables (edit these!)
export WANDB_API_KEY="your_wandb_key_here"
export HF_TOKEN="your_hf_token_here"

# Start Stage 2 training
echo "Starting Stage 2 instruction tuning..."
python train.py \
  --stage 2 \
  --preset 4_5b \
  --dataset code_feedback \
  --dataset-path "" \
  --checkpoint checkpoints/stage1_step5000.pt \
  --lr 3e-5 \
  --lr-schedule cosine \
  --warmup-steps 100 \
  --max-steps 2000 \
  --batch-size 16 \
  --seq-len 512 \
  --eval-every 100 \
  --checkpoint-dir checkpoints_stage2

echo "✓ Stage 2 training started"
