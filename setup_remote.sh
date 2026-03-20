#!/usr/bin/env bash
# One-time setup script for RunPod B200 pod.
# Run on the remote machine: bash setup_remote.sh
set -euo pipefail

PROJECT_DIR="/workspace/mighty-giant"
DATA_DIR="/workspace/data"
DATASET_DIR="${DATA_DIR}/fineweb-edu"
TOKENIZER_DIR="${DATA_DIR}/tokenizer-llama3"

echo "==> Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"

echo "==> Checking Python..."
python3 --version

# Install pip packages (RunPod images have Python + CUDA pre-installed)
echo "==> Installing dependencies..."
pip install --upgrade pip

# B200 requires sm_100 support — needs PyTorch nightly with CUDA 13.0
echo "==> Upgrading PyTorch for B200 (sm_100 / CUDA 13.0) support..."
pip install --force-reinstall --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
pip install transformers pyarrow wandb huggingface_hub

echo "==> Verifying PyTorch + CUDA..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# WandB login
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "==> Logging into WandB..."
    python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}')"
else
    echo "==> WANDB_API_KEY not set, skipping WandB login"
fi

# HuggingFace login
if [ -n "${HF_TOKEN:-}" ]; then
    echo "==> Logging into HuggingFace..."
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "==> Logging into HuggingFace..."
    python3 -c "from huggingface_hub import login; login(token='${HUGGING_FACE_HUB_TOKEN}')"
fi

# Download tokenizer if not cached
if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "==> Downloading Llama-3.1 tokenizer..."
    python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
tok.save_pretrained('${TOKENIZER_DIR}')
print(f'Tokenizer saved to ${TOKENIZER_DIR}, vocab_size={len(tok)}')
"
else
    echo "==> Tokenizer already cached at ${TOKENIZER_DIR}"
fi

# Download dataset if not present
if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A $DATASET_DIR 2>/dev/null)" ]; then
    echo "==> Downloading FineWeb-Edu 350BT sample..."
    mkdir -p "$DATASET_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='HuggingFaceFW/fineweb-edu',
    repo_type='dataset',
    allow_patterns='sample/350BT/*.parquet',
    local_dir='${DATASET_DIR}',
)
print('Dataset download complete')
"
else
    PARQUET_COUNT=$(find "$DATASET_DIR" -name '*.parquet' | wc -l)
    echo "==> Dataset already present at ${DATASET_DIR} (${PARQUET_COUNT} parquet files)"
fi

echo ""
echo "==> Setup complete!"
echo "    To train:"
echo "    cd ${PROJECT_DIR} && python3 train.py --stage 1 --wandb --dataset fineweb --dataset-path ${DATASET_DIR}/sample/350BT --tokenizer ${TOKENIZER_DIR}"
