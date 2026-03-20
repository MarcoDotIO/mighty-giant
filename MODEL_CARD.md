---
language:
- en
license: mit
tags:
- mamba
- ssm
- state-space-models
- titans
- long-context
- code
datasets:
- allenai/dolma
- m-a-p/Code-Feedback
metrics:
- perplexity
pipeline_tag: text-generation
---

# Mighty Giant

Mighty Giant is a 5.9B parameter hybrid language model that combines the Mamba-3 state-space model backbone with Titans long-term memory architecture.

## Model Details

- **Architecture**: Mamba-3 + Titans Memory Hybrid
- **Parameters**: 5.9B (5.9B trainable)
- **Layers**: 28
- **Hidden Size**: 3456
- **SSM State Size**: 128
- **Segment Length**: 256 tokens
- **Vocabulary**: 128,256 (Llama-3 tokenizer)

## Architecture Overview

### Backbone: Mamba-3
- Interleaved Mamba-3 blocks with SwiGLU FFN layers
- MIMO (Multiple-Input Multiple-Output) SSM for enhanced expressivity
- Efficient O(L) complexity for sequence length L

### Memory System: Titans
- **MAG (Memory-Augmented Gate)**: Cheap memory injection at every layer
- **MAC (Memory-Augmented Cross-attention)**: Sparse fusion blocks every 14 layers
- **Low-rank fast weights**: Rank-48 memory with depth-1 storage
- **Persistent slots**: 4 slots for long-term context retention

## Training

### Stage 1: Backbone Pretraining
- **Dataset**: Dolma (high-quality web text)
- **Steps**: 100K
- **Batch Size**: 32 × 256 tokens = 8,192 tokens/step
- **Total Tokens**: ~820M
- **Learning Rate**: 4.8e-4 with cosine decay
- **Warmup**: 200 steps
- **Hardware**: NVIDIA B200 (192GB)
- **Training Time**: ~15 hours

**Loss Curve**: 12.48 → 8.29 (eval) at step 5000

### Stage 2: Instruction Tuning (In Progress)
- **Dataset**: m-a-p/Code-Feedback (157K instruction pairs)
- **Steps**: 2,000
- **Batch Size**: 16 × 512 tokens
- **Learning Rate**: 3e-5 with cosine decay
- **Hardware**: NVIDIA H100 (80GB)

## Usage

```python
import torch
from mighty_giant import MightyGiantLM, HybridConfig

# Load config and checkpoint
config = HybridConfig.from_dict({
    "vocab_size": 128256,
    "d_model": 3456,
    "n_layers": 28,
    "d_state": 128,
    # ... see config.json for full config
})

model = MightyGiantLM(config, memory_enabled=False)
checkpoint = torch.load("stage1_step5000.pt")
model.load_state_dict(checkpoint["model"])

# Generate
input_ids = tokenizer("Once upon a time", return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=100)
```

## Performance

### Stage 1 (Backbone Only)
- **Validation Loss**: 8.29
- **Validation Perplexity**: 4,279
- **Throughput**: 13.9K tokens/sec on B200

### Stage 2 (Instruction Tuned)
Coming soon after instruction tuning completes.

## Limitations

- Currently trained on only 820M tokens (0.69% of Chinchilla optimal)
- Memory system not yet enabled (Stage 3 pending)
- No RLHF or safety tuning
- May generate biased or incorrect content

## Intended Use

- Research on hybrid SSM + memory architectures
- Code generation benchmarks (HumanEval, MBPP)
- Long-context modeling experiments
- Educational purposes

## Training Infrastructure

- **Stage 1**: RunPod B200 (192GB VRAM)
- **Stage 2**: RunPod H100 (80GB VRAM)
- **Framework**: PyTorch 2.4 + CUDA 12.4
- **Precision**: bfloat16 mixed precision

## Citation

```bibtex
@software{mighty_giant_2026,
  title={Mighty Giant: A Mamba-3 + Titans Hybrid Language Model},
  author={Marco},
  year={2026},
  url={https://github.com/MarcoDotIO/mighty-giant}
}
```

## References

- **Mamba-3**: [arXiv:2501.xxxxx](https://arxiv.org)
- **Titans**: Grattafiori et al. "Titans: Learning to Memorize at Test Time" [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- **Dolma**: Soldaini et al. "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research"

## License

MIT License - See LICENSE file for details.

## Contact

- GitHub: [@MarcoDotIO](https://github.com/MarcoDotIO)
- Model Repository: [huggingface.co/MarcoDotIO/mighty-giant-checkpoints](https://huggingface.co/MarcoDotIO/mighty-giant-checkpoints)
