# Nemotron Next 8B LoRA Fine-tuning with FiQA Dataset

Fine-tune [NVIDIA Nemotron Next 8B](https://huggingface.co/nvidia/Nemotron-Next-8B) using LoRA (Low-Rank Adaptation) on the [FiQA dataset](https://huggingface.co/datasets/explodinggradients/fiqa) for financial question answering.

## Overview

This project demonstrates:
- Loading Nemotron Next 8B with NeMo AutoModel
- Fine-tuning with LoRA for memory efficiency
- Evaluation on financial domain Q&A tasks
- Comparison of baseline vs fine-tuned performance

## Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (A100, H100, or RTX 4090 recommended)
- **CUDA**: 12.0+
- **Python**: 3.10+

## Setup

```bash
# Clone NeMo AutoModel
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
uv venv && uv pip install -e .

# Install additional dependencies
cd ../nemotron-next-fiqa
uv pip install -r requirements.txt
```

## Usage

```bash
# Run the notebook
jupyter notebook examples/nemotron_fiqa_lora.ipynb
```

## Project Structure

```
nemotron-next-fiqa/
├── examples/
│   └── nemotron_fiqa_lora.ipynb   # Main training notebook
├── configs/
│   └── lora_config.yaml           # LoRA hyperparameters
├── requirements.txt
├── README.md
└── nemotron-next-8b-fiqa.plan.md  # Implementation plan
```

## Dataset

**FiQA (Financial Question Answering)**
- Source: [explodinggradients/fiqa](https://huggingface.co/datasets/explodinggradients/fiqa)
- Task: Financial domain question answering
- Format: Question-answer pairs

## Time Estimates

| Phase | Time |
|-------|------|
| Model Loading | 2-5 min |
| Dataset Prep | 10-25 min |
| Baseline Eval | 30-60 min |
| LoRA Training | 3-6 hours |
| Final Eval | 30-60 min |

## License

Apache 2.0

