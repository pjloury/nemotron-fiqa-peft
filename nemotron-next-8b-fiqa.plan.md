---
name: Nemotron Nano 8B LoRA with FiQA Dataset
overview: Create a Jupyter notebook for LoRA fine-tuning of nvidia/Llama-3.1-Nemotron-Nano-8B-v1 model using the FiQA dataset (Financial Question Answering). Uses standard HuggingFace transformers + PEFT. Implementation follows 8 incremental phases with git checkpoints.
model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1
framework: transformers + peft (standard HuggingFace, NOT NeMo AutoModel)
phases:
  - id: phase-1-setup
    name: Project Setup & Environment
    status: verified
    tasks:
      - Create directory structure (examples/)
      - Create requirements.txt with transformers/peft dependencies
      - Create notebook skeleton with imports
      - Verify GPU detection and transformers imports
    test: "from transformers import AutoModelForCausalLM works"
    commit: "Phase 1: Project setup with transformers dependencies"

  - id: phase-2-model-loading
    name: Model Loading
    status: verified
    dependencies: [phase-1-setup]
    tasks:
      - Load nvidia/Llama-3.1-Nemotron-Nano-8B-v1 using AutoModelForCausalLM
      - Configure tokenizer and generation settings
      - Test basic text generation
    test: "Model generates coherent text response"
    commit: "Phase 2: Nemotron Nano 8B model loading and basic inference"
    time_estimate: "~6 seconds"
    result: "8.03B parameters loaded successfully"

  - id: phase-3-dataset
    name: Dataset Loading & Preprocessing
    status: verified
    dependencies: [phase-2-model-loading]
    tasks:
      - Load FiQA from HuggingFace datasets (main config)
      - Uses existing train/val/test splits (5500/500/648)
      - Format for instruction-tuning
      - Tokenize with model tokenizer
    test: "Print sample formatted data, verify split sizes"
    commit: "Phase 3: FiQA dataset loading with train/val/test splits"
    time_estimate: "~10 seconds"

  - id: phase-4-baseline
    name: Baseline Evaluation
    status: pending
    dependencies: [phase-3-dataset]
    tasks:
      - Run inference on test set subset
      - Calculate metrics (EM, F1, BLEU, perplexity)
      - Save baseline results to JSON
    test: "Baseline metrics computed and saved"
    commit: "Phase 4: Baseline model evaluation on FiQA test set"
    time_estimate: "30-60 minutes"

  - id: phase-5-lora-config
    name: LoRA Configuration
    status: verified
    dependencies: [phase-2-model-loading]
    tasks:
      - Configure LoRA parameters (rank=8, alpha=32, dropout=0.05)
      - Apply LoRA to model using HuggingFace PEFT
      - Verify trainable parameter count
      - Test model still generates after LoRA
    test: "Only LoRA params trainable (0.26%), inference works"
    commit: "Phase 5: LoRA adapter configuration and application"
    result: "21M trainable / 8.03B total (0.26%)"

  - id: phase-6-training
    name: LoRA Training
    status: verified
    dependencies: [phase-5-lora-config, phase-3-dataset]
    tasks:
      - Set up HuggingFace Trainer (optimizer, scheduler)
      - Train on FiQA train split
      - Validate on validation split
      - Save checkpoints and training logs
    test: "Training loss decreases, validation improves"
    commit: "Phase 6: LoRA training on FiQA with validation monitoring"
    time_estimate: "3-6 hours"
    result: "Test run: 10 steps, loss 3.25→2.97, 2.4 it/s"

  - id: phase-7-evaluation
    name: Fine-tuned Evaluation & Comparison
    status: pending
    dependencies: [phase-6-training, phase-4-baseline]
    tasks:
      - Evaluate fine-tuned model on test set
      - Compare metrics with baseline
      - Generate comparison table
    test: "Improvement shown in metrics vs baseline"
    commit: "Phase 7: Fine-tuned model evaluation and baseline comparison"
    time_estimate: "30-60 minutes"

  - id: phase-8-final
    name: Visualization & Documentation
    status: pending
    dependencies: [phase-7-evaluation]
    tasks:
      - Create loss curves and metric charts
      - Add example Q&A predictions (baseline vs fine-tuned)
      - Complete README with GPU requirements
      - Final cleanup and polish
    test: "Notebook runs end-to-end from scratch"
    commit: "Phase 8: Visualization, documentation, and final polish"
---

# Plan: LoRA Fine-tuning Notebook for NVIDIA Nemotron Next 8B with FiQA Dataset

## Overview

This plan creates a Jupyter notebook that fine-tunes the [NVIDIA Nemotron Next 8B](https://huggingface.co/nvidia/Nemotron-Next-8B) language model using LoRA (Low-Rank Adaptation) with the **FiQA dataset** (Financial Question Answering) from [Hugging Face](https://huggingface.co/datasets/explodinggradients/fiqa).

The implementation follows **8 incremental phases**, each with a git checkpoint to ensure components work before proceeding.

## Implementation Phases

### Phase 1: Project Setup & Environment
**Git Tag: `phase-1-setup`** | **Time: ~5 minutes**

| Task | Details |
|------|---------|
| Directory structure | Create `examples/` directory |
| Dependencies | `requirements.txt` with NeMo Framework, NeMo AutoModel |
| Notebook skeleton | Imports, GPU detection, markdown structure |
| **Checkpoint Test** | `import nemo_automodel` succeeds |

**Files created:**
- `requirements.txt`
- `examples/nemotron_fiqa_lora.ipynb` (skeleton)
- `README.md` (initial)

---

### Phase 2: Model Loading
**Git Tag: `phase-2-model-loading`** | **Time: 2-5 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Load model | `NeMoAutoModelForCausalLM.from_pretrained("nvidia/Nemotron-Next-8B")` |
| Configure | Tokenizer, generation config, dtype (bfloat16) |
| Test inference | Generate simple response to verify model works |
| **Checkpoint Test** | Model generates coherent text |

---

### Phase 3: Dataset Loading & Preprocessing
**Git Tag: `phase-3-dataset`** | **Time: 10-25 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Load FiQA | `datasets.load_dataset("explodinggradients/fiqa")` |
| Create splits | Train (80%) / Validation (10%) / Test (10%) |
| Format data | `"Question: {q}\nAnswer: {a}"` instruction format |
| Tokenize | Using NeMo preprocessing pipeline |
| **Checkpoint Test** | Print sample data, verify split sizes |

---

### Phase 4: Baseline Evaluation
**Git Tag: `phase-4-baseline`** | **Time: 30-60 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Inference | Run base model on test set |
| Calculate metrics | Exact Match, F1, BLEU, Perplexity |
| Save results | `baseline_results.json` |
| **Checkpoint Test** | Metrics computed and saved |

---

### Phase 5: LoRA Configuration
**Git Tag: `phase-5-lora-config`** | **Time: ~5 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Configure LoRA | `rank=8`, `alpha=32`, target attention layers |
| Apply adapter | Using NeMo AutoModel PEFT APIs |
| Verify params | Print trainable vs frozen parameter counts |
| **Checkpoint Test** | Only LoRA params trainable, model generates text |

**LoRA Hyperparameters:**
- Rank (r): 8
- Alpha: 32
- Target modules: query, key, value projections
- Dropout: 0.05

---

### Phase 6: LoRA Training
**Git Tag: `phase-6-training`** | **Time: 3-6 hours** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Training loop | AdamW optimizer, cosine LR scheduler |
| Train | On FiQA train split |
| Validate | On validation split every N steps |
| Checkpoints | Save best model based on validation loss |
| **Checkpoint Test** | Loss decreases, validation improves |

**Training Configuration:**
- Learning rate: 1e-4 to 5e-4
- Batch size: 4-8 (adjust for GPU memory)
- Epochs: 2-5
- Gradient accumulation: as needed

---

### Phase 7: Fine-tuned Evaluation & Comparison
**Git Tag: `phase-7-evaluation`** | **Time: 30-60 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Evaluate | Run fine-tuned model on test set |
| Compare | Side-by-side with baseline metrics |
| Generate table | Baseline vs LoRA comparison |
| **Checkpoint Test** | Improvement shown in metrics |

---

### Phase 8: Visualization & Documentation
**Git Tag: `phase-8-final`** | **Time: 1-2 hours**

| Task | Details |
|------|---------|
| Visualizations | Training/validation loss curves, metric charts |
| Examples | Side-by-side Q&A predictions |
| README | Complete with GPU requirements, time estimates |
| Polish | Remove debug code, clean markdown |
| **Checkpoint Test** | Notebook runs end-to-end from scratch |

---

## Git Workflow

```
main
├── phase-1-setup        ← Project skeleton
├── phase-2-model-loading ← Model loads and generates
├── phase-3-dataset      ← Data pipeline works
├── phase-4-baseline     ← Baseline metrics captured
├── phase-5-lora-config  ← LoRA applied correctly
├── phase-6-training     ← Training completes successfully
├── phase-7-evaluation   ← Improvement verified
└── phase-8-final        ← Merge to main (production ready)
```

---

## FiQA Dataset Reference

### Why FiQA for Nemotron Next 8B

1. **Financial Domain Focus**: Financial Q&A pairs align with Nemotron's text generation capabilities
2. **Structured Format**: Question-answer pairs suitable for instruction fine-tuning
3. **Domain Specialization**: Allows model to specialize in financial knowledge

### Dataset Structure

| Field | Description |
|-------|-------------|
| Source | [explodinggradients/fiqa](https://huggingface.co/datasets/explodinggradients/fiqa) |
| Format | Text-based Q&A pairs |
| Topics | Trading, investing, market analysis, financial planning |

### Proposed Splits

| Split | Source | Size |
|-------|--------|------|
| Train | 80% of original train | ~80% |
| Validation | 20% of original train | ~10% |
| Test | Original test set (held out) | ~10% |

---

## Evaluation Metrics

### Primary Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | Strict string match between predicted and ground truth |
| **F1 Score** | Token-level F1 (accounts for partial matches) |
| **BLEU Score** | N-gram overlap for answer quality |
| **Perplexity** | Model confidence in generated answers |

### Secondary Metrics

- Training/validation loss curves
- Learning rate schedules
- Token generation statistics

---

## Technical Notes

### Memory Considerations

8B parameter model requires significant GPU memory. Mitigations:
- Q-LoRA with 4-bit or 8-bit quantization
- Gradient checkpointing
- Smaller batch sizes with gradient accumulation

### API Usage

- Use **NeMo AutoModel APIs** throughout (not HuggingFace AutoModel directly)
- Follow NeMo AutoModel repository workflow patterns
- Include GPU requirement annotations on all relevant cells

### Data Formatting

Format FiQA data for instruction fine-tuning:
```
Question: {question}
Answer: {answer}
```

---

## Success Criteria

### Per-Phase Success

| Phase | Success Criteria |
|-------|------------------|
| 1 | Imports work, GPU detected |
| 2 | Model loads and generates text |
| 3 | Data splits created, samples formatted correctly |
| 4 | Baseline metrics computed and saved |
| 5 | LoRA applied, only adapter params trainable |
| 6 | Training completes, loss decreases |
| 7 | Fine-tuned model shows improvement over baseline |
| 8 | Notebook runs end-to-end, documentation complete |

### Final Success

- Clear improvement in financial QA performance (EM, F1, BLEU)
- Notebook is self-contained and reproducible
- Ready for merge as an example in NeMo AutoModel repository
