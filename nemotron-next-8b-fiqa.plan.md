---
name: Nemotron Nano 8B LoRA Training with FiQA Dataset
overview: Create a Jupyter notebook for LoRA fine-tuning of nvidia/Llama-3.1-Nemotron-Nano-8B-v1 model using the FiQA dataset (Financial Question Answering). Uses standard HuggingFace transformers + PEFT. Implementation follows 5 incremental phases with git checkpoints focused on training workflow.
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
    status: pending
    dependencies: [phase-2-model-loading]
    tasks:
      - Load chosen FiQA dataset (corpus, queries, qrels)
      - Preprocess documents based on analysis (extract snippets, truncate, or use as-is)
      - Convert retrieval format to Q&A pairs using qrels
      - Create train/eval splits (80/20 split of query-document pairs)
      - Format for instruction-tuning (Question: {query}\nAnswer: {document_text})
      - Tokenize with model tokenizer
      - Save processed datasets for reproducibility
    test: "Print sample formatted data, verify split sizes, check Q&A format"
    commit: "Phase 3: FiQA dataset conversion and train/eval splits"
    time_estimate: "10-30 minutes"

  - id: phase-4-lora-config
    name: LoRA Configuration
    status: verified
    dependencies: [phase-2-model-loading]
    tasks:
      - Configure LoRA parameters (rank=8, alpha=32, dropout=0.05)
      - Apply LoRA to model using HuggingFace PEFT
      - Verify trainable parameter count
      - Test model still generates after LoRA
    test: "Only LoRA params trainable (0.26%), inference works"
    commit: "Phase 4: LoRA adapter configuration and application"
    result: "21M trainable / 8.03B total (0.26%)"

  - id: phase-5-training
    name: LoRA Training
    status: verified
    dependencies: [phase-4-lora-config, phase-3-dataset]
    tasks:
      - Set up HuggingFace Trainer (optimizer, scheduler)
      - Train on FiQA train split
      - Validate on validation split
      - Save checkpoints and training logs
    test: "Training loss decreases, validation improves"
    commit: "Phase 5: LoRA training on FiQA with validation monitoring"
    time_estimate: "3-6 hours"
    result: "Test run: 10 steps, loss 3.25→2.97, 2.4 it/s"

  - id: phase-6-visualization
    name: Training Visualization & Documentation
    status: pending
    dependencies: [phase-5-training]
    tasks:
      - Create training/validation loss curves
      - Visualize learning rate schedules
      - Document training configuration and results
      - Complete README with GPU requirements and training notes
      - Final cleanup and polish
    test: "Training notebook runs end-to-end from scratch"
    commit: "Phase 6: Training visualization, documentation, and final polish"
---

# Plan: LoRA Fine-tuning Training Notebook for NVIDIA Nemotron Next 8B with FiQA Dataset

## Overview

This plan creates a Jupyter notebook for **training** a LoRA fine-tuned version of the [NVIDIA Nemotron Next 8B](https://huggingface.co/nvidia/Nemotron-Next-8B) language model using the **FiQA dataset** (Financial Question Answering) from [Hugging Face](https://huggingface.co/datasets/BeIR/fiqa).

The implementation follows **6 incremental phases**, each with a git checkpoint to ensure components work before proceeding. This plan focuses on the training workflow. For evaluation, see the separate evaluation plan.

## Implementation Phases

### Phase 1: Project Setup & Environment
**Git Tag: `phase-1-setup`** | **Time: ~5 minutes**

| Task | Details |
|------|---------|
| Directory structure | Create `examples/` directory |
| Dependencies | `requirements.txt` with transformers/peft dependencies |
| Notebook skeleton | Imports, GPU detection, markdown structure |
| **Checkpoint Test** | `from transformers import AutoModelForCausalLM` works |

**Files created:**
- `requirements.txt`
- `examples/nemotron_fiqa_lora.ipynb` (skeleton)
- `README.md` (initial)

---

### Phase 2: Model Loading
**Git Tag: `phase-2-model-loading`** | **Time: 2-5 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Load model | `AutoModelForCausalLM.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")` |
| Configure | Tokenizer, generation config, dtype (bfloat16) |
| Test inference | Generate simple response to verify model works |
| **Checkpoint Test** | Model generates coherent text |

---

### Phase 3: Dataset Loading & Preprocessing
**Git Tag: `phase-3-dataset`** | **Time: 10-30 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Load FiQA dataset | `load_dataset("explodinggradients/fiqa", "main")` |
| Extract Q&A pairs | Use `question` and first `ground_truths` item |
| **Use full answers** | **NO truncation** - use complete ground truth answers |
| Format for training | Convert to instruction format with system prompt encouraging comprehensive answers |
| Use existing splits | Train (5500), Validation (500), Test (648) |
| Tokenize | Tokenize with model tokenizer for training |
| Save processed data | Save processed train/validation splits for reproducibility |
| **Checkpoint Test** | Print sample Q&A pairs, verify split sizes, check format and answer lengths |

**Data Conversion Process:**
1. Load `explodinggradients/fiqa` with `main` config (already has train/val/test splits)
2. For each example, extract: question = `question` field, answer = first item from `ground_truths` list
3. **Use full answers (no truncation)** - Critical for consistency with evaluation ground truth
4. Format as instruction-tuning prompt with system prompt:
   ```
   "You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, and explanations when appropriate.

   Question: {question}

   Answer: {answer}"
   ```
5. Use existing splits: train (5500), validation (500), test (648)

**Prompt Engineering Benefits:**
- Guides model to generate comprehensive answers similar to ground truth length/detail
- Ensures model outputs are more comparable to ground truth for evaluation
- Can be adjusted to match desired answer style

**⚠️ Evaluation Consistency:** Using full answers ensures training and evaluation ground truth are identical, enabling fair semantic similarity evaluation.

---

### Phase 4: LoRA Configuration
**Git Tag: `phase-4-lora-config`** | **Time: ~5 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Configure LoRA | `rank=8`, `alpha=32`, target attention layers |
| Apply adapter | Using HuggingFace PEFT |
| Verify params | Print trainable vs frozen parameter counts |
| **Checkpoint Test** | Only LoRA params trainable, model generates text |

**LoRA Hyperparameters:**
- Rank (r): 8
- Alpha: 32
- Target modules: query, key, value projections
- Dropout: 0.05

---

### Phase 5: LoRA Training
**Git Tag: `phase-5-training`** | **Time: 3-6 hours** | **GPU REQUIRED**

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

### Phase 6: Training Visualization & Documentation
**Git Tag: `phase-6-visualization`** | **Time: 1-2 hours**

| Task | Details |
|------|---------|
| Visualizations | Training/validation loss curves, learning rate schedules |
| Training metrics | Document training statistics and checkpoints |
| README | Complete with GPU requirements, training time estimates |
| Polish | Remove debug code, clean markdown |
| **Checkpoint Test** | Training notebook runs end-to-end from scratch |

---

## Git Workflow

```
main
├── phase-1-setup        ← Project skeleton
├── phase-2-model-loading ← Model loads and generates
├── phase-3-dataset      ← Data pipeline works
├── phase-4-lora-config  ← LoRA applied correctly
├── phase-5-training     ← Training completes successfully
└── phase-6-visualization ← Merge to main (training ready)
```

---

## FiQA Dataset Reference

### Why FiQA for Nemotron Next 8B

1. **Financial Domain Focus**: Financial Q&A pairs align with Nemotron's text generation capabilities
2. **High-Quality Q&A Data**: BeIR/fiqa contains curated financial questions with relevant answer documents
3. **Domain Specialization**: Allows model to specialize in financial knowledge through instruction-tuning
4. **Repurposing Retrieval Data**: While BeIR/fiqa is designed as a retrieval benchmark, we extract the underlying Q&A pairs for fine-tuning (not evaluating retrieval performance)

### Dataset Structure

| Field | Description |
|-------|-------------|
| Source | [explodinggradients/fiqa](https://huggingface.co/datasets/explodinggradients/fiqa) (main config) |
| Format | Direct Q&A pairs (question, ground_truths) |
| Train Size | 5,500 examples |
| Validation Size | 500 examples |
| Test Size | 648 examples |
| Topics | Trading, investing, market analysis, financial planning, tax, business expenses |

### Dataset Overview

**Selected Dataset:** `explodinggradients/fiqa` (main config)

This dataset provides direct Q&A pairs suitable for instruction-tuning, with pre-existing train/validation/test splits. The answers are comprehensive financial responses that may benefit from truncation to create more concise training examples.

### Dataset Selection

**Dataset chosen:** `explodinggradients/fiqa` (main config)

**Analysis Results:**
- **Train size:** 5,500 examples
- **Answer length:** Median 154 words (mean 199 words)
- **Distribution:** 29.4% concise (<100 words), 52.1% medium (100-300 words), 18.6% long (≥300 words)
- **Assessment:** Moderately suitable - answers are somewhat long but manageable

**Preprocessing Strategy:**
- **Use full answers as-is (NO truncation)** - This ensures training and evaluation ground truth are consistent
- Model will learn to generate answers in the same style/length as ground truth
- Evaluation semantic similarity will be fair since model outputs and ground truth are comparable

**Why this dataset:**
- Has direct Q&A pairs (not retrieval format)
- Pre-existing train/validation/test splits (5500/500/648)
- Financial domain focus
- Full answers provide comprehensive context (median 154 words is reasonable)

**⚠️ Important:** For semantic similarity evaluation, training and evaluation ground truth must be in the same format. Using full answers for both ensures fair evaluation.

### Data Conversion Strategy

**Goal:** Format Q&A pairs for instruction-tuning

1. **Load Dataset:**
   - Load `explodinggradients/fiqa` with `main` config
   - Dataset already has train/validation/test splits (5500/500/648)
   - Each example has: `question` (string) and `ground_truths` (list of strings)

2. **Extract Answers:**
   - Use first item from `ground_truths` list as the answer
   - Answers are typically 150-200 words (median 154 words)
   - **Use full answers (NO truncation)** to maintain consistency with evaluation

3. **Preprocessing:**
   - **Use answers as-is (full length)** - No truncation
   - This ensures training data matches evaluation ground truth format
   - Model will learn to generate answers in the same style/length as ground truth
   - Critical for fair semantic similarity evaluation

4. **Format for Instruction-Tuning with Prompt Engineering:**
   - Use a system prompt that encourages comprehensive, detailed answers
   - Question = `question` field
   - Answer = first `ground_truths` item (full length, no truncation)
   - Format with prompt engineering:
     ```
     "You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, and explanations when appropriate.

     Question: {question}

     Answer: {answer}"
     ```
   - This prompt guides the model to generate answers similar in length and detail to ground truth

5. **Use Existing Splits:**
   - Train: 5,500 examples (use as-is)
   - Validation: 500 examples (use for validation during training)
   - Test: 648 examples (use for final evaluation)

**Key Point:** This dataset has direct Q&A pairs (not retrieval format), making it more suitable for instruction-tuning than BeIR/fiqa. **Use full answers (no truncation) to ensure training and evaluation ground truth are consistent for fair semantic similarity evaluation.**

### Dataset Splits

| Split | Size | Usage |
|-------|------|-------|
| Train | 5,500 examples | Training data |
| Validation | 500 examples | Validation during training |
| Test | 648 examples | Final evaluation |

**Note:** Dataset already has predefined splits. Use as-is - no need to create custom splits.

---

## Technical Notes

### Memory Considerations

8B parameter model requires significant GPU memory. Mitigations:
- Q-LoRA with 4-bit or 8-bit quantization
- Gradient checkpointing
- Smaller batch sizes with gradient accumulation

### API Usage

- Use **HuggingFace transformers APIs** throughout (standard transformers + PEFT)
- Follow HuggingFace best practices for LoRA fine-tuning
- Include GPU requirement annotations on all relevant cells

### Data Formatting

**Conversion from explodinggradients/fiqa format:**
```python
# Load dataset
from datasets import load_dataset
dataset = load_dataset("explodinggradients/fiqa", "main")

# Extract Q&A pairs (use full answers - no truncation)
def format_qa_pair(example):
    question = example["question"]
    answer = example["ground_truths"][0]  # Use first ground truth (full length)
    
    # IMPORTANT: Use full answer to match evaluation ground truth
    # No truncation - ensures training and evaluation are consistent
    
    return {
        "question": question,
        "answer": answer  # Full answer, no truncation
    }

# Format for instruction-tuning with prompt engineering
def create_instruction(example):
    system_prompt = "You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, and explanations when appropriate."
    return f"{system_prompt}\n\nQuestion: {example['question']}\n\nAnswer: {example['answer']}"
```

**Format for instruction fine-tuning (with prompt engineering):**
```
You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, and explanations when appropriate.

Question: {question}

Answer: {answer}
```

**Prompt Engineering Rationale:**
- System prompt encourages comprehensive, detailed answers
- Guides model to generate responses similar in length/detail to ground truth (median 154 words)
- Helps ensure model outputs are comparable to ground truth for semantic similarity evaluation
- Can adjust prompt based on desired answer style (e.g., more concise vs. more detailed)

**Implementation Notes:**
- Dataset already has train/validation/test splits - use as-is
- Answers are in `ground_truths` list - use first item
- **CRITICAL:** Use full answers (no truncation) to ensure consistency with evaluation
- Training and evaluation must use the same ground truth format for fair semantic similarity
- Save processed splits to avoid reprocessing
- Median answer length (154 words) is reasonable for training

---

## Success Criteria

### Per-Phase Success

| Phase | Success Criteria |
|-------|------------------|
| 1 | Imports work, GPU detected |
| 2 | Model loads and generates text |
| 3 | explodinggradients/fiqa loaded, answers truncated/preprocessed, train/val/test splits ready, samples formatted correctly |
| 4 | LoRA applied, only adapter params trainable |
| 5 | Training completes, loss decreases |
| 6 | Training notebook runs end-to-end, documentation complete |

### Final Success

- Training completes successfully with decreasing loss
- LoRA adapter saved and ready for evaluation
- Notebook is self-contained and reproducible
- Ready for evaluation workflow (see separate evaluation plan)
