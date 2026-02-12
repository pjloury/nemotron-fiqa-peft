---
name: Nemotron Nano 8B LoRA Evaluation with FiQA Dataset
overview: Create a Jupyter notebook for evaluating LoRA fine-tuned nvidia/Llama-3.1-Nemotron-Nano-8B-v1 model using the FiQA dataset (Financial Question Answering). Uses NeMo Evaluator SDK for metrics calculation and comparison between base and PEFT models. Implementation follows 6 incremental phases with git checkpoints focused on evaluation workflow.
model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1
framework: transformers + peft (standard HuggingFace) + NeMo Evaluator SDK
phases:
  - id: phase-1-setup
    name: Project Setup & Environment
    status: pending
    tasks:
      - Create directory structure (examples/)
      - Create requirements.txt with transformers/peft and NeMo Evaluator SDK dependencies
      - Create notebook skeleton with imports
      - Verify GPU detection and transformers imports
      - Verify NeMo Evaluator SDK installation
    test: "from transformers import AutoModelForCausalLM works and NeMo Evaluator SDK imports successfully"
    commit: "Phase 1: Project setup with transformers, PEFT, and NeMo Evaluator SDK dependencies"

  - id: phase-2-model-loading
    name: Model Loading
    status: pending
    dependencies: [phase-1-setup]
    tasks:
      - Load nvidia/Llama-3.1-Nemotron-Nano-8B-v1 using AutoModelForCausalLM
      - Configure tokenizer and generation settings
      - Test basic text generation
    test: "Model generates coherent text response"
    commit: "Phase 2: Nemotron Nano 8B model loading and basic inference"
    time_estimate: "~6 seconds"

  - id: phase-3-dataset
    name: Dataset Loading & Preprocessing
    status: pending
    dependencies: [phase-2-model-loading]
    tasks:
      - Load explodinggradients/fiqa dataset (main config)
      - Extract Q&A pairs using question and first ground_truths item
      - Use full answers (NO truncation) to match training ground truth format
      - Load test split (648 examples) for evaluation
      - Format for evaluation (Question: {question}\nAnswer: {answer})
      - Verify ground truth format matches training data
    test: "Print sample formatted data, verify split sizes, check Q&A format matches training, confirm full answers used"
    commit: "Phase 3: explodinggradients/fiqa dataset loading with full ground truth"
    time_estimate: "10-30 minutes"

  - id: phase-4-baseline
    name: Baseline Evaluation with NeMo Evaluator SDK
    status: pending
    dependencies: [phase-3-dataset]
    tasks:
      - Run inference on test set using base model
      - Use NeMo Evaluator SDK to calculate evaluation metrics
      - Configure NeMo Evaluator with appropriate metrics for QA task
      - Use full ground truth answers (matching training format)
      - Save baseline results to JSON
    test: "Baseline metrics computed using NeMo Evaluator SDK and saved"
    commit: "Phase 4: Baseline model evaluation using NeMo Evaluator SDK"
    time_estimate: "30-60 minutes"

  - id: phase-5-finetuned-eval
    name: Fine-tuned Evaluation & Comparison with NeMo Evaluator SDK
    status: pending
    dependencies: [phase-4-baseline]
    tasks:
      - Load fine-tuned LoRA adapter
      - Run inference on test set using fine-tuned model
      - Use NeMo Evaluator SDK to calculate evaluation metrics (same metrics as baseline)
      - Compare PEFT model metrics with base model metrics using NeMo Evaluator SDK
      - Generate comparison table and analysis
    test: "PEFT model metrics computed using NeMo Evaluator SDK, comparison with baseline shows improvement"
    commit: "Phase 5: Fine-tuned model evaluation and comparison using NeMo Evaluator SDK"
    time_estimate: "30-60 minutes"

  - id: phase-6-visualization
    name: Evaluation Visualization & Documentation
    status: pending
    dependencies: [phase-5-finetuned-eval]
    tasks:
      - Create metric comparison charts
      - Add example Q&A predictions (baseline vs fine-tuned)
      - Generate evaluation report
      - Complete README with evaluation results
      - Final cleanup and polish
    test: "Evaluation notebook runs end-to-end from scratch"
    commit: "Phase 6: Evaluation visualization, documentation, and final polish"
---

# Plan: LoRA Fine-tuned Model Evaluation Notebook for NVIDIA Nemotron Next 8B with FiQA Dataset

## Overview

This plan creates a Jupyter notebook for **evaluating** a LoRA fine-tuned version of the [NVIDIA Nemotron Next 8B](https://huggingface.co/nvidia/Nemotron-Next-8B) language model using the **FiQA dataset** (Financial Question Answering) from [Hugging Face](https://huggingface.co/datasets/explodinggradients/fiqa).

The implementation follows **6 incremental phases**, each with a git checkpoint to ensure components work before proceeding. This plan focuses on the evaluation workflow, including baseline and fine-tuned model comparison. For training, see the separate training plan.

## Implementation Phases

### Phase 1: Project Setup & Environment
**Git Tag: `phase-1-setup`** | **Time: ~5 minutes**

| Task | Details |
|------|---------|
| Directory structure | Create `examples/` directory |
| Dependencies | `requirements.txt` with transformers/peft and NeMo Evaluator SDK dependencies |
| Notebook skeleton | Imports, GPU detection, markdown structure |
| **Checkpoint Test** | `from transformers import AutoModelForCausalLM` works |

**Files created:**
- `requirements.txt`
- `examples/nemotron_fiqa_evaluation.ipynb` (skeleton)
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
| **Use full answers** | **NO truncation** - must match training ground truth format exactly |
| Load test split | Use test split (648 examples) for evaluation |
| Format for evaluation | Convert to instruction format: `"Question: {q}\nAnswer: {a}"` |
| Verify consistency | Ensure ground truth format matches training data (full answers) |
| **Checkpoint Test** | Print sample Q&A pairs, verify format matches training, confirm full answers |

**Data Conversion Process:**
1. Load `explodinggradients/fiqa` with `main` config (same as training)
2. For each example, extract: question = `question` field, answer = first item from `ground_truths` list
3. **CRITICAL:** Use full answers (no truncation) - must match training ground truth format
4. Load test split (648 examples) for evaluation
5. Format as evaluation prompt: `"Question: {question}\nAnswer: {answer}"`

**⚠️ Critical for Semantic Similarity:**
1. Ground truth format must exactly match training data (full answers, no truncation)
2. Use the same prompt format during inference as during training
3. This ensures model generates answers similar in length/detail to ground truth
4. Enables fair semantic similarity evaluation

---

### Phase 4: Baseline Evaluation
**Git Tag: `phase-4-baseline`** | **Time: 30-60 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Inference | Run base model on test set using same prompt format as training |
| Use training prompt | Apply same system prompt: "You are a helpful financial assistant. Provide comprehensive, detailed answers..." |
| **NeMo Evaluator SDK** | Use NeMo Evaluator SDK to calculate evaluation metrics |
| Configure metrics | Set up NeMO Evaluator with metrics appropriate for QA task |
| Use full ground truth | Compare model outputs against full ground truth answers (matching training format) |
| Save results | `baseline_results.json` with NeMo Evaluator SDK metrics |
| **Checkpoint Test** | Baseline metrics computed using NeMo Evaluator SDK and saved |

**NeMo Evaluator SDK Metrics Configuration:**
- **Semantic Similarity:** Use NeMo Evaluator's semantic similarity metrics (e.g., BERTScore, SBERT-based)
- **ROUGE Scores:** ROUGE-1, ROUGE-2, ROUGE-L for answer quality
- **BLEU Score:** N-gram overlap metric
- **Exact Match (EM):** Strict string match
- **F1 Score:** Token-level F1 for partial matches
- **Perplexity:** Model confidence (if available in SDK)

**NeMo Evaluator SDK Benefits:**
- Standardized metric calculation across base and PEFT models
- Consistent evaluation methodology
- Built-in support for semantic similarity and QA metrics
- Easy comparison between model versions

**⚠️ Important:** Use the same NeMo Evaluator SDK configuration for both base and PEFT models to ensure fair comparison.

---

### Phase 5: Fine-tuned Evaluation & Comparison
**Git Tag: `phase-5-finetuned-eval`** | **Time: 30-60 minutes** | **GPU REQUIRED**

| Task | Details |
|------|---------|
| Load adapter | Load trained LoRA adapter weights |
| Evaluate | Run fine-tuned model on test set using same prompt format as training |
| Use training prompt | Apply same system prompt used during training |
| **NeMo Evaluator SDK** | Use NeMo Evaluator SDK with same metric configuration as baseline |
| Compare | Use NeMo Evaluator SDK comparison tools to compare PEFT vs base metrics |
| Generate table | Baseline vs PEFT comparison table using NeMo Evaluator SDK results |
| Analyze improvements | Identify which metrics show improvement and by how much |
| **Checkpoint Test** | PEFT model shows improvement over baseline across key metrics |

**Expected Improvements (using NeMo Evaluator SDK metrics):**
- Higher semantic similarity scores (primary metric)
- Higher ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Higher BLEU scores
- Higher EM and F1 scores
- Better financial domain understanding reflected in all metrics
- Consistent improvement across NeMo Evaluator SDK metrics

---

### Phase 6: Evaluation Visualization & Documentation
**Git Tag: `phase-6-visualization`** | **Time: 1-2 hours**

| Task | Details |
|------|---------|
| Visualizations | Metric comparison charts, score distributions |
| Examples | Side-by-side Q&A predictions (baseline vs fine-tuned) |
| Report | Generate evaluation report with key findings |
| README | Complete with evaluation results and metrics |
| Polish | Remove debug code, clean markdown |
| **Checkpoint Test** | Evaluation notebook runs end-to-end from scratch |

---

## Git Workflow

```
main
├── phase-1-setup        ← Project skeleton
├── phase-2-model-loading ← Model loads and generates
├── phase-3-dataset      ← Data pipeline works
├── phase-4-baseline     ← Baseline metrics captured
├── phase-5-finetuned-eval ← Fine-tuned evaluation complete
└── phase-6-visualization ← Merge to main (evaluation ready)
```

---

## FiQA Dataset Reference

### Why FiQA for Evaluation

1. **Financial Domain Focus**: Financial Q&A pairs test domain-specific knowledge
2. **Structured Format**: Question-answer pairs suitable for metric calculation
3. **High-Quality Data**: BeIR/fiqa contains curated financial questions with relevant answer documents
4. **Reproducible**: Same dataset format as training ensures consistent evaluation

**Note:** While BeIR/fiqa is designed as a retrieval benchmark, we're using it to evaluate **question-answering performance** (not retrieval ranking). We extract Q&A pairs and measure how well the model generates answers, not how well it ranks documents.

### Dataset Structure

| Field | Description |
|-------|-------------|
| Source | [explodinggradients/fiqa](https://huggingface.co/datasets/explodinggradients/fiqa) (main config) |
| Format | Direct Q&A pairs (question, ground_truths) |
| Test Size | 648 examples (predefined split) |
| Topics | Trading, investing, market analysis, financial planning, tax, business expenses |
| Ground Truth | Full answers (no truncation) - must match training format |

### Understanding BeIR/fiqa for Evaluation

**What we're evaluating:**
- **Question-answering performance**: How well the model generates answers to financial questions
- **Not retrieval performance**: We're not measuring document ranking or retrieval metrics

**How we use the retrieval structure:**
- The qrels (relevance judgments) tell us which documents contain answers to each question
- We extract these Q&A pairs and evaluate answer quality (EM, F1, BLEU, perplexity)
- The retrieval structure is just a way to identify which documents answer which questions

### Data Conversion Strategy

**Goal:** Convert retrieval format → Q&A format for evaluation (same as training)

1. **Load Components:**
   - `corpus`: Dictionary of documents with `_id`, `title`, `text` (answer sources)
   - `queries`: Dictionary of queries with `_id`, `text` (questions)
   - `qrels`: Relevance judgments linking query IDs to relevant document IDs (Q&A mappings)

2. **Create Q&A Pairs:**
   - For each query in qrels, extract the relevant document text
   - Question = query text
   - Answer = relevant document text (ground truth for evaluation)

3. **Use Matching Eval Split:**
   - **Critical:** Must use the same train/eval split as the training plan
   - Load saved split indices or reproduce using same random seed
   - Ensures fair comparison between baseline and fine-tuned models

---

## Evaluation Metrics Using NeMo Evaluator SDK

### NeMo Evaluator SDK Integration

The evaluation uses **NeMo Evaluator SDK** to calculate metrics and compare base vs PEFT model performance. This ensures:
- Standardized metric calculation
- Consistent evaluation methodology
- Easy comparison between model versions
- Built-in support for QA-specific metrics

### Primary Metrics (NeMo Evaluator SDK)

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Semantic Similarity** | NeMo Evaluator's semantic similarity metrics (BERTScore, SBERT-based) | Primary metric for comparing answer quality |
| **ROUGE-1** | Unigram overlap between predicted and ground truth | Measures content overlap |
| **ROUGE-2** | Bigram overlap between predicted and ground truth | Measures phrase-level similarity |
| **ROUGE-L** | Longest common subsequence | Measures sentence-level similarity |
| **BLEU Score** | N-gram overlap with precision | Measures answer quality |
| **Exact Match (EM)** | Strict string match | Binary correctness metric |
| **F1 Score** | Token-level F1 (precision/recall) | Partial match metric |

### Metric Selection Rationale

**For Base vs PEFT Comparison:**
1. **Semantic Similarity (Primary):** Best captures meaning similarity, handles length differences
2. **ROUGE Scores:** Standard QA evaluation metrics, multiple granularities
3. **BLEU Score:** Widely used for text generation evaluation
4. **EM & F1:** Traditional QA metrics for completeness

### NeMo Evaluator SDK Configuration

```python
# Example NeMo Evaluator SDK usage
from nemo_evaluator import Evaluator

evaluator = Evaluator(
    metrics=['semantic_similarity', 'rouge', 'bleu', 'exact_match', 'f1'],
    semantic_similarity_model='sbert',  # or 'bertscore'
    rouge_types=['rouge1', 'rouge2', 'rougeL']
)

# Evaluate base model
base_results = evaluator.evaluate(
    predictions=base_model_outputs,
    references=ground_truths
)

# Evaluate PEFT model
peft_results = evaluator.evaluate(
    predictions=peft_model_outputs,
    references=ground_truths
)

# Compare
comparison = evaluator.compare(base_results, peft_results)
```

### Secondary Metrics

- Answer length statistics (mean, median, distribution)
- Generation time per example
- Token generation statistics
- Error analysis (common failure modes)
- Per-example metric breakdown

### Metric Comparison Strategy

1. **Calculate all metrics for both models** using NeMo Evaluator SDK
2. **Compare metrics side-by-side** to identify improvements
3. **Focus on semantic similarity** as primary improvement indicator
4. **Analyze ROUGE scores** to understand content overlap improvements
5. **Report aggregate statistics** (mean, median, std) for each metric

---

## Technical Notes

### Memory Considerations

8B parameter model requires significant GPU memory. Mitigations:
- Batch inference with smaller batch sizes
- Gradient checkpointing disabled (inference only)
- FP16/BF16 precision for faster inference

### API Usage

- Use **HuggingFace transformers APIs** for model loading and inference
- Load LoRA adapters using PEFT `PeftModel.from_pretrained()`
- Use **NeMo Evaluator SDK** for all metric calculations and comparisons
- Include GPU requirement annotations on all relevant cells

### NeMo Evaluator SDK Setup

```python
# Install NeMo Evaluator SDK
# pip install nemo-evaluator

from nemo_evaluator import Evaluator

# Configure evaluator
evaluator = Evaluator(
    metrics=['semantic_similarity', 'rouge', 'bleu', 'exact_match', 'f1'],
    semantic_similarity_model='sbert',  # or 'bertscore'
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

### Data Formatting

**Conversion from explodinggradients/fiqa format (must match training):**
```python
# Load dataset (same as training)
from datasets import load_dataset
dataset = load_dataset("explodinggradients/fiqa", "main")

# Extract Q&A pairs (use full answers - no truncation)
def format_qa_pair(example):
    question = example["question"]
    answer = example["ground_truths"][0]  # Use first ground truth (full length)
    
    # CRITICAL: Use full answer to match training ground truth format
    # No truncation - ensures fair semantic similarity evaluation
    
    return {
        "question": question,
        "answer": answer  # Full answer, no truncation
    }
```

**Format for evaluation (must match training format exactly):**
```
You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, and explanations when appropriate.

Question: {question}

Answer: {answer}
```

**Prompt Consistency:**
- Use the same system prompt during inference as during training
- This ensures model generates answers in the same style/length as training
- Helps ensure outputs are comparable to ground truth for semantic similarity

**Implementation Notes:**
- **CRITICAL:** Use the same dataset and preprocessing as training plan
- **Use full answers (no truncation)** - must match training ground truth format
- Load test split (648 examples) for evaluation
- Ground truth format must be identical to training for fair semantic similarity
- Save processed test data to avoid reprocessing

---

## Success Criteria

### Per-Phase Success

| Phase | Success Criteria |
|-------|------------------|
| 1 | Imports work, GPU detected |
| 2 | Model loads and generates text |
| 3 | BeIR/fiqa converted to Q&A format, eval split loaded (matching training), samples formatted correctly |
| 4 | Baseline metrics computed using NeMo Evaluator SDK and saved |
| 5 | PEFT model metrics computed using NeMo Evaluator SDK, shows improvement over baseline |
| 6 | Evaluation notebook runs end-to-end, documentation complete |

### Final Success

- Clear improvement in financial QA performance across NeMo Evaluator SDK metrics
- Semantic similarity shows improvement (primary metric)
- ROUGE scores show improvement across all variants
- Comprehensive evaluation report with visualizations using NeMo Evaluator SDK results
- Side-by-side comparison of base vs PEFT models using standardized metrics
- Notebook is self-contained and reproducible
- Ready for comparison with other fine-tuning approaches

