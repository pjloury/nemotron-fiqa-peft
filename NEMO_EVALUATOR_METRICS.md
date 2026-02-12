# NeMo Evaluator SDK Metrics for Base vs PEFT Model Comparison

## Overview

The evaluation plan uses **NeMo Evaluator SDK** to calculate metrics and compare performance between the base Nemotron-8B model and the PEFT fine-tuned model on the FiQA dataset.

## Metrics Configuration

### Primary Metrics (NeMo Evaluator SDK)

1. **Semantic Similarity (Primary Metric)**
   - **Purpose:** Measure meaning similarity between model outputs and ground truth
   - **Implementation:** BERTScore or SBERT-based embeddings via NeMo Evaluator SDK
   - **Why:** Best captures answer quality, handles length differences
   - **Expected:** PEFT model should show higher semantic similarity scores

2. **ROUGE Scores**
   - **ROUGE-1:** Unigram overlap (content coverage)
   - **ROUGE-2:** Bigram overlap (phrase-level similarity)
   - **ROUGE-L:** Longest common subsequence (sentence-level similarity)
   - **Purpose:** Standard QA evaluation metrics at multiple granularities
   - **Expected:** PEFT model should show improvements across all ROUGE variants

3. **BLEU Score**
   - **Purpose:** N-gram precision for answer quality
   - **Implementation:** Standard BLEU calculation via NeMo Evaluator SDK
   - **Expected:** PEFT model should show higher BLEU scores

4. **Exact Match (EM)**
   - **Purpose:** Binary correctness metric
   - **Implementation:** Strict string match
   - **Expected:** PEFT model may show modest improvements (EM is strict)

5. **F1 Score**
   - **Purpose:** Token-level precision and recall
   - **Implementation:** Partial match metric
   - **Expected:** PEFT model should show higher F1 scores

## NeMo Evaluator SDK Usage

### Setup

```python
from nemo_evaluator import Evaluator

# Configure evaluator with metrics for QA task
evaluator = Evaluator(
    metrics=['semantic_similarity', 'rouge', 'bleu', 'exact_match', 'f1'],
    semantic_similarity_model='sbert',  # or 'bertscore'
    rouge_types=['rouge1', 'rouge2', 'rougeL'],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

### Base Model Evaluation

```python
# Run inference on test set
base_predictions = run_inference(base_model, test_set)

# Evaluate using NeMo Evaluator SDK
base_results = evaluator.evaluate(
    predictions=base_predictions,
    references=ground_truths
)

# Save results
save_results(base_results, 'baseline_results.json')
```

### PEFT Model Evaluation

```python
# Load PEFT adapter
peft_model = load_peft_model(base_model, adapter_path)

# Run inference on test set
peft_predictions = run_inference(peft_model, test_set)

# Evaluate using NeMo Evaluator SDK (same configuration)
peft_results = evaluator.evaluate(
    predictions=peft_predictions,
    references=ground_truths
)

# Save results
save_results(peft_results, 'peft_results.json')
```

### Comparison

```python
# Compare base vs PEFT using NeMo Evaluator SDK
comparison = evaluator.compare(base_results, peft_results)

# Generate comparison report
print_comparison_table(comparison)

# Analyze improvements
analyze_improvements(comparison)
```

## Expected Results

### Metric Improvements (PEFT vs Base)

| Metric | Expected Improvement | Rationale |
|--------|---------------------|------------|
| Semantic Similarity | **+5-15%** | Primary metric - should show clear improvement |
| ROUGE-1 | **+3-10%** | Better content coverage |
| ROUGE-2 | **+3-10%** | Better phrase-level similarity |
| ROUGE-L | **+3-10%** | Better sentence-level similarity |
| BLEU | **+2-8%** | Better n-gram precision |
| F1 | **+2-8%** | Better token-level matching |
| Exact Match | **+1-5%** | Modest improvement (strict metric) |

### Success Criteria

- **Semantic Similarity:** Clear improvement (primary metric)
- **ROUGE Scores:** Improvement across all variants
- **Consistency:** Improvements across multiple metrics validate fine-tuning success
- **Statistical Significance:** Improvements should be consistent across test set

## Benefits of Using NeMo Evaluator SDK

1. **Standardization:** Consistent metric calculation across both models
2. **Reproducibility:** Same SDK ensures fair comparison
3. **Comprehensive:** Multiple metrics provide holistic view of performance
4. **Built-in Support:** QA-specific metrics optimized for question answering
5. **Easy Comparison:** Built-in comparison tools for base vs PEFT

## Implementation Notes

- Use **same NeMo Evaluator SDK configuration** for both base and PEFT models
- Ensure **same ground truth format** for both evaluations
- Use **same prompt format** during inference for both models
- Save **raw metric values** for detailed analysis
- Generate **comparison visualizations** using NeMo Evaluator SDK results

## Output Format

The evaluation will produce:
- `baseline_results.json`: Base model metrics
- `peft_results.json`: PEFT model metrics
- `comparison_report.json`: Side-by-side comparison
- Visualization charts comparing metrics
- Summary table showing improvements

