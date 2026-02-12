# Prompt Engineering for FiQA Fine-tuning

## Overview

To ensure model outputs are comparable to ground truth answers for semantic similarity evaluation, we use prompt engineering to guide the model to generate comprehensive, detailed answers similar in length and style to the ground truth.

## Ground Truth Analysis

- **Median answer length:** 154 words
- **Mean answer length:** 199 words
- **Answer style:** Comprehensive, detailed financial explanations with context and examples

## Prompt Template

### Training Prompt Format

```
You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, and explanations when appropriate.

Question: {question}

Answer: {answer}
```

### Inference Prompt Format (Same as Training)

```
You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, and explanations when appropriate.

Question: {question}

Answer:
```

## Rationale

1. **System Prompt:** Establishes role and expectations
   - "You are a helpful financial assistant" - Sets domain context
   - "Provide comprehensive, detailed answers" - Encourages longer, detailed responses
   - "Include relevant context, examples, and explanations" - Guides toward ground truth style

2. **Consistency:** Using the same prompt during training and inference ensures:
   - Model learns to generate answers in the expected format
   - Inference outputs are comparable to training examples
   - Fair semantic similarity evaluation

3. **Length Matching:** The prompt encourages answers similar to ground truth length (median 154 words)

## Implementation

### During Training

```python
def create_instruction(example):
    system_prompt = "You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, and explanations when appropriate."
    return f"{system_prompt}\n\nQuestion: {example['question']}\n\nAnswer: {example['answer']}"
```

### During Inference

```python
def create_inference_prompt(question):
    system_prompt = "You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, and explanations when appropriate."
    return f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"
```

## Alternative Prompt Variations

If you want to adjust the answer style, you can modify the system prompt:

### More Concise (if needed)
```
You are a helpful financial assistant. Provide clear, direct answers to financial questions.
```

### More Detailed
```
You are a helpful financial assistant. Provide comprehensive, detailed answers to financial questions. Include relevant context, examples, explanations, and any relevant regulations or guidelines when appropriate.
```

### Domain-Specific
```
You are a financial expert. Provide detailed, accurate answers to financial questions. Include relevant tax implications, regulations, and practical examples when applicable.
```

## Benefits

1. **Consistency:** Same prompt format during training and evaluation
2. **Comparability:** Model outputs are more similar to ground truth in length/style
3. **Fair Evaluation:** Semantic similarity metrics are more meaningful
4. **Flexibility:** Can adjust prompt to match desired answer style

## Notes

- The prompt should match between training and inference
- Ground truth format (full answers) should match training format
- Prompt engineering complements using full ground truth (no truncation)
- Together, these ensure fair semantic similarity evaluation

