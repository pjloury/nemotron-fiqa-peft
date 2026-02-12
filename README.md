# Nemotron Next 8B LoRA Fine-tuning with FiQA Dataset

This project fine-tunes the [NVIDIA Nemotron Next 8B](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) language model using LoRA (Low-Rank Adaptation) on the FiQA (Financial Question Answering) dataset for financial domain specialization.

## Overview

The project implements instruction-tuning using:
- **Model:** NVIDIA Llama-3.1-Nemotron-Nano-8B-v1 (8.03B parameters)
- **Framework:** NeMo AutoModel with HuggingFace PEFT
- **Dataset:** [explodinggradients/fiqa](https://huggingface.co/datasets/explodinggradients/fiqa) (main config)
- **Method:** LoRA fine-tuning (0.08% trainable parameters)

## Why explodinggradients/fiqa?

We selected `explodinggradients/fiqa` for the following reasons:

1. **Direct Q&A Format**: The dataset provides direct question-answer pairs (`question` and `ground_truths` fields), making it ideal for instruction-tuning without complex data conversion.

2. **Pre-existing Splits**: The dataset comes with predefined train/validation/test splits (5,500/500/648 examples), ensuring consistent evaluation and reproducibility.

3. **Financial Domain Focus**: Contains curated financial Q&A pairs covering topics like trading, investing, market analysis, tax, and business expenses - perfect for financial domain specialization.

4. **Appropriate Answer Length**: 
   - Median answer length: 154 words
   - Mean answer length: 199 words
   - Distribution: 29.4% concise, 52.1% medium, 18.6% long
   - Answers are comprehensive yet manageable for training

5. **Evaluation Consistency**: Using full answers (no truncation) ensures training and evaluation ground truth are identical, enabling fair semantic similarity evaluation.

## Project Structure

```
nemotron-fiqa-peft/
├── train_fiqa_peft.py          # Main training script
├── requirements.txt              # Python dependencies
├── checkpoints/                 # Saved model checkpoints
│   └── final_model/            # Final trained LoRA adapter
├── data/                        # Processed datasets
│   ├── train_processed.json
│   ├── val_processed.json
│   └── test_processed.json
├── outputs/                     # Training logs and outputs
├── examples/                    # Example notebooks
└── README.md                    # This file
```

## Requirements

### Hardware
- **GPU:** NVIDIA GPU with 24GB+ VRAM (A100, H100, or RTX 4090 recommended)
- **Memory:** Sufficient system RAM for dataset processing

### Software
- Python 3.12+
- CUDA 12.8+
- NeMo AutoModel
- PyTorch 2.9.0+

## Installation

1. **Clone the repository:**
   ```bash
   cd nemotron-fiqa-peft
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### Training

Run the training script:

```bash
source venv/bin/activate
python3 train_fiqa_peft.py
```

The script will:
1. ✅ Verify GPU availability and environment
2. 📥 Load and preprocess the FiQA dataset
3. 📥 Load Nemotron Next 8B model using NeMo AutoModel
4. 🔧 Configure and apply LoRA adapter
5. 🔄 Tokenize datasets
6. 🚀 Train the LoRA adapter
7. 💾 Save final model and checkpoints

### Training Configuration

Default configuration (can be modified in `train_fiqa_peft.py`):

- **LoRA Parameters:**
  - Rank: 8
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj

- **Training Parameters:**
  - Learning rate: 2e-4
  - Batch size: 4
  - Gradient accumulation: 4 (effective batch size: 16)
  - Epochs: 3
  - Max sequence length: 2048
  - Warmup ratio: 0.1

### Output

After training, you'll find:
- **Final model:** `checkpoints/final_model/` (LoRA adapter ~27MB)
- **Checkpoints:** Saved at regular intervals in `checkpoints/`
- **Training logs:** TensorBoard logs in `outputs/logs/`
- **Processed data:** Saved datasets in `data/`

## Dataset Details

### explodinggradients/fiqa

- **Source:** [HuggingFace Datasets](https://huggingface.co/datasets/explodinggradients/fiqa)
- **Config:** `main`
- **Format:** Direct Q&A pairs
- **Splits:**
  - Train: 5,500 examples
  - Validation: 500 examples
  - Test: 648 examples (held out for evaluation)

### Data Format

Each example contains:
- `question`: Financial question (string)
- `ground_truths`: List of answer strings (we use the first one)

### Prompt Engineering

The training uses a system prompt to guide the model:

```
You are a helpful financial assistant. Provide comprehensive, detailed answers 
to financial questions. Include relevant context, examples, and explanations when appropriate.

Question: {question}

Answer: {answer}
```

This ensures the model learns to generate comprehensive answers similar to the ground truth format.

## Training Results

The training completed successfully with:
- **Total steps:** 1,032
- **Epochs:** 3
- **Final training loss:** 2.2302
- **Best validation loss:** 2.3628
- **LoRA adapter size:** 27MB
- **Trainable parameters:** 0.08% of total (6.8M / 8.03B)

## Evaluation

For evaluation, see the separate evaluation plan (`nemotron-next-8b-fiqa-evaluation.plan.md`). The evaluation uses:
- NeMo Evaluator SDK for metrics calculation
- Semantic similarity as primary metric
- Comparison between base and PEFT models

## Model Usage

### Loading the Fine-tuned Model

```python
from nemo_automodel import NeMoAutoModelForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer

# Load base model
model = NeMoAutoModelForCausalLM.from_pretrained(
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "./checkpoints/final_model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/final_model")
```

### Inference

```python
system_prompt = (
    "You are a helpful financial assistant. Provide comprehensive, detailed answers "
    "to financial questions. Include relevant context, examples, and explanations when appropriate."
)

question = "What is the difference between a stock and a bond?"
prompt = f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Technical Notes

### Memory Considerations

The 8B parameter model requires significant GPU memory. The implementation uses:
- bfloat16 precision
- LoRA (only 0.08% parameters trainable)
- Gradient accumulation for effective larger batch sizes

### NeMo AutoModel

This project uses **NeMo AutoModel** (not standard HuggingFace transformers) for:
- Optimized model loading and inference
- Better integration with NVIDIA hardware
- Enhanced training efficiency

## License

Please refer to the licenses of:
- Nemotron Next 8B model
- explodinggradients/fiqa dataset
- NeMo AutoModel framework

## References

- [Nemotron Next 8B Model](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1)
- [explodinggradients/fiqa Dataset](https://huggingface.co/datasets/explodinggradients/fiqa)
- [NeMo AutoModel](https://github.com/NVIDIA/NeMo-AutoModel)
- [PEFT Library](https://github.com/huggingface/peft)

## Contributing

This is a research/educational project. For issues or improvements, please open an issue or submit a pull request.
