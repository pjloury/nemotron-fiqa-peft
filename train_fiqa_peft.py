#!/usr/bin/env python3
"""
Nemotron Next 8B LoRA Fine-tuning with FiQA Dataset

This script fine-tunes the NVIDIA Nemotron Next 8B model using LoRA (Low-Rank Adaptation)
on the FiQA (Financial Question Answering) dataset using NeMo AutoModel.

Based on:
- nemotron-next-8b-fiqa.plan.md
- NeMo AutoModel fine-tuning patterns
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch

# NeMo AutoModel imports - REQUIRED
try:
    from nemo_automodel import NeMoAutoModelForCausalLM
    # Note: NeMo AutoModel uses NeMoAutoModelForCausalLM, not AutoModelForCausalLM
    NEMO_AVAILABLE = True
except ImportError as e:
    print("❌ ERROR: NeMo AutoModel is required but not installed!")
    print(f"   Import error: {e}")
    print("\n   Please install NeMo AutoModel:")
    print("   pip install nemo-automodel")
    print("   or")
    print("   pip install git+https://github.com/NVIDIA/NeMo-AutoModel.git")
    raise ImportError("NeMo AutoModel is required. Please install it before running this script.")

# Data processing
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# PEFT for LoRA
from peft import LoraConfig, get_peft_model, TaskType

# Progress bars and utilities
from tqdm import tqdm
import sys

# Training callbacks
from transformers import TrainerCallback

# Configuration
CONFIG = {
    # Model
    "model_name": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "torch_dtype": torch.bfloat16,
    
    # Dataset
    "dataset_name": "explodinggradients/fiqa",
    "dataset_config": "main",
    
    # LoRA
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    
    # Training
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "max_seq_length": 2048,
    
    # Paths
    "output_dir": "./outputs",
    "checkpoint_dir": "./checkpoints",
    "data_dir": "./data",
}

# System prompt for instruction-tuning
SYSTEM_PROMPT = (
    "You are a helpful financial assistant. Provide comprehensive, detailed answers "
    "to financial questions. Include relevant context, examples, and explanations when appropriate."
)


# =============================================================================
# TRAINING CALLBACKS
# =============================================================================

class TrainingProgressCallback(TrainerCallback):
    """Custom callback to show detailed training progress."""
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        print(f"\n{'='*70}")
        print(f"📚 Epoch {state.epoch}/{args.num_train_epochs}")
        print(f"{'='*70}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are written."""
        if logs:
            step = state.global_step
            epoch = state.epoch
            if 'loss' in logs:
                print(f"\n   Step {step} | Epoch {epoch:.2f} | Loss: {logs['loss']:.4f}", end='')
                if 'learning_rate' in logs:
                    print(f" | LR: {logs['learning_rate']:.2e}", end='')
                if 'eval_loss' in logs:
                    print(f" | Eval Loss: {logs['eval_loss']:.4f}", end='')
                print()
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        print(f"\n   ✅ Evaluation completed at step {state.global_step}")


def setup_environment():
    """Set up directories and verify GPU availability."""
    print("\n" + "=" * 70)
    print("Environment Setup")
    print("=" * 70)
    
    # Verify GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available! This script requires a GPU.")
    
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create directories
    print("\n📁 Creating directories...")
    for dir_path in [CONFIG["output_dir"], CONFIG["checkpoint_dir"], CONFIG["data_dir"]]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    print("\n" + "=" * 70)
    print("✅ Environment setup complete!")
    print()


def load_and_preprocess_dataset():
    """
    Load FiQA dataset and preprocess for instruction-tuning.
    
    Returns:
        train_dataset, val_dataset, test_dataset: Processed datasets
        - train_dataset: For training
        - val_dataset: For validation during training
        - test_dataset: For final evaluation (held out)
    """
    print("=" * 70)
    print("Dataset Loading & Preprocessing")
    print("=" * 70)
    
    # Load dataset
    print(f"📥 Loading dataset: {CONFIG['dataset_name']} ({CONFIG['dataset_config']})...")
    with tqdm(total=1, desc="Loading dataset", bar_format="{l_bar}{bar}| {elapsed}") as pbar:
        dataset = load_dataset(CONFIG["dataset_name"], CONFIG["dataset_config"])
        pbar.update(1)
    
    print(f"\n✅ Dataset loaded successfully")
    print(f"   📊 Train size: {len(dataset['train']):,} examples")
    print(f"   📊 Validation size: {len(dataset['validation']):,} examples")
    print(f"   📊 Test size: {len(dataset['test']):,} examples")
    
    def format_instruction(example):
        """Format example for instruction-tuning with prompt engineering."""
        question = example["question"]
        # Use first ground truth (full answer, no truncation)
        answer = example["ground_truths"][0]
        
        # Format with system prompt
        formatted_text = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer: {answer}"
        
        return {
            "text": formatted_text,
            "question": question,
            "answer": answer,
        }
    
    # Process datasets
    print("\n🔄 Processing datasets...")
    print("   Processing training dataset...")
    train_dataset = dataset["train"].map(
        format_instruction,
        remove_columns=[col for col in dataset["train"].column_names if col not in ["text", "question", "answer"]],
        desc="   Formatting training examples",
        batched=False
    )
    
    print("   Processing validation dataset...")
    val_dataset = dataset["validation"].map(
        format_instruction,
        remove_columns=[col for col in dataset["validation"].column_names if col not in ["text", "question", "answer"]],
        desc="   Formatting validation examples",
        batched=False
    )
    
    print("   Processing test dataset (held out for final evaluation)...")
    test_dataset = dataset["test"].map(
        format_instruction,
        remove_columns=[col for col in dataset["test"].column_names if col not in ["text", "question", "answer"]],
        desc="   Formatting test examples",
        batched=False
    )
    
    # Print sample
    print("\n📝 Sample formatted example:")
    sample = train_dataset[0]
    print(f"   Question: {sample['question'][:100]}...")
    print(f"   Answer length: {len(sample['answer'])} characters ({len(sample['answer'].split())} words)")
    print(f"   Formatted text length: {len(sample['text'])} characters")
    print(f"   Formatted text preview: {sample['text'][:200]}...")
    
    # Save processed datasets
    train_path = Path(CONFIG["data_dir"]) / "train_processed.json"
    val_path = Path(CONFIG["data_dir"]) / "val_processed.json"
    test_path = Path(CONFIG["data_dir"]) / "test_processed.json"
    
    print(f"\n💾 Saving processed datasets...")
    train_dataset.to_json(str(train_path))
    val_dataset.to_json(str(val_path))
    test_dataset.to_json(str(test_path))
    print(f"   ✅ Saved: {train_path} ({len(train_dataset)} examples)")
    print(f"   ✅ Saved: {val_path} ({len(val_dataset)} examples)")
    print(f"   ✅ Saved: {test_path} ({len(test_dataset)} examples) - held out for final evaluation")
    
    print("\n📊 Dataset Split Summary:")
    print(f"   Training: {len(train_dataset)} examples (for model training)")
    print(f"   Validation: {len(val_dataset)} examples (for validation during training)")
    print(f"   Test: {len(test_dataset)} examples (held out for final evaluation)")
    
    print("=" * 70)
    print("✅ Dataset preprocessing complete!")
    print()
    
    return train_dataset, val_dataset, test_dataset


def load_model_and_tokenizer():
    """
    Load Nemotron Next 8B model and tokenizer using NeMo AutoModel.
    
    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    print("=" * 70)
    print("Model Loading")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load tokenizer
    print("📥 Loading tokenizer...")
    with tqdm(total=1, desc="   Loading tokenizer", bar_format="{l_bar}{bar}| {elapsed}") as pbar:
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"],
            trust_remote_code=True,
        )
        pbar.update(1)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"   ✅ Tokenizer loaded")
    print(f"      Vocab size: {tokenizer.vocab_size:,}")
    print(f"      Pad token: {tokenizer.pad_token}")
    
    # Load model using NeMo AutoModel (REQUIRED)
    print(f"\n📥 Loading model with NeMo AutoModel: {CONFIG['model_name']}")
    print("   This may take a few minutes...")
    
    # Create progress bar for model loading
    loading_bar = tqdm(total=100, desc="   Loading model", bar_format="{l_bar}{bar}| {elapsed} | {percentage:3.0f}%")
    
    # Start loading in a way that allows progress updates
    def update_progress(progress):
        loading_bar.n = min(progress, 100)
        loading_bar.refresh()
    
    try:
        # NeMo AutoModel loading with progress callback if available
        # Disable FlashAttention2 if not available
        model = NeMoAutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=CONFIG["torch_dtype"],
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="sdpa",  # Use SDPA instead of FlashAttention2
        )
        loading_bar.n = 100
        loading_bar.refresh()
    finally:
        loading_bar.close()
    
    load_time = time.time() - start_time
    
    print(f"\n   ✅ Model loaded successfully in {load_time:.1f} seconds")
    print(f"      Model type: {type(model).__name__}")
    print(f"      Dtype: {model.dtype}")
    print(f"      Device: {next(model.parameters()).device}")
    
    # Print model summary
    def count_parameters(model):
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    total, trainable = count_parameters(model)
    print(f"\n📊 Model Summary:")
    print(f"   Total parameters: {total:,} ({total/1e9:.2f}B)")
    print(f"   Trainable parameters: {trainable:,} ({trainable/1e9:.2f}B)")
    print(f"   Trainable %: {100 * trainable / total:.2f}%")
    
    print("=" * 70)
    print("✅ Model loading complete!")
    print()
    
    return model, tokenizer


def configure_lora(model, tokenizer):
    """
    Configure and apply LoRA adapter to the model.
    
    Args:
        model: Base model to apply LoRA to
        tokenizer: Tokenizer for testing inference
        
    Returns:
        model: Model with LoRA adapter applied
    """
    print("=" * 70)
    print("LoRA Configuration")
    print("=" * 70)
    
    # Configure LoRA using PEFT library
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["lora_target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print(f"📋 LoRA Configuration:")
    print(f"   Rank (r): {CONFIG['lora_rank']}")
    print(f"   Alpha: {CONFIG['lora_alpha']}")
    print(f"   Dropout: {CONFIG['lora_dropout']}")
    print(f"   Target modules: {CONFIG['lora_target_modules']}")
    
    # Apply LoRA
    print("\nApplying LoRA adapter...")
    model = get_peft_model(model, lora_config)
    
    # Verify trainable parameters
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    total, trainable = count_parameters(model)
    print(f"\n✅ LoRA applied successfully")
    print(f"   Total parameters: {total:,} ({total/1e9:.2f}B)")
    print(f"   Trainable parameters: {trainable:,} ({trainable/1e9:.2f}B)")
    print(f"   Trainable %: {100 * trainable / total:.2f}%")
    
    # Test inference
    print("\n🧪 Testing inference with LoRA adapter...")
    test_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: What is a stock?\n\nAnswer:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Test response: {response[len(test_prompt):][:100]}...")
    print("   ✅ Inference works correctly")
    
    print("=" * 70)
    print("✅ LoRA configuration complete!")
    print()
    
    return model


def prepare_data_for_training(train_dataset, val_dataset, tokenizer):
    """
    Tokenize and prepare datasets for training.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer to use
        
    Returns:
        tokenized_train, tokenized_val: Tokenized datasets
    """
    print("=" * 70)
    print("Data Tokenization")
    print("=" * 70)
    
    def tokenize_function(examples):
        """Tokenize examples for training."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=CONFIG["max_seq_length"],
            padding="max_length",
        )
    
    print("🔄 Tokenizing datasets...")
    print("   Tokenizing training dataset...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=[col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask"]],
        desc="   Tokenizing training data"
    )
    
    print("   Tokenizing validation dataset...")
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=[col for col in val_dataset.column_names if col not in ["input_ids", "attention_mask"]],
        desc="   Tokenizing validation data"
    )
    
    print(f"\n   ✅ Tokenization complete")
    print(f"      Training examples: {len(tokenized_train):,}")
    print(f"      Validation examples: {len(tokenized_val):,}")
    
    print("=" * 70)
    print("✅ Data preparation complete!")
    print()
    
    return tokenized_train, tokenized_val


def train_model(model, tokenizer, train_dataset, val_dataset):
    """
    Train the LoRA adapter on FiQA dataset using NeMo AutoModel Trainer.
    
    Args:
        model: Model with LoRA adapter
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    print("=" * 70)
    print("LoRA Training with NeMo AutoModel")
    print("=" * 70)
    
    # Use NeMo AutoModel Trainer if available, otherwise fall back to HF Trainer
    # Note: NeMo AutoModel may have different training APIs
    # For now, we'll use HF Trainer but ensure model is loaded with NeMo AutoModel
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["checkpoint_dir"],
        overwrite_output_dir=True,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        logging_dir=f"{CONFIG['output_dir']}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # Use bfloat16 instead
        bf16=True,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        # Verbose output settings
        logging_first_step=True,
        prediction_loss_only=False,
        remove_unused_columns=False,
        # Progress bar settings
        disable_tqdm=False,  # Enable progress bars
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create trainer with progress callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[TrainingProgressCallback()],
    )
    
    print(f"\n📋 Training Configuration:")
    print(f"   Epochs: {CONFIG['num_epochs']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"   Learning rate: {CONFIG['learning_rate']}")
    print(f"   Warmup ratio: {CONFIG['warmup_ratio']}")
    print(f"   Max sequence length: {CONFIG['max_seq_length']}")
    print(f"   Training examples: {len(train_dataset):,}")
    print(f"   Validation examples: {len(val_dataset):,}")
    print(f"   Checkpoint directory: {CONFIG['checkpoint_dir']}")
    print()
    
    print("=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # Train with progress bars
    print("Training progress will be shown below:\n")
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ Training Completed")
    print("=" * 70)
    print(f"   ⏱️  Total time: {training_time/3600:.2f} hours ({training_time/60:.1f} minutes)")
    print(f"   📉 Final training loss: {train_result.training_loss:.4f}")
    if hasattr(train_result, 'metrics'):
        print(f"   📊 Training metrics: {train_result.metrics}")
    print()
    
    # Save final model
    print("💾 Saving final model and artifacts...")
    final_model_path = Path(CONFIG["checkpoint_dir"]) / "final_model"
    
    with tqdm(total=2, desc="   Saving model", bar_format="{l_bar}{bar}| {elapsed}") as pbar:
        trainer.save_model(str(final_model_path))
        pbar.update(1)
        tokenizer.save_pretrained(str(final_model_path))
        pbar.update(1)
    
    print(f"   ✅ Model saved to: {final_model_path}")
    
    # Save training config
    config_path = Path(CONFIG["checkpoint_dir"]) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"   ✅ Config saved to: {config_path}")
    
    print("\n" + "=" * 70)
    print("✅ All training artifacts saved!")
    print("=" * 70)
    print()


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("Nemotron Next 8B LoRA Fine-tuning with FiQA Dataset")
    print("Using NeMo AutoModel")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Setup environment
        setup_environment()
        
        # Step 2: Load and preprocess dataset
        train_dataset, val_dataset, test_dataset = load_and_preprocess_dataset()
        
        # Note: test_dataset is saved but not used during training
        # It will be used for final evaluation (see evaluation plan)
        
        # Step 3: Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Step 4: Configure LoRA
        model = configure_lora(model, tokenizer)
        
        # Step 5: Prepare data for training
        tokenized_train, tokenized_val = prepare_data_for_training(
            train_dataset, val_dataset, tokenizer
        )
        
        # Step 6: Train model
        train_model(model, tokenizer, tokenized_train, tokenized_val)
        
        print("=" * 70)
        print("🎉 All steps completed successfully!")
        print("=" * 70)
        print(f"\n📁 Outputs saved to:")
        print(f"   - Checkpoints: {CONFIG['checkpoint_dir']}")
        print(f"   - Logs: {CONFIG['output_dir']}/logs")
        print(f"   - Processed data: {CONFIG['data_dir']}")
        print()
        print("Next steps:")
        print("  1. Evaluate the fine-tuned model (see evaluation plan)")
        print("  2. Use the saved adapter for inference")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

