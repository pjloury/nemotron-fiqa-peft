#!/usr/bin/env python3
"""
Evaluation script for Nemotron Next 8B PEFT model on FiQA test set.

Features:
- Uses NeMo Evaluator SDK for metrics calculation
- Compares base model vs PEFT fine-tuned model
- Semantic similarity as primary metric
- ROUGE, BLEU, EM, F1 scores
- Saves detailed results to JSON
- Checkpoint/resume support for long-running evaluations

Usage:
    python evaluate_fiqa.py --compare-base --max-samples 10 --verbose
    python evaluate_fiqa.py --peft-model none --compare-base  # Base model only
    python evaluate_fiqa.py --resume  # Resume from checkpoint
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from collections import defaultdict

# NeMo AutoModel
try:
    from nemo_automodel import NeMoAutoModelForCausalLM
except ImportError as e:
    print("❌ ERROR: NeMo AutoModel is required but not installed!")
    print(f"   Import error: {e}")
    raise ImportError("NeMo AutoModel is required. Please install it before running this script.")

# Evaluation Metrics
# Note: NeMo Evaluator is primarily a CLI launcher for API endpoints, not suitable
# for direct local model evaluation. We use fallback metrics instead.
# If you want to use NeMo Evaluator, you'd need to:
# 1. Set up your model as an OpenAI-compatible endpoint (e.g., using vLLM, TGI)
# 2. Use nemo-evaluator-launcher CLI tool with YAML config
# For local evaluation, we use: sentence-transformers, nltk, scikit-learn
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    FALLBACK_SEMANTIC = True
except ImportError:
    FALLBACK_SEMANTIC = False
    print("⚠️  sentence-transformers not available, semantic similarity will be skipped")
    # Fallback: use sentence-transformers for semantic similarity
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        FALLBACK_SEMANTIC = True
    except ImportError:
        FALLBACK_SEMANTIC = False
        print("⚠️  No semantic similarity fallback available")

# System prompt (must match training)
SYSTEM_PROMPT = (
    "You are a helpful financial assistant. Provide comprehensive, detailed answers "
    "to financial questions. Include relevant context, examples, and explanations when appropriate."
)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_id: str, peft_model_path: str = None, device: str = "auto"):
    """
    Load Nemotron Next 8B model with optional PEFT adapter.
    
    Args:
        model_id: Base model ID
        peft_model_path: Path to PEFT adapter (optional)
        device: Device to load model on
        
    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    print(f"📥 Loading base model: {model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model using NeMo AutoModel
    model = NeMoAutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
        attn_implementation="sdpa",  # Use SDPA instead of FlashAttention2
    )
    
    if peft_model_path:
        print(f"📥 Loading PEFT adapter: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path)
    
    model.eval()
    return model, tokenizer


# =============================================================================
# INFERENCE
# =============================================================================

def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512, temperature: float = 0.7):
    """
    Generate answer for a question using the model.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer
        question: Question string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated answer text
    """
    # Format prompt (same as training)
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (remove prompt)
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        # Fallback: take everything after the prompt
        answer = response[len(prompt):].strip()
    
    return answer


# =============================================================================
# EVALUATION
# =============================================================================

def setup_evaluator():
    """
    Set up evaluation metrics.
    
    Note: NeMo Evaluator is a CLI launcher for API endpoints, not suitable for
    direct local model evaluation. We use fallback metrics instead.
    """
    if FALLBACK_SEMANTIC:
        print("📊 Setting up fallback semantic similarity (sentence-transformers)...")
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        return None, semantic_model
    else:
        print("⚠️  No semantic similarity available, using basic metrics only")
        return None, None


def compute_semantic_similarity_fallback(predictions: List[str], references: List[str], model):
    """Compute semantic similarity using sentence-transformers fallback."""
    print("   Computing semantic similarity...")
    pred_embeddings = model.encode(predictions, show_progress_bar=True)
    ref_embeddings = model.encode(references, show_progress_bar=True)
    
    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
        similarities.append(float(sim))
    
    return {
        'semantic_similarity': {
            'mean': sum(similarities) / len(similarities),
            'scores': similarities
        }
    }


def compute_basic_metrics(predictions: List[str], references: List[str]):
    """Compute basic metrics (EM, F1) without NeMo Evaluator SDK."""
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            # Fallback: try punkt if punkt_tab fails
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
    
    from nltk.tokenize import word_tokenize
    
    exact_matches = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        # Exact match
        exact_matches.append(1.0 if pred.strip() == ref.strip() else 0.0)
        
        # F1 score (token-level)
        pred_tokens = set(word_tokenize(pred.lower()))
        ref_tokens = set(word_tokenize(ref.lower()))
        
        if len(ref_tokens) == 0:
            f1_scores.append(1.0 if len(pred_tokens) == 0 else 0.0)
        else:
            intersection = pred_tokens & ref_tokens
            precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
            recall = len(intersection) / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
    
    return {
        'exact_match': {
            'mean': sum(exact_matches) / len(exact_matches),
            'scores': exact_matches
        },
        'f1': {
            'mean': sum(f1_scores) / len(f1_scores),
            'scores': f1_scores
        }
    }


def save_checkpoint(checkpoint_dir: str, model_type: str, predictions: List[str], 
                   ground_truths: List[str], completed_indices: List[int], 
                   max_samples: int, metadata: Dict = None):
    """Save evaluation checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"{model_type}_checkpoint.json")
    
    checkpoint_data = {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "completed_indices": completed_indices,
        "max_samples": max_samples,
        "num_completed": len(predictions),
        "metadata": metadata or {},
        "timestamp": time.time()
    }
    
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    
    return checkpoint_file


def load_checkpoint(checkpoint_dir: str, model_type: str):
    """Load evaluation checkpoint if it exists."""
    checkpoint_file = os.path.join(checkpoint_dir, f"{model_type}_checkpoint.json")
    
    if not os.path.exists(checkpoint_file):
        return None
    
    with open(checkpoint_file, "r") as f:
        checkpoint_data = json.load(f)
    
    return checkpoint_data


def evaluate_model(model, tokenizer, dataset, max_samples: int = None, 
                  verbose: bool = False, checkpoint_dir: str = None, 
                  model_type: str = "peft", checkpoint_interval: int = 50):
    """
    Evaluate model on dataset and return predictions and metrics.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Test dataset
        max_samples: Maximum samples to evaluate (None for all)
        verbose: Print verbose output
        checkpoint_dir: Directory to save/load checkpoints
        model_type: Type of model ("peft" or "base") for checkpoint naming
        checkpoint_interval: Save checkpoint every N samples
        
    Returns:
        predictions, ground_truths, metrics
    """
    model_name = model_type.upper()
    print(f"\n🔄 Running inference on {model_name} model...")
    
    # Determine sample range
    total_samples = min(max_samples, len(dataset)) if max_samples else len(dataset)
    samples = dataset if max_samples is None else dataset.select(range(total_samples))
    
    # Try to load checkpoint
    start_idx = 0
    predictions = []
    ground_truths = []
    completed_indices = []
    
    if checkpoint_dir:
        checkpoint = load_checkpoint(checkpoint_dir, model_type)
        if checkpoint:
            predictions = checkpoint.get("predictions", [])
            ground_truths = checkpoint.get("ground_truths", [])
            completed_indices = checkpoint.get("completed_indices", list(range(len(predictions))))
            start_idx = len(predictions)
            
            # Verify checkpoint matches current run
            if checkpoint.get("max_samples") == max_samples:
                pct = (len(predictions) / total_samples) * 100
                print(f"   ✅ Resuming from checkpoint: {len(predictions)}/{total_samples} samples ({pct:.1f}%)")
            else:
                print(f"   ⚠️  Checkpoint max_samples mismatch, starting fresh")
                predictions = []
                ground_truths = []
                completed_indices = []
                start_idx = 0
    
    # Process remaining samples
    sample_list = list(samples)
    remaining_samples = sample_list[start_idx:]
    
    if remaining_samples:
        # Create progress bar with better formatting
        pbar = tqdm(
            remaining_samples,
            desc=f"   {model_name:4s} Progress",
            initial=start_idx,
            total=total_samples,
            unit="sample",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for local_idx, sample in enumerate(pbar):
            global_idx = start_idx + local_idx
            question = sample["question"]
            ground_truth = sample["ground_truths"][0]  # Use first ground truth
            
            # Update progress bar description with percentage
            pct = ((global_idx + 1) / total_samples) * 100
            pbar.set_description(f"   {model_name:4s} Progress ({pct:.1f}%)")
            
            # Generate prediction
            prediction = generate_answer(model, tokenizer, question)
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            completed_indices.append(global_idx)
            
            if verbose and global_idx < 3:
                print(f"\n   --- Sample {global_idx} ---")
                print(f"   Question: {question[:100]}...")
                print(f"   Ground Truth: {ground_truth[:150]}...")
                print(f"   Prediction: {prediction[:150]}...")
            
            # Save checkpoint periodically and show notification
            if checkpoint_dir and (global_idx + 1) % checkpoint_interval == 0:
                save_checkpoint(checkpoint_dir, model_type, predictions, ground_truths, 
                              completed_indices, max_samples, 
                              {"last_saved_idx": global_idx})
                # Print checkpoint notification (will appear above progress bar)
                pbar.write(f"   💾 Checkpoint saved at sample {global_idx + 1}/{total_samples} ({pct:.1f}%)")
        
        pbar.close()
        print(f"   ✅ Completed {total_samples}/{total_samples} samples (100.0%)")
    else:
        print(f"   ✅ All {total_samples} samples already completed from checkpoint")
    
    # Final checkpoint save
    if checkpoint_dir and len(predictions) > 0:
        save_checkpoint(checkpoint_dir, model_type, predictions, ground_truths, 
                       completed_indices, max_samples,
                       {"last_saved_idx": len(predictions) - 1, "completed": True})
    
    print(f"\n📊 Computing metrics on {len(predictions)} predictions...")
    
    # Set up evaluator
    evaluator, fallback_model = setup_evaluator()
    
    # Compute metrics using fallback approach
    # (NeMo Evaluator requires API endpoints, not suitable for local models)
    metrics = {}
    
    if fallback_model:
        print("   Computing semantic similarity...")
        semantic_metrics = compute_semantic_similarity_fallback(predictions, ground_truths, fallback_model)
        metrics.update(semantic_metrics)
    
    print("   Computing basic metrics (EM, F1)...")
    basic_metrics = compute_basic_metrics(predictions, ground_truths)
    metrics.update(basic_metrics)
    
    return predictions, ground_truths, metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Nemotron Next 8B PEFT on FiQA")
    parser.add_argument("--base-model", type=str, default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    parser.add_argument("--peft-model", type=str, default="./checkpoints/final_model",
                        help="Path to PEFT adapter, or 'none' for base only")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--compare-base", action="store_true", help="Also evaluate base model")
    parser.add_argument("--output", type=str, default="./results/eval_results.json", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--checkpoint-dir", type=str, default="./results/checkpoints",
                        help="Directory to save/load checkpoints for resuming")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Save checkpoint every N samples (default: 50)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable checkpointing (faster but no resume capability)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Nemotron Next 8B FiQA Evaluation")
    print("=" * 70)
    
    # Load dataset
    print(f"\n📥 Loading FiQA dataset (test split)...")
    dataset = load_dataset("explodinggradients/fiqa", "main")
    test_dataset = dataset["test"]
    print(f"   ✅ Loaded {len(test_dataset)} test examples")
    
    # Format dataset (same as training)
    def format_for_eval(example):
        return {
            "question": example["question"],
            "ground_truths": example["ground_truths"],
        }
    
    test_dataset = test_dataset.map(format_for_eval, desc="Formatting test data")
    
    all_results = {}
    
    # Setup checkpoint directory
    checkpoint_dir = None if args.no_checkpoint else args.checkpoint_dir
    if checkpoint_dir and args.resume:
        print(f"   📂 Checkpoint directory: {checkpoint_dir}")
    
    # Evaluate PEFT model
    if args.peft_model.lower() != "none" and os.path.exists(args.peft_model):
        print("\n" + "=" * 70)
        print("Evaluating PEFT Model")
        print("=" * 70)
        if args.compare_base:
            print(f"   (Step 1 of 2: PEFT model evaluation)")
        print()
        
        model, tokenizer = load_model(args.base_model, args.peft_model)
        peft_predictions, peft_ground_truths, peft_metrics = evaluate_model(
            model, tokenizer, test_dataset, args.max_samples, args.verbose,
            checkpoint_dir=checkpoint_dir, model_type="peft",
            checkpoint_interval=args.checkpoint_interval
        )
        
        print("\n--- PEFT Model Results ---")
        if isinstance(peft_metrics, dict):
            for key, value in peft_metrics.items():
                if isinstance(value, dict) and 'mean' in value:
                    print(f"   {key}: {value['mean']:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        all_results["peft"] = {
            "metrics": peft_metrics,
            "predictions": peft_predictions[:10] if args.verbose else [],  # Save first 10 for inspection
            "ground_truths": peft_ground_truths[:10] if args.verbose else [],
            "num_samples": len(peft_predictions)
        }
        
        del model
        torch.cuda.empty_cache()
    
    # Evaluate base model
    if args.compare_base:
        print("\n" + "=" * 70)
        print("Evaluating Base Model")
        print("=" * 70)
        print(f"   (Step 2 of 2: Base model evaluation)")
        print()
        
        model, tokenizer = load_model(args.base_model, peft_model_path=None)
        base_predictions, base_ground_truths, base_metrics = evaluate_model(
            model, tokenizer, test_dataset, args.max_samples, args.verbose,
            checkpoint_dir=checkpoint_dir, model_type="base",
            checkpoint_interval=args.checkpoint_interval
        )
        
        print("\n--- Base Model Results ---")
        if isinstance(base_metrics, dict):
            for key, value in base_metrics.items():
                if isinstance(value, dict) and 'mean' in value:
                    print(f"   {key}: {value['mean']:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        all_results["base"] = {
            "metrics": base_metrics,
            "predictions": base_predictions[:10] if args.verbose else [],
            "ground_truths": base_ground_truths[:10] if args.verbose else [],
            "num_samples": len(base_predictions)
        }
        
        # Comparison
        if "peft" in all_results:
            print("\n" + "=" * 70)
            print("Comparison: PEFT vs Base")
            print("=" * 70)
            
            p_metrics = all_results["peft"]["metrics"]
            b_metrics = all_results["base"]["metrics"]
            
            # Compare metrics
            comparison = {}
            for key in set(list(p_metrics.keys()) + list(b_metrics.keys())):
                if isinstance(p_metrics.get(key, {}), dict) and 'mean' in p_metrics.get(key, {}):
                    p_val = p_metrics[key]['mean']
                    b_val = b_metrics.get(key, {}).get('mean', 0.0)
                    diff = p_val - b_val
                    comparison[key] = {
                        "peft": p_val,
                        "base": b_val,
                        "improvement": diff,
                        "improvement_pct": (diff / b_val * 100) if b_val > 0 else 0.0
                    }
                    print(f"   {key:25s} PEFT={p_val:.4f}  Base={b_val:.4f}  Δ={diff:+.4f} ({diff/b_val*100:+.2f}%)" if b_val > 0 else f"   {key:25s} PEFT={p_val:.4f}  Base={b_val:.4f}  Δ={diff:+.4f}")
            
            all_results["comparison"] = comparison
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

