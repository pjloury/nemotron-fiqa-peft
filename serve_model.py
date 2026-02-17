#!/usr/bin/env python3
"""
Simple OpenAI-compatible API server for PEFT model.

This server loads your fine-tuned PEFT model and exposes an OpenAI-compatible
endpoint that NeMo Evaluator can use.

Usage:
    python serve_model.py --port 8000 --host 0.0.0.0
    # Or with specific model paths:
    python serve_model.py --base-model nvidia/Llama-3.1-Nemotron-Nano-8B-v1 --peft-model ./checkpoints/final_model
"""

import argparse
import os
import time
from typing import List, Optional, Dict, Any
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# NeMo AutoModel
try:
    from nemo_automodel import NeMoAutoModelForCausalLM
except ImportError as e:
    print("❌ ERROR: NeMo AutoModel is required!")
    print(f"   Import error: {e}")
    raise ImportError("NeMo AutoModel is required. Please install it before running this script.")

from transformers import AutoTokenizer
from peft import PeftModel

# System prompt (must match training)
SYSTEM_PROMPT = (
    "You are a helpful financial assistant. Provide comprehensive, detailed answers "
    "to financial questions. Include relevant context, examples, and explanations when appropriate."
)

# =============================================================================
# OpenAI-compatible request/response models
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]

# =============================================================================
# Model loading
# =============================================================================

def load_model(model_id: str, peft_model_path: str = None, device: str = "auto"):
    """Load Nemotron Next 8B model with optional PEFT adapter."""
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
# FastAPI app
# =============================================================================

app = FastAPI(title="Nemotron Next 8B PEFT API", version="1.0.0")

# Enable CORS for NeMo Evaluator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None
model_name = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, tokenizer, model_name
    
    base_model = os.getenv("BASE_MODEL", "nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    peft_model_path = os.getenv("PEFT_MODEL_PATH", "./checkpoints/final_model")
    
    if not os.path.exists(peft_model_path):
        print(f"⚠️  PEFT model path not found: {peft_model_path}")
        print("   Starting with base model only...")
        peft_model_path = None
    
    model_name = base_model
    if peft_model_path:
        model_name = f"{base_model}-peft"
    
    model, tokenizer = load_model(base_model, peft_model_path)
    print(f"✅ Model loaded and ready! Serving as: {model_name}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": model_name,
        "service": "Nemotron Next 8B PEFT API"
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nvidia"
        }]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extract messages
    messages = request.messages
    
    # Build prompt from messages
    # Format: system prompt + user messages
    prompt_parts = [SYSTEM_PROMPT]
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        if role == "system":
            # System message - prepend to prompt
            prompt_parts.insert(0, content)
        elif role == "user":
            prompt_parts.append(f"\n\nQuestion: {content}")
        elif role == "assistant":
            prompt_parts.append(f"\n\nAnswer: {content}")
    
    prompt_parts.append("\n\nAnswer:")
    prompt = "".join(prompt_parts)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
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
    
    # Calculate token usage (approximate)
    input_tokens = len(inputs.input_ids[0])
    output_tokens = len(outputs[0]) - input_tokens
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model or model_name,
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=answer),
            finish_reason="stop"
        )],
        usage={
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    )

def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible API server for PEFT model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--base-model", type=str, default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                        help="Base model ID")
    parser.add_argument("--peft-model", type=str, default="./checkpoints/final_model",
                        help="Path to PEFT adapter")
    args = parser.parse_args()
    
    # Set environment variables for startup
    os.environ["BASE_MODEL"] = args.base_model
    os.environ["PEFT_MODEL_PATH"] = args.peft_model
    
    print("=" * 70)
    print("Nemotron Next 8B PEFT API Server")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"PEFT model: {args.peft_model}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 70)
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

