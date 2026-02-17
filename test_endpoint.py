#!/usr/bin/env python3
"""Test script for the OpenAI-compatible API endpoint."""

import requests
import json
import time
import sys

ENDPOINT_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    print("=" * 70)
    print("Testing Health Check")
    print("=" * 70)
    try:
        response = requests.get(f"{ENDPOINT_URL}/", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_models():
    """Test models list endpoint."""
    print("\n" + "=" * 70)
    print("Testing Models List")
    print("=" * 70)
    try:
        response = requests.get(f"{ENDPOINT_URL}/v1/models", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chat_completion():
    """Test chat completions endpoint."""
    print("\n" + "=" * 70)
    print("Testing Chat Completions")
    print("=" * 70)
    
    payload = {
        "model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1-peft",
        "messages": [
            {"role": "user", "content": "What is a stock?"}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("\nSending request...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{ENDPOINT_URL}/v1/chat/completions",
            json=payload,
            timeout=120  # Allow time for model inference
        )
        elapsed = time.time() - start_time
        
        print(f"\nStatus: {response.status_code}")
        print(f"Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse:")
            print(json.dumps(result, indent=2))
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"\n✅ Generated Answer:")
                print(f"{'='*70}")
                print(content[:500])
                if len(content) > 500:
                    print("...")
                print(f"{'='*70}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("Testing OpenAI-Compatible API Endpoint")
    print(f"Endpoint: {ENDPOINT_URL}")
    
    # Wait for server to be ready
    print("\n⏳ Waiting for server to be ready...")
    max_wait = 120
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{ENDPOINT_URL}/", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except:
            time.sleep(2)
            print(".", end="", flush=True)
    else:
        print(f"\n❌ Server not responding after {max_wait}s")
        print("   Make sure the server is running:")
        print("   python serve_model.py --port 8000")
        sys.exit(1)
    
    print()
    
    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Models List", test_models()))
    results.append(("Chat Completion", test_chat_completion()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30s} {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All tests passed! Endpoint is ready for NeMo Evaluator.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

