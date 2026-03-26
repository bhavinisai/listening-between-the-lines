#!/usr/bin/env python3
"""
Check NVIDIA models available on OpenRouter.

Usage:
  python check_nvidia_models.py --api-key YOUR_KEY
"""

import requests
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Check NVIDIA models on OpenRouter")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    args = parser.parse_args()
    
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Get models list
        resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=30)
        resp.raise_for_status()
        
        models = resp.json()
        
        print("NVIDIA Models on OpenRouter:")
        print("=" * 50)
        
        nvidia_models = []
        for model in models.get("data", []):
            model_id = model.get("id", "")
            model_name = model.get("name", "")
            provider = model.get("provider", {}).get("name", "")
            
            if "nvidia" in provider.lower() or "nvidia" in model_id.lower() or "nemotron" in model_id.lower():
                nvidia_models.append(model)
        
        if nvidia_models:
            print("✅ Found NVIDIA models:")
            for model in nvidia_models:
                model_id = model.get("id", "Unknown")
                model_name = model.get("name", "Unknown")
                context_length = model.get("context_length", "Unknown")
                pricing = model.get("pricing", {})
                
                print(f"  {model_id}")
                print(f"    Name: {model_name}")
                print(f"    Context: {context_length} tokens")
                
                if pricing:
                    prompt_price = pricing.get("prompt", "0")
                    completion_price = pricing.get("completion", "0")
                    if prompt_price and completion_price:
                        print(f"    Pricing: ${prompt_price}/1M prompt, ${completion_price}/1M completion")
                        if float(prompt_price) == 0 and float(completion_price) == 0:
                            print(f"    ✅ FREE!")
                        else:
                            print(f"    💰 PAID")
                print()
        else:
            print("❌ No NVIDIA models found")
            print("\nAll free models available:")
            for model in models.get("data", []):
                model_id = model.get("id", "")
                pricing = model.get("pricing", {})
                if pricing:
                    prompt_price = pricing.get("prompt", "0")
                    completion_price = pricing.get("completion", "0")
                    if float(prompt_price) == 0 and float(completion_price) == 0:
                        print(f"  {model_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
