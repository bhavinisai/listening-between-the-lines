#!/usr/bin/env python3
"""
Check available OpenRouter models and their correct IDs.

Usage:
  python check_openrouter_models.py --api-key YOUR_KEY
"""

import requests
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Check OpenRouter available models")
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
        
        print("Available OpenRouter Models:")
        print("=" * 50)
        
        # Group by provider
        providers = {}
        for model in models.get("data", []):
            provider = model.get("provider", {}).get("name", "Unknown")
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model)
        
        # Print by provider
        for provider, model_list in providers.items():
            print(f"\n{provider}:")
            for model in model_list:
                model_id = model.get("id", "Unknown")
                model_name = model.get("name", "Unknown")
                context_length = model.get("context_length", "Unknown")
                pricing = model.get("pricing", {})
                
                print(f"  {model_id:20} {model_name:40}")
                if context_length != "Unknown":
                    print(f"    Context: {context_length:15} tokens")
                if pricing:
                    prompt_price = pricing.get("prompt", "0")
                    completion_price = pricing.get("completion", "0")
                    if prompt_price and completion_price:
                        print(f"    Pricing: ${prompt_price}/1M prompt, ${completion_price}/1M completion")
        
        print("\n" + "=" * 50)
        print("Model ID Format:")
        print("  Use the 'id' field from above in your --model parameter")
        print("  Example: --model openrouter/meta-llama/llama-3.2-3b-instruct:free")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
