#!/usr/bin/env python3
"""
Check if gpt-4o-mini is available on OpenRouter.

Usage:
  python check_gpt4o_mini.py --api-key YOUR_KEY
"""

import requests
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Check gpt-4o-mini availability")
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
        
        print("Checking for gpt-4o-mini models...")
        print("=" * 50)
        
        gpt4o_models = []
        for model in models.get("data", []):
            model_id = model.get("id", "")
            model_name = model.get("name", "")
            
            if "gpt-4o-mini" in model_id.lower() or "gpt-4o-mini" in model_name.lower():
                gpt4o_models.append(model)
        
        if gpt4o_models:
            print("✅ Found gpt-4o-mini models:")
            for model in gpt4o_models:
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
            print("❌ No gpt-4o-mini models found")
            print("\nAvailable GPT models:")
            for model in models.get("data", []):
                model_id = model.get("id", "")
                if "gpt" in model_id.lower():
                    print(f"  {model_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
