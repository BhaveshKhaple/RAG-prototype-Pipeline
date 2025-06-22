#!/usr/bin/env python3
"""
Check Available Gemini Models

This script lists all available Gemini models to help identify the correct model name.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

def main():
    """Check available Gemini models."""
    print("🔍 Checking Available Gemini Models")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("❌ No API key found!")
        return
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # List available models
        print("Available models:")
        print("-" * 30)
        
        models = genai.list_models()
        for model in models:
            print(f"Model: {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Supported Methods: {model.supported_generation_methods}")
            print()
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    main()