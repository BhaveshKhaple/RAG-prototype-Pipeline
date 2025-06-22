#!/usr/bin/env python3
"""
API Key Test Script for RAG Prototype

This script specifically tests whether the Google Gemini API key is working correctly.
Run this script to verify your API key configuration before using the main application.

Usage:
    python test_api_key.py
"""

import os
import sys
from utils import test_gemini_api_key, get_api_key_status

def main():
    """Main function to test API key functionality."""
    print("🔑 Google Gemini API Key Test")
    print("=" * 50)
    
    # Get comprehensive API key status
    status = get_api_key_status()
    
    print(f"API Key Present: {'✅ Yes' if status['api_key_present'] else '❌ No'}")
    
    if status['api_key_present']:
        print(f"API Key (masked): {status['api_key_masked']}")
        print(f"API Key Length: {status['api_key_length']} characters")
        print(f"Google AI Configured: {'✅ Yes' if status['genai_configured'] else '❌ No'}")
        
        print("\n" + "=" * 50)
        print("🧪 Testing API Key Functionality...")
        print("=" * 50)
        
        # Test the API key
        is_working, message = test_gemini_api_key()
        print(message)
        
        if is_working:
            print("\n🎉 SUCCESS: Your API key is working correctly!")
            print("You can now use the RAG prototype application.")
            print("\nTo start the application, run:")
            print("streamlit run app.py")
        else:
            print("\n⚠️  FAILURE: API key is not working properly.")
            print("\n🔧 Troubleshooting Steps:")
            print("1. Check your .env file exists in the project directory")
            print("2. Verify your API key is correct (no extra spaces or characters)")
            print("3. Ensure your Google Cloud project has Gemini API enabled")
            print("4. Check your API quotas and billing settings")
            print("5. Verify your internet connection")
            
            print("\n📝 Expected .env file format:")
            print("GOOGLE_API_KEY=your_actual_api_key_here")
            print("# OR")
            print("GEMINI_API_KEY=your_actual_api_key_here")
    else:
        print("\n❌ No API key found!")
        print("\n🔧 Setup Instructions:")
        print("1. Create a .env file in the project directory")
        print("2. Add your Google Gemini API key to the file:")
        print("   GOOGLE_API_KEY=your_actual_api_key_here")
        print("3. Get your API key from: https://makersuite.google.com/app/apikey")
        print("4. Make sure the .env file is in the same directory as this script")
    
    print("\n" + "=" * 50)
    print("📊 Summary")
    print("=" * 50)
    
    if status['api_key_present'] and status.get('test_result'):
        print("Status: ✅ READY - API key is configured and working")
        return 0
    elif status['api_key_present']:
        print("Status: ⚠️  CONFIGURED BUT NOT WORKING - Check API key validity")
        return 1
    else:
        print("Status: ❌ NOT CONFIGURED - API key missing")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)