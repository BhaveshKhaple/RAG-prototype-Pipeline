#!/usr/bin/env python3
"""
Test script to verify RAG prototype setup.
Run this script to check if all dependencies and components are working correctly.
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import tiktoken
        print("✅ TikToken imported successfully")
    except ImportError as e:
        print(f"❌ TikToken import failed: {e}")
        return False
    
    return True

def test_env_file():
    """Test if .env file exists and contains API key."""
    print("\nTesting environment setup...")
    
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if api_key:
        print("✅ Google API key found in environment")
        return True
    else:
        print("❌ Google API key not found")
        print("Please create a .env file with your GOOGLE_API_KEY")
        return False

def test_embedding_model():
    """Test if embedding model can be loaded."""
    print("\nTesting embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
        # Test encoding
        test_text = ["This is a test sentence."]
        embeddings = model.encode(test_text)
        print(f"✅ Embedding generation successful (shape: {embeddings.shape})")
        
        return True
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

def test_gemini_connection():
    """Test if Gemini API connection works."""
    print("\nTesting Gemini API connection...")
    
    try:
        import google.generativeai as genai
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            print("❌ No API key available for Gemini test")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Simple test prompt
        response = model.generate_content("Say 'Hello, RAG prototype!'")
        print("✅ Gemini API connection successful")
        print(f"Response: {response.text}")
        
        return True
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

def test_file_structure():
    """Test if required directories exist."""
    print("\nTesting file structure...")
    
    required_dirs = ['data', 'embeddings']
    required_files = ['app.py', 'utils.py', 'requirements.txt', 'README.md']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory '{dir_name}' exists")
        else:
            print(f"❌ Directory '{dir_name}' missing")
            return False
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ File '{file_name}' exists")
        else:
            print(f"❌ File '{file_name}' missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 RAG Prototype Setup Test")
    print("=" * 40)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Package Imports", test_imports),
        ("Environment Setup", test_env_file),
        ("Embedding Model", test_embedding_model),
        ("Gemini API", test_gemini_connection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG prototype is ready to use.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above and fix them.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Create .env file with your GOOGLE_API_KEY")
        print("3. Check your internet connection for model downloads")

if __name__ == "__main__":
    main() 