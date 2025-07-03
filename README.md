# RAG Prototype - Document Q&A System

A complete Retrieval-Augmented Generation (RAG) pipeline prototype that allows you to upload documents (PDFs and text files) and ask questions about their content using Google Gemini as the language model.

---

## 🚀 Quick Start

1. **Create `.env` file** in the `rag_prototype` directory:
   ```env
   GEMINI_API_KEY="your_google_gemini_api_key_here"
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

## Features
- Document Processing: PDF and TXT support
- Smart Chunking: Overlapping text splitting
- Vector Embeddings: Sentence Transformers
- FAISS Index: Fast similarity search
- Google Gemini Integration: LLM answers
- Streamlit Interface: User-friendly UI
- Source Citation: Traceable answers
- Persistent Storage: Embeddings/index reuse

---

## Project Structure
```
rag_prototype/
├── data/              # Raw PDFs and text files
├── embeddings/        # Persistent embedding storage
├── app.py             # Main Streamlit app
├── utils.py           # Helper functions
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── test_setup.py      # Setup verification script
```

---

## Setup & Usage

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key

### Environment Setup
1. Create a `.env` file in `rag_prototype`:
   ```env
   GEMINI_API_KEY="your_google_gemini_api_key_here"
   ```
   Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Upload & Process Documents
- Use the sidebar to upload PDF/TXT files
- The system extracts, chunks, embeds, and indexes content

### Ask Questions
- Type your question in the chat
- The system retrieves relevant chunks and generates an answer with Gemini
- Expand "View Sources" to see which document parts were used

---

## Code Quality & Best Practices

### Documentation
- Module-level and function-level docstrings
- Inline comments for complex logic
- Clear variable names and section headers

### Error Handling
- File existence and permission checks
- PDF encryption and API error handling
- User-friendly error messages

### Modularity
- Single-responsibility functions and classes
- Clear separation between utility and app logic

### Security
- API keys loaded from `.env` (never hardcoded)
- Input validation and file path sanitization

### User Experience
- Streamlit spinners, success/error/info messages
- Conditional UI elements and tooltips

---

## Troubleshooting
- **API Key Error**: Check `.env` for correct key
- **Import Errors**: Ensure all dependencies are installed
- **Memory Issues**: Reduce chunk size for large docs
- **PDF Errors**: Some PDFs may be image-based or encrypted

---

## Testing
Run the test script:
```bash
python test_setup.py
```
Checks package installation, environment, model loading, API connection, and file structure.

---

## Limitations & Future Enhancements
- Only PDF/TXT supported (for now)
- Image-based PDFs may not extract text
- Requires internet for Gemini API
- Embedding model downloads on first use
- Planned: More formats, web scraping, multi-language, advanced chunking, export, user feedback

---

## License
This project is for educational and prototyping purposes.

## Contributing
Feel free to submit issues and enhancement requests! 