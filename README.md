# RAG Prototype - Document Q&A System

A complete Retrieval-Augmented Generation (RAG) pipeline prototype that allows you to upload documents (PDFs and text files) and ask questions about their content using Google Gemini as the language model.

## 🚀 Quick Start

For detailed setup instructions, see [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md).
check version2 branch for all deployments errors


### Quick Setup (3 steps):

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

## Features

- 📄 **Document Processing**: Support for PDF and TXT files
- 🔍 **Smart Chunking**: Intelligent text splitting with overlap
- 🧠 **Vector Embeddings**: Using Sentence Transformers for semantic search
- 🔎 **FAISS Index**: Fast similarity search for document retrieval
- 🤖 **Google Gemini Integration**: Advanced language model for answer generation
- 💬 **Streamlit Interface**: User-friendly chatbot interface
- 📚 **Source Citation**: View the source documents for each answer
- 💾 **Persistent Storage**: Embeddings and index are saved for reuse

## Project Structure

```
rag_prototype/
├── data/              # For raw PDFs and video transcripts (initially empty)
├── embeddings/        # For persistent embedding storage (initially empty)
├── app.py            # Main Streamlit application
├── utils.py          # Helper functions (chunking, embedding, storage, Gemini integration)
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── SETUP_INSTRUCTIONS.md # Detailed setup guide
└── test_setup.py     # Setup verification script
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### 2. Install Dependencies

```bash
cd rag_prototype
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the `rag_prototype` directory with your Google API key:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

**To get a Google Gemini API key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Add it to your `.env` file

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage Guide

### 1. Upload Documents

1. Use the sidebar to upload PDF or TXT files
2. The system will automatically:
   - Extract text from the documents
   - Split them into manageable chunks
   - Generate vector embeddings
   - Create a searchable index

### 2. Ask Questions

1. Once documents are uploaded, you can start asking questions
2. Type your question in the chat input
3. The system will:
   - Search for relevant document sections
   - Generate an answer using Google Gemini
   - Show you the source documents used

### 3. View Sources

- Each answer includes an expandable "View Sources" section
- Click to see which parts of your documents were used to generate the answer
- This helps verify the accuracy and traceability of responses

## Technical Details

### Components

- **Document Processor**: Extracts text from PDFs and TXT files
- **Text Chunker**: Splits documents into overlapping chunks (500 chars with 50 char overlap)
- **Embedding Model**: Uses `all-MiniLM-L6-v2` for generating 384-dimensional embeddings
- **Vector Index**: In-memory vector store with cosine similarity search
- **Language Model**: Google Gemini Pro for answer generation
- **Storage**: Pickle files for metadata and embeddings

### Configuration

You can modify the following parameters in `utils.py`:

- `chunk_size`: Size of text chunks (default: 500 characters)
- `chunk_overlap`: Overlap between chunks (default: 50 characters)
- `vector_dimension`: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
- `k`: Number of context chunks to retrieve (default: 5)

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `.env` file contains the correct Google API key
2. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Memory Issues**: For large documents, consider reducing chunk size
4. **PDF Reading Errors**: Some PDFs may have security restrictions or be image-based

### Performance Tips

- Start with smaller documents to test the system
- The first run will download the embedding model (~80MB)
- Subsequent runs will be faster as embeddings are cached
- Large documents may take longer to process initially

## Testing

Run the test script to verify your setup:

```bash
python test_setup.py
```

This will check:
- ✅ All required packages are installed
- ✅ Environment variables are configured
- ✅ Embedding model can be loaded
- ✅ Gemini API connection works
- ✅ File structure is correct

## Limitations

- Currently supports only PDF and TXT files
- Image-based PDFs may not extract text properly
- Requires internet connection for Google Gemini API
- Embedding model is downloaded on first use

## Future Enhancements

- Support for more document formats (DOCX, PPTX, etc.)
- Web scraping capabilities
- Multi-language support
- Advanced chunking strategies
- Document summarization
- Export functionality

## License

This project is for educational and prototyping purposes.

## Contributing

Feel free to submit issues and enhancement requests! 
