# 🚀 RAG Prototype Setup and Running Instructions

This guide will walk you through setting up and running the RAG (Retrieval-Augmented Generation) prototype step by step.

## 📋 Prerequisites

Before you begin, make sure you have:
- Python 3.8 or higher installed
- A Google Gemini API key (free tier available)

## 🔑 Step 1: Environment Setup (.env File)

### Create the .env File

1. **Navigate to the rag_prototype directory** in your terminal/command prompt:
   ```bash
   cd rag_prototype
   ```

2. **Create a new file named `.env`** in the rag_prototype directory:
   - **Windows**: Right-click in the folder → New → Text Document → Name it `.env` (make sure to include the dot)
   - **Mac/Linux**: Use the command: `touch .env`

3. **Open the .env file** in any text editor and add the following line:
   ```env
   GEMINI_API_KEY="YOUR_API_KEY_HERE"
   ```

4. **Replace `YOUR_API_KEY_HERE`** with your actual Google Gemini API key:
   ```env
   GEMINI_API_KEY="AIzaSyC1234567890abcdefghijklmnopqrstuvwxyz"
   ```

### Getting a Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Paste it in your `.env` file

**⚠️ Important**: Never share your API key or commit the `.env` file to version control!

## 📦 Step 2: Install Dependencies

### Install Required Packages

Run the following command in your terminal from the `rag_prototype` directory:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages:
- `streamlit` - Web interface
- `google-generativeai` - Google Gemini integration
- `sentence-transformers` - Text embeddings
- `PyPDF2` - PDF text extraction
- `python-dotenv` - Environment variable management
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning utilities
- `langchain-text-splitters` - Advanced text chunking

### Troubleshooting Installation Issues

If you encounter installation errors:

1. **Update pip first**:
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install packages one by one** if needed:
   ```bash
   pip install streamlit
   pip install google-generativeai
   pip install sentence-transformers
   pip install PyPDF2
   pip install python-dotenv
   pip install numpy
   pip install scikit-learn
   pip install langchain-text-splitters
   ```

3. **For Windows users** with compilation issues, you might need to install Visual Studio Build Tools.

## 📁 Step 3: Prepare Sample Documents

### Add Documents to the data/ Directory

1. **Place your PDF or TXT files** in the `rag_prototype/data/` directory
2. **Supported formats**: PDF and TXT files
3. **Sample document**: A sample document about AI is already included in `data/sample_document.txt`

### Document Preparation Tips

- **PDF files**: Should contain extractable text (not scanned images)
- **TXT files**: Plain text format works best
- **File size**: Start with smaller files (< 10MB) for testing
- **Content**: Documents with clear sections and paragraphs work well

## 🚀 Step 4: Run the Application

### Start the Streamlit App

Run the following command from the `rag_prototype` directory:

```bash
streamlit run app.py
```

### What Happens Next

1. **Streamlit will start** and show output similar to:
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.1.100:8501
   ```

2. **Your web browser will automatically open** to the application
3. **If it doesn't open automatically**, copy and paste the Local URL into your browser

## 🧪 Step 5: Testing the Application

### Test 1: Upload and Process Documents

1. **In the sidebar**, under "Upload & Process New Documents":
   - Click "Browse files" and select your PDF/TXT files
   - Enter a name for your embedding set (e.g., "my_documents")
   - Click "Process & Generate Embeddings"

2. **Watch for success messages**:
   - ✅ "Successfully processed X files and generated Y embeddings!"

3. **Check the "Current Status" section** to see:
   - Loaded Embedding Set name
   - Number of chunks in store

### Test 2: Save and Load Embedding Sets

1. **After processing documents**, the embeddings are automatically saved
2. **To test loading**:
   - Clear the current session (refresh the page)
   - In the sidebar, under "Load Existing Embeddings"
   - Select your saved embedding set from the dropdown
   - Click "Load Selected Embeddings"
   - Verify the status shows your loaded set

### Test 3: Chat with Your Documents

1. **Once documents are loaded**, you can start asking questions
2. **Type a question** in the chat input at the bottom
3. **Example questions** for the sample AI document:
   - "What is artificial intelligence?"
   - "What are the types of AI?"
   - "What are the applications of AI?"
   - "What challenges does AI face?"

4. **Observe the response**:
   - The AI will answer based on your documents
   - Responses include source citations
   - You can expand the "View Sources" section to see which parts of your documents were used

### Test 4: Clear Chat History

- Click the "Clear Chat History" button to start a fresh conversation
- This doesn't delete your documents, just clears the chat

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Embedding model failed to load"
- **Solution**: Check your internet connection (first run downloads the model)
- **Alternative**: Restart the application

#### 2. "GEMINI_API_KEY not configured"
- **Solution**: Check your `.env` file exists and contains the correct API key
- **Verify**: The key should be in quotes: `GEMINI_API_KEY="your_key_here"`

#### 3. "No documents loaded"
- **Solution**: Upload and process documents first
- **Check**: Look at the "Current Status" section in the sidebar

#### 4. Installation errors
- **Solution**: Try installing packages individually
- **Alternative**: Use a virtual environment

#### 5. PDF reading errors
- **Solution**: Ensure PDF contains extractable text (not scanned images)
- **Alternative**: Convert to TXT format

### Performance Tips

- **Start small**: Begin with 1-2 documents for testing
- **Chunk size**: Default is 500 characters with 50 character overlap
- **Model loading**: First run takes longer as it downloads the embedding model
- **Memory usage**: Large documents may require more RAM

## 📚 Understanding the System

### How It Works

1. **Document Processing**: 
   - Extracts text from PDFs/TXT files
   - Splits into manageable chunks
   - Generates vector embeddings

2. **Question Answering**:
   - Converts your question to a vector
   - Finds most similar document chunks
   - Uses Google Gemini to generate an answer
   - Cites sources from your documents

3. **Storage**:
   - Embeddings are saved to `embeddings/` directory
   - Can be reloaded for future sessions

### Key Features

- ✅ **Source Citation**: Every answer shows which documents were used
- ✅ **Persistent Storage**: Embeddings are saved and can be reused
- ✅ **Multiple Documents**: Process and query multiple files
- ✅ **Real-time Chat**: Interactive question-answering interface
- ✅ **Error Handling**: Clear error messages and validation

## 🎯 Next Steps

Once you're comfortable with the basic functionality:

1. **Try different document types** (research papers, manuals, etc.)
2. **Experiment with different questions** to test the system's capabilities
3. **Explore the source citations** to understand how the system finds relevant information
4. **Consider the limitations** and potential improvements

## 📞 Support

If you encounter issues not covered in this guide:

1. Check the console output for detailed error messages
2. Verify all prerequisites are met
3. Ensure your API key is valid and has sufficient quota
4. Try restarting the application

---

**Happy exploring with your RAG prototype! 🎉** 