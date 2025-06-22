"""
RAG Prototype - Streamlit Application

This module provides a user-friendly web interface for the RAG (Retrieval-Augmented Generation)
pipeline. It allows users to upload documents, generate embeddings, and interact with
their documents through a chatbot interface powered by Google Gemini.

Features:
- Document upload and processing (PDF/TXT)
- Embedding generation and storage
- Interactive chat interface with source citations
- Session state management
- Comprehensive error handling and user feedback

Author: RAG Prototype Team
Date: 2024
"""

import streamlit as st
import os
import uuid
import glob
import tempfile
from utils import (
    InMemoryVectorStore, 
    load_document, 
    chunk_document, 
    generate_embeddings, 
    save_embeddings, 
    load_embeddings, 
    get_available_embedding_sets, 
    generate_answer_with_gemini,
    embedding_model,
    get_api_key_status,
    test_gemini_api_key
)

# =============================================================================
# STREAMLIT UI CONFIGURATION
# =============================================================================

# Configure the Streamlit page with appropriate title and layout
st.set_page_config(
    page_title="RAG Prototype - Document Q&A",
    page_icon="📚",
    layout="wide"
)

# =============================================================================
# MAIN TITLE AND DESCRIPTION
# =============================================================================

# Display main title with emoji for visual appeal
st.title("📚 RAG Prototype - Document Q&A System")

# Provide clear description of the application's capabilities
st.markdown("""
This application allows you to upload documents (PDF/TXT), generate embeddings, and ask questions about your documents using Google Gemini.

**Key Features:**
- 📄 Upload and process PDF/TXT documents
- 🧠 Generate semantic embeddings for intelligent search
- 💬 Chat with your documents using natural language
- 📚 View source citations for every answer
- 💾 Save and reload embedding sets for future use
""")

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize vector store for in-memory storage of embeddings
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore()

# Track the currently loaded embedding set name
if 'current_embeddings_name' not in st.session_state:
    st.session_state.current_embeddings_name = None

# Maintain chat history for conversation continuity
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Track embedding model loading status for UI state management
if 'embedding_model_loaded' not in st.session_state:
    st.session_state.embedding_model_loaded = embedding_model is not None

# =============================================================================
# ERROR HANDLING AND STATUS DISPLAY
# =============================================================================

# Display error if embedding model failed to load
if not st.session_state.embedding_model_loaded:
    st.error("❌ Embedding model failed to load. Please check your installation and try again.")
    st.info("💡 **Solution:** Ensure you have an internet connection for the first run (model download required).")

# =============================================================================
# SIDEBAR - DOCUMENT PROCESSING & EMBEDDING MANAGEMENT
# =============================================================================

with st.sidebar:
    st.header("📄 Document Processing & Embedding Management")
    
    # =====================================================================
    # UPLOAD & PROCESS NEW DOCUMENTS SECTION
    # =====================================================================
    
    st.subheader("Upload & Process New Documents")
    
    # File uploader with multiple file support and type restrictions
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Select one or more PDF or TXT files to process"
    )
    
    # Text input for naming the embedding set
    new_embedding_set_name = st.text_input(
        "New Embedding Set Name",
        placeholder="Enter a name for this embedding set...",
        help="Choose a descriptive name for your embedding set (e.g., 'research_papers', 'manuals')"
    )
    
    # Process button with conditional enabling based on model status
    process_button = st.button(
        "Process & Generate Embeddings",
        disabled=not st.session_state.embedding_model_loaded,
        help="Click to process uploaded documents and generate embeddings"
    )
    
    # Document processing logic with comprehensive error handling
    if process_button:
        # Validate user inputs before processing
        if not uploaded_files:
            st.error("Please upload at least one file.")
        elif not new_embedding_set_name.strip():
            st.error("Please enter a name for the embedding set.")
        else:
            # Process documents with loading indicator
            with st.spinner("Processing documents and generating embeddings..."):
                all_chunks = []
                
                try:
                    # Loop through uploaded files for processing
                    for uploaded_file in uploaded_files:
                        # Create temporary file in data directory
                        os.makedirs("data", exist_ok=True)
                        temp_file_path = os.path.join("data", uploaded_file.name)
                        
                        # Save uploaded file temporarily
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Generate unique document ID using UUID for consistency
                        doc_id = str(uuid.uuid4())
                        
                        # Load document content with error handling
                        text, error = load_document(temp_file_path)
                        if error:
                            st.error(f"Error loading {uploaded_file.name}: {error}")
                            continue
                        
                        # Chunk the document into manageable pieces
                        chunks = chunk_document(text, doc_id, uploaded_file.name)
                        
                        # Generate embeddings for the chunks
                        chunks_with_embeddings = generate_embeddings(chunks)
                        
                        # Accumulate all processed chunks
                        all_chunks.extend(chunks_with_embeddings)
                        
                        # Clean up temporary file
                        os.remove(temp_file_path)
                    
                    # Save embeddings if chunks were successfully generated
                    if all_chunks:
                        success, error = save_embeddings(all_chunks, new_embedding_set_name)
                        if success:
                            # Update session state with new data
                            st.session_state.vector_store = InMemoryVectorStore()
                            st.session_state.vector_store.add_vectors(all_chunks)
                            st.session_state.current_embeddings_name = new_embedding_set_name
                            
                            # Display success message with details
                            st.success(f"✅ Successfully processed {len(uploaded_files)} files and generated {len(all_chunks)} embeddings!")
                            st.rerun()
                        else:
                            st.error(f"Error saving embeddings: {error}")
                    else:
                        st.error("No chunks were generated from the uploaded files.")
                        
                except Exception as e:
                    st.error(f"Error during processing: {e}")
                    st.info("💡 **Troubleshooting:** Check file formats and ensure files contain extractable text.")
    
    st.divider()
    
    # =====================================================================
    # LOAD EXISTING EMBEDDINGS SECTION
    # =====================================================================
    
    st.subheader("Load Existing Embeddings")
    
    # Get available embedding sets for selection
    available_sets = get_available_embedding_sets()
    if available_sets:
        # Dropdown for selecting embedding set
        selected_embedding_set = st.selectbox(
            "Choose embedding set to load:",
            available_sets,
            help="Select a previously saved embedding set to load"
        )
        
        # Load button for selected embeddings
        load_button = st.button("Load Selected Embeddings")
        
        if load_button:
            # Validate selection before loading
            if not selected_embedding_set:
                st.error("Please select an embedding set to load.")
            else:
                # Load embeddings with loading indicator
                with st.spinner("Loading embeddings..."):
                    embeddings_data, error = load_embeddings(selected_embedding_set)
                    if error:
                        st.error(f"Error loading embeddings: {error}")
                    else:
                        # Update session state with loaded data
                        st.session_state.vector_store = InMemoryVectorStore()
                        st.session_state.vector_store.add_vectors(embeddings_data)
                        st.session_state.current_embeddings_name = selected_embedding_set
                        st.session_state.chat_history = []  # Clear chat history for new context
                        
                        # Display success message
                        st.success(f"✅ Successfully loaded embedding set: {selected_embedding_set}")
                        st.rerun()
    else:
        st.info("No existing embedding sets found.")
    
    st.divider()
    
    # =====================================================================
    # CURRENT STATUS SECTION
    # =====================================================================
    
    st.subheader("Current Status")
    
    # Display current system status
    if st.session_state.current_embeddings_name:
        st.info(f"**Loaded Embedding Set:** {st.session_state.current_embeddings_name}")
        st.info(f"**Number of chunks in store:** {len(st.session_state.vector_store.chunks_data)}")
    else:
        st.info("No embedding set currently loaded.")
    
    # Reiterate embedding model status
    if not st.session_state.embedding_model_loaded:
        st.error("❌ Embedding model not loaded - functionality limited.")
    
    st.divider()
    
    # =====================================================================
    # API KEY STATUS SECTION
    # =====================================================================
    
    st.subheader("🔑 API Key Status")
    
    # Get API key status
    api_status = get_api_key_status()
    
    if api_status['api_key_present']:
        st.success(f"✅ API Key: {api_status['api_key_masked']}")
        
        # Test API key button
        if st.button("Test API Key", help="Click to test if your API key is working"):
            with st.spinner("Testing API key..."):
                is_working, message = test_gemini_api_key()
                if is_working:
                    st.success(message)
                else:
                    st.error(message)
                    st.info("💡 **Troubleshooting:** Check your .env file and ensure your API key is valid.")
    else:
        st.error("❌ No API key found")
        st.info("💡 **Setup:** Create a .env file with GOOGLE_API_KEY or GEMINI_API_KEY")
        
        # Show setup instructions
        with st.expander("📝 Setup Instructions"):
            st.markdown("""
            1. Create a `.env` file in the project directory
            2. Add your Google Gemini API key:
               ```
               GOOGLE_API_KEY=your_actual_api_key_here
               ```
            3. Get your API key from: [Google AI Studio](https://makersuite.google.com/app/apikey)
            4. Restart the application after adding the API key
            """)

# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

st.header("💬 Chatbot")

# Display chat history with proper formatting
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =============================================================================
# CHAT INPUT AND RESPONSE GENERATION
# =============================================================================

# Determine if chat input should be disabled
chat_input_disabled = not st.session_state.embedding_model_loaded or len(st.session_state.vector_store.chunks_data) == 0

# Display appropriate warning messages
if chat_input_disabled:
    if not st.session_state.embedding_model_loaded:
        st.warning("Chat is disabled because the embedding model is not loaded.")
    else:
        st.warning("Chat is disabled because no documents are loaded. Please upload and process documents first.")

# Chat input with conditional enabling
user_query = st.chat_input(
    "Ask a question...",
    disabled=chat_input_disabled
)

# Process user query and generate response
if user_query:
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate assistant response with comprehensive error handling
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check system prerequisites
                if not st.session_state.embedding_model_loaded:
                    assistant_response = "Sorry, I cannot process your question because the embedding model is not loaded."
                elif len(st.session_state.vector_store.chunks_data) == 0:
                    assistant_response = "Sorry, I don't have any documents loaded to answer your question. Please upload and process some documents first."
                else:
                    # Generate query embedding with error handling
                    try:
                        query_embedding = embedding_model.encode([user_query])[0]
                    except Exception as e:
                        st.error(f"Error generating query embedding: {e}")
                        assistant_response = "Sorry, I encountered an error while processing your question."
                    else:
                        # Search for relevant chunks using cosine similarity
                        retrieved_chunks = st.session_state.vector_store.search(query_embedding, k=5)
                        
                        if retrieved_chunks:
                            # Generate answer using Gemini with context
                            assistant_response = generate_answer_with_gemini(user_query, retrieved_chunks)
                        else:
                            assistant_response = "I couldn't find any relevant information in the loaded documents to answer your question."
                
                # Display the assistant response
                st.markdown(assistant_response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                error_msg = f"An error occurred while processing your question: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# =============================================================================
# CHAT MANAGEMENT
# =============================================================================

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun() 