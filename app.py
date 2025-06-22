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
    embedding_model, # This might still be None initially if configure is not called before its definition
    calculate_retrieval_metrics,
    calculate_response_metrics,
    calculate_system_performance_metrics,
    format_metrics_for_display,
    calculate_overall_system_accuracy
)

# =============================================================================
# GOOGLE AI API KEY CONFIGURATION (NEW ADDITION)
# =============================================================================

import google.generativeai as genai

# Access the Google AI API key from Streamlit secrets
try:
    GOOGLE_AI_API_KEY = st.secrets["GOOGLE_AI_API_KEY"]
    # Configure the Google Generative AI library
    genai.configure(api_key=GOOGLE_AI_API_KEY)
except KeyError:
    st.error("Google AI API Key not found in Streamlit secrets. Please ensure 'GOOGLE_AI_API_KEY' is set in your app's secrets.")
    st.stop() # Stop the app if the key is not available, as core functionality depends on it.
except Exception as e:
    st.error(f"Error configuring Google Generative AI: {e}")
    st.stop()


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
# IMPORTANT: 'embedding_model' from utils needs to be defined AFTER genai.configure()
# If embedding_model relies on genai.configure, it should be loaded/initialized after it.
# Assuming 'embedding_model' in utils.py is just the sentence-transformer, it's fine.
# If it's a Google embedding model, ensure genai.configure() happens first.
if 'embedding_model_loaded' not in st.session_state:
    st.session_state.embedding_model_loaded = embedding_model is not None


# Store metrics for the last query
if 'last_query_metrics' not in st.session_state:
    st.session_state.last_query_metrics = None

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
                        # This `generate_embeddings` function in utils.py needs to use the configured genai.
                        # Make sure it's using the correct embedding model (e.g., from genai.get_embedding_model())
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
    # SYSTEM METRICS SECTION
    # =====================================================================

    st.subheader("📊 System Metrics")

    # Calculate and display system performance metrics
    if len(st.session_state.vector_store.chunks_data) > 0:
        system_metrics = calculate_system_performance_metrics(st.session_state.vector_store)

        # Display metrics in columns
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Documents", system_metrics['total_documents'])
            st.metric("Total Chunks", system_metrics['total_chunks'])
            st.metric("Unique Sources", system_metrics['unique_sources'])

        with col2:
            st.metric("Avg Chunk Size", f"{system_metrics['avg_chunk_size']:.0f} chars")
            st.metric("Total Content", f"{system_metrics['total_content_size']:,} chars")
            st.metric("Embedding Dims", system_metrics['embedding_dimensions'])

        # Show detailed metrics in expander
        with st.expander("📈 Detailed System Metrics"):
            st.markdown(format_metrics_for_display(system_metrics))
    else:
        st.info("No documents loaded - metrics unavailable")

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
                        # Ensure 'embedding_model' is correctly initialized and available
                        # If embedding_model is a global variable from utils, ensure it's configured
                        # For Gemini embeddings, it would be genai.get_embedding_model() or similar.
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

                            # Calculate metrics for this query
                            retrieval_metrics = calculate_retrieval_metrics(user_query, retrieved_chunks, query_embedding)
                            response_metrics = calculate_response_metrics(assistant_response, user_query, retrieved_chunks)

                            # Store metrics in session state
                            st.session_state.last_query_metrics = {
                                'retrieval': retrieval_metrics,
                                'response': response_metrics,
                                'query': user_query,
                                'timestamp': str(uuid.uuid4())[:8]  # Short timestamp for identification
                            }
                        else:
                            assistant_response = "I couldn't find any relevant information in the loaded documents to answer your question."
                            st.session_state.last_query_metrics = None

                # Display the assistant response
                st.markdown(assistant_response)

                # Display metrics if available
                if st.session_state.last_query_metrics and retrieved_chunks:
                    with st.expander("📊 Query Performance Metrics", expanded=False):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("🔍 Retrieval Metrics")
                            retrieval_metrics = st.session_state.last_query_metrics['retrieval']
                            st.metric("Chunks Retrieved", retrieval_metrics['num_chunks_retrieved'])
                            st.metric("Avg Similarity", f"{retrieval_metrics['avg_similarity']:.3f}")
                            st.metric("🎯 Retrieval Accuracy", f"{retrieval_metrics['retrieval_accuracy']:.1%}")
                            st.metric("🔗 Semantic Coherence", f"{retrieval_metrics['semantic_coherence']:.3f}")

                        with col2:
                            st.subheader("💬 Response Metrics")
                            response_metrics = st.session_state.last_query_metrics['response']
                            st.metric("Response Length", f"{response_metrics['response_word_count']} words")
                            st.metric("🎯 Response Accuracy", f"{response_metrics['response_accuracy']:.1%}")
                            st.metric("📊 Context Utilization", f"{response_metrics['context_utilization']:.1%}")
                            st.metric("✅ Answer Completeness", f"{response_metrics['answer_completeness']:.1%}")

                        # Detailed metrics
                        st.subheader("📈 Detailed Metrics")
                        detailed_tab1, detailed_tab2 = st.tabs(["Retrieval Details", "Response Details"])

                        with detailed_tab1:
                            st.markdown(format_metrics_for_display(retrieval_metrics))

                        with detailed_tab2:
                            st.markdown(format_metrics_for_display(response_metrics))

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

# =============================================================================
# METRICS DASHBOARD
# =============================================================================

st.header("📊 Performance Dashboard")

# Create tabs for different metric views
if len(st.session_state.vector_store.chunks_data) > 0:
    tab1, tab2, tab3 = st.tabs(["System Overview", "Last Query Analysis", "Historical Trends"])

    with tab1:
        st.subheader("🏗️ System Overview")

        # System metrics
        system_metrics = calculate_system_performance_metrics(st.session_state.vector_store)
        accuracy_metrics = calculate_overall_system_accuracy(st.session_state.vector_store)

        # Display overall system accuracy prominently
        overall_readiness = accuracy_metrics['overall_readiness']
        if overall_readiness >= 0.8:
            readiness_color = "🟢"
            readiness_status = "Excellent"
        elif overall_readiness >= 0.6:
            readiness_color = "🟡"
            readiness_status = "Good"
        elif overall_readiness >= 0.4:
            readiness_color = "🟠"
            readiness_status = "Fair"
        else:
            readiness_color = "🔴"
            readiness_status = "Poor"

        st.markdown(f"""
        ### {readiness_color} System Accuracy Score: {overall_readiness:.1%} ({readiness_status})

        This score reflects the overall quality and readiness of your RAG system based on data quality,
        embedding coverage, and recent query performance.
        """)

        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📄 Total Documents",
                value=system_metrics['total_documents'],
                help="Number of unique documents in the system"
            )

        with col2:
            st.metric(
                label="🧩 Total Chunks",
                value=system_metrics['total_chunks'],
                help="Number of text chunks created from documents"
            )

        with col3:
            st.metric(
                label="📏 Avg Chunk Size",
                value=f"{system_metrics['avg_chunk_size']:.0f}",
                help="Average size of text chunks in characters"
            )

        with col4:
            st.metric(
                label="🎯 Embedding Dims",
                value=system_metrics['embedding_dimensions'],
                help="Dimensionality of the embedding vectors"
            )

        # Additional system info
        st.subheader("📋 System Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"**Total Content Size:** {system_metrics['total_content_size']:,} characters")
            st.info(f"**Unique Sources:** {system_metrics['unique_sources']} files")

        with col2:
            st.info(f"**Current Embedding Set:** {st.session_state.current_embeddings_name or 'None'}")
            st.info(f"**Embedding Model:** all-MiniLM-L6-v2") # This will still be used for local embeddings
        with col3:
            st.info(f"**Data Quality:** {accuracy_metrics['data_quality']:.1%}")
            st.info(f"**Embedding Coverage:** {accuracy_metrics['embedding_quality']:.1%}")

        # Accuracy breakdown
        st.subheader("🎯 Accuracy Breakdown")
        acc_col1, acc_col2, acc_col3 = st.columns(3)

        with acc_col1:
            st.metric(
                label="🏗️ System Health",
                value=f"{accuracy_metrics['system_health']:.1%}",
                help="Overall health based on data quality and structure"
            )

        with acc_col2:
            st.metric(
                label="📊 Data Quality",
                value=f"{accuracy_metrics['data_quality']:.1%}",
                help="Percentage of chunks with valid content"
            )

        with acc_col3:
            st.metric(
                label="🎯 Embedding Quality",
                value=f"{accuracy_metrics['embedding_quality']:.1%}",
                help="Percentage of chunks with embeddings"
            )

    with tab2:
        st.subheader("🔍 Last Query Analysis")

        if st.session_state.last_query_metrics:
            metrics = st.session_state.last_query_metrics

            # Query info
            st.info(f"**Query:** {metrics['query']}")
            st.info(f"**Analysis ID:** {metrics['timestamp']}")

            # Overall accuracy score
            retrieval_acc = metrics['retrieval']['retrieval_accuracy']
            response_acc = metrics['response']['response_accuracy']
            overall_accuracy = (retrieval_acc + response_acc) / 2

            # Display overall accuracy prominently
            if overall_accuracy >= 0.8:
                overall_color = "🟢"
                overall_status = "Excellent"
            elif overall_accuracy >= 0.6:
                overall_color = "🟡"
                overall_status = "Good"
            elif overall_accuracy >= 0.4:
                overall_color = "🟠"
                overall_status = "Fair"
            else:
                overall_color = "🔴"
                overall_status = "Poor"

            st.markdown(f"""
            ### {overall_color} Overall System Accuracy: {overall_accuracy:.1%} ({overall_status})

            This score combines retrieval accuracy (how well relevant chunks were found)
            and response accuracy (how well the answer addresses the query).
            """)

            # Metrics visualization
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎯 Retrieval Performance")
                retrieval = metrics['retrieval']

                # Retrieval accuracy gauge
                retrieval_accuracy = retrieval['retrieval_accuracy']
                if retrieval_accuracy >= 0.8:
                    accuracy_color = "🟢"
                elif retrieval_accuracy >= 0.6:
                    accuracy_color = "🟡"
                else:
                    accuracy_color = "🔴"

                st.metric(
                    label=f"{accuracy_color} Retrieval Accuracy",
                    value=f"{retrieval_accuracy:.1%}",
                    help="Percentage of retrieved chunks that are relevant (similarity > 0.5)"
                )

                st.metric("📊 Avg Similarity", f"{retrieval['avg_similarity']:.3f}")
                st.metric("🔗 Semantic Coherence", f"{retrieval['semantic_coherence']:.3f}")
                st.metric("📏 Context Used", f"{retrieval['total_context_length']:,} chars")

            with col2:
                st.subheader("💬 Response Quality")
                response = metrics['response']

                # Response accuracy
                response_accuracy = response['response_accuracy']
                if response_accuracy >= 0.8:
                    response_color = "🟢"
                elif response_accuracy >= 0.6:
                    response_color = "🟡"
                else:
                    response_color = "🔴"

                st.metric(
                    label=f"{response_color} Response Accuracy",
                    value=f"{response_accuracy:.1%}",
                    help="How well the response addresses the query based on term coverage and structure"
                )

                st.metric("📊 Context Utilization", f"{response['context_utilization']:.1%}")
                st.metric("✅ Answer Completeness", f"{response['answer_completeness']:.1%}")
                st.metric("⏱️ Reading Time", f"{response['estimated_reading_time']:.1f} min")

            # Detailed breakdown
            with st.expander("🔬 Detailed Analysis"):
                st.json({
                    "retrieval_metrics": metrics['retrieval'],
                    "response_metrics": metrics['response']
                })
        else:
            st.info("No query metrics available. Ask a question to see analysis!")

    with tab3:
        st.subheader("📈 Historical Trends")
        st.info("Historical trend analysis will be available in future versions.")
        st.markdown("""
        **Coming Soon:**
        - Query performance over time
        - Most frequently asked topics
        - Retrieval accuracy trends
        - Response quality metrics
        - User interaction patterns
        """)

else:
    st.info("📊 **Metrics Dashboard:** Upload and process documents to see performance metrics and analytics.")