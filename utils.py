"""
RAG Prototype - Utility Functions Module

This module contains all core functionality for the RAG (Retrieval-Augmented Generation) pipeline:
- Document loading and processing
- Text chunking and embedding generation
- Vector storage and similarity search
- Google Gemini integration for answer generation

Author: RAG Prototype Team
Date: 2024
"""

import os
import json
import pickle
import numpy as np
import hashlib
import uuid
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Directory paths for data storage
EMBEDDINGS_DIR = "embeddings"
DATA_DIR = "data"

# Load environment variables from .env file for security
load_dotenv()

# Load Gemini API key from environment variables (security best practice)
# Support both GEMINI_API_KEY and GOOGLE_API_KEY for compatibility
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables")
    print("Please create a .env file with your Google Gemini API key")

# Configure Google Generative AI with API key
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: Google Generative AI not configured due to missing API key")

# Initialize SentenceTransformer model with error handling
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ SentenceTransformer model loaded successfully")
except Exception as e:
    print(f"❌ Error loading SentenceTransformer model: {e}")
    embedding_model = None

# =============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# =============================================================================

def load_document(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Loads text content from PDF or TXT files with comprehensive error handling.
    
    This function supports multiple file formats and provides detailed error messages
    for troubleshooting. It uses PyPDF2 for PDF processing and direct text reading
    for TXT files.
    
    Args:
        file_path (str): Path to the document file to be processed
        
    Returns:
        Tuple[Optional[str], Optional[str]]: 
            - First element: Extracted text content (None if error)
            - Second element: Error message (None if successful)
            
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        PermissionError: If the file cannot be accessed due to permissions
        
    Example:
        >>> text, error = load_document("data/sample.pdf")
        >>> if error:
        ...     print(f"Error: {error}")
        ... else:
        ...     print(f"Successfully loaded {len(text)} characters")
    """
    try:
        # Validate file existence
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"
        
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            return None, f"Permission denied: Cannot read {file_path}"
        
        # Process PDF files using PyPDF2
        if file_path.lower().endswith('.pdf'):
            text = ""
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # Check if PDF is encrypted
                    if pdf_reader.is_encrypted:
                        return None, f"PDF is encrypted: {file_path}"
                    
                    # Extract text from each page
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        except Exception as e:
                            print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                            continue
                
                return text.strip(), None
                
            except Exception as e:
                return None, f"Error reading PDF {file_path}: {e}"
        
        # Process TXT files with UTF-8 encoding
        elif file_path.lower().endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                return text.strip(), None
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        text = file.read()
                    return text.strip(), None
                except Exception as e:
                    return None, f"Error reading TXT file {file_path}: {e}"
            except Exception as e:
                return None, f"Error reading TXT file {file_path}: {e}"
        
        # Unsupported file type
        else:
            return None, f"Unsupported file type: {file_path}. Supported formats: PDF, TXT"
            
    except Exception as e:
        return None, f"Unexpected error reading file {file_path}: {e}"

def chunk_document(text: str, doc_id: str, filename: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    Splits text into context-aware chunks with comprehensive metadata.
    
    This function uses LangChain's RecursiveCharacterTextSplitter for intelligent
    text chunking that respects natural boundaries like paragraphs and sentences.
    Each chunk is assigned a unique identifier and metadata for traceability.
    
    Args:
        text (str): The text content to be chunked
        doc_id (str): Unique identifier for the source document (UUID)
        filename (str): Original filename for reference
        chunk_size (int): Maximum size of each chunk in characters (default: 500)
        chunk_overlap (int): Overlap between consecutive chunks in characters (default: 50)
        
    Returns:
        List[Dict]: List of chunk dictionaries with metadata
        
    Note:
        - chunk_size should be larger than chunk_overlap
        - Smaller chunks provide more precise retrieval but may lose context
        - Larger overlaps help maintain context across chunk boundaries
        
    Example:
        >>> chunks = chunk_document("Long text content...", "doc_123", "sample.txt")
        >>> print(f"Created {len(chunks)} chunks")
    """
    # Validate input parameters
    if not text or not text.strip():
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    # Use RecursiveCharacterTextSplitter for intelligent chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Prefer paragraph breaks, then line breaks, then spaces
    )
    
    # Split the text into chunks
    text_chunks = text_splitter.split_text(text)
    
    # Create chunk dictionaries with comprehensive metadata
    chunks = []
    for i, chunk_content in enumerate(text_chunks):
        # Generate unique chunk ID using SHA256 hash for consistency
        chunk_id = hashlib.sha256(f"{doc_id}-{i}-{chunk_content}".encode('utf-8')).hexdigest()
        
        chunk_dict = {
            'content': chunk_content,
            'doc_id': doc_id,
            'source': filename,
            'chunk_id': chunk_id,
            'chunk_index': i,
            'chunk_size': len(chunk_content),
            'page_number': None,  # Placeholder for future PDF page tracking
            'timestamp_hint': None  # Placeholder for future timestamp tracking
        }
        chunks.append(chunk_dict)
    
    return chunks

# =============================================================================
# EMBEDDING GENERATION AND STORAGE FUNCTIONS
# =============================================================================

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """
    Generates vector embeddings for a list of chunk dictionaries.
    
    This function uses the SentenceTransformer model to convert text chunks into
    high-dimensional vector representations that capture semantic meaning. The
    embeddings are added to each chunk dictionary for storage and retrieval.
    
    Args:
        chunks (List[Dict]): List of chunk dictionaries containing 'content' field
        
    Returns:
        List[Dict]: List of chunk dictionaries with embeddings added
        
    Note:
        - Requires internet connection for first-time model download
        - Embeddings are 384-dimensional vectors (all-MiniLM-L6-v2 model)
        - Processed in batches for efficiency
        
    Example:
        >>> chunks_with_embeddings = generate_embeddings(text_chunks)
        >>> print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    """
    # Validate embedding model availability
    if embedding_model is None:
        print("Error: Embedding model not loaded")
        return chunks
    
    if not chunks:
        print("Warning: No chunks provided for embedding generation")
        return chunks
    
    try:
        # Extract content for embedding generation
        contents = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings using SentenceTransformer
        print(f"Generating embeddings for {len(contents)} chunks...")
        embeddings = embedding_model.encode(contents, show_progress_bar=True)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        print(f"✅ Generated embeddings for {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return chunks

def save_embeddings(embeddings: List[Dict], storage_name: str) -> Tuple[bool, Optional[str]]:
    """
    Persistently saves a list of embedding-chunk dictionaries to disk.
    
    This function ensures the embeddings directory exists and saves the data
    using pickle format for efficient storage and retrieval. It provides
    comprehensive error handling for file operations.
    
    Args:
        embeddings (List[Dict]): List of chunk dictionaries with embeddings
        storage_name (str): Name for the storage file (without extension)
        
    Returns:
        Tuple[bool, Optional[str]]: 
            - First element: True if successful, False otherwise
            - Second element: Error message if failed, None if successful
            
    Note:
        - Files are saved in the embeddings/ directory
        - Storage name should be alphanumeric and safe for filenames
        - Data is serialized using pickle for efficiency
        
    Example:
        >>> success, error = save_embeddings(chunks, "my_documents")
        >>> if success:
        ...     print("Embeddings saved successfully")
        ... else:
        ...     print(f"Error: {error}")
    """
    try:
        # Validate input parameters
        if not embeddings:
            return False, "No embeddings provided for saving"
        
        if not storage_name or not storage_name.strip():
            return False, "Storage name cannot be empty"
        
        # Sanitize storage name for filename safety
        safe_name = "".join(c for c in storage_name if c.isalnum() or c in ('-', '_')).rstrip()
        if not safe_name:
            return False, "Storage name contains no valid characters"
        
        # Ensure embeddings directory exists
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        
        # Create file path
        file_path = os.path.join(EMBEDDINGS_DIR, f"{safe_name}.pkl")
        
        # Save to pickle file with error handling
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"✅ Embeddings saved to {file_path}")
        return True, None
        
    except PermissionError:
        error_msg = f"Permission denied: Cannot write to {EMBEDDINGS_DIR}"
        print(f"❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Error saving embeddings: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg

def load_embeddings(storage_name: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Loads a list of embedding-chunk dictionaries from a pickle file.
    
    This function provides comprehensive error handling for file operations
    and validates the loaded data structure. It supports both relative and
    absolute file paths.
    
    Args:
        storage_name (str): Name of the storage file (without .pkl extension)
        
    Returns:
        Tuple[Optional[List[Dict]], Optional[str]]: 
            - First element: Loaded embeddings data if successful, None if failed
            - Second element: Error message if failed, None if successful
            
    Note:
        - Files are expected to be in the embeddings/ directory
        - Data is deserialized using pickle
        - Validates data structure after loading
        
    Example:
        >>> data, error = load_embeddings("my_documents")
        >>> if error:
        ...     print(f"Error loading: {error}")
        ... else:
        ...     print(f"Loaded {len(data)} chunks")
    """
    try:
        # Validate input parameter
        if not storage_name or not storage_name.strip():
            return None, "Storage name cannot be empty"
        
        # Sanitize storage name
        safe_name = "".join(c for c in storage_name if c.isalnum() or c in ('-', '_')).rstrip()
        if not safe_name:
            return None, "Storage name contains no valid characters"
        
        # Create file path
        file_path = os.path.join(EMBEDDINGS_DIR, f"{safe_name}.pkl")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return None, f"Embedding file not found: {file_path}"
        
        # Load data from pickle file
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Validate loaded data structure
        if not isinstance(embeddings, list):
            return None, "Invalid data format: expected list of dictionaries"
        
        if embeddings and not isinstance(embeddings[0], dict):
            return None, "Invalid data format: expected list of dictionaries"
        
        print(f"✅ Embeddings loaded from {file_path}")
        return embeddings, None
        
    except PermissionError:
        error_msg = f"Permission denied: Cannot read from {file_path}"
        print(f"❌ {error_msg}")
        return None, error_msg
    except pickle.UnpicklingError as e:
        error_msg = f"Error unpickling file: {e}"
        print(f"❌ {error_msg}")
        return None, error_msg
    except Exception as e:
        error_msg = f"Error loading embeddings: {e}"
        print(f"❌ {error_msg}")
        return None, error_msg

def get_available_embedding_sets() -> List[str]:
    """
    Returns a list of available embedding set names from the embeddings directory.
    
    This function scans the embeddings directory for .pkl files and returns
    their names without the file extension. It provides error handling for
    directory access issues.
    
    Returns:
        List[str]: List of available embedding set names (without .pkl extension)
        
    Note:
        - Only returns .pkl files from the embeddings/ directory
        - File extensions are automatically removed
        - Returns empty list if directory doesn't exist or is empty
        
    Example:
        >>> available_sets = get_available_embedding_sets()
        >>> print(f"Available sets: {available_sets}")
    """
    try:
        # Check if embeddings directory exists
        if not os.path.exists(EMBEDDINGS_DIR):
            return []
        
        # List all files in the directory
        files = os.listdir(EMBEDDINGS_DIR)
        
        # Filter for .pkl files and remove extensions
        embedding_files = [f for f in files if f.endswith('.pkl')]
        return [f[:-4] for f in embedding_files]  # Remove .pkl extension
        
    except PermissionError:
        print(f"Permission denied: Cannot access {EMBEDDINGS_DIR}")
        return []
    except Exception as e:
        print(f"Error listing embedding sets: {e}")
        return []

# =============================================================================
# VECTOR STORE CLASS
# =============================================================================

class InMemoryVectorStore:
    """
    A simple in-memory vector store for efficient similarity search.
    
    This class provides an in-memory implementation of a vector store that
    supports adding embeddings and performing cosine similarity search.
    It's designed for prototyping and small to medium-sized datasets.
    
    Attributes:
        chunks_data (List[Dict]): List of chunk dictionaries with metadata
        embeddings_matrix (np.ndarray): 2D array of embeddings for similarity search
        
    Note:
        - Uses cosine similarity for semantic search
        - Stores embeddings as NumPy arrays for efficient computation
        - Supports incremental addition of new embeddings
        - Memory usage scales with number of embeddings and vector dimension
    """
    
    def __init__(self):
        """
        Initialize an empty vector store.
        
        Creates empty containers for chunks data and embeddings matrix.
        The embeddings matrix will be initialized when first vectors are added.
        """
        self.chunks_data = []  # List of dictionaries containing chunk metadata
        self.embeddings_matrix = np.array([])  # 2D NumPy array of embeddings
    
    def add_vectors(self, embeddings_data: List[Dict]):
        """
        Adds a list of chunk dictionaries with embeddings to the vector store.
        
        This method filters out chunks without embeddings, extracts the embedding
        vectors, and appends them to the existing embeddings matrix. It maintains
        the relationship between chunks_data and embeddings_matrix.
        
        Args:
            embeddings_data (List[Dict]): List of chunk dictionaries with 'embedding' field
            
        Note:
            - Only chunks with valid embeddings are added
            - Embeddings are converted to NumPy arrays for efficient computation
            - Matrix is extended vertically to accommodate new embeddings
            - Maintains data integrity between chunks and embeddings
            
        Example:
            >>> vector_store = InMemoryVectorStore()
            >>> vector_store.add_vectors(chunks_with_embeddings)
            >>> print(f"Added {len(vector_store.chunks_data)} vectors")
        """
        # Filter out chunks missing embeddings
        valid_chunks = [chunk for chunk in embeddings_data if 'embedding' in chunk]
        
        if not valid_chunks:
            print("Warning: No valid chunks with embeddings found")
            return
        
        # Extract embeddings and convert to NumPy array
        new_embeddings = np.array([chunk['embedding'] for chunk in valid_chunks])
        
        # Append to chunks_data
        self.chunks_data.extend(valid_chunks)
        
        # Stack embeddings into embeddings_matrix
        if self.embeddings_matrix.size == 0:
            # First addition: initialize the matrix
            self.embeddings_matrix = new_embeddings
        else:
            # Subsequent additions: stack vertically
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_embeddings])
        
        print(f"✅ Added {len(valid_chunks)} vectors to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Performs cosine similarity search to find the most similar chunks.
        
        This method computes cosine similarity between the query embedding and
        all stored embeddings, then returns the top-k most similar chunks.
        It handles edge cases like empty stores and invalid inputs.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of top results to return (default: 5)
            
        Returns:
            List[Dict]: List of top-k most similar chunk dictionaries
            
        Note:
            - Uses sklearn's cosine_similarity for efficient computation
            - Returns empty list if no embeddings are stored
            - k is automatically adjusted if fewer embeddings are available
            - Results are sorted by similarity score (highest first)
            
        Example:
            >>> query_vec = embedding_model.encode(["What is AI?"])[0]
            >>> results = vector_store.search(query_vec, k=3)
            >>> print(f"Found {len(results)} similar chunks")
        """
        # Check if vector store has embeddings
        if self.embeddings_matrix.size == 0:
            print("Warning: No embeddings in vector store")
            return []
        
        # Validate input parameters
        if k <= 0:
            print("Warning: k must be positive, using k=1")
            k = 1
        
        # Adjust k if fewer embeddings are available
        k = min(k, len(self.chunks_data))
        
        try:
            # Reshape query embedding if needed (ensure 2D array)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarities using sklearn
            similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
            
            # Get top k indices (highest similarity first)
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Return top k chunks
            results = [self.chunks_data[i] for i in top_indices]
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

# =============================================================================
# GEMINI INTEGRATION FUNCTION
# =============================================================================

def generate_answer_with_gemini(query: str, context_chunks: List[Dict]) -> str:
    """
    Generates a context-rich answer using Google Gemini with source citations.
    
    This function constructs a comprehensive prompt that includes the user's
    question and relevant context chunks. It uses Google Gemini to generate
    an answer that cites specific sources from the provided context.
    
    Args:
        query (str): User's question to be answered
        context_chunks (List[Dict]): Retrieved context chunks with metadata
        
    Returns:
        str: Generated answer text or error message
        
    Note:
        - Requires valid GEMINI_API_KEY in environment variables
        - Constructs detailed prompt with source citations
        - Handles API errors gracefully
        - Returns user-friendly error messages
        
    Example:
        >>> answer = generate_answer_with_gemini("What is AI?", context_chunks)
        >>> print(answer)
    """
    # Check for GEMINI_API_KEY configuration
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured. Please check your .env file."
    
    # Validate input parameters
    if not query or not query.strip():
        return "Error: Query cannot be empty"
    
    if not context_chunks:
        return "I couldn't find any relevant information in the loaded documents to answer your question."
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Format context chunks with source information
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"Source {i} (File: {chunk['source']}, Chunk ID: {chunk['chunk_id']}):\n"
            context_text += f"{chunk['content']}\n\n"
        
        # Construct comprehensive prompt with clear instructions
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Instructions:
1. Answer the question using ONLY the information provided in the context below.
2. If the context doesn't contain enough information to answer the question, say so clearly.
3. Cite your sources by mentioning the filename and chunk ID when referencing information.
4. Provide a comprehensive and accurate answer based on the available context.
5. If the question is ambiguous, ask for clarification.

User Question: {query}

Context:
{context_text}

Please provide a comprehensive answer based on the context above:"""
        
        # Generate response using Gemini API
        response = model.generate_content(prompt)
        
        # Validate and return response
        if response and response.text:
            return response.text
        else:
            return "Error: Empty response from Gemini API"
            
    except Exception as e:
        return f"Error generating answer with Gemini: {e}"

# =============================================================================
# METRICS AND EVALUATION FUNCTIONS
# =============================================================================

def calculate_retrieval_metrics(query: str, retrieved_chunks: List[Dict], query_embedding: np.ndarray) -> Dict[str, float]:
    """
    Calculate various retrieval metrics for the RAG system including accuracy measures.
    
    Args:
        query (str): The user's query
        retrieved_chunks (List[Dict]): List of retrieved chunks with embeddings
        query_embedding (np.ndarray): Embedding of the query
        
    Returns:
        Dict[str, float]: Dictionary containing various metrics including accuracy
    """
    if not retrieved_chunks or query_embedding is None:
        return {
            'avg_similarity': 0.0,
            'max_similarity': 0.0,
            'min_similarity': 0.0,
            'similarity_std': 0.0,
            'num_chunks_retrieved': 0,
            'avg_chunk_length': 0.0,
            'total_context_length': 0,
            'retrieval_accuracy': 0.0,
            'precision_at_k': 0.0,
            'semantic_coherence': 0.0
        }
    
    # Calculate similarity scores
    similarities = []
    chunk_lengths = []
    total_context_length = 0
    
    for chunk in retrieved_chunks:
        if 'embedding' in chunk:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            similarities.append(similarity)
        
        chunk_length = len(chunk.get('content', ''))
        chunk_lengths.append(chunk_length)
        total_context_length += chunk_length
    
    # Calculate accuracy metrics
    retrieval_accuracy = calculate_retrieval_accuracy(similarities)
    precision_at_k = calculate_precision_at_k(similarities, k=len(similarities))
    semantic_coherence = calculate_semantic_coherence(retrieved_chunks, query_embedding)
    
    # Calculate metrics
    metrics = {
        'avg_similarity': np.mean(similarities) if similarities else 0.0,
        'max_similarity': np.max(similarities) if similarities else 0.0,
        'min_similarity': np.min(similarities) if similarities else 0.0,
        'similarity_std': np.std(similarities) if similarities else 0.0,
        'num_chunks_retrieved': len(retrieved_chunks),
        'avg_chunk_length': np.mean(chunk_lengths) if chunk_lengths else 0.0,
        'total_context_length': total_context_length,
        'retrieval_accuracy': retrieval_accuracy,
        'precision_at_k': precision_at_k,
        'semantic_coherence': semantic_coherence
    }
    
    return metrics

def calculate_retrieval_accuracy(similarities: List[float], threshold: float = 0.5) -> float:
    """
    Calculate retrieval accuracy based on similarity threshold.
    
    Args:
        similarities (List[float]): List of similarity scores
        threshold (float): Minimum similarity threshold for relevant results
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    if not similarities:
        return 0.0
    
    relevant_count = sum(1 for sim in similarities if sim >= threshold)
    accuracy = relevant_count / len(similarities)
    return accuracy

def calculate_precision_at_k(similarities: List[float], k: int, threshold: float = 0.5) -> float:
    """
    Calculate Precision@K metric for retrieval quality.
    
    Args:
        similarities (List[float]): List of similarity scores
        k (int): Number of top results to consider
        threshold (float): Minimum similarity threshold for relevance
        
    Returns:
        float: Precision@K score (0.0 to 1.0)
    """
    if not similarities or k <= 0:
        return 0.0
    
    # Take top k similarities
    top_k_similarities = similarities[:k]
    relevant_count = sum(1 for sim in top_k_similarities if sim >= threshold)
    precision = relevant_count / len(top_k_similarities)
    return precision

def calculate_semantic_coherence(retrieved_chunks: List[Dict], query_embedding: np.ndarray) -> float:
    """
    Calculate semantic coherence between retrieved chunks and query.
    
    Args:
        retrieved_chunks (List[Dict]): List of retrieved chunks with embeddings
        query_embedding (np.ndarray): Query embedding vector
        
    Returns:
        float: Semantic coherence score (0.0 to 1.0)
    """
    if not retrieved_chunks or query_embedding is None:
        return 0.0
    
    # Calculate inter-chunk similarities
    chunk_embeddings = []
    for chunk in retrieved_chunks:
        if 'embedding' in chunk:
            chunk_embeddings.append(np.array(chunk['embedding']))
    
    if len(chunk_embeddings) < 2:
        return 0.0
    
    # Calculate average similarity between chunks
    inter_similarities = []
    for i in range(len(chunk_embeddings)):
        for j in range(i + 1, len(chunk_embeddings)):
            sim = cosine_similarity([chunk_embeddings[i]], [chunk_embeddings[j]])[0][0]
            inter_similarities.append(sim)
    
    # Calculate query-chunk coherence
    query_similarities = []
    for embedding in chunk_embeddings:
        sim = cosine_similarity([query_embedding], [embedding])[0][0]
        query_similarities.append(sim)
    
    # Combine inter-chunk and query-chunk coherence
    inter_coherence = np.mean(inter_similarities) if inter_similarities else 0.0
    query_coherence = np.mean(query_similarities) if query_similarities else 0.0
    
    # Weighted average (query coherence is more important)
    semantic_coherence = 0.7 * query_coherence + 0.3 * inter_coherence
    return semantic_coherence

def calculate_response_metrics(response: str, query: str, retrieved_chunks: List[Dict] = None) -> Dict[str, any]:
    """
    Calculate metrics for the generated response including accuracy measures.
    
    Args:
        response (str): Generated response
        query (str): Original query
        retrieved_chunks (List[Dict]): Retrieved chunks used for response generation
        
    Returns:
        Dict[str, any]: Dictionary containing response metrics including accuracy
    """
    if not response or not query:
        return {
            'response_length': 0,
            'response_word_count': 0,
            'query_length': 0,
            'query_word_count': 0,
            'response_to_query_ratio': 0.0,
            'has_sources': False,
            'estimated_reading_time': 0.0,
            'response_accuracy': 0.0,
            'context_utilization': 0.0,
            'answer_completeness': 0.0
        }
    
    # Basic text metrics
    response_length = len(response)
    response_words = len(response.split())
    query_length = len(query)
    query_words = len(query.split())
    
    # Check if response contains source citations
    has_sources = any(keyword in response.lower() for keyword in ['source', 'file:', 'chunk', 'document'])
    
    # Estimated reading time (average 200 words per minute)
    reading_time = response_words / 200.0
    
    # Calculate accuracy metrics
    response_accuracy = calculate_response_accuracy(response, query)
    context_utilization = calculate_context_utilization(response, retrieved_chunks) if retrieved_chunks else 0.0
    answer_completeness = calculate_answer_completeness(response, query)
    
    metrics = {
        'response_length': response_length,
        'response_word_count': response_words,
        'query_length': query_length,
        'query_word_count': query_words,
        'response_to_query_ratio': response_length / query_length if query_length > 0 else 0.0,
        'has_sources': has_sources,
        'estimated_reading_time': reading_time,
        'response_accuracy': response_accuracy,
        'context_utilization': context_utilization,
        'answer_completeness': answer_completeness
    }
    
    return metrics

def calculate_response_accuracy(response: str, query: str) -> float:
    """
    Calculate response accuracy based on query-response alignment.
    
    Args:
        response (str): Generated response
        query (str): Original query
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    if not response or not query:
        return 0.0
    
    # Convert to lowercase for comparison
    response_lower = response.lower()
    query_lower = query.lower()
    
    # Extract key terms from query (simple approach)
    query_words = set(word.strip('.,!?;:') for word in query_lower.split() if len(word) > 2)
    
    # Count how many query terms appear in response
    matched_terms = sum(1 for word in query_words if word in response_lower)
    
    # Calculate accuracy based on term coverage
    if len(query_words) == 0:
        return 0.0
    
    term_coverage = matched_terms / len(query_words)
    
    # Bonus for having sources and structured response
    structure_bonus = 0.0
    if any(keyword in response_lower for keyword in ['source', 'according to', 'based on']):
        structure_bonus += 0.1
    if len(response.split()) >= 20:  # Reasonable response length
        structure_bonus += 0.1
    
    # Penalty for error messages
    error_penalty = 0.0
    if any(keyword in response_lower for keyword in ['error', 'sorry', 'cannot', "don't have"]):
        error_penalty = 0.3
    
    accuracy = min(1.0, term_coverage + structure_bonus - error_penalty)
    return max(0.0, accuracy)

def calculate_context_utilization(response: str, retrieved_chunks: List[Dict]) -> float:
    """
    Calculate how well the response utilizes the retrieved context.
    
    Args:
        response (str): Generated response
        retrieved_chunks (List[Dict]): Retrieved chunks
        
    Returns:
        float: Context utilization score (0.0 to 1.0)
    """
    if not response or not retrieved_chunks:
        return 0.0
    
    response_lower = response.lower()
    total_chunks = len(retrieved_chunks)
    utilized_chunks = 0
    
    # Check how many chunks have content referenced in the response
    for chunk in retrieved_chunks:
        chunk_content = chunk.get('content', '').lower()
        if not chunk_content:
            continue
        
        # Extract key phrases from chunk (simple approach)
        chunk_words = set(word.strip('.,!?;:') for word in chunk_content.split() if len(word) > 3)
        
        # Check if any significant words from chunk appear in response
        matches = sum(1 for word in chunk_words if word in response_lower)
        
        # If enough matches, consider chunk utilized
        if matches >= min(3, len(chunk_words) * 0.1):
            utilized_chunks += 1
    
    utilization = utilized_chunks / total_chunks if total_chunks > 0 else 0.0
    return utilization

def calculate_answer_completeness(response: str, query: str) -> float:
    """
    Calculate how complete the answer is relative to the query.
    
    Args:
        response (str): Generated response
        query (str): Original query
        
    Returns:
        float: Completeness score (0.0 to 1.0)
    """
    if not response or not query:
        return 0.0
    
    response_lower = response.lower()
    query_lower = query.lower()
    
    # Identify query type and expected completeness
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
    query_type = None
    
    for word in question_words:
        if word in query_lower:
            query_type = word
            break
    
    # Base completeness on response length and structure
    word_count = len(response.split())
    
    # Scoring based on response characteristics
    completeness = 0.0
    
    # Length-based scoring
    if word_count >= 50:
        completeness += 0.4
    elif word_count >= 20:
        completeness += 0.3
    elif word_count >= 10:
        completeness += 0.2
    
    # Structure-based scoring
    if '.' in response and len(response.split('.')) >= 2:  # Multiple sentences
        completeness += 0.2
    
    # Content-based scoring
    if any(keyword in response_lower for keyword in ['because', 'due to', 'therefore', 'as a result']):
        completeness += 0.1  # Explanatory content
    
    if any(keyword in response_lower for keyword in ['first', 'second', 'next', 'finally', 'steps']):
        completeness += 0.1  # Structured content
    
    # Source citation bonus
    if any(keyword in response_lower for keyword in ['source', 'according to', 'document']):
        completeness += 0.2
    
    return min(1.0, completeness)

def calculate_overall_system_accuracy(vector_store, recent_queries: List[Dict] = None) -> Dict[str, float]:
    """
    Calculate overall system accuracy metrics.
    
    Args:
        vector_store: The vector store containing embeddings
        recent_queries (List[Dict]): Recent query metrics for trend analysis
        
    Returns:
        Dict[str, float]: Overall accuracy metrics
    """
    if not hasattr(vector_store, 'chunks_data') or not vector_store.chunks_data:
        return {
            'system_health': 0.0,
            'data_quality': 0.0,
            'embedding_quality': 0.0,
            'overall_readiness': 0.0
        }
    
    chunks = vector_store.chunks_data
    
    # Calculate data quality metrics
    total_chunks = len(chunks)
    valid_chunks = sum(1 for chunk in chunks if chunk.get('content', '').strip())
    chunks_with_embeddings = sum(1 for chunk in chunks if 'embedding' in chunk)
    
    data_quality = valid_chunks / total_chunks if total_chunks > 0 else 0.0
    embedding_quality = chunks_with_embeddings / total_chunks if total_chunks > 0 else 0.0
    
    # Calculate system health based on data distribution
    chunk_sizes = [len(chunk.get('content', '')) for chunk in chunks]
    avg_chunk_size = np.mean(chunk_sizes) if chunk_sizes else 0
    
    # Ideal chunk size is between 200-800 characters
    size_score = 1.0
    if avg_chunk_size < 100 or avg_chunk_size > 1000:
        size_score = 0.6
    elif avg_chunk_size < 200 or avg_chunk_size > 800:
        size_score = 0.8
    
    system_health = (data_quality + embedding_quality + size_score) / 3
    
    # Calculate recent performance if available
    recent_performance = 0.8  # Default assumption
    if recent_queries:
        recent_accuracies = []
        for query_metrics in recent_queries[-10:]:  # Last 10 queries
            if 'retrieval' in query_metrics and 'response' in query_metrics:
                ret_acc = query_metrics['retrieval'].get('retrieval_accuracy', 0.0)
                resp_acc = query_metrics['response'].get('response_accuracy', 0.0)
                overall_acc = (ret_acc + resp_acc) / 2
                recent_accuracies.append(overall_acc)
        
        if recent_accuracies:
            recent_performance = np.mean(recent_accuracies)
    
    # Overall readiness score
    overall_readiness = (system_health + recent_performance) / 2
    
    return {
        'system_health': system_health,
        'data_quality': data_quality,
        'embedding_quality': embedding_quality,
        'overall_readiness': overall_readiness,
        'recent_performance': recent_performance
    }

def calculate_system_performance_metrics(vector_store) -> Dict[str, any]:
    """
    Calculate overall system performance metrics.
    
    Args:
        vector_store: The vector store containing embeddings
        
    Returns:
        Dict[str, any]: Dictionary containing system metrics
    """
    if not hasattr(vector_store, 'chunks_data') or not vector_store.chunks_data:
        return {
            'total_documents': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0.0,
            'total_content_size': 0,
            'unique_sources': 0,
            'embedding_dimensions': 0
        }
    
    chunks = vector_store.chunks_data
    
    # Calculate metrics
    total_chunks = len(chunks)
    chunk_sizes = [len(chunk.get('content', '')) for chunk in chunks]
    total_content_size = sum(chunk_sizes)
    unique_sources = len(set(chunk.get('source', 'unknown') for chunk in chunks))
    
    # Get embedding dimensions
    embedding_dims = 0
    if chunks and 'embedding' in chunks[0]:
        embedding_dims = len(chunks[0]['embedding'])
    
    # Count unique documents
    unique_docs = len(set(chunk.get('doc_id', 'unknown') for chunk in chunks))
    
    metrics = {
        'total_documents': unique_docs,
        'total_chunks': total_chunks,
        'avg_chunk_size': np.mean(chunk_sizes) if chunk_sizes else 0.0,
        'total_content_size': total_content_size,
        'unique_sources': unique_sources,
        'embedding_dimensions': embedding_dims
    }
    
    return metrics

def format_metrics_for_display(metrics: Dict[str, any]) -> str:
    """
    Format metrics dictionary for display in Streamlit.
    
    Args:
        metrics (Dict[str, any]): Metrics dictionary
        
    Returns:
        str: Formatted metrics string
    """
    formatted_lines = []
    
    for key, value in metrics.items():
        # Format key (convert snake_case to Title Case)
        formatted_key = key.replace('_', ' ').title()
        
        # Format value based on type
        if isinstance(value, float):
            if 0 < value < 1:
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = f"{value:.2f}"
        elif isinstance(value, bool):
            formatted_value = "✅ Yes" if value else "❌ No"
        else:
            formatted_value = str(value)
        
        formatted_lines.append(f"**{formatted_key}:** {formatted_value}")
    
    return "\n".join(formatted_lines)