"""
Configuration module for the RAG system.
This centralizes all settings and makes them easy to modify.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This keeps sensitive data like API keys out of code
load_dotenv()

class Config:
    """
    Central configuration class for the RAG system.
    """
    
    # ===== API KEYS =====
    # Groq provides free access to fast LLMs like Llama and Mixtral
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # ===== MODEL SETTINGS =====
    # Using Groq's free Mixtral model - excellent for reasoning tasks
    # Other free options: "llama2-70b-4096", "gemma-7b-it"
    LLM_MODEL = "llama-3.1-8b-instant"
    
    # Temperature controls randomness (0 = deterministic, 1 = creative)
    LLM_TEMPERATURE = 0.1  # Low for factual responses
    
    # Maximum tokens in the LLM response
    LLM_MAX_TOKENS = 1024
    
    # ===== EMBEDDING SETTINGS =====
    # Using sentence-transformers - free, runs locally, no API needed
    # This model balances quality and speed
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Embedding dimension for the chosen model (384 for MiniLM)
    EMBEDDING_DIMENSION = 384
    
    # ===== TEXT PROCESSING =====
    # Chunk size: how many characters per document chunk
    # Smaller = more precise retrieval, but may lose context
    # Larger = more context, but less precise
    CHUNK_SIZE = 500
    
    # Chunk overlap: characters shared between consecutive chunks
    # Prevents information loss at chunk boundaries
    CHUNK_OVERLAP = 50
    
    # ===== RETRIEVAL SETTINGS =====
    # Number of relevant documents to retrieve for each query
    # More = more context but may include irrelevant info
    TOP_K = 3
    
    # ===== VECTOR STORE =====
    # Path to store ChromaDB data
    VECTOR_STORE_PATH = "./vectorstore/chroma_db"

    UPLOAD_DIR = "./data/uploaded_docs"  # NEW: For uploaded files
    
    # Collection name in ChromaDB
    COLLECTION_NAME = "langchain_rag_collection"
    
    # ===== DATA PATHS =====
    # DATA_DIR = "./data/sample_docs"
    
    @classmethod
    def validate(cls):
        """
        Validate that all required configurations are set.
        Raises ValueError if any critical config is missing.
        """
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Please add it to your .env file. "
                "Get a free key at: https://console.groq.com/"
            )
        
        
                # Create upload directory if it doesn't exist
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        print("✅ Configuration validated successfully!")