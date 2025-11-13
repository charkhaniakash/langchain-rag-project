"""
Module for managing the vector database (ChromaDB).
Handles storing embeddings and enabling similarity search.
"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List
import os
from src.config import Config

class VectorStoreManager:
    """
    Manages the ChromaDB vector store for storing and retrieving embeddings.
    """
    
    def __init__(self, embeddings):
        """
        Initialize the vector store manager.
        
        Args:
            embeddings: The embeddings model to use for vectorization
        
        ChromaDB is chosen because:
        - Free and open-source
        - Runs locally (no cloud required)
        - Easy to set up (no server needed)
        - Persistent storage (survives restarts)
        - Good performance for small-to-medium datasets
        """
        self.embeddings = embeddings
        self.persist_directory = Config.VECTOR_STORE_PATH
        self.collection_name = Config.COLLECTION_NAME
        self.vectorstore = None
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of document chunks to embed and store
        
        Returns:
            ChromaDB vector store instance
        
        What happens here:
        1. Each document chunk is converted to embedding
        2. Embeddings are stored in ChromaDB with metadata
        3. ChromaDB creates an index for fast similarity search
        4. Data is persisted to disk for later use
        """
        print(f"\n💾 Creating vector store with {len(documents)} documents...")
        
        # Chroma.from_documents() is a convenience method that:
        # - Calls embeddings.embed_documents() on all chunks
        # - Stores vectors in ChromaDB
        # - Creates a searchable index
        # - Saves everything to disk
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        print(f"✅ Vector store created and saved to: {self.persist_directory}")
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        Load an existing vector store from disk.
        
        Returns:
            Loaded ChromaDB instance, or None if doesn't exist
        
        This allows you to:
        - Avoid re-embedding documents every time
        - Start querying immediately
        - Save time and compute resources
        """
        if os.path.exists(self.persist_directory):
            print(f"📖 Loading existing vector store from: {self.persist_directory}")
            # Chroma.from_documents() → create a new database and store documents
            # Chroma(...) → load an existing one from disk
            # Load the vector store from disk
            # It reconstructs the index and makes it ready for queries
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print("✅ Vector store loaded successfully")
            return self.vectorstore
        else:
            print(f"⚠️  No existing vector store found at: {self.persist_directory}")
            return None
    
    def get_vectorstore(self) -> Chroma:
        """
        Get the vector store instance (load if exists, otherwise return None).
        """
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform similarity search to find relevant documents.
        
        Args:
            query: The search query
            k: Number of results to return (default from config)
        
        Returns:
            List of most similar documents
        
        How it works:
        1. Query is converted to embedding
        2. Cosine similarity computed with all stored embeddings
        3. Top-k most similar documents returned
        4. Results include both content and metadata
        """
        if k is None:
            k = Config.TOP_K
        
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore() first.")
        
        print(f"\n🔍 Searching for: '{query}' (returning top {k} results)")
        
        # similarity_search() returns documents ordered by relevance
        results = self.vectorstore.similarity_search(query, k=k)
        
        print(f"✅ Found {len(results)} relevant documents")
        return results
    
    def similarity_search_with_score(self, query: str, k: int = None):
        """
        Perform similarity search and return relevance scores.
        
        Args:
            query: The search query
            k: Number of results to return
        
        Returns:
            List of (document, score) tuples
        
        Scores indicate how similar each document is to the query.
        Lower scores = more similar (distance metric)
        """
        if k is None:
            k = Config.TOP_K
        
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        # similarity_search_with_score() returns (Document, float) tuples
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        print(f"\n🔍 Search results with scores:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.4f} | Preview: {doc.page_content[:100]}...")
        
        return results