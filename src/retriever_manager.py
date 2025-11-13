"""
Module for managing the retriever component.
Retriever is the interface between queries and the vector store.
"""

from langchain_core.retrievers import BaseRetriever
from src.config import Config

class RetrieverManager:
    """
    Manages the retriever that finds relevant documents for queries.
    """
    
    def __init__(self, vectorstore):
        """
        Initialize the retriever from a vector store.
        
        Args:
            vectorstore: The ChromaDB instance to retrieve from
        
        What is a retriever?
        - Abstraction over vector stores
        - Provides a consistent interface for different backends
        - Can be configured with different search strategies
        - Integrates seamlessly with LangChain chains
        """
        self.vectorstore = vectorstore
        self.retriever = None
    
    def create_retriever(self, search_type: str = "similarity", **kwargs) -> BaseRetriever:
        """
        Create a retriever with specified configuration.
        
        Args:
            search_type: Type of search to perform
                - "similarity": Standard cosine similarity (default)
                - "mmr": Maximum Marginal Relevance (diverse results)
                - "similarity_score_threshold": Filter by minimum score
            **kwargs: Additional arguments passed to the retriever
        
        Returns:
            Configured retriever instance
        
        Search type comparison:
        - similarity: Best for most cases, finds most relevant docs
        - mmr: Good when you want diverse perspectives on a topic
        - similarity_score_threshold: Ensures quality by filtering low-relevance results
        """
        print(f"\n🔗 Creating retriever with search_type='{search_type}'")
        
        # Default search kwargs from config
        search_kwargs = {
            "k": Config.TOP_K,  # Number of documents to retrieve
            **kwargs  # Allow override through function arguments
        }
        
        # as_retriever() converts vector store to LangChain retriever
        # This is necessary for integration with chains
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        print(f"✅ Retriever created (will return top {search_kwargs['k']} results)")
        return self.retriever
    
    def get_retriever(self) -> BaseRetriever:
        """
        Get the retriever instance.
        """
        if self.retriever is None:
            raise ValueError("Retriever not created. Call create_retriever() first.")
        return self.retriever
    
    def test_retrieval(self, query: str):
        """
        Test the retriever with a sample query.
        
        Args:
            query: Test query string
        
        This helps you understand:
        - What documents are being retrieved
        - Whether retrieval quality is good
        - If chunk size/overlap needs adjustment
        """
        if self.retriever is None:
            raise ValueError("Retriever not created.")
        
        print(f"\n🧪 Testing retrieval for query: '{query}'")
        
        # get_relevant_documents() is the main retriever method
        # It returns a list of Document objects
        docs = self.retriever.get_relevant_documents(query)
        
        print(f"✅ Retrieved {len(docs)} documents:\n")
        
        for i, doc in enumerate(docs, 1):
            print(f"--- Document {i} ---")
            print(f"Content (first 200 chars): {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            print()



# Retriever vs Vector Store:

# Vector Store: Low-level storage and search
# Retriever: High-level interface used by chains
# Retriever can combine multiple vector stores
# Retriever can apply additional filtering/ranking logic