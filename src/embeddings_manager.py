"""
Module for managing embeddings generation.
Uses sentence-transformers (free, local, no API required).
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import Config

class EmbeddingsManager:
    """
    Manages the creation of text embeddings using HuggingFace models.
    """
    
    def __init__(self):
        """
        Initialize the embeddings model.
        
        Why HuggingFaceEmbeddings?
        - Completely free (no API costs)
        - Runs locally (no data sent to external servers)
        - Fast enough for most use cases
        - Good quality embeddings
        - No rate limits
        """
        print(f"🔤 Initializing embeddings model: {Config.EMBEDDING_MODEL}")
        
        # HuggingFaceEmbeddings wraps sentence-transformers models
        # The model is downloaded on first use and cached locally
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            
            # Run on CPU (set to "cuda" if you have GPU)
            model_kwargs={'device': 'cpu'},
            
            # Normalize embeddings to unit length for cosine similarity
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("✅ Embeddings model loaded")
    
    def get_embeddings(self):
        """
        Return the embeddings model for use by other components.
        
        This model can:
        - embed_documents(texts): Convert list of texts to embeddings
        - embed_query(text): Convert a single query to embedding
        
        LangChain's vectorstores automatically call these methods.
        """
        return self.embeddings
    
    def test_embedding(self, text: str = "This is a test sentence."):
        """
        Test the embeddings model with a sample text.
        Useful for debugging and understanding embeddings.
        
        Args:
            text: Sample text to embed
        """
        print(f"\n🧪 Testing embeddings with: '{text}'")
        
        # embed_query() returns a list of floats (the embedding vector)
        embedding = self.embeddings.embed_query(text)
        
        print(f"✅ Embedding dimension: {len(embedding)}")
        print(f"📊 First 5 values: {embedding[:5]}")
        print(f"📊 Embedding norm (should be ~1.0): {sum(x**2 for x in embedding)**0.5:.4f}")



# What makes embeddings work:

# Converts text → high-dimensional vectors (here: 384 dimensions)
# Similar texts get similar vectors (measured by cosine similarity)
# "cat" and "kitten" have vectors closer than "cat" and "airplane"
# This enables semantic search: find meaning, not just keywords