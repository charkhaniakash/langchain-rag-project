"""
Module for managing the Large Language Model integration.
Uses Groq for fast, free LLM inference.
"""

from langchain_groq import ChatGroq
from src.config import Config

class LLMManager:
    """
    Manages the LLM (Large Language Model) used for generation.
    """
    
    def __init__(self):
        """
        Initialize the LLM with Groq.
        
        Why Groq?
        - Completely FREE (no credit card required)
        - FAST inference (optimized hardware)
        - Access to powerful models (Llama 2, Mixtral, Gemma)
        - Generous rate limits for free tier
        - Easy API (compatible with OpenAI format)
        
        Get API key at: https://console.groq.com/
        """
        print(f"🤖 Initializing LLM: {Config.LLM_MODEL}")
        
        # ChatGroq is LangChain's wrapper for Groq's API
        # It provides a consistent interface with other LLM providers
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL,
            
            # Temperature: controls randomness
            # 0.0 = deterministic, always same answer
            # 1.0 = creative, varied answers
            # For RAG, we want factual answers, so low temperature
            temperature=Config.LLM_TEMPERATURE,
            
            # Max tokens in response
            # Prevents overly long answers
            max_tokens=Config.LLM_MAX_TOKENS,
            
            # Streaming allows token-by-token response (like ChatGPT typing)
            # Set to False for complete response at once
            streaming=False
        )
        
        print("✅ LLM initialized")
    
    def get_llm(self):
        """
        Return the LLM instance for use in chains.
        """
        return self.llm
    
    def test_llm(self, prompt: str = "Explain RAG in one sentence."):
        """
        Test the LLM with a simple prompt.
        
        Args:
            prompt: Test prompt to send to the LLM
        
        This verifies:
        - API key is valid
        - Model is accessible
        - Responses are reasonable
        """
        print(f"\n🧪 Testing LLM with prompt: '{prompt}'")
        
        try:
            # invoke() sends prompt to LLM and waits for response
            # Returns a message object with .content attribute
            response = self.llm.invoke(prompt)
            
            print(f"✅ LLM Response: {response.content}")
            return response.content
            
        except Exception as e:
            print(f"❌ LLM test failed: {e}")
            raise