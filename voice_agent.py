"""
Voice Agent - Main Application
===============================

This is the main entry point that brings everything together:
STT → Orchestrator (LLM + RAG + MCP) → TTS

Usage:
    python voice_agent.py --input input.wav --output response.wav
    python voice_agent.py --interactive  (for continuous conversation)

Architecture:
┌─────────────┐
│ Voice Input │
└──────┬──────┘
       │
       v
┌─────────────┐
│     STT     │  (Whisper)
└──────┬──────┘
       │
       v
┌─────────────┐
│Orchestrator │  (LLM + RAG + MCP)
│  - Intent   │
│  - RAG      │
│  - Tools    │
└──────┬──────┘
       │
       v
┌─────────────┐
│     TTS     │  (pyttsx3)
└──────┬──────┘
       │
       v
┌─────────────┐
│Voice Output │
└─────────────┘
"""

import argparse
import logging
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


# Import our modules
from src.embeddings_manager import EmbeddingsManager
from src.rag_chain import RAGChain
from src.retriever_manager import RetrieverManager
from src.vectorstore_manager import VectorStoreManager
from stt import SpeechToText
from tts import TextToSpeech
from mcp_tools import MCPTools
from orchestrator import Orchestrator
from src.llm_manager import LLMManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceAgent:
    """
    Main Voice Agent that integrates all components.
    
    This class:
    1. Initializes STT, TTS, MCP, and Orchestrator
    2. Processes voice input
    3. Generates voice output
    4. Manages the conversation flow
    """
    
    def __init__(
        self,
        model_name: str = "mistral",
        whisper_model: str = "base",
        tts_rate: int = 150,
        rag_chain=None
    ):
        """
        Initialize the Voice Agent with all components.
        
        Args:
            model_name: Ollama model name for LLM
            whisper_model: Whisper model size for STT
            tts_rate: Speech rate for TTS (words per minute)
            rag_chain: Your existing RAG chain instance
        """
        logger.info("Initializing Voice Agent...")
        
        # Initialize Speech-to-Text
        logger.info("Loading STT model...")
        self.stt = SpeechToText(model_size=whisper_model)
        
        # Initialize Text-to-Speech
        logger.info("Loading TTS engine...")
        self.tts = TextToSpeech(rate=tts_rate)
        
        # Initialize MCP Tools
        logger.info("Initializing MCP tools...")
        self.mcp_tools = MCPTools()
        
        # Initialize Orchestrator
        logger.info("Initializing orchestrator...")
        self.orchestrator = Orchestrator(
            model_name=model_name,
            rag_chain=load_rag_chain(),
            mcp_tools=self.mcp_tools
        )
        
        logger.info("Voice Agent initialized successfully!")
    
    def process_voice_input(self, audio_file: str, output_file: str = "response.wav"):
        """
        Process a single voice input and generate voice response.
        
        This is the complete pipeline:
        1. Convert audio to text (STT)
        2. Process query through orchestrator (LLM + RAG + MCP)
        3. Convert response to speech (TTS)
        
        Args:
            audio_file: Path to input audio file (wav, mp3, etc.)
            output_file: Path to save response audio
        
        Returns:
            Dictionary with:
            - transcribed_text: What the user said
            - response_text: Agent's text response
            - audio_file: Path to response audio
        """
        logger.info(f"Processing voice input: {audio_file}")
        
        # Step 1: Speech-to-Text
        logger.info("Step 1: Transcribing audio...")
        try:
            stt_result = self.stt.transcribe(audio_file)
            user_text = stt_result['text']
            logger.info(f"Transcribed: {user_text}")
        except Exception as e:
            logger.error(f"STT failed: {e}")
            return {"error": f"Speech recognition failed: {str(e)}"}
        
        if not user_text.strip():
            logger.warning("Empty transcription")
            return {"error": "No speech detected in audio"}
        
        # Step 2: Process through orchestrator
        logger.info("Step 2: Processing through orchestrator...")
        try:
            response_text = self.orchestrator.process_query(user_text)
            logger.info(f"Response generated: {response_text[:100]}...")
        except Exception as e:
            logger.error(f"Orchestrator failed: {e}")
            return {"error": f"Processing failed: {str(e)}"}
        
        # Step 3: Text-to-Speech
        logger.info("Step 3: Generating speech...")
        try:
            success = self.tts.speak(response_text, output_file=output_file)
            if not success:
                logger.error("TTS failed")
                return {"error": "Speech generation failed"}
            logger.info(f"Response audio saved: {output_file}")
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return {"error": f"Speech generation failed: {str(e)}"}
        
        # Return complete results
        return {
            "success": True,
            "transcribed_text": user_text,
            "response_text": response_text,
            "audio_file": output_file
        }
    
    def interactive_mode(self):
        """
        Run in interactive mode: continuous conversation.
        
        In this mode:
        1. User provides audio file path
        2. System processes and responds
        3. Loop continues until user types 'quit'
        
        Useful for testing and development.
        """
        print("\n" + "="*60)
        print("Voice Agent - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  - Enter path to audio file to process")
        print("  - Type 'reset' to clear conversation history")
        print("  - Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\nEnter audio file path (or command): ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'reset':
                    self.orchestrator.reset_conversation()
                    print("Conversation history cleared.")
                    continue
                
                if not user_input:
                    continue
                
                # Check if file exists
                if not os.path.exists(user_input):
                    print(f"Error: File not found: {user_input}")
                    continue
                
                # Process the audio
                print("\nProcessing...")
                result = self.process_voice_input(user_input)
                
                if "error" in result:
                    print(f"\nError: {result['error']}")
                else:
                    print(f"\nYou said: {result['transcribed_text']}")
                    print(f"\nAgent response: {result['response_text']}")
                    print(f"\nAudio saved: {result['audio_file']}")
                    
                    # Ask if user wants to play the audio
                    play = input("\nPlay response audio? (y/n): ").strip().lower()
                    if play == 'y':
                        self.tts.speak(result['response_text'])
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {str(e)}")


def load_rag_chain():
    """
    Load your existing RAG chain from rag_chain.py.
    
    This function should import and initialize your RAG pipeline.
    Adjust this based on your actual rag_chain.py implementation.
    
    Returns:
        Initialized RAG chain instance
    """
    try:
        logger.info("Initializing RAG components...")
        
        # Initialize all components
        llm = LLMManager().get_llm()
        embeddings = EmbeddingsManager().get_embeddings()
        vectorstore = VectorStoreManager(embeddings).get_vectorstore()
        
        retriever_manager = RetrieverManager(vectorstore)
        retriever = retriever_manager.create_retriever(search_type="similarity")
        
        # Create RAG chain object
        rag_chain = RAGChain(llm=llm, retriever=retriever)
        rag_chain.create_chain(chain_type="stuff", return_source_documents=True)
        
        logger.info("✓ RAG chain loaded successfully")
        return rag_chain 
    except Exception as e:
        logger.error(f"Failed to load RAG chain: {e}")
        return None


def main():
    """
    Main entry point for the Voice Agent.
    
    Supports two modes:
    1. Single file mode: Process one audio file
    2. Interactive mode: Continuous conversation
    """
    parser = argparse.ArgumentParser(
        description="Voice Agent with RAG and MCP Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single audio file
  python voice_agent.py --input input.wav --output response.wav
  
  # Interactive mode
  python voice_agent.py --interactive
  
  # Custom model settings
  python voice_agent.py --model llama2 --whisper small --interactive
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='/Users/akash/Downloads/what_happen.wav'
    )
    
    # python voice_agent.py --input /Users/akash/Downloads/what_happen.wav --output response.wav
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='response.wav',
        help='Output audio file path (default: response.wav)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='mistral',
        help='Ollama model name (default: mistral)'
    )
    
    parser.add_argument(
        '--whisper',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base)'
    )
    
    parser.add_argument(
        '--tts-rate',
        type=int,
        default=150,
        help='TTS speech rate in WPM (default: 150)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.input:
        parser.error("Either --input or --interactive is required")
    
    try:
        # Load RAG chain
        logger.info("Loading RAG chain...")
        # rag_chain = rag_chain,
        
        # Initialize Voice Agent
        agent = VoiceAgent(
            model_name=args.model,
            whisper_model=args.whisper,
            tts_rate=args.tts_rate,
            rag_chain=None
        )
        
        # Run in appropriate mode
        if args.interactive:
            agent.interactive_mode()
        else:
            # Single file mode
            logger.info(f"Processing audio file: {args.input}")
            result = agent.process_voice_input(args.input, args.output)
            
            if "error" in result:
                logger.error(f"Processing failed: {result['error']}")
                sys.exit(1)
            
            print("\n" + "="*60)
            print("Processing Complete!")
            print("="*60)
            print(f"\nTranscribed: {result['transcribed_text']}")
            print(f"\nResponse: {result['response_text']}")
            print(f"\nAudio saved: {result['audio_file']}")
            print()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()