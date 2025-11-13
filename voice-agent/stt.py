"""
Speech-to-Text Module using OpenAI Whisper
============================================

This module handles converting audio files to text using the Whisper model.
Whisper is a robust, open-source speech recognition model that works offline.

Key Features:
- Supports multiple audio formats (wav, mp3, m4a, etc.)
- Works completely offline (no API calls)
- High accuracy for multiple languages
- Automatic language detection

How it works:
1. Load the Whisper model (base model is good balance of speed/accuracy)
2. Load audio file from disk
3. Transcribe audio to text
4. Return transcribed text
"""

import whisper
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechToText:
    """
    Handles speech-to-text conversion using Whisper.
    
    The Whisper model sizes:
    - tiny: fastest, least accurate (~1GB RAM)
    - base: good balance (recommended, ~1GB RAM)
    - small: better accuracy (~2GB RAM)
    - medium: high accuracy (~5GB RAM)
    - large: best accuracy (~10GB RAM)
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the Whisper model.
        
        Args:
            model_size: Size of the Whisper model to use.
                       Options: "tiny", "base", "small", "medium", "large"
        
        The model is downloaded on first use and cached locally.
        """
        logger.info(f"Loading Whisper model: {model_size}")
        try:
            # Load the Whisper model
            # This downloads the model on first run (~140MB for base)
            self.model = whisper.load_model(model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file (wav, mp3, m4a, etc.)
            language: Optional language code (e.g., "en", "es", "fr")
                     If None, Whisper will auto-detect the language
        
        Returns:
            Dictionary containing:
            - "text": The transcribed text
            - "language": Detected language
            - "segments": Detailed segments with timestamps
        
        How Whisper works:
        1. Loads audio file and converts to 16kHz mono
        2. Processes audio in 30-second chunks
        3. Uses transformer model to predict text
        4. Returns full transcription with metadata
        """
        
        # Validate file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio file: {audio_path}")
        
        try:
            # Transcribe the audio
            # Options:
            # - fp16=False: Use 32-bit precision (more compatible)
            # - language: Force a specific language or auto-detect
            # - task: "transcribe" or "translate" (to English)
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False,  # Use FP32 for better compatibility
                verbose=False  # Set True for detailed output
            )
            
            logger.info(f"Transcription successful. Language: {result['language']}")
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_with_timestamps(self, audio_path: str) -> list:
        """
        Transcribe with detailed timestamps for each segment.
        
        Useful for understanding when specific words were spoken.
        
        Returns:
            List of segments with:
            - start: Start time in seconds
            - end: End time in seconds
            - text: Transcribed text for this segment
        """
        result = self.transcribe(audio_path)
        
        return [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result.get("segments", [])
        ]


# Example usage
if __name__ == "__main__":
    # Initialize STT
    stt = SpeechToText(model_size="base")
    
    # Test with an audio file
    test_audio = "input.wav"
    
    if os.path.exists(test_audio):
        # Simple transcription
        result = stt.transcribe(test_audio)
        print(f"Transcribed text: {result['text']}")
        print(f"Detected language: {result['language']}")
        
        # With timestamps
        segments = stt.transcribe_with_timestamps(test_audio)
        print("\nSegments with timestamps:")
        for seg in segments:
            print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")
    else:
        print(f"Test audio file '{test_audio}' not found")
        print("Please provide an audio file to test transcription")