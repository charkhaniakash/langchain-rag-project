"""
Text-to-Speech Module using pyttsx3
====================================

This module converts text responses into speech audio files.
pyttsx3 is a lightweight, offline TTS library that works cross-platform.

Key Features:
- Completely offline (no API calls)
- Cross-platform (Windows, Mac, Linux)
- Adjustable voice, rate, and volume
- No dependencies on external services

How it works:
1. Initialize the pyttsx3 engine
2. Configure voice properties (rate, volume, voice)
3. Convert text to speech and save as audio file
"""

import pyttsx3
import os
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Handles text-to-speech conversion using pyttsx3.
    
    pyttsx3 uses platform-specific TTS engines:
    - Windows: SAPI5
    - Mac: NSSpeechSynthesizer
    - Linux: espeak
    """
    
    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize the TTS engine.
        
        Args:
            rate: Speaking rate (words per minute)
                 Default is 150. Range: 50-300
                 - Lower = slower speech
                 - Higher = faster speech
            volume: Volume level (0.0 to 1.0)
                   Default is 1.0 (maximum)
        
        The engine is initialized once and reused for efficiency.
        """
        logger.info("Initializing TTS engine")
        try:
            # Initialize pyttsx3 engine
            # This creates a connection to the platform's TTS system
            self.engine = pyttsx3.init()
            
            # Set speech rate (words per minute)
            self.engine.setProperty('rate', rate)
            
            # Set volume (0.0 to 1.0)
            self.engine.setProperty('volume', volume)
            
            # Get available voices
            self.voices = self.engine.getProperty('voices')
            
            logger.info(f"TTS engine initialized. Available voices: {len(self.voices)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            raise
    
    def list_voices(self) -> list:
        """
        List all available voices on the system.
        
        Returns:
            List of voice objects with properties:
            - id: Unique voice identifier
            - name: Human-readable voice name
            - languages: Supported languages
            - gender: Voice gender (if available)
        """
        voices_info = []
        for i, voice in enumerate(self.voices):
            voices_info.append({
                "index": i,
                "id": voice.id,
                "name": voice.name,
                "languages": voice.languages,
            })
        return voices_info
    
    def set_voice(self, voice_index: int = 0):
        """
        Set the voice to use for speech.
        
        Args:
            voice_index: Index of the voice from list_voices()
                        Default is 0 (first available voice)
        
        Most systems have multiple voices (male/female, different accents).
        Use list_voices() to see available options.
        """
        if 0 <= voice_index < len(self.voices):
            self.engine.setProperty('voice', self.voices[voice_index].id)
            logger.info(f"Voice set to: {self.voices[voice_index].name}")
        else:
            logger.warning(f"Invalid voice index: {voice_index}")
    
    def speak(self, text: str, output_file: Optional[str] = None) -> bool:
        """
        Convert text to speech.
        
        Args:
            text: The text to convert to speech
            output_file: Optional path to save audio file
                        If None, speaks immediately without saving
                        If provided, saves to file (e.g., "output.wav")
        
        Returns:
            True if successful, False otherwise
        
        How it works:
        1. If output_file is provided:
           - Uses save_to_file() to queue the text
           - Calls runAndWait() to generate audio file
        2. If no output_file:
           - Uses say() to queue the text
           - Calls runAndWait() to speak immediately
        """
        if not text:
            logger.warning("Empty text provided for TTS")
            return False
        
        try:
            if output_file:
                # Save to audio file
                logger.info(f"Generating speech audio: {output_file}")
                
                # Queue the text to be saved to file
                self.engine.save_to_file(text, output_file)
                
                # Process the queue and generate the file
                self.engine.runAndWait()
                
                # Verify file was created
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    logger.info(f"Audio file created: {output_file} ({file_size} bytes)")
                    return True
                else:
                    logger.error("Audio file was not created")
                    return False
            else:
                # Speak immediately (no file)
                logger.info("Speaking text immediately")
                self.engine.say(text)
                self.engine.runAndWait()
                return True
                
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False
    
    def speak_async(self, text: str):
        """
        Speak text asynchronously (non-blocking).
        
        Useful for real-time applications where you don't want to wait
        for speech to complete before continuing.
        """
        try:
            self.engine.say(text)
            # Start speaking in background
            self.engine.startLoop(False)
            self.engine.iterate()
            self.engine.endLoop()
        except Exception as e:
            logger.error(f"Async TTS failed: {e}")
    
    def adjust_rate(self, rate: int):
        """
        Adjust speaking rate dynamically.
        
        Args:
            rate: Words per minute (50-300)
        """
        self.engine.setProperty('rate', rate)
        logger.info(f"Speech rate adjusted to: {rate} WPM")
    
    def adjust_volume(self, volume: float):
        """
        Adjust volume dynamically.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
        self.engine.setProperty('volume', volume)
        logger.info(f"Volume adjusted to: {volume}")


# Example usage
if __name__ == "__main__":
    # Initialize TTS with custom settings
    tts = TextToSpeech(rate=150, volume=0.9)
    
    # List available voices
    print("Available voices:")
    for voice in tts.list_voices():
        print(f"  [{voice['index']}] {voice['name']}")
    
    # Test text
    test_text = "I want to create one bot please can you help me"
    
    # Option 1: Speak immediately (no file saved)
    print("\nSpeaking text...")
    tts.speak(test_text)
    
    # Option 2: Save to audio file
    print("\nGenerating audio file...")
    output_file = "response.wav"
    success = tts.speak(test_text, output_file=output_file)
    
    if success:
        print(f"Audio saved to: {output_file}")
    
    # Option 3: Change voice and speak again
    print("\nChanging voice...")
    tts.set_voice(1 if len(tts.voices) > 1 else 0)
    tts.speak("Now speaking with a different voice!", "response_alt.wav")