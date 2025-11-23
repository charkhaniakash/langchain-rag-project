import pyttsx3
import subprocess
import os
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSpeech:
    """
    Updated TTS class that reinitializes engine for each speak()
    to avoid pyttsx3 engine freeze issues.
    """

    def __init__(self, rate: int = 150, volume: float = 1.0, play_audio: bool = True):
        self.rate = rate
        self.volume = volume
        self.play_audio = play_audio

    def _create_engine(self):
        """Create a fresh engine every time."""
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        return engine

    def speak(self, text: str, output_file: Optional[str] = None) -> bool:
        if not text:
            return False
        
        try:
            engine = self._create_engine()

            if output_file:
                engine.save_to_file(text, output_file)
                engine.runAndWait()

                if os.path.exists(output_file) and self.play_audio:
                    subprocess.run(["afplay", output_file])
                
                return True

            # Speak directly
            engine.say(text)
            engine.runAndWait()
            return True

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False
