# backend/modules/voice/service.py
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import os
from voice_agent import VoiceAgent

class VoiceService:
    def __init__(self):
        self.voice_agent = VoiceAgent()
        # Create a temp directory for audio files if it doesn't exist
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)

    def process_voice_input(self, audio_data: bytes, output_format: str = "wav") -> Dict[str, Any]:
        """
        Process voice input and get voice response.
        
        Args:
            audio_data: Binary audio data
            output_format: Format for the output audio (wav, mp3, etc.)
            
        Returns:
            Dictionary containing:
            - transcribed_text: The transcribed text from the input audio
            - response_text: The generated response text
            - audio_data: The response audio in bytes
        """
        try:
            # Save input audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", 
                                           delete=False, 
                                           dir=self.temp_dir) as temp_audio:
                temp_audio.write(audio_data)
                input_path = temp_audio.name
            
            # Process the audio file
            output_path = str(self.temp_dir / f"response_{os.path.basename(input_path)}")
            result = self.voice_agent.process_voice_input(input_path, output_path)
            
            # Read the output audio
            with open(output_path, "rb") as f:
                audio_response = f.read()
            
            # Clean up temporary files
            os.unlink(input_path)
            os.unlink(output_path)
            
            return {
                "transcribed_text": result.get("transcribed_text", ""),
                "response_text": result.get("response_text", ""),
                "audio_data": audio_response
            }
            
        except Exception as e:
            # Clean up in case of error
            if 'input_path' in locals() and os.path.exists(input_path):
                os.unlink(input_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)
            raise e

voice_service = VoiceService()