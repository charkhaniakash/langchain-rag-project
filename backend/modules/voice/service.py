# backend/modules/voice/service.py
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import os
import base64
from voice_agent import VoiceAgent
import logging

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        self.voice_agent = VoiceAgent()
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)

    def process_input(
        self, 
        audio_data: Optional[bytes] = None, 
        file_extension: Optional[str] = None,
        text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process voice or text input using voice_agent.process_voice_input().
        
        Args:
            audio_data: Binary audio data (for voice input)
            file_extension: Audio file extension (wav, mp3, etc.)
            text: Text input (alternative to voice)
            
        Returns:
            Dictionary containing:
            - success: Boolean
            - transcribed_text: Input text (transcribed or provided)
            - response_text: Agent's response
            - audio_data: Response audio in base64
        """
        input_path = None
        output_path = None
        
        try:
            # Prepare output path
            output_path = str(self.temp_dir / f"response_{os.urandom(8).hex()}.wav")
            
            # Handle voice input
            if audio_data:
                logger.info(f"Processing voice input: {len(audio_data)} bytes")
                
                with tempfile.NamedTemporaryFile(
                    suffix=f".{file_extension}", 
                    delete=False, 
                    dir=self.temp_dir
                ) as temp_audio:
                    temp_audio.write(audio_data)
                    input_path = temp_audio.name
                
                logger.info(f"Audio saved to: {input_path}")
                
                # Process through voice_agent
                result = self.voice_agent.process_voice_input(
                    audio_file=input_path,
                    output_file=output_path
                )
            
            # Handle text input
            elif text:
                logger.info(f"Processing text input: {text}")
                
                # Process through voice_agent with text_input parameter
                result = self.voice_agent.process_voice_input(
                    text_input=text,
                    output_file=output_path
                )
            
            else:
                raise ValueError("Either audio_data or text must be provided")
            
            # Check if processing was successful
            if not result.get("success"):
                error_msg = result.get("error", "Processing failed")
                raise Exception(error_msg)
            
            # Verify output file exists
            if not os.path.exists(output_path):
                raise Exception(f"Output file not created: {output_path}")
            
            logger.info(f"Output file size: {os.path.getsize(output_path)} bytes")
            
            # Read the output audio
            with open(output_path, "rb") as f:
                audio_response = f.read()
            
            if len(audio_response) == 0:
                raise Exception("Empty audio file generated")
            
            # Clean up temporary files
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
            
            return {
                "success": True,
                "transcribed_text": result.get("transcribed_text", ""),
                "response_text": result.get("response_text", ""),
                "audio_data": base64.b64encode(audio_response).decode("utf-8")
            }
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            # Clean up in case of error
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
            raise e

voice_service = VoiceService()