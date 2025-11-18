# backend/modules/voice/models.py
from pydantic import BaseModel
from typing import Optional

class TextToSpeechRequest(BaseModel):
    text: str
    voice_model: Optional[str] = None
    speed: Optional[float] = None

class SpeechToTextRequest(BaseModel):
    audio_data: bytes
    language: Optional[str] = "en-US"