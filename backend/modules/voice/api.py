# backend/modules/voice/api.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from .service import voice_service
import os

router = APIRouter()

@router.post("/process")
async def process_input(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """
    Unified endpoint for processing voice or text input.
    Returns voice response with transcription and response text.
    """
    try:
        # Validate that exactly one input is provided
        if file and text:
            raise HTTPException(status_code=400, detail="Provide either file or text, not both")
        if not file and not text:
            raise HTTPException(status_code=400, detail="Provide either file or text input")
        
        # Process the input (voice or text)
        if file:
            # Validate file type
            allowed_extensions = {'.wav', '.mp3', '.ogg', '.m4a'}
            file_extension = os.path.splitext(file.filename)[1].lower()
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
                )
            audio_data = await file.read()
            result = voice_service.process_input(audio_data=audio_data, file_extension=file_extension[1:])
        else:
            result = voice_service.process_input(text=text)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")