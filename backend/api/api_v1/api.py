# backend/api/api_v1/api.py
from fastapi import APIRouter
from backend.modules.rag.api import router as rag_router
from backend.modules.voice.api import router as voice_router

api_router = APIRouter()

api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(voice_router, prefix="/voice", tags=["voice"])

@api_router.get("/health")
async def health_check():
    return {"status": "healthy"}