# backend/modules/rag/api.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from .models import QueryRequest, QueryResponse
from .service import rag_service
import os
from src.config import Config

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        result = rag_service.query(
            question=request.question,
            top_k=request.top_k or Config.TOP_K
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # The upload manager will automatically trigger vector store rebuild
        return {"filename": file.filename, "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))