# backend/modules/rag/models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from datetime import datetime, timezone
from typing import Dict, Any

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class DocumentResponse(BaseModel):
    content: str
    metadata: dict
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    documents: List[DocumentResponse]
    timestamp: datetime = datetime.now(timezone.utc)
class UploadResponse(BaseModel):
    filename: str
    status: str
    file_path: str
    timestamp: datetime = datetime.now(timezone.utc)    


class DocumentModel(BaseModel):
    page_content: str
    metadata: Dict[str, Any]


class QueryRequest(BaseModel):
    question: str  # instead of 'query'
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    documents: List[DocumentModel]
