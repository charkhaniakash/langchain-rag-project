# backend/modules/rag/models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from datetime import datetime, timezone


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
