# backend/modules/rag/api.py
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import JSONResponse

from main import rebuild_vectorstore
from src import upload_manager
from src.data_loader import DataLoader
from src.embeddings_manager import EmbeddingsManager
from src.llm_manager import LLMManager
from src.rag_chain import RAGChain
from src.retriever_manager import RetrieverManager
from src.upload_manager import UploadManager
from src.vectorstore_manager import VectorStoreManager
from .models import DocumentModel, QueryRequest, QueryResponse, UploadResponse
from .service import rag_service
import os
from src.config import Config

router = APIRouter()
document_registry = {}

@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = rag_service.query(request.question, top_k=request.top_k)
    docs = [
        DocumentModel(page_content=doc.page_content, metadata=doc.metadata)
        for doc in result["source_documents"]
    ]
    return QueryResponse(answer=result["result"], documents=docs)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        upload_dir = Config.UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)

        # Generate a unique document ID
        document_id = str(uuid.uuid4())
        filename = file.filename
        file_path = os.path.join(upload_dir, f"{document_id}_{filename}")

        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Register document
        document_registry[document_id] = {
            "filename": filename,
            "file_path": file_path
        }

        # Rebuild vectorstore manually (optional)
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.get_embeddings()
        data_loader = DataLoader()
        chunks = data_loader.load_and_split()
        vectorstore_manager = VectorStoreManager(embeddings)
        vectorstore = vectorstore_manager.create_vectorstore(chunks)
        retriever_manager = RetrieverManager(vectorstore)
        retriever = retriever_manager.create_retriever()
        llm = LLMManager().get_llm()
        rag_chain = RAGChain(llm, retriever)
        rag_chain.create_chain()

        return {
            "document_id": document_id,
            "filename": filename,
            "status": "uploaded",
            "file_path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# List endpoint
@router.get("/list-documents")
async def list_documents():
    try:
        documents = [
            {"document_id": doc_id, "filename": info["filename"]}
            for doc_id, info in document_registry.items()
        ]
        return {"status": "success", "uploaded_files": documents}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Delete endpoint
@router.delete("/delete-document/{document_id}")
async def delete_document(document_id: str):
    try:
        if document_id not in document_registry:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = document_registry[document_id]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from registry
        del document_registry[document_id]

        return {"status": "success", "document_id": document_id, "message": "Deleted successfully"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)