# backend/modules/rag/api.py
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import JSONResponse

from src.data_loader import DataLoader
from src.embeddings_manager import EmbeddingsManager
from src.llm_manager import LLMManager
from src.rag_chain import RAGChain
from src.retriever_manager import RetrieverManager
from src.vectorstore_manager import VectorStoreManager
from .models import DocumentModel, QueryRequest, QueryResponse
from src.config import Config

# Import your orchestrator and MCP tools
from orchestrator import Orchestrator
from mcp_tools import MCPTools

import os

router = APIRouter()
document_registry = {}

# Initialize the orchestrator ONCE at startup
# This avoids reinitializing on every request
_orchestrator = None
_rag_chain = None

def get_orchestrator():
    """Lazy initialization of orchestrator"""
    global _orchestrator, _rag_chain
    
    if _orchestrator is None:
        # Initialize all components
        llm = LLMManager().get_llm()
        embeddings = EmbeddingsManager().get_embeddings()
        vectorstore = VectorStoreManager(embeddings).get_vectorstore()
        retriever_manager = RetrieverManager(vectorstore)
        retriever = retriever_manager.create_retriever(search_type="similarity")
        
        # Create RAG chain - IMPORTANT: Must call create_chain()!
        _rag_chain = RAGChain(llm=llm, retriever=retriever)
        _rag_chain.create_chain()  # ← THIS WAS MISSING!
        
        # Initialize MCP tools
        mcp_tools = MCPTools()
        
        # Create orchestrator with both RAG and MCP tools
        _orchestrator = Orchestrator(
            model_name="gemini-2.5-flash",
            rag_chain=_rag_chain,
            mcp_tools=mcp_tools,
            max_history=5
        )
    
    return _orchestrator

@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Main query endpoint - now uses orchestrator for intelligent routing
    between RAG and MCP tools
    """
    try:
        # Get orchestrator instance
        orchestrator = get_orchestrator()
        
        # Process query through orchestrator
        # This will automatically decide whether to use RAG, MCP tools, or both
        answer = orchestrator.process_query(request.question)
        
        # Fetch source documents directly via RAG chain for transparency
        # This ensures clients receive context documents when available
        source_docs = []
        try:
            global _rag_chain
            if _rag_chain is None:
                # Ensure RAG chain is initialized
                orchestrator = get_orchestrator()
            rag_result = _rag_chain.query(request.question)
            docs = rag_result.get("source_documents", [])
            source_docs = [
                DocumentModel(page_content=doc.page_content, metadata=doc.metadata)
                for doc in docs
            ]
        except Exception:
            # If RAG isn't applicable or fails, return empty documents
            source_docs = []
        
        return QueryResponse(
            answer=answer,
            documents=source_docs
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and rebuild the vectorstore
    After upload, the orchestrator will have access to the new documents
    """
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

        # Rebuild vectorstore with new document
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.get_embeddings()
        data_loader = DataLoader()
        chunks = data_loader.load_and_split()
        vectorstore_manager = VectorStoreManager(embeddings)
        vectorstore = vectorstore_manager.create_vectorstore(chunks)
        retriever_manager = RetrieverManager(vectorstore)
        retriever = retriever_manager.create_retriever()
        llm = LLMManager().get_llm()
        
        # Update the global RAG chain
        global _rag_chain, _orchestrator
        _rag_chain = RAGChain(llm, retriever)
        _rag_chain.create_chain()  # ← MUST call this!
        
        # Reinitialize orchestrator with new RAG chain
        mcp_tools = MCPTools()
        _orchestrator = Orchestrator(
            model_name="gemini-2.5-flash",
            rag_chain=_rag_chain,
            mcp_tools=mcp_tools,
            max_history=5
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "status": "uploaded",
            "file_path": file_path,
            "message": "Document uploaded and vectorstore rebuilt"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = [
            {"document_id": doc_id, "filename": info["filename"]}
            for doc_id, info in document_registry.items()
        ]
        return {"status": "success", "uploaded_files": documents}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@router.delete("/delete-document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and rebuild vectorstore"""
    try:
        if document_id not in document_registry:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = document_registry[document_id]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from registry
        del document_registry[document_id]

        # Rebuild vectorstore without the deleted document
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.get_embeddings()
        data_loader = DataLoader()
        chunks = data_loader.load_and_split()
        vectorstore_manager = VectorStoreManager(embeddings)
        vectorstore = vectorstore_manager.create_vectorstore(chunks)
        retriever_manager = RetrieverManager(vectorstore)
        retriever = retriever_manager.create_retriever()
        llm = LLMManager().get_llm()
        
        # Update the global RAG chain
        global _rag_chain, _orchestrator
        _rag_chain = RAGChain(llm, retriever)
        _rag_chain.create_chain()  # ← MUST call this!
        
        # Reinitialize orchestrator
        mcp_tools = MCPTools()
        _orchestrator = Orchestrator(
            model_name="gemini-2.5-flash",
            rag_chain=_rag_chain,
            mcp_tools=mcp_tools,
            max_history=5
        )

        return {
            "status": "success",
            "document_id": document_id,
            "message": "Document deleted and vectorstore rebuilt"
        }
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@router.get("/health")
async def health_check():
    """Check if orchestrator and all components are initialized"""
    try:
        orchestrator = get_orchestrator()
        return {
            "status": "healthy",
            "orchestrator": "initialized",
            "rag_chain": "available" if _rag_chain else "not available",
            "mcp_tools": "available" if orchestrator.mcp_tools else "not available"
        }
    except Exception as e:
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)},
            status_code=500
        )

@router.post("/test-rag-direct")
async def test_rag_direct(request: QueryRequest):
    """
    Test RAG chain directly without orchestrator
    Useful for debugging RAG issues
    """
    try:
        if _rag_chain is None:
            return {
                "status": "error",
                "message": "RAG chain not initialized"
            }
        
        # Test different methods
        result = {}
        
        # Try .query() method
        if hasattr(_rag_chain, 'query'):
            try:
                query_result = _rag_chain.query(request.question)
                result['query_method'] = {
                    'success': True,
                    'result': query_result,
                    'type': str(type(query_result))
                }
            except Exception as e:
                result['query_method'] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Try .invoke() method
        if hasattr(_rag_chain, 'invoke'):
            try:
                invoke_result = _rag_chain.invoke({"question": request.question})
                result['invoke_method'] = {
                    'success': True,
                    'result': invoke_result,
                    'type': str(type(invoke_result))
                }
            except Exception as e:
                result['invoke_method'] = {
                    'success': False,
                    'error': str(e)
                }
        
        # List available methods
        result['available_methods'] = [
            method for method in dir(_rag_chain) 
            if not method.startswith('_') and callable(getattr(_rag_chain, method))
        ]
        
        return {
            "status": "success",
            "results": result
        }
        
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": str(e)},
            status_code=500
        )
