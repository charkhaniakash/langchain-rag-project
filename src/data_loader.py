"""
Module for loading and processing documents.
Handles reading files and splitting them into chunks.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List
from langchain_core.documents import Document
from src.config import Config
from pathlib import Path

class DataLoader:
    """
    Handles loading documents from files and splitting them into chunks.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the DataLoader with configuration settings.
             Args:
            data_dir: Directory to load documents from (defaults to uploaded docs)
        """
        # self.data_dir = Config.DATA_DIR
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        
        

        self.data_dir = data_dir or Config.UPLOAD_DIR
        # RecursiveCharacterTextSplitter is smart about splitting
        # It tries to split on natural boundaries (paragraphs, sentences)
        # rather than arbitrary character counts
        # RecursiveCharacterTextSplitter is a text splitting utility from LangChain,
        # mainly used to break large text documents into smaller, structured chunks for LLMs (like GPT) to process efficiently.
        # It recursively tries to split the text using a list of separators — from bigger to smaller units — until each chunk fits 
        # within your desired size limit.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,  # Use character count as length metric
            separators=["\n\n", "\n", ". ", " ", ""]  # Split hierarchy
        )
    
    def load_documents(self) -> List[Document]:
        """
        Load documents from the data directory.
        
        Returns:
            List of LangChain Document objects containing text and metadata
        """
        print(f"📂 Loading documents from: {self.data_dir}")
        documents = []
        
        # Load text files
        try:
            text_loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            text_docs = text_loader.load()
            documents.extend(text_docs)
            print(f"✅ Loaded {len(text_docs)} text documents")
        except Exception as e:
            print(f"⚠️  Error loading text files: {e}")
        
        # Load PDF files one by one with error handling
        pdf_files = list(Path(self.data_dir).glob("**/*.pdf"))
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                documents.extend(pdf_docs)
                print(f"✅ Processed PDF: {pdf_file.name} ({len(pdf_docs)} pages)")
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {str(e)}")
        
        print(f"\n📊 Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects to split
        
        Returns:
            List of Document chunks
        
        Why splitting is important:
        - Embeddings work better on focused, coherent chunks
        - Allows retrieving specific relevant sections, not entire docs
        - Prevents exceeding LLM context window limits
        - Improves retrieval precision
        """
        print(f"✂️  Splitting documents (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        
        # split_documents() intelligently splits while preserving metadata
        # Each chunk inherits the parent document's metadata
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Display sample chunk for verification
        if chunks:
            print(f"\n📄 Sample chunk (first 200 chars):")
            print(f"{chunks[0].page_content[:200]}...")
        
        return chunks
    
    def load_and_split(self) -> List[Document]:
        """
        Convenience method: load documents and split them in one call.
        
        Returns:
            List of document chunks ready for embedding
        """
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        return chunks