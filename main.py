"""
Main application with document upload support.
"""

import os
from src.config import Config
from src.upload_manager import UploadManager
from src.data_loader import DataLoader
from src.embeddings_manager import EmbeddingsManager
from src.vectorstore_manager import VectorStoreManager
from src.retriever_manager import RetrieverManager
from src.llm_manager import LLMManager
from src.rag_chain import RAGChain



# Initialize
llm = LLMManager().get_llm()
embeddings = EmbeddingsManager().get_embeddings()
vectorstore = VectorStoreManager(embeddings).get_vectorstore()
retriever_manager = RetrieverManager(vectorstore)
retriever = retriever_manager.create_retriever(search_type="similarity")

rag_chain = RAGChain(llm=llm, retriever=retriever)
rag_chain.create_chain()

# Test both queries
print("\n=== Test 1: Original query ===")
result1 = rag_chain.query("What was the narrator's greatest fear as he moved towards the school?")
print(result1['result'])

print("\n=== Test 2: LLM modified query ===")
result2 = rag_chain.query("Narrator's fear in the story approaching the school")
print(result2['result'])

# Global variables for initialized components
rag_chain = None
vectorstore_manager = None
upload_manager = None

def upload_documents_interactive():
    """
    Interactive document upload interface.
    """
    global upload_manager
    
    print("\n" + "="*60)
    print("📤 Document Upload")
    print("="*60)
    
    print("\nOptions:")
    print("1. Upload single file")
    print("2. Upload multiple files")
    print("3. List uploaded files")
    print("4. Delete a file")
    print("5. Clear all uploads")
    print("6. Back to main menu")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == '1':
        file_path = input("Enter full file path: ").strip()
        upload_manager.upload_file(file_path)
        
    elif choice == '2':
        print("Enter file paths (one per line, empty line to finish):")
        file_paths = []
        while True:
            path = input().strip()
            if not path:
                break
            file_paths.append(path)
        if file_paths:
            upload_manager.upload_multiple_files(file_paths)
        else:
            print("No files entered.")
        
    elif choice == '3':
        files = upload_manager.list_uploaded_files()
        print(f"\n📁 Uploaded files ({len(files)}):")
        for i, f in enumerate(files, 1):
            file_path = os.path.join(upload_manager.upload_dir, f)
            size = os.path.getsize(file_path)
            print(f"  {i}. {f} ({size} bytes)")
        if not files:
            print("  No files uploaded yet.")
            print(f"  Upload directory: {upload_manager.upload_dir}")
            
    elif choice == '4':
        files = upload_manager.list_uploaded_files()
        if not files:
            print("No files to delete.")
        else:
            print("\nAvailable files:")
            for i, f in enumerate(files, 1):
                print(f"{i}. {f}")
            filename = input("Enter filename to delete: ").strip()
            upload_manager.delete_file(filename)
        
    elif choice == '5':
        confirm = input("⚠️  Delete ALL files? (yes/no): ").strip().lower()
        if confirm == 'yes':
            upload_manager.clear_all_uploads()
        else:
            print("Cancelled.")
            
    elif choice == '6':
        return
    else:
        print("❌ Invalid choice!")
    
    input("\nPress Enter to continue...")

def rebuild_vectorstore():
    """
    Rebuild vector store from uploaded documents.
    """
    global vectorstore_manager, rag_chain
    
    print("\n" + "="*60)
    print("🔄 Rebuilding Vector Store")
    print("="*60)
    
    # Check if there are files to process
    files = upload_manager.list_uploaded_files()
    if not files:
        print("❌ No documents found in upload directory!")
        print(f"   Upload directory: {Config.UPLOAD_DIR}")
        print("   Please upload documents first (Option 1)")
        input("\nPress Enter to continue...")
        return False
    
    print(f"Found {len(files)} files to process...")
    
    # Initialize embeddings
    print("\n🔤 Initializing embeddings...")
    embeddings_manager = EmbeddingsManager()
    embeddings = embeddings_manager.get_embeddings()
    
    # Load documents from upload directory
    print("\n📚 Loading documents...")
    data_loader = DataLoader()  # Uses upload directory by default
    chunks = data_loader.load_and_split()
    
    if not chunks:
        print("❌ No content to index!")
        input("\nPress Enter to continue...")
        return False
    
    # Create vector store
    print("\n💾 Creating vector store...")
    vectorstore_manager = VectorStoreManager(embeddings)
    vectorstore = vectorstore_manager.create_vectorstore(chunks)
    
    # Recreate RAG chain
    print("\n🔗 Creating RAG chain...")
    retriever_manager = RetrieverManager(vectorstore)
    retriever = retriever_manager.create_retriever()
    
    llm_manager = LLMManager()
    llm = llm_manager.get_llm()
    
    rag_chain = RAGChain(llm, retriever)
    rag_chain.create_chain()
    
    print("\n" + "="*60)
    print("✅ Vector store rebuilt successfully!")
    print("="*60)
    input("\nPress Enter to continue...")
    return True

def interactive_mode():
    """
    Interactive Q&A mode.
    """
    global rag_chain
    
    if rag_chain is None:
        print("\n❌ System not initialized!")
        print("   Please rebuild vector store first (Option 2)")
        input("\nPress Enter to continue...")
        return
    
    print("\n" + "="*60)
    print("💬 Interactive Q&A Mode")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q', 'back']:
            break
        
        if not question:
            continue
        
        try:
            response = rag_chain.query(question)
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """
    Main application entry point.
    """
    global upload_manager, rag_chain, vectorstore_manager
    
    print("\n" + "="*60)
    print("🚀 RAG System with Document Upload")
    print("="*60)
    
    # Validate config
    Config.validate()
    
    # Initialize upload manager
    upload_manager = UploadManager()
    
    print(f"\n📁 Upload directory: {Config.UPLOAD_DIR}")
    print(f"📁 Vector store: {Config.VECTOR_STORE_PATH}")
    
    while True:
        print("\n" + "="*60)
        print("Main Menu")
        print("="*60)
        print("1. Upload documents")
        print("2. Rebuild vector store (after uploading)")
        print("3. Ask questions (Interactive Q&A)")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            upload_documents_interactive()
            
        elif choice == '2':
            rebuild_vectorstore()
            
        elif choice == '3':
            interactive_mode()
            
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()