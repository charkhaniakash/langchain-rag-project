# LangChain RAG Project - Complete Implementation

A full-featured Retrieval-Augmented Generation (RAG) system built with LangChain, using only free tools and APIs.

## 🌟 Features

- **Document Loading**: Automatically loads and processes text documents
- **Smart Chunking**: Intelligently splits documents for optimal retrieval
- **Semantic Search**: Uses embeddings for meaning-based search (not just keywords)
- **Free LLM**: Powered by Groq's fast, free API
- **Local Vector DB**: ChromaDB runs entirely on your machine
- **Source Tracking**: See which documents were used for each answer
- **Evaluation Tools**: Measure and improve system performance
- **Interactive & Demo Modes**: Multiple ways to use the system

## 🛠️ Technology Stack

- **Framework**: LangChain 0.1.0
- **LLM**: Groq API (Mixtral-8x7b) - Free
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) - Local, free
- **Vector Store**: ChromaDB - Local, free
- **Language**: Python 3.8+

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Groq API key (free from https://console.groq.com/)

### Step-by-step Setup

1. **Clone or download this project**
```bash
   cd langchain-rag-project
```

2. **Create virtual environment**
```bash
   python -m venv venv
   
   # Activate it:
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your Groq API key
   # Get free key at: https://console.groq.com/
```

5. **Add your documents**
```bash
   # Place your .txt files in:
   data/sample_docs/
   
   # Sample documents are already included!
```

## 🚀 Usage

### Quick Start
```bash
python main.py
```

Then select your preferred mode:
- **Interactive Q&A**: Ask questions in real-time
- **Demo Mode**: See example questions and answers
- **Evaluation Mode**: Test system performance

### Example Usage
```python
from src.config import Config
from main import initialize_rag_system

# Initialize system
rag_chain, _ = initialize_rag_system()

# Ask a question
response = rag_chain.query("What is LangChain?")

# Access the answer
print(response['result'])

# See source documents
for doc in response['source_documents']:
    print(doc.page_content)
```

## 📂 Project Structure
```
langchain-rag-project/
├── data/                    # Document storage
│   └── sample_docs/         # Your text documents go here
├── vectorstore/             # Vector database (auto-created)
├── src/                     # Source code
│   ├── config.py            # Configuration settings
│   ├── data_loader.py       # Document loading & chunking
│   ├── embeddings_manager.py# Embedding generation
│   ├── vectorstore_manager.py# Vector database management
│   ├── retriever_manager.py # Retrieval logic
│   ├── llm_manager.py       # LLM integration
│   ├── rag_chain.py         # Main RAG pipeline
│   └── evaluator.py         # Performance evaluation
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Configuration

Edit `src/config.py` to customize:

- **LLM Model**: Change to different Groq models
- **Chunk Size**: Adjust document splitting (default: 500 chars)
- **Top-K Retrieval**: Number of documents to retrieve (default: 3)
- **Temperature**: LLM creativity (default: 0.1 for factual)

## 📊 Evaluation

Run evaluation mode to measure:
- Retrieval quality (speed, relevance)
- Answer quality (length, keyword matching)
- Context relevance (are retrieved docs useful?)
```bash
python main.py
# Select option 3 for evaluation
```

## 🆓 Free Resources Used

1. **Groq API**: Free fast LLM inference
   - Sign up: https://console.groq.com/
   - No credit card required

2. **Sentence Transformers**: Free local embeddings
   - No API needed
   - Runs on your machine

3. **ChromaDB**: Free local vector database
   - No cloud, no costs
   - Persistent storage

## 🐛 Troubleshooting

### "GROQ_API_KEY not found"
- Make sure you created `.env` file
- Add your API key: `GROQ_API_KEY=your_key_here`

### "No documents found"
- Add .txt files to `data/sample_docs/`
- Sample documents are included

### Slow performance
- First run downloads embedding model (one-time, ~80MB)
- Subsequent runs are much faster

### Import errors
- Make sure virtual environment is activated
- Reinstall: `pip install -r requirements.txt --upgrade`

## 📚 Learning Resources

- LangChain Docs: https://python.langchain.com/
- Groq Documentation: https://console.groq.com/docs
- ChromaDB Guide: https://docs.trychroma.com/
- RAG Explanation: See included sample documents!

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new features
- Improve evaluation metrics
- Try different models
- Share your learnings

## 📄 License

MIT License - Feel free to use for learning and projects!

## 🎓 What You'll Learn

By studying this project, you'll understand:
- How RAG systems work end-to-end
- LangChain's abstractions and components
- Vector databases and semantic search
- Prompt engineering for better answers
- Evaluation and system improvement
- Production-ready code structure

Happy learning! 🚀