"""
The core RAG chain that connects all components together.
This is where the magic happens!
"""


from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA




from src.config import Config

class RAGChain:
    """
    Orchestrates the complete RAG pipeline:
    Query → Retrieval → Context → LLM → Answer
    """
    
    def __init__(self, llm, retriever):
        """
        Initialize the RAG chain.
        
        Args:
            llm: The language model for generation
            retriever: The retriever for finding relevant documents
        
        This chain combines:
        1. Retriever: finds relevant context
        2. Prompt: formats context and question
        3. LLM: generates answer based on context
        """
        self.llm = llm
        self.retriever = retriever
        self.chain = None
        
        # Custom prompt template for better control
        # This tells the LLM how to use the retrieved context
        self.prompt_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.

                                    If you don't know the answer based on the context provided, just say "I don't have enough information to answer that question based on the provided context." Don't try to make up an answer.

                                    Always cite which part of the context you used to answer the question.

                                    Context:
                                    {context}

                                    Question: {question}

                                    Helpful Answer:
                                """
        
        # Create PromptTemplate object
        # This structures how context and question are passed to LLM
        # PromptTemplate is a LangChain class for creating reusable prompt templates with variables. 
        # It lets you define a prompt structure once and dynamically insert values (like user questions) at runtime.
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def create_chain(self, chain_type: str = "stuff", return_source_documents: bool = True):
        """
        Create the RAG chain with specified configuration.
        
        Args:
            chain_type: How to combine documents
                - "stuff": Put all docs into one prompt (best for small contexts)
                - "map_reduce": Summarize each doc, then combine summaries
                - "refine": Iteratively refine answer with each doc
                - "map_rerank": Score each doc, use highest scoring
            
            return_source_documents: Whether to return source docs with answer
        
        Returns:
            Configured RetrievalQA chain
        
        Chain type comparison:
        - stuff: Fastest, works well when retrieved docs fit in context
        - map_reduce: Good for many documents, but slower
        - refine: Best quality, but slowest
        - map_rerank: When you want best single document
        """
        print(f"\n🔗 Creating RAG chain (type: {chain_type})")
        
        # RetrievalQA is LangChain's pre-built RAG chain
        # It handles the entire pipeline automatically:
        # 1. Takes user question
        # 2. Passes to retriever → gets relevant docs
        # 3. Formats docs + question using prompt
        # 4. Sends to LLM → gets answer
        # 5. Returns answer (and optionally source docs)
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            
            # return_source_documents=True adds retrieved docs to output
            # Useful for transparency and debugging
            return_source_documents=return_source_documents,
            
            # chain_type_kwargs passes our custom prompt
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        print("✅ RAG chain created successfully")
        return self.chain
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
        
        Returns:
            Dictionary with:
                - 'result': The generated answer
                - 'source_documents': List of retrieved documents (if enabled)
        
        This is the main method you'll use to interact with the RAG system.
        """
        if self.chain is None:
            raise ValueError("Chain not created. Call create_chain() first.")
        
        print(f"\n❓ Question: {question}")
        print("🔄 Processing...")
        
        # invoke() runs the entire RAG pipeline:
        # 1. Retrieves relevant documents
        # 2. Formats prompt with context
        # 3. Sends to LLM
        # 4. Returns structured result
        response = self.chain.invoke({"query": question})
        
        print(f"\n✅ Answer: {response['result']}\n")
        
        # Display source documents if available
        if 'source_documents' in response and response['source_documents']:
            print(f"📚 Sources used ({len(response['source_documents'])} documents):")
            for i, doc in enumerate(response['source_documents'], 1):
                print(f"\n  Source {i}:")
                print(f"  Content: {doc.page_content[:150]}...")
                print(f"  Metadata: {doc.metadata}")
        
        return response
    
    def batch_query(self, questions: list) -> list:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of question strings
        
        Returns:
            List of response dictionaries
        
        Useful for:
        - Evaluation on multiple questions
        - Batch processing of user queries
        - Testing system performance
        """
        if self.chain is None:
            raise ValueError("Chain not created. Call create_chain() first.")
        
        print(f"\n📊 Processing {len(questions)} questions in batch...")
        
        responses = []
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            response = self.query(question)
            responses.append(response)
        
        return responses
    
    def get_chain(self):
        """
        Get the chain instance.
        """
        return self.chain