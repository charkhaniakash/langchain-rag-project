"""
Orchestrator Module
===================

This is the brain of the voice agent. It coordinates:
1. LLM (Ollama) for understanding and generating responses
2. RAG pipeline for knowledge retrieval
3. MCP tools for actions (weather, Wikipedia, etc.)
4. Conversation memory for context

Flow:
User Query → LLM analyzes intent → 
    If knowledge needed → Call RAG
    If action needed → Call MCP tools
    → LLM generates final response

The orchestrator uses a multi-step reasoning approach:
1. Intent detection: What does the user want?
2. Tool selection: Which tools/RAG are needed?
3. Tool execution: Call the appropriate tools
4. Response generation: Combine results into coherent answer
"""

import re
import ollama
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys

from torch import embedding


sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.embeddings_manager import EmbeddingsManager
from src.llm_manager import LLMManager
from src.rag_chain import RAGChain
from src.retriever_manager import RetrieverManager
from src.vectorstore_manager import VectorStoreManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates the interaction between LLM, RAG, and MCP tools.
    
    This is the central coordinator that:
    - Maintains conversation history
    - Decides when to use RAG vs MCP tools
    - Manages the flow of information
    - Generates final responses
    """
    
    def __init__(
        self,
        model_name: str = "mistral",
        rag_chain=None,
        mcp_tools=None,
        max_history: int = 5
    ):
        """
        Initialize the orchestrator.
        
        Args:
            model_name: Name of the Ollama model to use
                       Popular options: "mistral", "llama2", "codellama"
            rag_chain: Your existing RAG chain instance from rag_chain.py
            mcp_tools: MCPTools instance for tool execution
            max_history: Maximum conversation turns to remember (default: 5)
        
        The orchestrator needs:
        1. An LLM (via Ollama) for reasoning
        2. A RAG pipeline for knowledge retrieval
        3. MCP tools for actions
        """
        self.model_name = model_name
        self.rag_chain = rag_chain
        self.mcp_tools = mcp_tools
        self.max_history = max_history
        
        # Conversation memory: stores recent exchanges
        # Each entry: {"role": "user" | "assistant", "content": "text"}
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"Orchestrator initialized with model: {model_name}")
        
        # Verify Ollama is available
        try:
            ollama.list()
            logger.info("Ollama connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def add_to_history(self, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            role: "user" or "assistant"
            content: The message content
        
        Maintains a sliding window of recent conversations.
        Old messages are removed when max_history is exceeded.
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Keep only recent history (sliding window)
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def build_system_prompt(self) -> str:
        """
        Build the system prompt that guides the LLM's behavior.
        
        The system prompt:
        1. Defines the LLM's role and capabilities
        2. Explains available tools (RAG + MCP)
        3. Provides instructions for tool usage
        4. Sets the response format
        
        Returns:
            Complete system prompt string
        """
        tools_description = ""
        if self.mcp_tools:
            tools_description = "\n\nAvailable MCP Tools:\n"
            for tool in self.mcp_tools.get_tool_definitions():
                tools_description += f"- {tool['name']}: {tool['description']}\n"
        
        system_prompt = f"""You are a helpful voice assistant with access to:

            1. KNOWLEDGE BASE (RAG): You can search a document knowledge base for information.
            - Use this when the user asks about specific documents or stored information.
            - When using RAG_SEARCH, preserve the user's original question as much as possible
            - Format: [RAG_SEARCH: user's original question]
            - Example: User asks "What was the narrator's fear?" → [RAG_SEARCH: What was the narrator's fear?]

            2. MCP TOOLS: You have access to external tools:{tools_description}
            - Use tools by specifying: [TOOL: tool_name] with parameters in JSON
            - Example: [TOOL: get_weather] {{"location": "London"}}

            IMPORTANT INSTRUCTIONS:
            - First, analyze what the user needs
            - If you need information from the knowledge base, use [RAG_SEARCH: original user question]
            - DO NOT rephrase or shorten the user's question when using RAG_SEARCH
            - If you need to perform an action (weather, Wikipedia), use [TOOL: tool_name]
            - You can use multiple tools in one response if needed
            - Always provide a complete, conversational response to the user
            - Be concise but informative

            Response Format:
            1. Think about what's needed (internally)
            2. If needed, request tools: [RAG_SEARCH: ...] or [TOOL: ...]
            3. Provide your final answer in natural language

            Current date: {datetime.now().strftime('%Y-%m-%d')}
            """
        return system_prompt
    
    def parse_tool_requests(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM's response to detect tool requests.
        
        Looks for patterns:
        - [RAG_SEARCH: query text] → Search knowledge base
        - [TOOL: tool_name] {"param": "value"} → Execute MCP tool
        
        Args:
            llm_response: Raw response from the LLM
        
        Returns:
            Dictionary containing:
            - rag_queries: List of RAG search queries
            - tool_calls: List of MCP tool calls
            - clean_response: LLM response with tool markers removed
        """
        rag_queries = []
        tool_calls = []
        
        # Pattern for RAG search: [RAG_SEARCH: query]
        rag_pattern = r'\[RAG_SEARCH:\s*([^\]]+)\]'
        rag_matches = re.findall(rag_pattern, llm_response)
        rag_queries.extend([q.strip() for q in rag_matches])
        
        # Pattern for MCP tool: [TOOL: tool_name] {parameters}
        tool_pattern = r'\[TOOL:\s*(\w+)\]\s*(\{[^}]+\})'
        tool_matches = re.findall(tool_pattern, llm_response)
        
        for tool_name, params_str in tool_matches:
            try:
                params = json.loads(params_str)
                tool_calls.append({
                    "name": tool_name,
                    "parameters": params
                })
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool parameters: {params_str}")
        
        # Remove tool markers from response
        clean_response = re.sub(rag_pattern, '', llm_response)
        clean_response = re.sub(tool_pattern, '', clean_response)
        clean_response = clean_response.strip()
        
        return {
            "rag_queries": rag_queries,
            "tool_calls": tool_calls,
            "clean_response": clean_response
        }
    
    def execute_rag_search(self, query: str) -> str:
        """
        Execute a RAG search using your existing pipeline.
        
        Args:
            query: Search query for the knowledge base
        
        Returns:
            Retrieved information as a string
        
        This integrates with your existing rag_chain.py
        """
        if not self.rag_chain:
            logger.warning("RAG chain not available")
            return "Knowledge base is not available."
        
        try:
            logger.info(f"Executing RAG search: {query}")
            
            # Call your existing RAG chain
            # Adjust this based on your actual rag_chain.py interface
            # Common patterns:
            # result = self.rag_chain.run(query)
            # result = self.rag_chain.invoke({"query": query})
            # result = self.rag_chain(query)
            
            # result = self.rag_chain.invoke(query)

            result = self.rag_chain.query(query)
            
            # Extract the relevant information
            # Adjust based on your RAG chain's return format
            if isinstance(result, dict):
                return result.get("answer", str(result))
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return f"Error searching knowledge base: {str(e)}"
    
    def process_query(self, user_query: str) -> str:
        """
        Main orchestration method: processes a user query end-to-end.
        
        Steps:
        1. Add query to conversation history
        2. Send to LLM for initial analysis
        3. Parse response for tool requests
        4. Execute requested tools (RAG/MCP)
        5. Send results back to LLM
        6. Generate final response
        7. Add to history and return
        
        Args:
            user_query: The user's input text
        
        Returns:
            Final response text to be converted to speech
        """
        logger.info(f"Processing query: {user_query}")
        
        # Add user query to history
        self.add_to_history("user", user_query)
        
        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Step 1: Get initial LLM response
        try:
            logger.info("Sending query to LLM...")
            llm_response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            
            initial_response = llm_response['message']['content']
            logger.info(f"LLM initial response: {initial_response[:200]}...")
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm sorry, I encountered an error processing your request."
        
        # Step 2: Parse for tool requests
        parsed = self.parse_tool_requests(initial_response)
        
        # Step 3: Execute tools if requested
        tool_results = []
        
        # Execute RAG searches
        for rag_query in parsed["rag_queries"]:
            logger.info(f"Executing RAG search: {rag_query}")
            # rag_result = self.execute_rag_search(rag_query)
            rag_result = self.execute_rag_search(user_query)
            tool_results.append(f"Knowledge Base Result for '{rag_query}':\n{rag_result}")
        
        # Execute MCP tools
        for tool_call in parsed["tool_calls"]:
            logger.info(f"Executing MCP tool: {tool_call['name']}")
            tool_result = self.mcp_tools.execute_tool(
                tool_call['name'],
                tool_call['parameters']
            )
            formatted_result = self.mcp_tools.format_tool_result(tool_result)
            tool_results.append(f"Tool Result ({tool_call['name']}):\n{formatted_result}")
        
        # Step 4: If tools were used, send results back to LLM for final response
        if tool_results:
            logger.info("Tools executed, generating final response...")
            
            # Add tool results to context
            tools_context = "\n\n".join(tool_results)
            
            final_messages = messages + [
                {"role": "assistant", "content": initial_response},
                {
                    "role": "user",
                    "content": f"Here are the results from the tools you requested:\n\n{tools_context}\n\nPlease provide a complete, conversational response to the user based on this information."
                }
            ]
            
            try:
                final_llm_response = ollama.chat(
                    model=self.model_name,
                    messages=final_messages
                )
                
                final_response = final_llm_response['message']['content']
                
            except Exception as e:
                logger.error(f"Final LLM response error: {e}")
                final_response = parsed["clean_response"] or "I found some information but had trouble forming a response."
        else:
            # No tools needed, use the initial response
            final_response = parsed["clean_response"] or initial_response
        
        # Add assistant response to history
        self.add_to_history("assistant", final_response)
        
        logger.info(f"Final response generated: {final_response[:200]}...")
        return final_response
    
    def reset_conversation(self):
        """Clear conversation history. Useful for starting a new conversation."""
        self.conversation_history = []
        logger.info("Conversation history reset")


# Example usage
if __name__ == "__main__":
    from mcp_tools import MCPTools
    
    # Initialize components
    mcp_tools = MCPTools()
    
    # rag_chain = load_rag_chain()

    llm = LLMManager().get_llm()
    embeddings = EmbeddingsManager().get_embeddings()
    vectorstore = VectorStoreManager(embeddings).get_vectorstore()
    retriever_manager = RetrieverManager(vectorstore)
    retriever = retriever_manager.create_retriever(search_type="similarity")
    rag_chain = RAGChain(llm=llm, retriever=retriever).create_chain()
    
    # Initialize orchestrator (without RAG for this example)
    orchestrator = Orchestrator(
        model_name="mistral",
        rag_chain=rag_chain,  # Replace with your actual RAG chain
        mcp_tools=mcp_tools
    )
    
    # Test queries
    test_queries = [
        "Custom Model Fine-tuning / Instruction Tuning"
        # "Tell me about artificial intelligence from Wikipedia",
        # "What's the weather in London and tell me about the city from Wikipedia"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"User: {query}")
        print(f"{'='*60}")
        
        response = orchestrator.process_query(query)
        
        print(f"\nAssistant: {response}")
        print()