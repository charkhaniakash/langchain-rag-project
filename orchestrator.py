"""
Fixed Orchestrator with proper tool selection and execution
"""

import re
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys
from google import genai

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.embeddings_manager import EmbeddingsManager
from src.llm_manager import LLMManager
from src.rag_chain import RAGChain
from src.retriever_manager import RetrieverManager
from src.vectorstore_manager import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        rag_chain=None,
        mcp_tools=None,
        max_history: int = 5
    ):
        self.model_name = model_name
        self.rag_chain = rag_chain
        self.mcp_tools = mcp_tools
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"Orchestrator initialized with model: {model_name}")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY is not set")
            raise RuntimeError("Missing GOOGLE_API_KEY")
        self.gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized")
    
    def send_to_gemini(self, prompt: str) -> str:
        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return getattr(response, "text", "") or ""
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def _compose_prompt(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)

    def _safe_json_loads(self, s: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned = s.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool parameters: {s}")
            return None

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def build_system_prompt(self) -> str:
        """Enhanced system prompt with better tool selection logic"""
        tools_description = ""
        if self.mcp_tools:
            tools_description = "\n\nAvailable External Tools:\n"
            for tool in self.mcp_tools.get_tool_definitions():
                tools_description += f"- {tool['name']}: {tool['description']}\n"
        
        system_prompt = f"""You are a helpful AI assistant with TWO types of information sources:

1. LOCAL KNOWLEDGE BASE (RAG): Contains uploaded documents about specific topics
   - Use format: [RAG_SEARCH: user's exact question]
   - Only use this when the question is about documents/content that was uploaded
   - Examples: "What does my document say about...", "Explain the content in my PDF..."

2. EXTERNAL TOOLS: For general world knowledge and real-time information{tools_description}
   - Use format: [TOOL: tool_name] {{"param": "value"}}
   - Use these for general knowledge questions NOT in your local documents
   - Examples: 
     * "Who is the founder of Jio" → [TOOL: search_wikipedia] {{"query": "Jio founder Mukesh Ambani"}}
     * "What's the weather in London" → [TOOL: get_weather] {{"location": "London"}}

DECISION MAKING:
Step 1: Analyze the question
  - Is it about uploaded documents/PDFs? → Use RAG_SEARCH
  - Is it general knowledge or real-time info? → Use appropriate TOOL
  - Is it a simple greeting? → Answer directly

Step 2: Execute tools if needed
  - You can use MULTIPLE tools in one response
  - Be specific with tool parameters

Step 3: Provide natural response
  - Synthesize information from tools
  - Be conversational and helpful

IMPORTANT RULES:
- ALWAYS use tools for questions you cannot answer from general knowledge
- For people, companies, places → use search_wikipedia
- For weather → use get_weather  
- For document content → use RAG_SEARCH
- Answer greetings directly without tools

Current date: {datetime.now().strftime('%Y-%m-%d')}"""
        
        return system_prompt
    
    def parse_tool_requests(self, llm_response: str) -> Dict[str, Any]:
        """Enhanced parsing with better pattern matching"""
        rag_queries = []
        tool_calls = []
        
        logger.info(f"Parsing LLM response: {llm_response}")
        
        # Pattern for RAG search: [RAG_SEARCH: query]
        rag_pattern = r'\[RAG_SEARCH:\s*([^\]]+)\]'
        rag_matches = re.findall(rag_pattern, llm_response, re.IGNORECASE)
        logger.info(f"Raggggg: {rag_matches}")
        rag_queries.extend([q.strip() for q in rag_matches])
        logger.info(f"Found RAG queries: {rag_queries}")
        
        # Pattern for MCP tool: [TOOL: tool_name] {parameters}
        # More flexible pattern to handle variations
        tool_pattern = re.compile(
            r'\[TOOL:\s*(\w+)\]\s*(\{[^}]*\})', 
            re.IGNORECASE | re.DOTALL
        )
        tool_matches = tool_pattern.findall(llm_response)
        logger.info(f"Found tool matches: {tool_matches}")
        
        for tool_name, params_str in tool_matches:
            params = self._safe_json_loads(params_str)
            if params is not None:
                tool_calls.append({
                    "name": tool_name,
                    "parameters": params
                })
                logger.info(f"Added tool call: {tool_name} with params: {params}")
            else:
                logger.warning(f"Failed to parse tool parameters: {params_str}")
        
        # Remove tool markers from response
        clean_response = re.sub(rag_pattern, '', llm_response, flags=re.IGNORECASE)
        clean_response = tool_pattern.sub('', clean_response)
        clean_response = clean_response.strip()
        
        return {
            "rag_queries": rag_queries,
            "tool_calls": tool_calls,
            "clean_response": clean_response
        }
    
    def execute_rag_search(self, query: str) -> str:
        if not self.rag_chain:
            logger.warning("RAG chain not available")
            return "Knowledge base is not available."
        
        try:
            logger.info(f"Executing RAG search: {query}")
            
            # Try different RAG chain methods
            result = None
            
            # Method 1: Try .query() method
            if hasattr(self.rag_chain, 'query'):
                result = self.rag_chain.query(query)
            # Method 2: Try .invoke() method (LangChain style)
            elif hasattr(self.rag_chain, 'invoke'):
                result = self.rag_chain.invoke({"question": query})
            # Method 3: Try calling directly
            elif callable(self.rag_chain):
                result = self.rag_chain(query)
            else:
                logger.error("RAG chain has no recognizable query method")
                return "Knowledge base method not found."
            
            logger.info(f"RAG result type: {type(result)}")
            logger.info(f"RAG result: {result}")
            
            # Parse different result formats
            if isinstance(result, dict):
                # Try common dictionary keys
                answer = (result.get("answer") or 
                         result.get("result") or 
                         result.get("output") or
                         result.get("response"))
                if answer:
                    return str(answer)
                else:
                    logger.warning(f"Unknown dict format: {result.keys()}")
                    return str(result)
            elif isinstance(result, str):
                return result
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"RAG search failed: {e}", exc_info=True)
            return f"Error searching knowledge base: {str(e)}"
    
    def process_query(self, user_query: str) -> str:
        """Main orchestration with enhanced logging"""
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
            logger.info("Sending query to Gemini...")
            prompt = self._compose_prompt(messages)
            initial_response = self.send_to_gemini(prompt)
            logger.info(f"LLM initial response: {initial_response}")
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm sorry, I encountered an error processing your request."
        
        # Step 2: Parse for tool requests
        parsed = self.parse_tool_requests(initial_response)
        logger.info(f"Parsed result - RAG queries: {parsed['rag_queries']}, Tool calls: {parsed['tool_calls']}")
        
        # Step 3: Execute tools if requested
        tool_results = []
        
        # Execute RAG searches
        for rag_query in parsed["rag_queries"]:
            logger.info(f"Executing RAG search: {rag_query}")
            rag_result = self.execute_rag_search(rag_query)
            tool_results.append(f"Knowledge Base Result for '{rag_query}':\n{rag_result}")
        
        # Execute MCP tools
        for tool_call in parsed["tool_calls"]:
            logger.info(f"Executing MCP tool: {tool_call['name']} with params: {tool_call['parameters']}")
            try:
                tool_result = self.mcp_tools.execute_tool(
                    tool_call['name'],
                    tool_call['parameters']
                )
                formatted_result = self.mcp_tools.format_tool_result(tool_result)
                tool_results.append(f"Tool Result ({tool_call['name']}):\n{formatted_result}")
                logger.info(f"Tool result: {formatted_result[:200]}...")
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                tool_results.append(f"Error executing {tool_call['name']}: {str(e)}")
        
        # Step 4: Generate final response
        if tool_results:
            logger.info(f"Tools executed ({len(tool_results)} results), generating final response...")
            
            tools_context = "\n\n".join(tool_results)
            
            final_messages = messages + [
                {"role": "assistant", "content": initial_response},
                {
                    "role": "user",
                    "content": f"Here are the results from the tools:\n\n{tools_context}\n\nProvide a clear, conversational answer based on this information."
                }
            ]
            
            try:
                final_prompt = self._compose_prompt(final_messages)
                final_response = self.send_to_gemini(final_prompt)
                logger.info(f"Final response: {final_response[:200]}...")
                
            except Exception as e:
                logger.error(f"Final LLM response error: {e}")
                final_response = tools_context  # Fallback to raw tool results
        else:
            # No tools needed, use the initial response
            logger.info("No tools requested, using initial response")
            final_response = parsed["clean_response"] or initial_response
        
        # Add assistant response to history
        self.add_to_history("assistant", final_response)
        
        return final_response
    
    def reset_conversation(self):
        self.conversation_history = []
        logger.info("Conversation history reset")


# Test script
if __name__ == "__main__":
    from mcp_tools import MCPTools
    
    # Initialize components
    mcp_tools = MCPTools()
    
    llm = LLMManager().get_llm()
    embeddings = EmbeddingsManager().get_embeddings()
    vectorstore = VectorStoreManager(embeddings).get_vectorstore()
    retriever_manager = RetrieverManager(vectorstore)
    retriever = retriever_manager.create_retriever(search_type="similarity")
    rag_chain = RAGChain(llm=llm, retriever=retriever).create_chain()
    
    # Initialize orchestrator
    orchestrator = Orchestrator(
        model_name="gemini-2.5-flash",
        rag_chain=rag_chain,
        mcp_tools=mcp_tools
    )
    
    # Test queries
    test_queries = [
        "Who is the founder of Jio?",  # Should use Wikipedia tool
        "What's the weather in Mumbai?",  # Should use weather tool
        "Custom Model Fine-tuning"  # Should use RAG if in documents
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"User: {query}")
        print(f"{'='*60}")
        
        response = orchestrator.process_query(query)
        
        print(f"\nAssistant: {response}")
        print()