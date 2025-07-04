from termcolor import colored
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_util_base import UtilBase
from typing import Any, Dict, List, Optional, Tuple, Coroutine
import logging
import traceback
import requests
import asyncio
import concurrent.futures
from pydantic import ValidationError
from py_classes.globals import g
from py_classes.utils.cls_utils_web import WebTools

def _run_async_safely(coro: Coroutine) -> Any:
    """
    Helper function to run async code from sync context safely.
    Handles both cases: when called from sync context and from async context.
    """
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
        # We're in an async context - this is the tricky case
        # Instead of using thread pools (which cause signal issues), 
        # let's try to use nest_asyncio to allow nested event loops
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
            return asyncio.run(coro)
        except ImportError:
            # nest_asyncio not available, fall back to a blocking wait approach
            # Create a task and busy-wait for it (not ideal but avoids threading issues)
            task = loop.create_task(coro)
            import time
            while not task.done():
                time.sleep(0.001)  # Small sleep to prevent busy waiting
            return task.result()
    except RuntimeError:
        # No event loop running, we can use asyncio.run() safely
        return asyncio.run(coro)

class SearchWeb(UtilBase):
    """
    A utility for performing web searches.
    
    This utility allows searching the web and summarizing the results.
    """
    
    # Initialize WebTools here to ensure it's available in sandbox
    web_tools = None
    
    @classmethod
    def initialize(cls):
        """Initialize the class resources if not already initialized"""
        if cls.web_tools is None:
            try:
                cls.web_tools = WebTools()
            except Exception as e:
                print(f"SearchWeb: Error initializing WebTools: {e}")
                # Create a minimal fallback implementation
                class MinimalWebTools:
                    def search_brave(self, query, num_results=3):
                        return [(f"Error: Could not initialize WebTools properly. Error: {e}", "https://error")]
                cls.web_tools = MinimalWebTools()
    
    @staticmethod
    def run(queries: List[str], depth: int = 0) -> str:
        """
        Perform a web search and receive a summary of the results.
        
        Args:
            queries: List of multiple search queries (IMPORTANT: include additional information for improved accuracy)
            
        Returns:
            A summary of the search results
        """
        MAX_RECURSION_DEPTH = 1

        # Initialize WebTools if not already done
        SearchWeb.initialize()
        
        # Handle both single string and list of strings
        if isinstance(queries, str):
            queries = [queries]
        
        # Main search function that can be called recursively if needed
        all_results: List[Tuple[str, str]] = []
        failed_queries: List[Tuple[str, str]] = []
        
        num_results = 3
        if (len(queries) == 1):
            num_results = 5

        for query in queries:
            try:
                print(colored(f"BRAVE: Searching {num_results} websites for: {query}", "green"))
                query_results = SearchWeb.web_tools.search_brave(query, num_results=num_results)
                if query_results:
                    all_results.extend(query_results)
                else:
                    failed_queries.append((query, "No results found"))
            except Exception as e:
                failed_queries.append((query, f"Unexpected error: {str(e)}"))
        
        # Format results
        inst = f"""You are a helpful AI assistant tasked with summarizing web search results.
        
Task: Analyze the following web search results and provide a clear, informative summary that:
1. Synthesizes the key information from all sources
2. Preserves important details and context
3. Resolves and highlights any contradictions between sources
4. Includes relevant dates or timestamps when available
5. Cites specific sources for key claims

Format your response as a coherent paragraph that flows naturally.
If sources conflict or information seems outdated, note this in your summary.


"""

        summarization_prompt = f"# Search Queries\n"
        query_list = '\n'.join([f'{i+1}. {query}' for i, query in enumerate(queries)])
        summarization_prompt += f"{query_list}\n# Search Results\n"
        for result, url in all_results:
            summarization_prompt += f"URL: {url}\n"
            summarization_prompt += f"Content: {result}\n"
            summarization_prompt += "-----------------------------------\n"

        # Add final instruction
        summarization_prompt += f"Search Queries: {', '.join(queries)}\nPlease provide a dense summary of the results, include key facts and relevant details extending beyond the required. Focus on relating the results to answer the original Search Queries: {', '.join(queries)}. Ensure trustworthy sources and interpolate details when necessary."

        # Add note about failed queries if some succeeded but others failed
        if failed_queries and all_results:
            failed_query_list = ", ".join([f"'{q}'" for q, _ in failed_queries])
            summarization_prompt += f"\n\nNote: Unable to retrieve results for the following queries: {failed_query_list}. The summary is based only on the available results."

        chat = Chat(inst, "SearchWebUtil")
        chat.add_message(Role.USER, summarization_prompt)
        
        try:
            summary = _run_async_safely(LlmRouter.generate_completion(
                chat,
                strengths=[AIStrengths.GENERAL],
                exclude_reasoning_tokens=True,
                hidden_reason="Condensing search results"
            ))
        except Exception as e:
            print(f"❌ SearchWeb: Failed to generate summary: {str(e)}")
            # Fallback to simple concatenation of results
            summary = "Search results summary failed to generate. Raw results:\n\n"
            for i, (result, url) in enumerate(all_results, 1):
                summary += f"{i}. {url}\n{result[:200]}...\n\n"
        
        # Check if the summary is relevant to the queries
        is_relevant_instruction = f"""You are a helpful assistant that checks if a summary is relevant to a list of search queries. 
Always return a tool_code block as shown in the following Examples:

Example 1:
```tool_code
return_summary_with_success()
```

Or if not relevant, suggest better queries:
```tool_code
rerun_web_search(new_queries=['new query 1', 'new query 2'])
```

Analyse the summary carefully. Only suggest rerunning the search if the current summary is not helpful for answering the queries.
"""
        is_relevant_prompt = f"""Is the below summary relevant to the original Search Queries: {', '.join(queries)}?

Summary:
{summary}

If the summary addresses any of the queries or provides information that helps answer them, use return_summary_with_success().
If the summary is completely unrelated or doesn't provide any useful information regarding the queries, suggest better queries with rerun_web_search().
Your suggested queries should be more specific or use alternative terminology that might yield better results.
"""
        is_relevant_chat = Chat(is_relevant_instruction)
        is_relevant_chat.add_message(
            role=Role.USER,
            content=is_relevant_prompt
        )
        try:
            is_relevant_tool_response = _run_async_safely(LlmRouter.generate_completion(
                is_relevant_chat,
                strengths=[AIStrengths.REASONING, AIStrengths.SMALL],
                exclude_reasoning_tokens=True,
                hidden_reason="SearchWebUtil: verifying relevance"
            ))
        except Exception as e:
            print(f"❌ SearchWeb: Failed to verify relevance: {str(e)}")
            # Default to returning the summary as-is
            is_relevant_tool_response = "return_summary_with_success()"
        
        def extract_relevance_action(response: str) -> Dict[str, Any]:
            """Extract action from relevance response"""
            if "return_summary_with_success()" in response:
                return {"action": "return_summary"}
            elif "rerun_web_search" in response:
                # Simple parsing for this example - in production would be more robust
                import re
                match = re.search(r"rerun_web_search\(new_queries=\[(.*?)\]\)", response)
                if match:
                    query_str = match.group(1)
                    queries = [q.strip().strip("'\"") for q in query_str.split(",")]
                    return {"action": "rerun_search", "new_queries": queries}
            return {"action": "return_summary"}  # Default fallback
        
        relevance_action = extract_relevance_action(is_relevant_tool_response)
        
        if relevance_action['action'] == 'rerun_search' and 'new_queries' in relevance_action and depth < MAX_RECURSION_DEPTH:
            print(f"SearchWebUtil: Summary not relevant, rerunning with new queries: {relevance_action['new_queries']}")
            summary = SearchWeb.run(relevance_action['new_queries'], depth=depth + 1)
        else:
            pass
        
        # Return a clean response with the summary as a string with clear formatting
        formatted_summary = f"SEARCH RESULT SUMMARY 1:\n{summary}\n"
        return formatted_summary