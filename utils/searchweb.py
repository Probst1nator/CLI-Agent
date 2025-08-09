from termcolor import colored
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_util_base import UtilBase
from typing import Any, Dict, List, Tuple, Coroutine
import asyncio
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
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["google search", "find information", "browse web", "research topic", "look up", "internet search", "huggingface", "model hub", "github", "repository", "latest version", "discover", "explore", "trending", "compare models", "benchmarks", "documentation", "technical specs", "download link", "release notes", "model card"],
            "use_cases": [
                "Search the web for the latest news on artificial intelligence.",
                "Find the official documentation for the Python requests library.",
                "What is the weather forecast for tomorrow?",
                "Find performance benchmarks comparing different LLM architectures.",
                "Search for tutorials on deploying models with Ollama."
            ],
            "arguments": {
                "queries": "A list of search terms or questions.",
                "depth": "Internal use for recursion depth, should not be set by the user."
            },
            "code_examples": [
                {
                    "description": "Search for latest AI news",
                    "code": """```python
from utils.searchweb import SearchWeb
result = SearchWeb.run(["latest AI developments 2025"])
print(result)
```"""
                },
                {
                    "description": "Using module-level function (CLI-Agent style)",
                    "code": """```python
import utils.searchweb as searchweb
result = searchweb.run(query="Python machine learning tutorials")
print(result)
```"""
                }
            ]
        }
    
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
    def _run_logic(queries: List[str], depth: int = 0) -> str:
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
        inst = """You are a helpful AI assistant tasked with summarizing web search results.
        
Task: Analyze the following web search results and provide a clear, informative summary that:
1. Synthesizes the key information from all sources
2. Preserves important details and context
3. Resolves and highlights any contradictions between sources
4. Includes relevant dates or timestamps when available
5. Cites specific sources for key claims

Format your response as a coherent paragraph that flows naturally.
If sources conflict or information seems outdated, note this in your summary.


"""

        summarization_prompt = "# Search Queries\n"
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
                hidden_reason="Condensing search results",
            ))
        except Exception as e:
            print(f"❌ SearchWeb: Failed to generate summary: {str(e)}")
            # Fallback to simple concatenation of results
            summary = "Search results summary failed to generate. Raw results:\n\n"
            for i, (result, url) in enumerate(all_results, 1):
                summary += f"{i}. {url}\n{result[:200]}...\n\n"
        
        # Check if the summary is relevant to the queries
        is_relevant_instruction = """You are a helpful assistant that checks if a summary is relevant to a list of search queries. 
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
                strengths=[AIStrengths.SMALL],
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
            summary = SearchWeb._run_logic(relevance_action['new_queries'], depth=depth + 1)
        else:
            pass
        
        if (summary):
            # Return a clean response with the summary as a string with clear formatting
            summary = f"# SEARCH SUMMARY\n{summary}\n"
        return summary


# Module-level run function for CLI-Agent compatibility
def run(query=None, queries=None, depth: int = 0) -> str:
    """
    Module-level wrapper for SearchWeb.run() to maintain compatibility with CLI-Agent.
    
    Args:
        query: Single query string (CLI-Agent style)
        queries: List of query strings (original style) 
        depth: Internal recursion depth parameter
        
    Returns:
        str: Search results summary
    """
    # Handle both calling patterns
    if query is not None:
        # CLI-Agent style: run(query="search term")
        search_queries = [query] if isinstance(query, str) else query
    elif queries is not None:
        # Original style: run(queries=["search term"])
        search_queries = queries
    else:
        raise ValueError("Either 'query' or 'queries' parameter must be provided")
    
    return SearchWeb._run_logic(search_queries, depth)


if __name__ == "__main__":
    """
    Test the SearchWeb utility functionality
    """
    # Add parent directory to path for imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing SearchWeb utility...")
    
    # Test 1: Basic functionality test
    print("\n=== Test 1: Basic Search ===")
    try:
        result = SearchWeb._run_logic(["latest agent LLMs under 30GB huggingface"])
        print(f"Search completed. Result length: {len(result) if result else 0} characters")
        if result:
            print("First 200 characters of result:")
            print(result[:200] + ("..." if len(result) > 200 else ""))
        else:
            print("No result returned")
    except Exception as e:
        print(f"Error in basic search test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Single string input (should be converted to list)
    print("\n=== Test 2: Single String Input ===")
    try:
        result = SearchWeb._run_logic("Python machine learning libraries")
        print(f"Search completed. Result length: {len(result) if result else 0} characters")
        if result:
            print("First 200 characters of result:")
            print(result[:200] + ("..." if len(result) > 200 else ""))
    except Exception as e:
        print(f"Error in single string test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check if WebTools initializes properly
    print("\n=== Test 3: WebTools Initialization ===")
    try:
        SearchWeb.initialize()
        print(f"WebTools initialized: {SearchWeb.web_tools is not None}")
        print(f"WebTools type: {type(SearchWeb.web_tools)}")
    except Exception as e:
        print(f"Error in WebTools initialization test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing Complete ===")
    
    # Test 4: Module-level access test (what the CLI-Agent is trying to do)
    print("\n=== Test 4: Module-level Access ===")
    print(f"SearchWeb class exists: {SearchWeb is not None}")
    print(f"SearchWeb has run method: {hasattr(SearchWeb, 'run')}")
    print(f"SearchWeb run method callable: {callable(getattr(SearchWeb, 'run', None))}")
    
    # Test the problematic import pattern from CLI-Agent
    print("\n=== Test 5: Import Pattern Test ===")
    try:
        # This simulates what happens in the CLI-Agent when it does "import utils.searchweb"
        import utils.searchweb as searchweb_module
        print(f"Module imported successfully: {searchweb_module}")
        print(f"Module has 'run' attribute: {hasattr(searchweb_module, 'run')}")
        print(f"Module has 'SearchWeb' class: {hasattr(searchweb_module, 'SearchWeb')}")
        
        if hasattr(searchweb_module, 'SearchWeb'):
            print("Trying to call SearchWeb.run directly...")
            result = searchweb_module.SearchWeb._run_logic(["test query"])
            print(f"Direct call successful, result length: {len(result) if result else 0}")
    except Exception as e:
        print(f"Import pattern test failed: {e}")
        import traceback
        traceback.print_exc()