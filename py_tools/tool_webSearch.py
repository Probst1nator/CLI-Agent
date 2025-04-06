from typing import Any, Dict, List, Union, Tuple
from py_classes.cls_chat import Chat, Role
from py_classes.cls_tooling_web import WebTools

from py_tools.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_methods.tooling import extract_blocks
import requests
from pydantic import ValidationError
import traceback
import logging
import re


def extract_relevance_action(text: str) -> Dict[str, Any]:
    """
    Extract relevance action from tool_code blocks that follow these patterns:
    ```tool_code
    return_summary_with_success()
    ```
    or
    ```tool_code
    rerun_web_search(new_queries=['new query 1', 'new query 2'])
    ```
    
    Args:
        text (str): The text containing the tool_code block
        
    Returns:
        Dict[str, Any]: A dictionary containing the action type and optional parameters
            - 'action': 'return_summary' or 'rerun_search'
            - 'new_queries': List of new queries if action is 'rerun_search'
    """
    blocks = extract_blocks(text)
    
    for block_type, content in blocks:
        if block_type.lower() == 'tool_code':
            # Check for return_summary_with_success()
            if re.search(r'return_summary_with_success\(\)', content, re.IGNORECASE):
                return {
                    'action': 'return_summary'
                }
            
            # Check for rerun_web_search with new queries
            rerun_match = re.search(r'rerun_web_search\(\s*new_queries\s*=\s*(\[.*?\])\s*\)', content, re.DOTALL)
            if rerun_match:
                queries_str = rerun_match.group(1)
                try:
                    # Try to parse the query list by replacing single quotes with double quotes for JSON compatibility
                    queries_str = queries_str.replace("'", '"')
                    new_queries = eval(queries_str)  # Use eval for simplicity, considering the controlled environment
                    if isinstance(new_queries, list):
                        return {
                            'action': 'rerun_search',
                            'new_queries': new_queries
                        }
                except:
                    # If parsing fails, return default action
                    pass
    
    # Default to returning the summary if no valid pattern is found
    return {
        'action': 'return_summary'
    }

class WebSearchTool(BaseTool):
    def __init__(self):
        self.web_tools = WebTools()

    @property
    def default_timeout(self) -> float:
        """Override default timeout to 60 seconds for web searches"""
        return 60.0

    async def _run(self, params: Dict[str, Any], context_chat: Chat, preferred_models: List[str] = [], timeout: float = None) -> ToolResponse:
        """Override to maintain backward compatibility with the preferred_models parameter"""
        # Pass the preferred_models to _execute by extending the params
        params_with_models = params.copy()
        params_with_models["preferred_models"] = preferred_models
        return await super().run(params_with_models, timeout=timeout)

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="web_search",
            description="Search the web for current information using Brave Search",
            detailed_description="""Use this tool when you need to:
- Research topics outside your knowledge
- Get real-time data or updates
- Fact check contradictions between sources
Perfect for:
- Information gathering
- News and weather
- Latest developments in any field
- Resolution of contradicting information""",
            constructor="""
def run(queries: List[str]) -> None:
    \"\"\"Search the web for information.
    
    Args:
        queries: A list of search queries to execute
    \"\"\"
"""
        )

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: queries"
            )

        try:
            # Extract preferred_models if it was passed via the run method
            preferred_models = params.pop("preferred_models", []) if "preferred_models" in params else []
            queries: Union[List[str], str] = params["parameters"]["queries"]
            
            # Handle both single string and list of strings
            if isinstance(queries, str):
                queries = [queries]
            
            # Main search function that can be called recursively if needed
            async def perform_search(search_queries: List[str]) -> ToolResponse:
                all_results: List[Tuple[str, str]] = []
                failed_queries: List[Tuple[str, str]] = []
                
                
                if (len(search_queries) == 1):
                    num_results = 3
                else:
                    num_results = 2

                for query in search_queries:
                    try:
                        query_results = self.web_tools.search_brave(query, num_results=num_results)
                        if query_results:
                            all_results.extend(query_results)
                        else:
                            failed_queries.append((query, "No results found"))
                    except Exception as e:
                        failed_queries.append((query, f"Unexpected error: {str(e)}"))
                
                if not all_results:
                    failure_details = "\n".join([f"- Query '{q}': {reason}" for q, reason in failed_queries])
                    return self.format_response(
                        summary=f"Could not retrieve search results for any of the queries:\n{failure_details}",
                        status="error",
                    )

                # Format results
                summarization_prompt = f"""You are a helpful AI assistant tasked with summarizing web search results.
                
Task: Analyze the following web search results and provide a clear, informative summary that:
1. Synthesizes the key information from all sources
2. Preserves important details and context
3. Resolves and highlights any contradictions between sources
4. Includes relevant dates or timestamps when available
5. Cites specific sources for key claims

Format your response as a coherent paragraph that flows naturally.
If sources conflict or information seems outdated, note this in your summary.

Search Queries: {", ".join(search_queries)}

Here are the search results to analyze:

"""
                for result, url in all_results:
                    summarization_prompt += f"URL: {url}\n"
                    summarization_prompt += f"Content: {result}\n"
                    summarization_prompt += "-----------------------------------\n"

                # Add final instruction
                summarization_prompt += f"\nPlease provide a summary of the results, including key facts and details extending beyond the required, to answer the original Search Queries: {', '.join(search_queries)}. Ensuring trusthworthy sources and interpolating between them when necessary."

                # Add note about failed queries if some succeeded but others failed
                if failed_queries and all_results:
                    failed_query_list = ", ".join([f"'{q}'" for q, _ in failed_queries])
                    summarization_prompt += f"\n\nNote: Unable to retrieve results for the following queries: {failed_query_list}. The summary is based only on the available results."

                summary = LlmRouter.generate_completion(
                    summarization_prompt,
                    preferred_models=preferred_models,
                    strength=[AIStrengths.GENERAL],
                    exclude_reasoning_tokens=True
                )
                
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

Analysis the summary carefully. Only suggest rerunning the search if the current summary is not helpful for answering the queries.
"""
                is_relevant_prompt = f"""Is the following summary relevant to the original Search Queries: {', '.join(search_queries)}?

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
                is_relevant_tool_response = LlmRouter.generate_completion(
                    is_relevant_chat,
                    preferred_models=preferred_models,
                    strength=[AIStrengths.REASONING, AIStrengths.FAST],
                    exclude_reasoning_tokens=True
                )
                
                relevance_action = extract_relevance_action(is_relevant_tool_response)
                
                if relevance_action['action'] == 'rerun_search' and 'new_queries' in relevance_action:
                    logging.info(f"Summary not relevant, rerunning with new queries: {relevance_action['new_queries']}")
                    return await perform_search(relevance_action['new_queries'])
                
                # Return a clean response with the summary as a string
                status = "success" if not failed_queries else "partial_success"
                return self.format_response(
                    summary=summary,
                    status=status,
                )
            
            # Start the search process
            return await perform_search(queries)

        except ValidationError as ve:
            # Handle Pydantic validation errors
            error_detail = str(ve)
            logging.error(f"Validation error in web search: {error_detail}\n{traceback.format_exc()}")
            return self.format_response(
                summary="Error validating search results. This is likely due to unexpected data format from the search API.",
                status="error",
            )
        except requests.RequestException as re:
            # Handle network/API errors
            error_detail = str(re)
            logging.error(f"Request error in web search: {error_detail}")
            return self.format_response(
                summary="Error connecting to search API. Please check your internet connection or try again later.",
                status="error",
            )
        except Exception as e:
            # Handle any other unexpected errors
            error_detail = str(e)
            logging.error(f"Unexpected error in web search: {error_detail}\n{traceback.format_exc()}")
            return self.format_response(
                summary=f"Error performing web search: {error_detail}",
                status="error",
            ) 