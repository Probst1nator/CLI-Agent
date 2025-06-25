from termcolor import colored
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_util_base import UtilBase
from typing import Any, Dict, List, Optional, Tuple
import logging
import traceback
import requests
from pydantic import ValidationError
from py_classes.globals import g
from py_classes.utils.cls_utils_web import WebTools

class SearchWeb(UtilBase):
    """
    A utility for performing web searches.
    
    This utility allows searching the web and summarizing the results.
    """
    
    # Initialize WebTools here to ensure it's available in sandbox
    web_tools = None
    

    
    @staticmethod
    async def run(queries: List[str], depth: int = 0) -> str:
        """
        Perform a web search, summarize the results, and handle failures gracefully.

        Args:
            queries (List[str]): A list of search queries.
            depth (int): The current recursion depth to prevent infinite loops.

        Returns:
            A string containing a summary of successful results and a list of failures.
        """
        MAX_RECURSION_DEPTH = 1
        SearchWeb._initialize()

        if isinstance(queries, str):
            queries = [queries]

        all_results: List[Tuple[str, str]] = []
        failed_queries: List[Tuple[str, str]] = []

        num_results_per_query = 5 if len(queries) == 1 else 3

        print(colored(f"SearchWeb: Performing {len(queries)} queries...", "cyan"))
        for query in queries:
            try:
                query_results = SearchWeb.web_tools.search_brave(query, num_results=num_results_per_query)
                if query_results:
                    all_results.extend(query_results)
                else:
                    failed_queries.append((query, "No results returned from search API."))
            except requests.exceptions.HTTPError as e:
                # This catches HTTP errors from the underlying web tool
                error_message = f"HTTP Error {e.response.status_code} while searching."
                print(colored(f"  └─ Failed query '{query}': {error_message}", "red"))
                failed_queries.append((query, error_message))
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}"
                print(colored(f"  └─ Failed query '{query}': {error_message}", "red"))
                failed_queries.append((query, error_message))

        # --- Summarization Phase ---
        instruction = """You are an AI assistant that synthesizes web search results.
Your task is to create a clear, concise summary based on the provided content.
If there are failed searches, list them clearly at the end.
If ALL searches failed, state that you were unable to find information and list the reasons.
Format the successful results into a coherent summary.
"""
        summarization_chat = Chat(instruction, debug_title="SearchWeb-Summarization")

        # Build the prompt for the summarization LLM
        prompt_content = f"# Search Query/Queries:\n- {'\n- '.join(queries)}\n\n"
        if all_results:
            prompt_content += "# Successfully Retrieved Content:\n"
            for i, (content, url) in enumerate(all_results):
                prompt_content += f"## Source {i+1}: {url}\n{content[:1500]}...\n\n"
        else:
            prompt_content += "# Successfully Retrieved Content:\nNone.\n\n"
        
        if failed_queries:
            prompt_content += "# Failed Queries:\n"
            for query, reason in failed_queries:
                prompt_content += f"- Query: '{query}'\n  Reason: {reason}\n"

        prompt_content += "\n## Summary:\nPlease provide your summary based on the information above."
        
        summarization_chat.add_message(Role.USER, prompt_content)

        try:
            summary = await LlmRouter.generate_completion(
                summarization_chat,
                strengths=[AIStrengths.GENERAL],
                hidden_reason="Summarizing search results"
            )
        except Exception as e:
            summary = f"Error during summarization: {e}. Raw data processing is required."
        
        # --- Relevance Check & Final Return ---
        relevance_check_prompt = f"""Analyze the following summary and determine if it sufficiently answers the original queries.
Original Queries: {queries}
Summary: {summary}
If the summary is helpful, respond with ONLY the word "return".
If the summary is unhelpful AND there is a clear path to improve the search (e.g., the query was too broad), respond with a JSON object containing better queries:
{{"new_queries": ["more specific query 1", "alternative angle query"]}}
"""
        relevance_chat = Chat("You are a search query relevance checker.", "SearchWeb-Relevance")
        relevance_chat.add_message(Role.USER, relevance_check_prompt)
        
        relevance_response = await LlmRouter.generate_completion(relevance_chat, strengths=[AIStrengths.SMALL], hidden_reason="Checking search relevance")

        try:
            # Try to parse as JSON for new queries
            import json
            new_queries_data = json.loads(relevance_response)
            if isinstance(new_queries_data, dict) and 'new_queries' in new_queries_data:
                if depth < MAX_RECURSION_DEPTH:
                    print(colored(f"SearchWeb: Rerunning with improved queries (depth {depth+1})...", "yellow"))
                    return await SearchWeb.run(new_queries_data['new_queries'], depth=depth + 1)
        except (json.JSONDecodeError, TypeError):
            # If it's not JSON, it's treated as a "return" signal.
            pass

        # If recursion limit is reached or relevance check passes, return the current summary.
        print(colored("SearchWeb: Finalizing summary.", "green"))
        return summary

    @classmethod
    def _initialize(cls) -> None:
        """Initialize WebTools if not already done."""
        if cls.web_tools is None:
            cls.web_tools = WebTools()