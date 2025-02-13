from typing import Any, Dict, List
from py_classes.cls_tooling_web import WebTools

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_llm_router import AIStrengths, LlmRouter

class WebSearchTool(BaseTool):
    def __init__(self):
        self.web_tools = WebTools()

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="web_search",
            description="Search the web for current information using Brave Search",
            parameters={
                "web_query": {
                    "type": "string",
                    "description": "The search query to execute"
                }
            },
            required_params=["web_query"],
            example_usage="""
            {
                "reasoning": "Need to find current information about a topic",
                "tool": "web_search",
                "web_query": "latest developments in AI 2024"
            }
            """
        )

    @property
    def prompt_template(self) -> str:
        return """
        Use the web search tool when you need to:
        1. Find time sensitive or current information
        2. Reason about information not present in the current context
        3. Resolve contradictions between sources
        
        The search will be performed using Brave Search.
        Provide specific, focused queries for better results.
        
        Example:
        {
            "reasoning": "The user wants to know if he needs to bring an umbrella today when visiting the Brandenburg Gate",
            "tool": "web_search",
            "web_query": "todays weather in Berlin"
        }
        """

    async def execute(self, params: Dict[str, Any], preferred_models: List[str] = ["mixtral"]) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                "Invalid parameters provided",
                status="error",
                error="Missing required parameter: web_query"
            )

        try:
            # Handle parameters both at root level and under 'params' key
            if 'params' in params:
                params = params['params']
            
            query = params["web_query"]
            results = self.web_tools.search_brave(query, num_results=3)
            
            if not results:
                return self.format_response(
                    summary="No search results found",
                    status="error",
                    error="No results found for the query"
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

Search Query: {query}

Here are the search results to analyze:

"""
            for result, url in results:
                summarization_prompt += f"URL: {url}\n"
                summarization_prompt += f"Content: {result}\n"
                summarization_prompt += "-----------------------------------\n"

            # Add final instruction
            summarization_prompt += "\nPlease provide a comprehensive summary of these search results, focusing on answering the original query while incorporating the most relevant and up-to-date information from the sources provided."

            summary = LlmRouter.generate_completion(
                summarization_prompt,
                preferred_models=preferred_models
            )

            # Return a clean response with the summary as a string
            return self.format_response(
                summary=summary,
                status="success"
            )

        except Exception as e:
            return self.format_response(
                summary="Error performing web search",
                status="error",
                error=str(e)
            ) 