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
            detailed_description="""Use this tool when you need to:
- Research topics outside your knowledge
- Get real-time data or updates
- Fact check contradictions between sources

Perfect for:
- Information gathering
- News and weather
- Latest developments in any field
- Resolution of contradicting information""",
            parameters={
                "search_query": {
                    "type": "string",
                    "description": "The search query to execute"
                }
            },
            required_params=["search_query"],
            example_usage="""
{
    "reasoning": "Need to find information about x",
    "tool": "web_search",
    "parameters": {
        "search_query": "detailed web query to retrieve the required information about x"
    }
}
"""
        )

    async def execute(self, params: Dict[str, Any], preferred_models: List[str] = []) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: search_query"
            )

        try:
            query = params["parameters"]["search_query"]
            results = self.web_tools.search_brave(query, num_results=3)
            
            if not results:
                return self.format_response(
                    summary="No search results found for the query",
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
                preferred_models=preferred_models,
                strength=AIStrengths.REASONING,
                exclude_reasoning_tokens=True
            )

            # Return a clean response with the summary as a string
            return self.format_response(
                summary=summary,
                status="success"
            )

        except Exception as e:
            return self.format_response(
                summary=f"Error performing web search: {str(e)}",
                status="error",
            ) 