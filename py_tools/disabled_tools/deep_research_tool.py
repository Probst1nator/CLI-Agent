from typing import Any, Dict, List
from py_classes.cls_tooling_web import WebTools

from py_tools.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_methods.tooling import extract_json
from py_classes.cls_chat import Chat

class DeepResearchTool(BaseTool):
    def __init__(self):
        self.web_tools = WebTools()

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="deep_research",
            description="Conduct comprehensive research to gather inclusive information across many sources",
            detailed_description="""Use this tool when you need to:
- Perform thorough research on complex topics
- Gather information from multiple specialized sources
- Compare perspectives across different domains
- Get detailed analysis beyond surface-level information
Perfect for:
- Comprehensive information gathering
- Academic or technical research
- In-depth analysis of topics
- Comparative studies""",
            constructor="""
def run(
    query: str, 
    depth_level: str = "moderate"
) -> None:
    \"\"\"Conduct comprehensive research on a topic.
    
    Args:
        query: A well defined query that requires multiple internet sources and deep analysis to answer reliably
        depth_level: Desired depth of research: 'basic', 'moderate', or 'comprehensive'
    \"\"\"
"""
        )

    async def _run(self, params: Dict[str, Any], context_chat: Chat, preferred_models: List[str] = []) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: query"
            )

        try:
            query = params["parameters"]["query"]
            depth_level = params["parameters"].get("depth_level", "moderate")
            num_results = 3 if depth_level == "basic" else 4 if depth_level == "moderate" else 5
            
            preliminary_research_query = f"""You are an assistant tasked with authoring a query to gather preliminary information for a research project.
The research objective is: {query}
Please author a query that searches for a condensed and comprehensive overview of the topic. Like a informed list or a report or similar, decide based on the topic.

After thinking please provide a final json object with the following structure:
{{
    "web_search_query": "concise web-search query"
}}
the query should be concise and to the point, and should be no more than 10 words.
"""
            preliminary_research_query_response = LlmRouter.generate_completion(
                preliminary_research_query,
                preferred_models=preferred_models,
                strength=AIStrengths.REASONING,
                exclude_reasoning_tokens=True
            )
            
            web_search_query = extract_json(preliminary_research_query_response, required_keys=["web_search_query"])["web_search_query"]
            
            web_results = self.web_tools.search_brave(web_search_query, num_results=num_results, top_k=num_results*2)
            
            if not web_results:
                return self.format_response(
                    summary="No research results found for the query",
                    status="error",
                )

            distil_insights_prompt = f"""You are a research assistant tasked with distilling insights and key information from a list of results from a search engine.
The research objective is: "{query}"

You have been provided with a list of results from a search engine that may be relevant to the research objective.
Find the most important and relevant information from the results and distill it into a concise and informative summary for focusing the research.
Results to analyze:
"""
            for result, url in web_results:
                distil_insights_prompt += f"Source: {url}\n"
                distil_insights_prompt += f"Content: {result}\n"
                distil_insights_prompt += "-----------------------------------\n"

            preliminary_insights = LlmRouter.generate_completion(
                distil_insights_prompt,
                preferred_models=preferred_models,
                strength=AIStrengths.REASONING,
                exclude_reasoning_tokens=True
            )

            research_prompt = f"""You are a research assistant tasked with authoring a research plan for a research project.
The research objective is: "{query}"
A prelimnary study has been conducted and the following insights have been distilled:
{preliminary_insights}

Based on these insights, suggest 3-4 clear and informed web research tasks that aim to gather comprehensive and detailed information focused on enriching available data and information to the research objective.
The research steps should be specific, actionable and self contained without any further information or context needed.
"""

            research_plan = LlmRouter.generate_completion(
                research_prompt,
                preferred_models=preferred_models,
                strength=AIStrengths.REASONING,
                exclude_reasoning_tokens=False
            )
            
            json_conversion_prompt = f"""Convert the following research analysis into a structured JSON format that can be used to automate further research steps.

RESEARCH ANALYSIS:
{research_plan}

Convert this into a JSON object with the following structure:
{{
  "query": "{query}",
  "research_steps": [
    {{
      "step_id": 1,
      "title": "brief descriptive title of this research step",
      "task": "specific research action to take",
      "reasoning": "why this step is valuable based on current findings",
      "query": "specific query to perform",
      "expected_outcome": "what information this step aims to discover"
    }},
    // Additional steps...
  ]
}}

Ensure the JSON is properly formatted and contains all the suggested research steps in a logical and increasingly insightful order."""

            json_response = LlmRouter.generate_completion(
                json_conversion_prompt,
                preferred_models=preferred_models,
                strength=AIStrengths.TOOLUSE
            )
            
            research_steps = extract_json(json_response, required_keys=["research_steps"])["research_steps"]
            
            research_results = []
            research_steps_tasks = []
            
            for step in research_steps:
                step_id = step["step_id"]
                title = step["title"]
                task = step["task"]
                reasoning = step["reasoning"]
                search_query = step["query"]
                expected_outcome = step["expected_outcome"]
                
                research_steps_tasks.append(task)
                
                web_results = self.web_tools.search_brave(search_query, num_results=num_results, top_k=num_results*2)
                
                step_summary_prompt = f"""{reasoning}
Please distill only the valuable contents from the following sources for the task: {task}
"""
                for result, url in web_results:
                    step_summary_prompt += f"Source: {url}\n"
                    step_summary_prompt += f"Content: {result}\n"
                    step_summary_prompt += "-----------------------------------\n"
                
                step_results = LlmRouter.generate_completion(
                    step_summary_prompt,
                    preferred_models=preferred_models,
                    strength=AIStrengths.REASONING,
                    exclude_reasoning_tokens=True
                )
                research_results.append(step_results)
                
            final_research_summary_prompt = f"""You are part of a research team that is conducting research on the topic: {query}
The following steps were taken to gather more information:
{research_steps_tasks}

We need to analyze the joined the results of the research and synthesize the information to provide a final response to the research query.
Here are the results of the research steps:
{research_results}

Please synthesize the information from the research steps and provide a final response to the research query.
Be aware that the research steps may have gone wrong, and the results need to be assessed critically given the context of the research objective.
The foremost priority is to provide a final response to the research query, excluding irrelevant findings and focusing on the reasoning and details of the objective relevant insights and data.
"""

            final_research_summary = LlmRouter.generate_completion(
                final_research_summary_prompt,
                preferred_models=preferred_models,
                strength=AIStrengths.REASONING,
                exclude_reasoning_tokens=True
            )
            
            # Return a clean response with the analysis as a string
            return self.format_response(
                summary=final_research_summary,
                status="success"
            )

        except Exception as e:
            return self.format_response(
                summary=f"Error performing deep research: {str(e)}",
                status="error",
            ) 