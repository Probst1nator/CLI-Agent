from typing import Any, Dict, List, Optional, Tuple, Protocol
import pyautogui
import json
import os
import time
from PIL import Image
from termcolor import colored
import base64
from dataclasses import dataclass
from enum import Enum, auto
import re

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_llm_router import LlmRouter
from py_classes.cls_chat import Chat, Role
from py_classes.globals import g
from py_methods.tooling import extract_blocks, extract_json

# Import Omniparser
import sys
sys.path.append(os.path.join(g.PROJ_DIR_PATH, "OmniParser"))

class UiAction(Enum):
    """Enumeration of supported UI actions"""
    COMPREHEND = auto()
    CLICK = auto()
    TYPE = auto()
    SCROLL = auto()
    DONE = auto()

@dataclass
class ActionResult:
    """Result of a UI action execution"""
    success: bool
    message: str
    screen_description: Optional[str] = None
    should_continue: bool = True

class ActionHandler(Protocol):
    """Protocol for action handlers"""
    async def handle(self, target: str, screenshot_path: str, screen_description: str) -> ActionResult:
        ...

class UiTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.parser = None
        self._action_handlers: Dict[UiAction, ActionHandler] = self._initialize_action_handlers()

    def _initialize_action_handlers(self) -> Dict[UiAction, ActionHandler]:
        """Initialize action handlers"""
        return {
            UiAction.COMPREHEND: self.ComprehendActionHandler(self),
            UiAction.CLICK: self.ClickActionHandler(self),
            UiAction.TYPE: self.TypeActionHandler(self),
            UiAction.SCROLL: self.ScrollActionHandler(self),
            UiAction.DONE: self.DoneActionHandler(self)
        }

    class BaseActionHandler:
        """Base class for action handlers"""
        def __init__(self, tool: 'UiTool'):
            self.tool = tool

    class ComprehendActionHandler(BaseActionHandler):
        async def handle(self, target: str, screenshot_path: str, screen_description: str) -> ActionResult:
            visual_analysis = await self.tool._analyze_screen_with_vlm(screenshot_path, target)
            if visual_analysis["status"] != "success":
                return ActionResult(False, f"Failed to comprehend target: {target}", screen_description)
            return ActionResult(True, visual_analysis["screen_description"], screen_description, False)

    class ClickActionHandler(BaseActionHandler):
        async def handle(self, target: str, screenshot_path: str, screen_description: str) -> ActionResult:
            screen_elements = await self.tool._analyze_screen_for_clicking()
            if not screen_elements:
                return ActionResult(False, f"Failed to analyze screen for clicking", screen_description)

            # Keep screen elements local to this operation
            target_element = self.tool._find_best_matching_element(target, screen_elements)
            if not target_element:
                return ActionResult(False, f"Could not find element matching description: {target}", screen_description)
            
            if not self.tool._click_element(target_element):
                return ActionResult(False, f"Failed to click element: {target_element['text']}", screen_description)
            
            return ActionResult(True, f"Successfully clicked element: {target_element['text']}", screen_description)

    class TypeActionHandler(BaseActionHandler):
        async def handle(self, target: str, screenshot_path: str, screen_description: str) -> ActionResult:
            pyautogui.write(target)
            return ActionResult(True, "Text input successful", screen_description)

    class ScrollActionHandler(BaseActionHandler):
        async def handle(self, target: str, screenshot_path: str, screen_description: str) -> ActionResult:
            if target not in ["up", "down"]:
                return ActionResult(False, f"Invalid scroll direction. Must be 'up' or 'down', got: {target}", screen_description)
            
            if not self.tool._scroll_screen(target):
                return ActionResult(False, f"Failed to scroll {target}", screen_description)
            
            return ActionResult(True, "Scroll successful", screen_description)

    class DoneActionHandler(BaseActionHandler):
        async def handle(self, target: str, screenshot_path: str, screen_description: str) -> ActionResult:
            return ActionResult(True, screen_description, screen_description, False)

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="ui",
            description="Analyze the current screen or control the UI",
            parameters={
                "intent": {
                    "type": "string",
                    "description": "Step by step description of what to achieve using the UI"
                }
            },
            required_params=["intent"],
            example_usage="""
            {
                "tool": "ui",
                "reasoning": "Need to display the video 'Never Gonna Give You Up'",
                "intent": "Open a browser, navigate to YouTube, search for 'Never Gonna Give You Up', and play the video"
            }
            
            {
                "tool": "ui",
                "reasoning": "I can use the UI tool to analyze the current screen and analyze what the user is engaging with",
                "intent": "Analyze the screen to identify the users current activities"
            }
            """
        )

    @property
    def prompt_template(self) -> str:
        return """Use the UI tool to analyze and interact with screen elements based on natural language intent.
The tool will automatically:
1. Analyze the screen before any action
2. Find relevant UI elements
3. Execute appropriate actions (click, type) based on the intent
4. Iterate until the provided intent is achieved or aborted

Example usage:
{
    "tool": "ui",
    "reasoning": "Need to display the video 'Never Gonna Give You Up'",
    "intent": "Open a browser, navigate to YouTube, search for 'Never Gonna Give You Up', and play the video"
}

{
    "tool": "ui",
    "reasoning": "Understand what the user is doing",
    "intent": "Analyze the current screen"
}"""

    def _take_screenshot(self) -> Tuple[str, str]:
        """Take a screenshot and save both original and annotated versions."""
        print(colored("📸 Taking screenshot...", "cyan"))
        timestamp = int(time.time())
        screenshot_dir = os.path.join(g.PROJ_VSCODE_DIR_PATH, "ui_tool")
        os.makedirs(screenshot_dir, exist_ok=True)
        
        screenshot_path = os.path.join(screenshot_dir, f"screen_{timestamp}.png")
        annotated_path = os.path.join(screenshot_dir, f"screen_{timestamp}_annotated.png")
        
        pyautogui.screenshot(screenshot_path)
        print(colored(f"✅ Screenshot saved to: {screenshot_path}", "green"))
        return screenshot_path, annotated_path

    def _get_base64_image(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    async def _analyze_screen_with_vlm(self, screenshot_path: str, intent: str) -> Dict[str, Any]:
        """Analyze the screen using LLM's vision capabilities."""
        base64_image = self._get_base64_image(screenshot_path)
        
        analysis_chat = Chat(debug_title="VLM Screenshot Analysis")
        
        analysis_chat.add_message(
            Role.USER,
            f"{intent}\nWhat windows are open and what are their contents?.",
        )
        
        screen_description = LlmRouter.generate_completion(analysis_chat, base64_images=[base64_image])
        
        return {
            "status": "success",
            "summary": screen_description
        }

    async def _analyze_screen_for_clicking(self) -> List[Dict[str, Any]]:
        """Analyze the screen using Omniparser to prepare for clicking elements."""
        screenshot_path, annotated_path = self._take_screenshot()
        if not self.parser:
            from OmniParser.omniparser import Omniparser, config as omniparser_config
            self.parser = Omniparser(omniparser_config)
        
        image, screen_elements = self.parser.parse(screenshot_path)
        
        # Always save the annotated image
        image.save(annotated_path)
        
        # Transform screen elements into simplified format
        simplified_elements = []
        for element in screen_elements:
            simplified_element = {
                'type': element.get('type', ''),
                'content': element.get('text', element.get('content', '')),  # Try text first, fall back to content
                'shape': {}
            }
            
            # Convert shape coordinates to integers if present
            if 'shape' in element:
                simplified_element['shape'] = {
                    k: int(v) for k, v in element['shape'].items()
                }
            
            simplified_elements.append(simplified_element)
        
        return simplified_elements

    def _parse_intent(self, reasoning: str, intent: str, screen_description: str) -> Dict[str, Any]:
        """Parse the intent string to determine required actions, with screen context."""
        # Create a chat context for intent parsing
        parse_chat = Chat(debug_title="UI Intent Parsing")
        
        parse_chat.add_message(
            Role.USER,
            f"""You are a UI interaction assistant. Your goal is to progress step by step towards the intent using the minimum number of actions necessary.
To interact with UI elements, follow these action guidelines:

COMPREHEND:
- Use ONLY if you need to read text or understand visual content
- Never use comprehend to find UI elements - that's handled automatically by the actions themselves

ACTIONS:
- CLICK for any interaction with buttons, links, or UI elements
- TYPE for text input
- SCROLL if content might be off-screen
- DONE when the goal is achieved

Action Priority (try in this order):
1. CLICK: For ANY interaction with UI elements (the tool will find the element automatically)
2. TYPE: For text input (the tool will find the input field automatically)
3. SCROLL: If target might be off-screen
4. COMPREHEND: ONLY for reading text or understanding visual content
5. DONE: When the goal is achieved

First, explain your thought process:
1. What is the current state based on the screen description?
2. What specific action is needed to progress toward the goal?
3. Why is this the best next action?

After explaining your reasoning, provide your action selection in a single, valid JSON format.

Action Guidelines:
- click: Use for ANY interaction with UI elements - the tool will locate the element
  target = "contextful description of what to click (e.g., 'youtube video', 'firefox search text input')"
- type: Use for text input - the tool will find the input field
  target = "exact text to type"
- scroll: Use when target might be off-screen
  target = "up" or "down"
- comprehend: ONLY for extracting text or understanding visual content
  target = "Question about the text/visual information to extract"
- done: Use when the intent is satisfied
  target = null

Response Format Example:
Let me think through this...
1. Based on the screen, [current state analysis]
2. To progress, we could [evaluation of different relevant actions and their respective targets]
3. The best approach seems to be [a final chosen action and its target]

Here's my action selection:
{{
    "reasoning": "brief explanation of why this specific action is the best next step",
    "action": "click" | "type" | "scroll" | "comprehend" | "done",
    "target": "what to click/type/analyze" | "up" | "down" | null
}}

Current screen context:
{screen_description}

Intent Reasoning: "{reasoning}"
Intent: "{intent}"

Return a single JSON object containing the most relevant next action."""
        )
        
        try:
            response = LlmRouter.generate_completion(parse_chat)
            parsed_json = extract_json(response, required_keys=["action", "target"])
            
            if parsed_json is None:
                print(colored("No valid JSON found in response", "red"))
                return None
                
            return parsed_json
            
        except Exception as e:
            print(colored(f"Error with UI action: {str(e)}", "red"))
            return None

    def _scroll_screen(self, direction: str) -> bool:
        """Scroll the screen up or down."""
        try:
            # Positive clicks for scrolling down, negative for up
            clicks = -1 if direction == "up" else 1
            pyautogui.scroll(clicks * 3)  # Multiply by 3 for more noticeable scrolling
            return True
        except Exception as e:
            print(colored(f"Error scrolling screen: {str(e)}", "red"))
            return False

    def _find_best_matching_element(self, target_description: str, screen_elements: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the UI element that best matches the target description."""
        if not screen_elements:
            print(colored("No screen elements available for matching", "red"))
            return None
            
        # Create a chat context for semantic matching
        match_chat = Chat(debug_title="UI Element Matching")
        match_chat.add_message(
            Role.SYSTEM,
            """You are an AI assistant that matches UI element descriptions to actual elements.
Your task is to find the best matching UI element based on semantic similarity and visual context.

Analyze the available elements and respond with a JSON object containing:
{
    "candidates": [
        {
            "index": number,
            "explanation": "why this element matches the target description"
        }
    ],
    "reasoning": "detailed explanation of why the best candidate was chosen",
    "best_match": number (index of the best candidate, or -1 if no good match found)
}

Guidelines:
1. Evaluate semantic similarity between target description and element text/type
2. Consider element type (button, link, text, etc.) and context
3. List at most 3 candidates, ordered by relevance
4. Return -1 if no suitable match is found

Example response:
{
    "candidates": [
        {
            "index": 5,
            "explanation": "Element text 'Play' exactly matches target 'play button' and is a button type"
        },
        {
            "index": 2,
            "explanation": "Element is a play icon but lacks text label"
        }
    ],
    "reasoning": "Selected index 5 because it has exact text match and correct element type",
    "best_match": 5
}"""
        )
        
        # Include more detailed element information for better matching
        elements_list = "\n".join(
            f"{i}. Type: {elem['type']}, Text: '{elem['text']}', "
            f"Location: (x={elem['shape']['x']:.0f}, y={elem['shape']['y']:.0f}), "
            f"Size: {elem['shape']['width']:.0f}x{elem['shape']['height']:.0f}"
            for i, elem in enumerate(screen_elements)
        )
        
        match_chat.add_message(
            Role.USER,
            f"""Target description: "{target_description}"
            
Available UI elements:
{elements_list}

Return a JSON object selecting the best matching element."""
        )
        
        try:
            response = LlmRouter.generate_completion(match_chat)
            parsed_response = json.loads(response)
            
            best_match_index = parsed_response.get("best_match", -1)
            if best_match_index >= 0 and best_match_index < len(screen_elements):
                # Log the candidates and reasoning for debugging
                candidates = parsed_response.get("candidates", [])
                reasoning = parsed_response.get("reasoning", "No reasoning provided")
                print(colored("Matching candidates:", "cyan"))
                for candidate in candidates:
                    print(colored(f"- Index {candidate['index']}: {candidate['explanation']}", "cyan"))
                print(colored("\nSelection reasoning:", "cyan"))
                print(colored(reasoning, "cyan"))
                
                matched_element = screen_elements[best_match_index]
                print(colored(f"\nSelected element: {matched_element['type']} - '{matched_element['text']}'", "green"))
                return matched_element
            
            # Log if no match was found
            print(colored("No matching element found. Analysis:", "yellow"))
            print(colored(response, "yellow"))
            
        except Exception as e:
            print(colored(f"Error during element matching: {str(e)}", "red"))
            
        return None

    def _click_element(self, element: Dict[str, Any]) -> bool:
        """Click on a UI element using its coordinates."""
        try:
            x = element['shape']['x'] + (element['shape']['width'] / 2)
            y = element['shape']['y'] + (element['shape']['height'] / 2)
            pyautogui.click(x, y)
            return True
        except Exception as e:
            print(colored(f"Error clicking element: {str(e)}", "red"))
            return False

    def _map_action_string_to_enum(self, action_str: str) -> Optional[UiAction]:
        """Map action string to UiAction enum"""
        try:
            return UiAction[action_str.upper()]
        except KeyError:
            return None

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the UI tool based on natural language intent."""
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: intent"
            )

        reasoning = params["reasoning"]
        intent = params["intent"]
        action_counter = 0
        MAX_ACTIONS = 10
        last_screen_description = None
        action_results = []

        try:
            while action_counter < MAX_ACTIONS:
                # Take and analyze screenshot
                screenshot_path, _ = self._take_screenshot()
                screen_analysis = await self._analyze_screen_with_vlm(screenshot_path, intent)
                
                if screen_analysis["status"] != "success":
                    return self.format_response(
                        status="error",
                        summary=f"Failed to analyze screen: {screen_analysis.get('error', 'Unknown error')}"
                    )

                screen_description = screen_analysis["summary"]
                last_screen_description = screen_description

                # Parse intent
                parsed_intent = self._parse_intent(reasoning, intent, screen_description)
                if parsed_intent is None:
                    return self.format_response(
                        status="success",
                        summary=f"Intent parsing failed, but screen analysis completed. Screen contents: {screen_description}"
                    )

                # Map action string to enum
                action_str = parsed_intent["action"]
                action = self._map_action_string_to_enum(action_str)
                if action is None:
                    return self.format_response(
                        status="error",
                        summary=f"Unsupported action: {action_str}. Screen contents: {screen_description}"
                    )

                # Get action handler
                handler = self._action_handlers.get(action)
                if handler is None:
                    return self.format_response(
                        status="error",
                        summary=f"No handler found for action: {action_str}. Screen contents: {screen_description}"
                    )

                # Execute action
                target = parsed_intent["target"]
                result = await handler.handle(target, screenshot_path, screen_description)
                action_results.append(result)

                if not result.success:
                    # Format all results into a comprehensive error message
                    error_summary = self._format_action_results(action_results)
                    return self.format_response(
                        status="error",
                        summary=f"Action failed: {result.message}\nPrevious actions:\n{error_summary}"
                    )

                if action == UiAction.DONE or not result.should_continue:
                    # Format all successful results into a final summary
                    success_summary = self._format_action_results(action_results)
                    return self.format_response(
                        status="success",
                        summary=success_summary
                    )

                action_counter += 1
                time.sleep(0.5)  # Allow UI to update

            # Max actions reached
            final_summary = self._format_action_results(action_results)
            return self.format_response(
                status="error",
                summary=f"Maximum actions ({MAX_ACTIONS}) reached without completing task.\nActions performed:\n{final_summary}\nLast screen state: {last_screen_description}"
            )

        except Exception as e:
            error_summary = self._format_action_results(action_results) if action_results else "No actions completed"
            return self.format_response(
                status="error",
                summary=f"Error executing UI tool: {str(e)}\nActions performed:\n{error_summary}"
            )

    def _format_action_results(self, results: List[ActionResult]) -> str:
        """Format a list of action results into a readable summary."""
        if not results:
            return "No actions performed"
        
        summary_lines = []
        for i, result in enumerate(results, 1):
            status = "✓" if result.success else "✗"
            summary_lines.append(f"{i}. [{status}] {result.message}")
        
        return "\n".join(summary_lines) 