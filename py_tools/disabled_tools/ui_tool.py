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
import mss
import mss.tools
import io

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_llm_router import AIStrengths, LlmRouter
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
        self.selected_screen_index: Optional[int] = None
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
            visual_analysis = await self.tool._analyze_screen_with_vlm(screenshot_path, self.tool.selected_screen_index)
            if visual_analysis["status"] != "success":
                return ActionResult(False, f"Failed to comprehend target: {target}", screen_description)
            return ActionResult(True, visual_analysis["screen_description"], screen_description, False)

    class ClickActionHandler(BaseActionHandler):
        async def handle(self, target: str, screenshot_path: str, screen_description: str) -> ActionResult:
            _, screen_elements = await self.tool._get_annotated_image_path(screenshot_path)
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
            description="Call the UI tool when you need to directly interact with the user interface or any of the user's graphical applications",
            detailed_description="""Use this tool when you need to:
- Understand what's currently on the user's screen
- Click buttons, links, or other UI elements
- Enter text into forms or input fields
- Navigate through content by scrolling
- Perform sequences of UI interactions

Perfect for tasks like:
- Navigating websites
- Interacting with desktop applications
- Filling out forms
- Analyzing screen content
- Automating UI workflows""",
            constructor="""
def run(
    component_type: str,
    content: str,
    properties: Dict[str, Any] = None
) -> None:
    \"\"\"Create UI components.
    
    Args:
        component_type: Type of UI component ('button', 'text', 'image', 'input', 'chart')
        content: The main content for the component (text, URL, data)
        properties: Optional dictionary of additional properties for the component
    \"\"\"
"""
        )

    def _get_monitor_center(self, monitor_index: Optional[int] = None) -> Tuple[int, int]:
        """Get the center coordinates of the specified monitor.
        
        Args:
            monitor_index: Optional index of specific monitor. If None, uses selected_screen_index.
            
        Returns:
            Tuple of (center_x, center_y) coordinates
        """
        monitors = mss.mss().monitors[1:]  # Skip first monitor (represents all monitors combined)
        monitor_idx = monitor_index if monitor_index is not None else self.selected_screen_index
        if monitor_idx is None or monitor_idx >= len(monitors):
            # Fallback to primary screen center
            screen_width, screen_height = pyautogui.size()
            return (screen_width // 2, screen_height // 2)
            
        monitor = monitors[monitor_idx]
        center_x = monitor["left"] + (monitor["width"] // 2)
        center_y = monitor["top"] + (monitor["height"] // 2)
        return (center_x, center_y)

    def _take_screenshot(self, monitor_index: Optional[int] = None) -> Tuple[List[str], int]:
        """Take screenshots of monitors."""
        print(colored("📸 Taking screenshots...", "cyan"))
        timestamp = int(time.time())
        screenshot_dir = os.path.join(g.PROJ_TEMP_STORAGE_PATH, "ui_tool")
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Save current mouse position
        original_x, original_y = pyautogui.position()
        
        screenshot_paths = []
        try:
            with mss.mss() as sct:
                monitors = sct.monitors[1:]  # Skip the first monitor which is the "all in one"
                
                if monitor_index is not None:
                    # Capture only the specified monitor
                    if 0 <= monitor_index < len(monitors):
                        monitor = monitors[monitor_index]
                        screenshot_path = os.path.join(screenshot_dir, f"screen_{timestamp}_display_{monitor_index}.png")
                        
                        # Move mouse to center of monitor
                        center_x, center_y = self._get_monitor_center(monitor_index)
                        pyautogui.moveTo(center_x, center_y)
                        time.sleep(0.5)  # Wait for any hover effects
                        
                        screenshot = sct.grab(monitor)
                        mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)
                        
                        screenshot_paths = [screenshot_path]
                        print(colored(f"✅ Screenshot saved for display {monitor_index}: {screenshot_path}", "green"))
                        return screenshot_paths, 1
                    else:
                        print(colored(f"⚠️ Invalid monitor index {monitor_index}, falling back to single screenshot", "yellow"))
                else:
                    # Capture all monitors
                    for i, monitor in enumerate(monitors):
                        screenshot_path = os.path.join(screenshot_dir, f"screen_{timestamp}_display_{i}.png")
                        
                        # Move mouse to center of monitor
                        center_x, center_y = self._get_monitor_center(i)
                        pyautogui.moveTo(center_x, center_y)
                        time.sleep(0.5)  # Wait for any hover effects
                        
                        screenshot = sct.grab(monitor)
                        mss.tools.to_png(screenshot.rgb, screenshot.size, output=screenshot_path)
                        
                        screenshot_paths.append(screenshot_path)
                        print(colored(f"✅ Screenshot saved for display {i}: {screenshot_path}", "green"))
            
            num_displays = len(screenshot_paths)
            if num_displays == 0:
                # Fallback to single screenshot if no monitors detected
                screenshot_path = os.path.join(screenshot_dir, f"screen_{timestamp}_display_0.png")
                
                # Move mouse to center of primary screen
                center_x, center_y = self._get_monitor_center()
                pyautogui.moveTo(center_x, center_y)
                time.sleep(0.5)  # Wait for any hover effects
                
                pyautogui.screenshot(screenshot_path)
                screenshot_paths = [screenshot_path]
                num_displays = 1
                print(colored("⚠️ No monitors detected, falling back to single screenshot", "yellow"))
            
            return screenshot_paths, num_displays
        finally:
            # Restore original mouse position
            pyautogui.moveTo(original_x, original_y)

    async def _get_annotated_image_path(self, screenshot_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate annotated image from a screenshot and return its path along with screen elements."""
        if not self.parser:
            from OmniParser.omniparser import Omniparser, config as omniparser_config
            self.parser = Omniparser(omniparser_config)
        
        image, screen_elements = self.parser.parse(screenshot_path)
        
        # Generate the annotated image path
        base_path = screenshot_path.rsplit('.', 1)[0]
        annotated_path = f"{base_path}_annotated.png"
        
        # Save the annotated image
        image.save(annotated_path)
        print(f"✅ Annotated image saved to: {annotated_path}")
        
        # Transform screen elements into simplified format
        simplified_elements = []
        unlabelled_elements = []
        for element in screen_elements:
            simplified_element = {
                'shape': element.get('shape', ''),
                'content': eval(element.get('text', '')).get('content', '') if element.get('text') else element.get('content', ''),  # Try text first, fall back to content
            }
            
            # Convert shape coordinates to integers if present
            if 'shape' in element:
                simplified_element['shape'] = {
                    k: int(v) for k, v in element['shape'].items()
                }
            
            # verify if shape and content values are present
            if simplified_element.get('shape') and simplified_element.get('content'):
                simplified_elements.append(simplified_element)
            else:
                unlabelled_elements.append(element)
        
        # # handle elements without text content        
        # print(colored("🔍 Capturing elements without text content...", "cyan"))
        # # save the original mouse position
        # original_x, original_y = pyautogui.position()
        # # move the mouse to the center of the screen
        # center_x, center_y = self._get_monitor_center()
        # pyautogui.moveTo(center_x, center_y)
        # time.sleep(0.5)
        # for index, unlabelled_element in enumerate(unlabelled_elements):
        #     print(colored(f"🔍 Capturing elements progress: {index}/{len(unlabelled_elements)}", "cyan"))
        #     shape = unlabelled_element['shape']
        #     # screenshot the element region
        #     # move the mouse to the center of the element
        #     center_x = int(shape['x'] + (shape['width'] / 2))
        #     center_y = int(shape['y'] + (shape['height'] / 2))
        #     pyautogui.moveTo(center_x, center_y)
        #     region = (int(shape['x']), int(shape['y']), int(shape['width']), int(shape['height']))
        #     screenshot = pyautogui.screenshot(region=region)
        #     # save the screenshot as content as file and put the path in the content
        #     temp_path = os.path.join(g.PROJ_TEMP_STORAGE_PATH, "ui_tool", f"element_{index}.png")
        #     screenshot.save(temp_path)
        #     unlabelled_element["content"] = temp_path
        # # move the mouse to its original position
        # pyautogui.moveTo(original_x, original_y)
        # # replace base64 content with vlm analysis
        # print(colored("🔍 Analyzing elements with VLM...", "cyan"))
        # for index, unlabelled_element in enumerate(unlabelled_elements):
        #     print(colored(f"🔍 VLM analyzing elements progress: {index}/{len(unlabelled_elements)}", "cyan"))
        #     vlm_analysis = await self._analyze_element_with_vlm(unlabelled_element["content"])
        #     unlabelled_element["content"] = vlm_analysis
        # simplified_elements.extend(unlabelled_elements)
        
        return annotated_path, simplified_elements

    def _get_base64_image(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    async def _analyze_screen_with_vlm(self, screenshot_path: str, screen_index: int) -> Dict[str, Any]:
        """Analyze a single screen using LLM's vision capabilities."""
        base64_image = self._get_base64_image(screenshot_path)
        
        analysis_chat = Chat(debug_title=f"VLM Screenshot Analysis - Display {screen_index}")
        
        analysis_chat.add_message(
            Role.USER,
            f"This is display {screen_index}. Are any windows open and if so, what are their contents?.",
        )
        
        screen_description = LlmRouter.generate_completion(analysis_chat, base64_images=[base64_image])
        
        return {
            "status": "success",
            "summary": screen_description,
            "display": screen_index
        }
    
    async def _analyze_element_with_vlm(self, screenshot_path: Optional[str] = None) -> Dict[str, str]:
        """Analyze a single screen using LLM's vision capabilities.
        
        Args:
            screenshot_path: Path to image file to analyze
            base64_image: Base64 encoded image to analyze
            
        Returns:
            Dict containing analysis status and summary
        """
        # Create temp dir if it doesn't exist
        temp_dir = os.path.join(g.PROJ_TEMP_STORAGE_PATH, "ui_tool", "temp")
        
        # Resize image to 672x672 (llava hotfix)
        image = Image.open(screenshot_path)
        image.thumbnail((672, 672), Image.Resampling.LANCZOS)
        
        # Save resized image
        image.save(screenshot_path, "PNG")
        
        # Create analysis chat
        analysis_chat = Chat(debug_title=f"VLM Screenshot Analysis - Element")
        analysis_chat.add_message(
            Role.USER,
            f"This is an graphical ui element. What is it?. Be concise and to the point.",
        )
        
        # Get base64 of resized image
        with open(screenshot_path, "rb") as f:
            resized_base64 = base64.b64encode(f.read()).decode()
        
        # Get analysis
        screen_description = LlmRouter.generate_completion(analysis_chat, base64_images=[resized_base64])
        
        return {
            "status": "success",
            "summary": screen_description,
        }

    def _decide_next_action(self, reasoning: str, intent: str, screen_description: str) -> Dict[str, Any]:
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
            response = LlmRouter.generate_completion(parse_chat, strength=AIStrengths.TOOLUSE)
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

        # Include more detailed element information for better matching
        elements_list = "\n".join(
            f"{i}. {json.dumps(elem)}"
            for i, elem in enumerate(screen_elements)
        )
        
        match_chat.add_message(
            Role.USER,
            f"""You are an AI assistant that matches UI element descriptions to actual elements.
Your task is to find the best matching UI element based on semantic similarity and visual context.

Please pick the UI element that best matches the following target description: "{target_description}"
            
Available UI elements:
{elements_list}

Instructions:
1. Read the target description and come up with a list of keywords that may indicate the best match
2. List promising available UI elements and shortly explain why they are promising
3. Evaluate each element's relevance to the target description

After explicitly working through these steps, provide a final selection of the best matching element in the below JSON format:
{{
    "reasoning": "Selected index 5 because it has exact text match and its position is common for such clickable element",
    "best_match": 5
}}

Return the valid JSON object as shown, populated with the best matching element."""
        )
        
        try:
            response = LlmRouter.generate_completion(match_chat, strength=AIStrengths.TOOLUSE)
            parsed_response = extract_json(response, required_keys=["reasoning", "best_match"])
            
            if parsed_response is None:
                print(colored("No valid JSON found in response", "red"))
                return None
            
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

    def _select_relevant_screen(self, screen_analyses: List[Dict[str, Any]], intent: str) -> Optional[Dict[str, Any]]:
        """Select the most relevant screen based on the intent and screen analyses.
        
        Args:
            screen_analyses: List of screen analysis results
            intent: The user's intent
            
        Returns:
            The most relevant screen analysis or None if no relevant screen found
        """
        selection_chat = Chat(debug_title="Screen Selection")
        
        # Format screen descriptions for analysis
        screen_descriptions = "\n\n".join([
            f"Display {analysis['display']}:\n{analysis['summary']}"
            for analysis in screen_analyses
        ])
        
        selection_chat.add_message(
            Role.USER,
            f"""Based on the following intent and screen descriptions, you MUST select ONE screen that is most relevant for achieving the intent.

Intent: {intent}

Available Screens:
{screen_descriptions}

Analyze each screen's content and determine which ONE is most relevant for the intent, following these steps:
1. Review the intent
2. Review each screen's content in the context of the intent
3. Systematically evaluate each screen's relevance to the intent
4. Provide a chain of thought for your final selection

IMPORTANT: Respond with valid JSON only. Do not include comments in the JSON.
Your response must end with this exact format:
{{
    "selected_display": 0,
    "reasoning": "Detailed explanation of selection"
}}"""
        )
        
        try:
            response = LlmRouter.generate_completion(selection_chat, strength=AIStrengths.TOOLUSE)
            parsed_json = extract_json(response, required_keys=["selected_display", "reasoning"])
            
            if parsed_json is None:
                print(colored("No valid JSON found in screen selection response", "red"))
                return None
            
            selected_display = parsed_json["selected_display"]
            reasoning = parsed_json["reasoning"]
            
            # Find the selected screen analysis
            for analysis in screen_analyses:
                if analysis["display"] == selected_display:
                    print(colored(f"\nSelected Display {selected_display}: {reasoning}", "cyan"))
                    return analysis
            
            print(colored(f"Selected display {selected_display} not found in analyses", "red"))
            return None
            
        except Exception as e:
            print(colored(f"Error selecting relevant screen: {str(e)}", "red"))
            return None

    async def run(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the UI tool based on natural language intent."""
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: intent"
            )

        try:
            parameters = params["parameters"]
            reasoning = params.get("reasoning", "No specific reasoning provided")
            intent = parameters["intent"]
            action_counter = 0
            MAX_ACTIONS = 10
            action_results = []
            self.selected_screen_index = None  # Reset at start of execution

            # Initial capture of all screens
            screenshot_paths, num_displays = self._take_screenshot()
            
            # Analyze all screens initially
            screen_analyses = []
            for i, screenshot_path in enumerate(screenshot_paths):
                screen_analysis = await self._analyze_screen_with_vlm(screenshot_path, i)
                if screen_analysis["status"] != "success":
                    return self.format_response(
                        status="error",
                        summary=f"Failed to analyze screen {i}: {screen_analysis.get('error', 'Unknown error')}"
                    )
                screen_analyses.append(screen_analysis)

            # Select the most relevant screen once
            selected_screen = self._select_relevant_screen(screen_analyses, intent)
            if selected_screen is None:
                return self.format_response(
                    status="error",
                    summary="Failed to select a relevant screen"
                )
            
            # Store the selected screen index globally
            self.selected_screen_index = selected_screen["display"]
            screen_description = selected_screen["summary"]

            # Main action loop
            while action_counter < MAX_ACTIONS:
                # Parse intent for the selected screen
                parsed_intent = self._decide_next_action(reasoning, intent, screen_description)
                if parsed_intent is None:
                    return self.format_response(
                        status="success",
                        summary=f"Intent parsing failed, but screen analyses completed. Screen contents:\n{screen_description}"
                    )

                # Map action string to enum
                action_str = parsed_intent["action"]
                action = self._map_action_string_to_enum(action_str)
                if action is None:
                    return self.format_response(
                        status="error",
                        summary=f"Unsupported action: {action_str}. Screen contents:\n{screen_description}"
                    )

                # Get action handler
                handler = self._action_handlers.get(action)
                if handler is None:
                    return self.format_response(
                        status="error",
                        summary=f"No handler found for action: {action_str}. Screen contents:\n{screen_description}"
                    )

                # Execute action on the selected screen
                target = parsed_intent["target"]
                screenshot_paths, _ = self._take_screenshot(self.selected_screen_index)
                screenshot_path = screenshot_paths[0]  # Only one screenshot now
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

                # Take new screenshot of only the selected screen
                screenshot_paths, _ = self._take_screenshot(self.selected_screen_index)
                screenshot_path = screenshot_paths[0]  # Only one screenshot now
                
                # Analyze only the selected screen
                screen_analysis = await self._analyze_screen_with_vlm(screenshot_path, self.selected_screen_index)
                if screen_analysis["status"] != "success":
                    return self.format_response(
                        status="error",
                        summary=f"Failed to analyze screen after action: {screen_analysis.get('error', 'Unknown error')}"
                    )
                screen_description = screen_analysis["summary"]

                action_counter += 1
                time.sleep(0.5)  # Allow UI to update

            # Max actions reached
            final_summary = self._format_action_results(action_results)
            return self.format_response(
                status="error",
                summary=f"Maximum actions ({MAX_ACTIONS}) reached without completing task.\nActions performed:\n{final_summary}\nLast screen state:\n{screen_description}"
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