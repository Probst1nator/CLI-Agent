import datetime
import importlib
import inspect
import json
import os
import subprocess
import sys
from typing import List, Optional, Tuple, Any, Dict
import venv

from termcolor import colored
from classes.ai_providers.cls_ollama_interface import OllamaClient, ToolCall, ollama_convert_method_to_tool
from classes.cls_chat import Chat, Role
from classes.cls_few_shot_factory import FewShotProvider
from classes.cls_llm_router import LlmRouter
from globals import g

class AgenticPythonProcess:
    def __init__(self, llm_key: str = "llama-3.1-70b-versatile"):
        self.llm_key = llm_key

    def run(self, user_request: Optional[str] = None):
        print(colored("Starting AgenticAI run...", "cyan", attrs=["bold"]))
        
        # Step 0: Handle user request
        if user_request:
            print(colored("Step 0/8: ", "yellow") + "Adding user request to memory...")
            self._STATE_update_memory("user_request", user_request)
            print(colored("✓ User request added to memory", "green"))

        # Step 1: Get and reason about memory
        print(colored("Step 1/8: ", "yellow") + "Getting and reasoning about agent's memory...")
        memory_reasoning = self._get_and_reason_about_memory()
        print(colored("✓ Memory retrieved and reasoning complete", "green"))

        # Step 2: Get available methods
        print(colored("Step 2/8: ", "yellow") + "Listing available methods...")
        available_methods = self._get_available_methods()
        print(colored("✓ Methods listed", "green"))

        # Step 3: Pick method
        print(colored("Step 3/8: ", "yellow") + "Picking method...")
        picked_method, reasoning = self._LLM_pick_relevant_methods(memory_reasoning, available_methods, 1)
        print(colored(f"✓ Method selected: {picked_method}", "green"))

        # Step 4: Critique action
        print(colored("Step 4/8: ", "yellow") + "Critiquing action...")
        action_accepted, critique_reasoning = self._critique_action(memory_reasoning, picked_method, reasoning, available_methods)
        if not action_accepted:
            picked_method = self._handle_rejected_action(critique_reasoning, memory_reasoning)
        print(colored("✓ Action critique complete", "green"))

        # Step 5: Generate method parameters
        print(colored("Step 5/8: ", "yellow") + "Generating method parameters...")
        method_args = self._generate_method_parameters(memory_reasoning, picked_method)
        print(colored("✓ Method parameters generated", "green"))

        # Step 6: Execute method
        print(colored("Step 6/8: ", "yellow") + "Executing method...")
        method_output = self._execute_method(picked_method, method_args)
        print(colored("✓ Method executed", "green"))

        # Step 7: Reflect on method output
        print(colored("Step 7/8: ", "yellow") + "Reflecting on method output...")
        chat = self._reflect_on_method_output(method_output)
        print(colored("✓ Reflection complete", "green"))

        # Step 8: Update memory
        print(colored("Step 8/8: ", "yellow") + "Updating memory...")
        self._update_memory(chat)
        print(colored("✓ Memory updated and saved", "green"))

        print(colored("AgenticAI run completed successfully!", "cyan", attrs=["bold"]))

    def _get_and_reason_about_memory(self) -> str:
        memory: List[Tuple[str, str, str]] = self._STATE_get_memory()
        memory_str = "\n".join([f"Timestamp: {datetime} Title: {title} Content: {contents}" for datetime, title, contents in memory])
        chat = Chat("You are a helpful agentic ai assistant. To provide optimal assistance you're currently reflecting on your memory. Think step by step to understand how your memory relates to your latest request and how to proceed accordingly. You are fully autonomous and can not interact with the user. Your only ability is to imagine strategic advanced python methods for task completion. Do not implement such methods yet, only describe the required task completion steps.")
        chat.add_message(Role.USER, f"Please review your memory and provide a summary of the most relevant information for the latest user request. \n# # # YOUR MEMORY # # #\n{memory_str}")
        return LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)

    def _get_available_methods(self) -> List[str]:
        sandbox_methods: List[str] = self._STATE_list_available_methods()
        atomic_methods: List[str] = self._STATE_list_available_methods("implement_new_method", os.path.join(g.PROJ_AGENTIC_PATH, "atomic_tools.py"))
        return sandbox_methods + atomic_methods

    def _LLM_pick_relevant_methods(self, memory_reasoning: str, tools: List[str], pick_count: int = 1) -> Tuple[str, str]:
        tools_str = "\n".join(tools)
        chat = Chat("You are a helpful agentic ai assistant. To provide optimal assistance you're currently reflecting on your memory and available methods for your next action. Think step by step to understand which methods provide the best tooling for your latest intentions.")
        prompt = f"Please review your memory and choose the single best suited method for your next action. Provide your reasoning for this choice."
        chat.add_message(Role.USER, prompt + f"\n # # # YOUR MEMORY # # #\n{memory_reasoning}\n\n# # # YOUR AVAILABLE METHODS # # #\n{tools_str}")
        response = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
        chat.add_message(Role.ASSISTANT, response)
        chat.add_message(Role.USER, "Please provide your method choice and reasoning in the following WELL FORMED JSON FORMAT, using this TEMPLATE: {\"method\": \"method_name\", \"reasoning\": \"Your reasoning here\"}\nDO NOT ADD ANY METHODS THAT WERE NOT SPECIFIED IN THE AVAILABLE METHODS. DO NOT INCLUDE PARAMS IN THE JSON.")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                method_json = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
                json_start = method_json.find('{')
                json_end = method_json.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    method_data = json.loads(method_json[json_start:json_end])
                    if "method" in method_data and "reasoning" in method_data:
                        return method_data["method"], method_data["reasoning"]
                raise ValueError("Invalid JSON structure")
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    chat.add_message(Role.USER, f"The previous response was not valid JSON or didn't contain the required fields. Please try again and ensure the response is in the correct JSON format with 'method' and 'reasoning' fields.")
                else:
                    raise ValueError(f"Failed to get a valid JSON response after {max_retries} attempts")

    def _critique_action(self, memory_reasoning: str, picked_method: str, method_reasoning: str, available_methods: List[str]) -> Tuple[bool, str]:
        chat = Chat("You are a critical AI assistant tasked with evaluating the chosen method for the next action. Your job is to thoroughly analyze the decision and determine if it's the best course of action.")
        prompt = f"Please review the chosen method and the reasoning behind it. Decide if this is truly the best course of action given the current memory and available methods. If you believe this is the best action, respond with 'ACCEPT'. If you believe a better action is possible, respond with 'REJECT'. Provide your reasoning for this decision.\n\nChosen Method: {picked_method}\nReasoning: {method_reasoning}\n\nMemory Reasoning: {memory_reasoning}\n\nAvailable Methods: {', '.join(available_methods)}"
        chat.add_message(Role.USER, prompt)
        critique = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
        chat.add_message(Role.ASSISTANT, critique)
        chat.add_message(Role.USER, "Please provide your decision and reasoning in the following WELL FORMED JSON FORMAT, using this TEMPLATE: {\"decision\": \"ACCEPT/REJECT\", \"reasoning\": \"Your reasoning here\"}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                decision_json = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
                json_start = decision_json.find('{')
                json_end = decision_json.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    decision_data = json.loads(decision_json[json_start:json_end])
                    if "decision" in decision_data and "reasoning" in decision_data:
                        return decision_data["decision"] == "ACCEPT", decision_data["reasoning"]
                raise ValueError("Invalid JSON structure")
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    chat.add_message(Role.USER, f"The previous response was not valid JSON or didn't contain the required fields. Please try again and ensure the response is in the correct JSON format with 'decision' and 'reasoning' fields.")
                else:
                    raise ValueError(f"Failed to get a valid JSON response after {max_retries} attempts")

    def _handle_rejected_action(self, critique_reasoning: str, memory_reasoning: str) -> str:
        chat = Chat("You are an advanced AI assistant capable of implementing new Python methods to achieve goals. You need to decide if implementing a new method could help achieve the current goal.")
        prompt = f"""Based on the critique and the current memory, decide if implementing a new Python method could help achieve the goal. If yes, implement the method. If no, choose the next best available method.

Critique Reasoning: {critique_reasoning}

Memory Reasoning: {memory_reasoning}

If you decide to implement a new method, please provide a complete Python method implementation. The method should be fully functional and ready to use. The implementation should include the imports and the method, without any classes or usage examples.

If you decide to use an existing method, please provide the name of the method you recommend using instead."""

        chat.add_message(Role.USER, prompt)
        response = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
        chat.add_message(Role.ASSISTANT, response)
        chat.add_message(Role.USER, """Please provide your decision and the new method (if applicable) in the following WELL FORMED JSON FORMAT, using this TEMPLATE:
{
    "decision": "IMPLEMENT_NEW" or "USE_EXISTING",
    "method": "If IMPLEMENT_NEW, provide the full method implementation. If USE_EXISTING, provide the method name."
}
Ensure that the JSON is properly formatted and escaped.""")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                decision_json = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
                json_start = decision_json.find('{')
                json_end = decision_json.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    decision_data = json.loads(decision_json[json_start:json_end])
                    if "decision" in decision_data and "method" in decision_data:
                        break
                else:
                    raise ValueError("No valid JSON object found in the response")
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    chat.add_message(Role.USER, f"The previous response was not valid JSON. Please try again and ensure the response is in the correct JSON format.")
                else:
                    raise ValueError(f"Failed to get a valid JSON response after {max_retries} attempts")

        if decision_data["decision"] == "IMPLEMENT_NEW":
            method_name = self._implement_new_method(decision_data["method"])
            return method_name
        else:
            return decision_data["method"]

    def _update_requirements(self, new_requirements: str, file_path: str) -> None:
        # Read existing requirements
        existing_requirements: Dict[str, str] = {}
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        package = line.split("==")[0]
                        existing_requirements[package] = line

        # Update with new requirements
        for req in new_requirements.split("\n"):
            req = req.strip()
            if req and not req.startswith("#"):
                package = req.split("==")[0]
                existing_requirements[package] = req

        # Write updated requirements
        with open(file_path, "w") as f:
            for req in existing_requirements.values():
                f.write(req + "\n")

    def _implement_new_method(self, method_implementation: str) -> str:
        # write requirments.txt
        requirements = FewShotProvider.few_shot_toPythonRequirements(method_implementation, preferred_model_keys=self.llm_key)
        self._update_requirements(requirements, os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "requirements.txt"))
        
        # Extract method name from the implementation
        method_name = method_implementation.split("def ")[1].split("(")[0].strip()
        
        tools_file_path = os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "tools.py")
        with open(tools_file_path, 'a') as file:
            file.write(f"\n\n{method_implementation}")
        print(colored(f"New method '{method_name}' implemented and added to tools.py", "green"))
        return method_name

    def _generate_method_parameters(self, memory_reasoning: str, picked_method: str) -> dict:
        if picked_method == "implement_new_method":
            method_full = self._STATE_list_available_methods("implement_new_method", os.path.join(g.PROJ_AGENTIC_PATH, "atomic_tools.py"), include_logic=True)[0]
        else:
            method_full = self._STATE_list_available_methods(picked_method, include_logic=True)[0]

        # Normalize indentation
        lines = method_full.split('\n')
        min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
        method_full = '\n'.join(line[min_indent:] if line.strip() else '' for line in lines)

        chat = Chat("You are a method parameter generator AI assistant. Your task is to generate the required parameters for the chosen method.")
        chat.add_message(Role.USER, "I will provide you with a metho which contains the method implementation, afterwards i will provide you with a description of what we want to achieve using this method. Only then you will be required to generate the required parameters in a WELL FORMED JSON for the method. Answer with 'READY' if you understand.")
        chat.add_message(Role.ASSISTANT, "READY")
        chat.add_message(Role.USER, f"Here's the task description, answer with ready if you're ready for the method we want to use to proceed according to plan. {memory_reasoning}")
        chat.add_message(Role.ASSISTANT, "READY")
        chat.add_message(Role.USER, "Provide parameters for the following method, respond only in this WELL FORMED JSON TEMPLATE: \{\"param1_title\": \"param1_value\", \"param2_title\": \"param2_value\", ...}\n\n" + method_full)
        response_json_str = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
        json_start = response_json_str.find('{')
        json_end = response_json_str.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            args_dict = json.loads(response_json_str[json_start:json_end])
            return args_dict
        
        # try:
        #     ollama_tool = ollama_convert_method_to_tool(method_full)
        # except IndentationError as e:
        #     print(f"IndentationError in method string: {e}")
        #     print("Method string causing the error:")
        #     print(method_full)
        #     raise

        # tool_calls = OllamaClient().generate_response(chat=chat, model="llama3.1", tools=[ollama_tool])
        # assert all(isinstance(call, ToolCall) for call in tool_calls)
        
        # generated_arguments = tool_calls[0].arguments
        # return generated_arguments

    def _execute_method(self, method_title: str, method_args: dict) -> Any:
        if method_title == "implement_new_method":
            file_path = os.path.join(g.PROJ_AGENTIC_PATH, "atomic_tools.py")
        else:
            file_path = os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "tools.py")

        dir_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        module_name = os.path.splitext(file_name)[0]

        # Create and manage Python environment
        env_path = os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "env")
        if not os.path.exists(env_path):
            print(colored("Creating new Python environment...", "yellow"))
            venv.create(env_path, with_pip=True)

        # Install requirements
        pip_path = os.path.join(env_path, "bin", "pip")
        if sys.platform == "win32":
            pip_path = os.path.join(env_path, "Scripts", "pip")
        subprocess.run([pip_path, "install", "-r", os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "requirements.txt")], check=True)

        # Prepare the Python executable path
        if sys.platform == "win32":
            python_executable = os.path.join(env_path, "Scripts", "python.exe")
        else:
            python_executable = os.path.join(env_path, "bin", "python")

        # Prepare the command to run the method
        command = [
            python_executable,
            "-c",
            f"import sys; sys.path.append('{dir_path}'); "
            f"import {module_name}; "
            f"print({module_name}.{method_title}(**{method_args}))"
        ]

        try:
            # Run the command in a subprocess
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(colored(f"Error executing method: {e}", "red"))
            print("Standard Error:")
            print(e.stderr)
            raise
        except Exception as e:
            print(colored(f"Unexpected error: {e}", "red"))
            raise

    def _reflect_on_method_output(self, method_output: Any) -> Chat:
            chat = Chat()
            method_output_str = str(method_output)
            if len(method_output_str) < 16384:
                chat.add_message(Role.IPYTHON, method_output_str)
            else:
                chat.add_message(Role.IPYTHON, f"Warning, output too large to display in full. Shortening to first 4096 characters:\n{method_output_str[:4096]}...")
            method_reflection = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
            chat.add_message(Role.ASSISTANT, method_reflection)
            return chat

    def _update_memory(self, chat: Chat):
        chat.add_message(Role.USER, "You are now required to save progress with a new memory. Please provide a title and the content for the new memory in the following WELL FORMED JSON FORMAT, using this TEMPLATE: {\"title\": \"new_memory_title_string\", \"content\": \"new_memory_content_string\"}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                new_memory_str = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
                json_start = new_memory_str.find('{')
                json_end = new_memory_str.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    new_memory_data = json.loads(new_memory_str[json_start:json_end])
                    if "title" in new_memory_data and "content" in new_memory_data:
                        self._STATE_update_memory(new_memory_data["title"], new_memory_data["content"])
                        return
                raise ValueError("Invalid JSON structure")
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    chat.add_message(Role.USER, f"The previous response was not valid JSON or didn't contain the required fields. Please try again and ensure the response is in the correct JSON format with 'title' and 'content' fields.")
                else:
                    raise ValueError(f"Failed to get a valid JSON response after {max_retries} attempts")

    def _STATE_get_memory(self) -> list:
        memory_file: str = os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "memory.json")
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                return json.load(f)
        else:
            print(colored("No stored memory found. Please provide an initial request or task:", "yellow"))
            user_input: str = input(colored("> ", "cyan")).strip()
            initial_memory = []
            self._STATE_update_memory("user_request", user_input, initial_memory)
            return initial_memory

    def _STATE_update_memory(self, memory_title: str, memory_contents: str, memory_data: Optional[list] = None) -> None:
        if not memory_title or not memory_contents:
            print(colored("Error: Memory title and contents cannot be empty.", "red"))
            raise ValueError(f"Memory title and contents cannot be empty: memory_title={memory_title}, memory_contents={memory_contents}")

        if memory_data is None:
            memory_data = self._STATE_get_memory()

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        memory_data.append((current_datetime, memory_title, memory_contents))

        memory_file: str = os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "memory.json")
        with open(memory_file, 'w') as f:
            json.dump(memory_data, f, indent=4)

    def _STATE_list_available_methods(self, method_title: Optional[str] = None, non_default_tools_file_path: Optional[str] = None, include_logic: bool = False) -> List[str]:
        if non_default_tools_file_path:
            tools_file_path = non_default_tools_file_path
        else:
            tools_file_path = os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "tools.py")
        
        with open(tools_file_path, 'r') as file:
            lines = file.readlines()

        methods: List[str] = []
        current_method: List[str] = []
        in_method = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("def "):
                if in_method:
                    methods.append("".join(current_method))
                    current_method = []
                in_method = True
                current_method.append(line)
                
                if method_title and not stripped_line.startswith(f"def {method_title}("):
                    in_method = False
                    current_method = []
            elif in_method:
                current_method.append(line)
                if stripped_line.startswith('"""') and not include_logic:
                    for docstring_line in current_method[1:]:
                        if docstring_line.strip().endswith('"""'):
                            current_method.append(docstring_line)
                            break
                    methods.append("".join(current_method))
                    in_method = False
                    current_method = []

        if in_method:
            methods.append("".join(current_method))

        return methods

if __name__ == "__main__":
    agent = AgenticPythonProcess()
    agent.run()