import datetime
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple, Any, Dict
import venv

from termcolor import colored
from py_classes.ai_providers.cls_ollama_interface import OllamaClient, ToolCall, ollama_convert_method_to_tool
from py_classes.cls_chat import Chat, Role
from py_classes.cls_few_shot_factory import FewShotProvider
from py_classes.cls_llm_router import LlmRouter
from py_classes.globals import g

class AgenticPythonProcess:
    def __init__(self, llm_key: str = "llama-3.1-70b-versatile"):
        self.llm_key = llm_key
        self.top_level_chat = Chat("You are a the commander component of a system of llm agents, tasked with implementing strategic Python methods to increase the system's capability for the user. Your goal is to analyze the system's memory state, explore ideas, envision strategies, and identify the most appropriate chain of actions to achieve the desired outcome.")

    def run(self, user_request: Optional[str] = None):
        print(colored("Starting AgenticAI run...", "cyan", attrs=["bold"]))
        for i in range(0, 3):
            try:
                # Step -1: Create Sandbox backup
                print(colored("Step 0/6: ", "yellow") + "Creating sandbox backup...")
                if os.path.exists(g.PROJ_AGENTIC_SANDBOX_BACKUP_PATH):
                    shutil.rmtree(g.PROJ_AGENTIC_SANDBOX_BACKUP_PATH)
                shutil.copytree(g.PROJ_AGENTIC_SANDBOX_PATH, g.PROJ_AGENTIC_SANDBOX_BACKUP_PATH)
                print(colored("✓ Sandbox backup created", "green"))

                # Step 1: Get and reason about memory
                print(colored("Step 1/6: ", "yellow") + "Reasoning about agent's memory...")
                memory_reasoning, reasoning_chat = self._LLM_memory_reasoning()
                print(colored("✓ Memory retrieved and reasoning complete", "green"))

                # Step 2: Get available methods
                print(colored("Step 2/6: ", "yellow") + "Reasoning about agent's available actions...")
                available_methods = self._get_available_methods()
                action_reasoning, reasoning_chat = self._LLM_action_reasoning(available_methods, reasoning_chat)
                print(colored("✓ Methods listed", "green"))
                
                print(colored("Step 3/6: ", "yellow") + "Envisioning strategy...")
                max_retries = 3
                while True:
                    max_retries -= 1
                    if max_retries == 0:
                        raise RuntimeError(f"Failed to get envision acceptable goal directed strategy after {max_retries} attempts")
                    print(colored("Step 3.1/6: ", "yellow") + "Envisioning goal directed strategy...")
                    reasoning_chat.add_message(Role.USER, "Please provide a goal directed strategy that the system should follow to achieve the desired outcome. Your suggested strategy must be implementable using consecutive Python methods. For now, only provide a deailed outline of the strategy.")
                    strategy_outline = LlmRouter.generate_completion(reasoning_chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
                    reasoning_chat.add_message(Role.ASSISTANT, strategy_outline)
                    print(colored("Step 3.2/6: ", "yellow") + "Critiquing vision...")
                    critique_chat = Chat("As a critical component in a system of llm agents, your task is to evaluate the proposed strategy. Review the system's memory state to determine if this is a viable course of action that aligns with the users intent. Respond in a chain of thought in natural language.")
                    critique_chat.add_message(Role.USER, f"# # # SYSTEM MEMORY # # #\n{memory_reasoning}\n\n# # # PROPOSED STRATEGY # # #\n{strategy_outline}")
                    critique_response = LlmRouter.generate_completion(critique_chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
                    accept_strategy = FewShotProvider.few_shot_YesNo(f"Is the proposed strategy acceptable? Please respond with 'yes' or 'no'.\n\n{critique_response}")
                    if accept_strategy:
                        break
                    reasoning_chat.add_message(Role.USER, critique_response)
                    
                print(colored("Step 4/6: ", "yellow") + "Choosing action...")
                print(colored("Step 5/6: ", "yellow") + "Executing action...")
                print(colored("Step 6/6: ", "yellow") + "Memorizing vision and strategy state...")
                
                
                # # # PERCEPTION END # # #
                
                # # # DIVERGENCE START # # #
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
                # # # DIVERGENCE END # # #
                
                # # # ACTION START # # #
                # Step 5: Generate method parameters
                print(colored("Step 5/8: ", "yellow") + "Generating method parameters...")
                method_args = self._LLM_generate_method_parameters(memory_reasoning, picked_method)
                print(colored("✓ Method parameters generated", "green"))

                # Step 6: Execute method
                print(colored("Step 6/8: ", "yellow") + "Executing method...")
                method_output = self._execute_method(picked_method, method_args)
                print(colored("✓ Method executed", "green"))
                # # # ACTION END # # #

                # # # PERCEPTION START # # #
                # Step 7: Reflect on method output
                print(colored("Step 7/8: ", "yellow") + "Reflecting on method output...")
                chat = self._reflect_on_method_output(method_output)
                print(colored("✓ Reflection complete", "green"))
            except Exception as e:
                chat = Chat("The agentic ai system has encountered an exception and is unable to continue. Please review the error message and take appropriate note.")
                chat.add_message(Role.USER, f"Here's the error message:\n{e}")
                print(colored(f"FATAL SYSTEM EXCEPTION: {e}", "yellow"))
                # Agent Exception Handling Step: Restoring Sandbox backup
                print(colored("Step X: Restoring Sandbox backup", "yellow"))
                shutil.rmtree(g.PROJ_AGENTIC_SANDBOX_PATH)
                shutil.copytree(g.PROJ_AGENTIC_SANDBOX_BACKUP_PATH, g.PROJ_AGENTIC_SANDBOX_PATH)
                print(colored("✓ Sandbox backup restored", "green"))
            # Step 8: Update memory
            print(colored("Step 8/8: ", "yellow") + "Updating memory...")
            self._update_memory(chat)
            print(colored("✓ Memory updated and saved", "green"))
            
        print(colored("AgenticAI run completed successfully!", "cyan", attrs=["bold"]))

    def _LLM_memory_reasoning(self) -> Tuple[str, Chat]:
        memory: List[Tuple[str, str, str]] = self._STATE_get_memory()
        memory_str = "\n".join([f"Timestamp: {datetime} Title: {title} Content: {contents}" for datetime, title, contents in memory])
        chat = Chat("You are part of a larger system of llm agents, tasked with reviewing the system's current memory state. The system's goal is to implement strategic Python methods to increase its helpfulness to the user. Analyze the memory to understand how it relates to the latest system state and how to proceed accordingly.")
        chat.add_message(Role.USER, f"Please review the system's memory and provide a summary of the most relevant information for the current system state. \n# # # SYSTEM MEMORY # # #\n{memory_str}")
        response = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
        return response, chat

    def _LLM_action_reasoning(self, available_methods: List[str], reasoning_chat: Chat) -> Tuple[str, Chat]:
        available_actions_str = "\n".join(available_methods)
        reasoning_chat.add_message(Role.USER, f"Interesting, consider the following available actions that are available to the system, please review the if any stand out as currently relevant:\n{available_actions_str}")
        response = LlmRouter.generate_completion(reasoning_chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
        return response, reasoning_chat

    def _get_available_methods(self) -> List[str]:
        sandbox_methods: List[str] = self._STATE_list_available_methods()
        atomic_methods: List[str] = self._STATE_list_available_methods("implement_new_method", os.path.join(g.PROJ_AGENTIC_PATH, "atomic_tools.py"))
        return sandbox_methods + atomic_methods

    def _LLM_pick_relevant_methods(self, memory_reasoning: str, tools: List[str], pick_count: int = 1) -> Tuple[str, str]:
        tools_str = "\n".join(tools)
        chat = Chat("You are a component in a system of llm agents, responsible for selecting methods based on the system's current memory state. Your goal is to choose the most appropriate Python method to enhance the system's ability to assist the user.")
        prompt = f"Please review the system's memory and choose the single best suited method for the next action. Please provide a reasoning about the context before providing your choice."
        chat.add_message(Role.USER, prompt + f"\n # # # SYSTEM MEMORY # # #\n{memory_reasoning}\n\n# # # AVAILABLE METHODS # # #\n{tools_str}")
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
        chat = Chat("As a critical component in a system of llm agents, your task is to evaluate the chosen method for the next action. Review the system's memory state and available methods to determine if this is truly the optimal course of action to increase helpfulness to the user.")
        prompt = f"Please review the chosen method and the reasoning behind it. Decide if this is truly the best course of action given the current system memory and available methods. If you believe this is the best action, respond with 'ACCEPT'. If you believe a better action is possible, respond with 'REJECT'. Provide your reasoning for this decision.\n\nChosen Method: {picked_method}\nReasoning: {method_reasoning}\n\nMemory Reasoning: {memory_reasoning}\n\nAvailable Methods: {', '.join(available_methods)}"
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
        chat = Chat("You are an advanced component in a system of llm agents, capable of implementing new Python methods or selecting existing ones. Based on the system's current memory state and goals, decide if implementing a new method or choosing an existing one would best serve to increase the system's helpfulness to the user.")
        prompt = f"""Based on the critique and the current system memory, decide if implementing a new Python method would be a better and viable alternative for the system.

Critique Reasoning: {critique_reasoning}

Memory Reasoning: {memory_reasoning}
"""

        chat.add_message(Role.USER, prompt)
        response = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
        implement_new_method = FewShotProvider.few_shot_YesNo(f"Should a new method be implemented? Please respond with 'yes' or 'no'. \n\n{response})")
        if implement_new_method:
            return "implement_new_method"
        else:
            return self._LLM_pick_relevant_methods(response, self._STATE_list_available_methods(), 1)[0]

    def _update_requirements(self, method_description: str, requirements_file_path: str) -> None:
        # write requirments.txt
        new_requirements = FewShotProvider.few_shot_toPythonRequirements(method_description, preferred_model_keys=self.llm_key)
        self._update_requirements(new_requirements, os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "requirements.txt"))
        # Read existing requirements
        existing_requirements: Dict[str, str] = {}
        if os.path.exists(requirements_file_path):
            with open(requirements_file_path, "r") as f:
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
        with open(requirements_file_path, "w") as f:
            for req in existing_requirements.values():
                f.write(req + "\n")

    def _LLM_generate_method_parameters(self, memory_reasoning: str, picked_method: str) -> dict:
        try:
            if picked_method == "implement_new_method":
                method_full = self._STATE_list_available_methods("implement_new_method", os.path.join(g.PROJ_AGENTIC_PATH, "atomic_tools.py"), include_logic=True)[0]
            else:
                method_full = self._STATE_list_available_methods(picked_method, include_logic=True)[0]
        except IndexError as e:
            raise IndexError(f"Method '{picked_method}' was not found in available methods")

        # Normalize indentation
        lines = method_full.split('\n')
        min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
        method_full = '\n'.join(line[min_indent:] if line.strip() else '' for line in lines)

        chat = Chat("As a specialized component in a system of llm agents, your role is to generate the required parameters for the chosen method. Consider the system's current memory state and the overall goal of implementing strategic Python methods to increase helpfulness to the user.")
        chat.add_message(Role.USER, "I will provide you with a method which contains the method implementation, afterwards I will provide you with a description of what we want to achieve using this method. Only then you will be required to generate the required parameters in a WELL FORMED JSON for the method. Answer with 'READY' if you understand.")
        chat.add_message(Role.ASSISTANT, "READY")
        chat.add_message(Role.USER, f"Here's the task description, answer with ready if you're ready for the method we want to use to proceed according to plan. {memory_reasoning}")
        chat.add_message(Role.ASSISTANT, "READY")
        chat.add_message(Role.USER, "Provide parameters for the following method, respond only in this WELL FORMED JSON TEMPLATE, ONLY PROVIDE A SINGLE OBJECT: {\"param1_title\": \"param1_value\", \"param2_title\": \"param2_value\", ...}\n\n" + method_full)
        response_json_str = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
        json_start = response_json_str.find('{')
        json_end = response_json_str.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            args_dict = json.loads(response_json_str[json_start:json_end])
            return args_dict
        raise ValueError("Failed to generate valid JSON for method parameters")

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
            try:
                venv.create(env_path, with_pip=True)
            except Exception as e:
                print(colored(f"Error creating virtual environment: {e}", "red"))
                raise

        # Install requirements
        if sys.platform == "win32":
            pip_path = os.path.join(env_path, "Scripts", "pip")
        else:
            pip_path = os.path.join(env_path, "bin", "pip")

        requirements_file = os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "requirements.txt")
        if os.path.exists(requirements_file):
            try:
                subprocess.run([pip_path, "install", "-r", requirements_file], check=True)
            except subprocess.CalledProcessError as e:
                raise Exception(f"Error installing requirements: {e}")
            print(colored(f"Requirements file not found at {requirements_file}", "yellow"))

        # Prepare the Python executable path
        if sys.platform == "win32":
            python_executable = os.path.join(env_path, "Scripts", "python.exe")
        else:
            python_executable = os.path.join(env_path, "bin", "python")

        if not os.path.exists(python_executable):
            raise FileNotFoundError(f"Python executable not found at {python_executable}")

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
        chat = Chat("As part of the system's reflection process, analyze the output of the executed method in the context of increasing helpfulness to the user.")
        method_output_str = str(method_output)
        if len(method_output_str) < 16384:
            chat.add_message(Role.IPYTHON, method_output_str)
        else:
            chat.add_message(Role.IPYTHON, f"Warning, output too large to display in full. Shortening to first 4096 characters:\n{method_output_str[:4096]}...")
        method_reflection = LlmRouter.generate_completion(chat, preferred_model_keys=[self.llm_key], force_preferred_model=True)
        chat.add_message(Role.ASSISTANT, method_reflection)
        return chat

    def _update_memory(self, chat: Chat):
        chat.add_message(Role.USER, "As part of the system's memory management, you need to update the system's memory with new information. Based on the current system state and the actions taken, provide a title and content for a new memory entry that reflects the progress in implementing strategic Python methods to increase helpfulness to the user. Please provide a title and the content for the new memory in the following WELL FORMED JSON FORMAT, using this TEMPLATE: {\"title\": \"new_memory_title_string\", \"content\": \"new_memory_content_string\"}")
        
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
            with open(memory_file, 'w') as f:
                json.dump([], f)
            return []

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