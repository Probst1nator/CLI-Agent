import subprocess
import os
import sys
import json
from collections import defaultdict
import re
from typing import Any, List, Dict, Tuple, Optional, DefaultDict
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from termcolor import colored
from py_agents.assistants import code_assistant
from py_classes.cls_chat import Chat, Role
from py_classes.cls_few_shot_factory import FewShotProvider
from py_classes.cls_llm_router import LlmRouter
from py_methods import tooling

@dataclass
class MakeError:
    file_path: str
    line: int
    full_message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

class BaseAssistantAgent(ABC):
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        pass

class MakeErrorCollectorAgent(BaseAssistantAgent):
    def execute(self, command_to_gen_errors: List[str], context_file_path: str) -> bool:
        # Initialize variables
        full_output: str = ""
        
        # Run the make command and process output
        process: subprocess.Popen = subprocess.Popen(
            command_to_gen_errors,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )

        output_lines: List[str] = []
        for line in iter(process.stdout.readline, ''):
            print(line, end='')  # Print the line in real-time
            output_lines.append(line.strip())

        process.stdout.close()
        return_code: int = process.wait()

        # If make command was successful, return True
        if return_code == 0:
            print(colored("Make Agent Execution has finished successfully!", "green"))
            return True

        # If make failed, process the errors
        full_output = '\n'.join(output_lines)
        error_output: str = full_output[full_output.find('error'):full_output.rfind('error')]
        
        error_example1 = {"file_path": "/home/prob/repos/python_project/image_processor.py", "line": "75", "full_message": "ImportError: No module named 'PIL'"}
        error_example2 = {"file_path": "/home/prob/repos/rust_project/src/main.rs", "line": "42", "full_message": "mismatched types: expected &str, found String"}
        error_example3 = {"file_path": "/home/prob/repos/rust_project/src/main.rs", "line": "80", "full_message": "no method named `unwrap` found for type `std::result::Result<_, _>` in the current scope"}
        error_example4 = {"file_path": "/home/prob/repos/aidev/src/robotObserver.cpp", "line": "26", "full_message": "error: 'btVector3' was not declared in this scope; did you forget to include <bullet/LinearMath/btVector3.h>?"}
        error_example5 = {"file_path": "/home/prob/repos/aidev/src/robotSimulation.cpp", "line": "18", "full_message": "error: 'btQuaternion' is not a member of 'btQuaternion'; did you mean 'btQuaternion::btQuaternion'?"}
        error_example5 = {"file_path": "/home/prob/repos/aidev/src/robotSimulation.cpp", "line": "25", "full_message": " error: no declaration matches ‘void RobotCommandAdaptor::handleRobotResponse(const QList<robot::StatusPayload>&)’"}
        error_example6 = {"file_path": "/home/prob/repos/aidev/src/theoreticalPhysics.cpp", "line": "42", "full_message": "error: invalid operands of types 'double' and 'const char [6]' to binary 'operator/'"}
        returned_object = FewShotProvider.few_shot_objectFromTemplate([{"found_errors": [error_example1]}, {"found_errors": [error_example2, error_example3]}, {"found_errors": [error_example4, error_example5, error_example6]}], f"```\n{error_output}\n```\n\nPlease extract all the errors which include their source file path from the output and provide them as a list of objects as specified in the template. Only include errors that stem from the project, ignore non-trivial system errors and errors which stem from the build directory.", preferred_models=["llama-3.1-70b-versatile"], silent=False)
        returned_object_list: List[Dict[str, str]] = self.normalize_few_shot_object(returned_object)
        
        # filter objects
        returned_object_list = [obj for obj in returned_object_list if obj['file_path'] and obj['line'] and obj['full_message']] # all fields must be present and populated
        returned_object_list = [obj for obj in returned_object_list if os.path.exists(obj['file_path'])] # remove entries which file does not exist
        returned_object_list = [obj for obj in returned_object_list if 'autogen' not in obj['file_path']] # # remove entries which contain 'autogen' in the path
        
        if not returned_object_list:
            raise Exception("No valid errors were found in the output.")
        
        error_dicts: List[Dict[str, str]] = self.process_make_errors(returned_object_list)
        
        self.handle_reimplementations(error_dicts, context_file_path)
        
        print(colored("Make Agent Execution Run Completed", "green"))
        return False

    def normalize_few_shot_object(self, returned_object: Any) -> List[Dict[str, str]]:
        if isinstance(returned_object, dict):
            return returned_object.get("found_errors", [])
        elif isinstance(returned_object, list):
            if returned_object and isinstance(returned_object[0], dict):
                return returned_object[0].get("found_errors", returned_object)
        return returned_object

    def process_make_errors(self, returned_object_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Processes a list of make errors and extracts relevant code chunks
        containing the errors along with error information for each chunk.
        
        Args:
        returned_object_list (List[Dict[str, str]]): A list of dictionaries where each dictionary
        represents an error with keys such as 'file_path', 'line', and 'full_message'.
        
        Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary contains:
        - 'file_path': The path to the file containing errors.
        - 'chunk': A string representing the code chunk containing the error(s).
        - 'errors': A string with error information including line numbers and messages.
        """
        # Group errors by file using a defaultdict
        errors_by_file: DefaultDict[str, List[MakeError]] = defaultdict(list)
        for error in returned_object_list:
            # Convert each error dict to a MakeError object and group by file path
            errors_by_file[error['file_path']].append(MakeError(**error))

        result: List[Dict[str, str]] = []

        # Process each file and its associated errors
        for file_path, errors in errors_by_file.items():
            with open(file_path, 'r') as file:
                file_content: str = file.read()

            # Use FewShotProvider to determine the appropriate chunking delimiter
            chunking_delimiter: str = FewShotProvider.few_shot_objectFromTemplate(
                [{"code_seperator": "def "}, {"code_seperator": "function "}],
                target_description=f"\n{file_content}\n\nA delimiter to split this code into small chunks like functions or methods. If the code is not a function or method, please provide a delimiter that would split the code into logical chunks.",
                preferred_models=["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
            )['code_seperator']
            
            # Optional: exclude license headers
            start_index = 0
            if "***************" in file_content:
                start_index = file_content.find('\n', file_content.rfind("***************")) + 1
            
            # Find all occurrences of the chunking delimiter in the file content
            chunk_start_indexes: List[int] = [start_index] + [match.start() for match in re.finditer(re.escape(chunking_delimiter), file_content)]
            chunk_start_indexes.append(len(file_content))  # Add end of file as last chunk boundary
            
            # Calculate absolute line numbers for each chunk start
            line_numbers: List[int] = [1]  # First chunk always starts at line 1
            for i in range(1, len(chunk_start_indexes)):
                line_numbers.append(file_content[:chunk_start_indexes[i]].count('\n') + 1)

            # Process each chunk separately
            for i in range(len(chunk_start_indexes) - 1):
                chunk_start_index = chunk_start_indexes[i]
                chunk_end_index = chunk_start_indexes[i + 1]
                chunk_start_line = line_numbers[i]
                chunk_end_line = line_numbers[i + 1] - 1

                # Extract the current code chunk
                chunk: str = file_content[chunk_start_index:chunk_end_index].strip()

                # Collect errors related to this chunk
                chunk_errors = [error for error in errors if chunk_start_line <= int(error.line) <= chunk_end_line]

                # If there are errors in this chunk, add it to the result
                if chunk_errors:
                    error_info: str = "\n\n".join([f'line: {error.line}, message: {error.full_message}' for error in chunk_errors])
                    if "has_is_blue" in error_info:
                        pass
                    result.append({
                        'file_path': file_path,
                        'chunk': chunk,
                        'errors': error_info,
                    })

        return result

    def handle_reimplementations(self, error_dicts: List[Dict[str, str]], file_path: str) -> None:
        diff_str: Optional[str] = self.get_local_file_diff(file_path)
        
        diff_chat: Chat = Chat("You are a highly intelligent assistant.")
        diff_chat.add_message(Role.USER, f"{diff_str}\n\nINSTRUCTION:\nPlease highlight what has changed and how. Then expertly predict what issues this is going to cause in dependent/related files and how to fix them properly.")
        response: str = LlmRouter.generate_completion(diff_chat, ["gpt-4o"], use_reasoning=False, silent=True)
        diff_chat.add_message(Role.ASSISTANT, response)
        
        reimplementation_originals_pair: List[Dict[str, str]] = []
        
        for error_dict in error_dicts:
            chat: Chat = diff_chat.deep_copy()
            file_path: str = error_dict['file_path']
            code_chunk: str = error_dict['chunk']
            error_messages: str = error_dict['errors']
            if not code_chunk or not error_messages or not file_path:
                pass
            
            block_to_reimplement: str = f"""```
{code_chunk}
```"""

            reimplement_prompt = f"{block_to_reimplement}\n\n{error_messages}\n\nINSTRUCTION: This snippet is now causing these errors, likely due to the discussed diffs. Please identify the issues and provide a fixed implementation. In your implementation provide the fixed error as comment above the modifed line. Please provide the implementation as a single block for drop-in replacement, pay attention to not leave any placeholders and do not remove or modify any existing comments. You must provide the replacement code block in full."
            chat.add_message(Role.USER, reimplement_prompt)
            response: str = LlmRouter.generate_completion(chat, ["qwen2.5-coder:7b-instruct", "gpt-4o-mini"], use_reasoning=True, silent_reasoning=False)
            chat.add_message(Role.ASSISTANT, response)
            
            # Guards
            contains_placeholders, is_drop_in_replacement = False, False
            contains_placeholders, _ = FewShotProvider.few_shot_YesNo(response + "\n\nQUESTION: \nDoes the response provide a incomplete implementation, eg. does it contain ANY placeholders like these: '// Further handling logic here...', '// Rest of the code remains unchanged' or '// Replace with actual method'", silent=False)
            if not contains_placeholders:
                is_drop_in_replacement, _ = FewShotProvider.few_shot_YesNo(response + "\n\nQUESTION: \nIs the response a valid drop-in replacement for the original code block?", silent=False)
            
            if contains_placeholders or (not is_drop_in_replacement):
                chat.add_message(Role.USER, "Your response was rejected, validate step by step that that it does not contain any placeholders and starts + finishes exactly like the code block I provided does. It should only contain imports or includes if the example also had them. It must be an exact drop in replacement for the original block. Please provide a corrected version of the block in full.")
                response = LlmRouter.generate_completion(chat, ["qwen2.5-coder:7b-instruct", "gpt-4o-mini"], use_reasoning=True, silent_reasoning=False)
                chat.add_message(Role.ASSISTANT, response)
            
            
            filetypes_blocks: List[Tuple[str, str]] = tooling.extract_blocks(response)
            longest_filetype_block: Tuple[str, str] = max(filetypes_blocks, key=lambda block: len(block[1])) if filetypes_blocks else ("", "")
            
            confidence: float = 1.0
            if abs(len(longest_filetype_block[1]) - len(block_to_reimplement)) > len(block_to_reimplement)*0.4:
                print(colored("The reimplemented blocks length is significantly different from the original block. Please review the reimplemented block and ensure it is correct.", "red"))
                chat.print_chat()
                confidence = 0.5
            
            # add contextful hq few shot example 
            if confidence == 1.0:
                diff_chat.add_message(Role.USER, reimplement_prompt)
                diff_chat.add_message(Role.ASSISTANT, response)
            
            reimplementation_originals_pair.append({"file_path": file_path, "reimplementation": longest_filetype_block[1], "original": code_chunk, "confidence": confidence})
        
        self.confirm_and_apply_reimplementations(reimplementation_originals_pair)

    def confirm_and_apply_reimplementations(self, reimplementation_originals_pair: List[Dict[str, str]]) -> None:
        confirmed_reimplementations: List[Dict[str, str]] = []
        total_reimplementations: int = len(reimplementation_originals_pair)

        print(f"\n{colored(f'Do you want to accept all {len(reimplementation_originals_pair)} reimplementations? (Y/n):', 'white', 'on_blue')} ", end='')
        accept_all: bool = not "n" in input().lower()
        
        only_low_confidence: bool = False
        if not accept_all:
            print(f"\n{colored('Do you want to review only the low confidence reimplementations? (Y/n):', 'white', 'on_blue')} ", end='')
            only_low_confidence = "y" in input().lower()
        
        for index, reimplementation in enumerate(reimplementation_originals_pair, start=1):
            if accept_all or (only_low_confidence and reimplementation['confidence'] < 1):
                confirmed_reimplementations.append(reimplementation)
                continue
            
            print("\n" + "# " * 50)
            print(f"{colored('Original:', 'cyan')}")
            print(f"{colored(reimplementation['original'], 'red')}\n")
            print(f"{colored(f'Reimplementation {index}/{total_reimplementations}', 'cyan')} for {colored(reimplementation['file_path'], 'yellow')}:")
            print(f"{colored(reimplementation['reimplementation'], 'green')}\n")
            print(f"{colored('Implementation confidence:', 'cyan')} {reimplementation['confidence']}")
            user_input: str = input(f"{colored(f'ACCEPT OR DECLINE ({total_reimplementations - index} left) (Y/n):', 'white', 'on_blue')} ")
            if user_input.lower() in ['', 'y', 'yes']:
                confirmed_reimplementations.append(reimplementation)
            
            print("# " * 50 + "\n")
        
        print(f"\n{colored('Confirmed reimplementations:', 'cyan')} {len(confirmed_reimplementations)}/{total_reimplementations}")
        
        for reimplementation in confirmed_reimplementations:
            original_file_content: str = open(reimplementation['file_path'], 'r').read()
            new_file_content: str = original_file_content.replace(reimplementation['original'], reimplementation['reimplementation'])
            open(reimplementation['file_path'], 'w').write(new_file_content)
            print(colored(f"Reimplementation applied to {reimplementation['file_path']}", "green"))

    def get_local_file_diff(self, file_path: str) -> Optional[str]:
        try:
            repo_root: str = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                                cwd=os.path.dirname(file_path),
                                                text=True).strip()
            
            relative_path: str = os.path.relpath(file_path, repo_root)
            
            print('git diff master --', relative_path)
            diff_output: str = subprocess.check_output(['git', 'diff', 'master', '--', relative_path],
                                                cwd=repo_root,
                                                text=True)
            return diff_output if diff_output else "No changes detected in the file."
        
        except subprocess.CalledProcessError as e:
            print(f"Error executing git command: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_agent.py <path_to_makefile_directory> <changed_file_path>")
        sys.exit(1)

    make_path: str = sys.argv[1]
    file_path: str = sys.argv[2]
    
    agent: MakeErrorCollectorAgent = MakeErrorCollectorAgent()
    success: bool = agent.execute(make_path, file_path)
    
    if success:
        print("Make command executed successfully.")
    else:
        print("Make command failed. Errors were processed and potential fixes were suggested.")