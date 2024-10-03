import shutil
import subprocess
import os
import sys
from collections import defaultdict
import re
from typing import Any, List, Dict, Tuple, Optional, DefaultDict
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from termcolor import colored
from py_classes.cls_chat import Chat, Role
from py_classes.cls_few_shot_factory import FewShotProvider
from py_classes.cls_llm_router import LlmRouter
from py_methods import tooling
from py_classes.globals import g

@dataclass
class MakeError:
    file_path: str
    line_number: int
    error_message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

class BaseAssistantAgent(ABC):
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        pass

class MakeErrorCollectorAgent(BaseAssistantAgent):
    def execute(self, build_dir_path: str, context_file_paths: str, init_sandbox: bool = True, force_local: bool = False) -> bool:
        proj_dir_path = os.path.dirname(build_dir_path)
        build_dir_title = os.path.basename(build_dir_path)
        proj_dir_title = os.path.basename(os.path.dirname(build_dir_path))
        sandboxed_build_path = os.path.join(g.PROJ_SANDBOX_PATH, proj_dir_title, build_dir_title)
        if init_sandbox:
            shutil.rmtree(g.PROJ_SANDBOX_PATH, ignore_errors=True)
            os.makedirs(g.PROJ_SANDBOX_PATH)
            subprocess.run(["cp", "-r", proj_dir_path, g.PROJ_SANDBOX_PATH], check=True)
            subprocess.run(["rm", "-rf", sandboxed_build_path], check=True)
            subprocess.run(['mkdir', sandboxed_build_path], check=True)
            subprocess.run(['cmake', '-DBUILD_FIRMWARE=TRUE', '-DDOWNLOAD_V8=FALSE', '..'], cwd=sandboxed_build_path, check=True)  
        
        command_to_gen_errors: List[str] = ['make', '-C', sandboxed_build_path, '-j', str(os.cpu_count())]

        # Run the make command and process output
        try:
            # Run the make command and process output
            process = subprocess.Popen(
                command_to_gen_errors,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_lines: List[str] = []
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Print the line in real-time
                sys.stdout.flush()
                output_lines.append(line.strip())

            process.stdout.close()
            return_code: int = process.wait()

            # If make command was successful, return True
            if return_code == 0:
                print(colored("Make Agent Execution has finished successfully! All errors cleared!", "green"))
                return True

            # Filter the errors to reduce the amount of text to process
            filtered_output_lines: List[str] = [
                line for line in output_lines 
                if not any(substring in line for substring in ["Building ", "Entering ", "Leaving ", "Installing: "])
            ]
            filtered_output_str: str = '\n'.join(filtered_output_lines)
            error_output: str = filtered_output_str[filtered_output_str.find('error'):filtered_output_str.rfind('error')]
            
            if not error_output:
                print(colored("Make Agent Execution Failure: No errors were found in the output.", "red"))
                return False
            
            error_example1 = {"file_path": "/home/prob/repos/python_project/image_processor.py", "line_number": "75", "error_message": "ImportError: No module named 'PIL'"}
            error_example2 = {"file_path": "/home/prob/repos/rust_project/src/main.rs", "line_number": "42", "error_message": "mismatched types: expected &str, found String"}
            error_example3 = {"file_path": "/home/prob/repos/rust_project/src/main.rs", "line_number": "80", "error_message": "no method named `unwrap` found for type `std::result::Result<_, _>` in the current scope"}
            error_example4 = {"file_path": "/home/prob/repos/aidev/src/robotObserver.cpp", "line_number": "26", "error_message": "btVector3' was not declared in this scope; did you forget to include <bullet/LinearMath/btVector3.h>?"}
            error_example5 = {"file_path": "/home/prob/repos/aidev/src/robotSimulation.cpp", "line_number": "18", "error_message": "'btQuaternion' is not a member of 'btQuaternion'; did you mean 'btQuaternion::btQuaternion'?"}
            error_example6 = {"file_path": "/home/prob/repos/aidev/src/robotSimulation.cpp", "line_number": "25", "error_message": "no declaration matches 'void RobotCommandAdaptor::handleRobotResponse(const QList<robot::StatusPayload>&)'"}
            error_example7 = {"file_path": "/home/prob/repos/aidev/src/theoreticalPhysics.cpp", "line_number": "42", "error_message": "invalid operands of types 'double' and 'const char [6]' to binary 'operator/'"}
            returned_object = FewShotProvider.few_shot_objectFromTemplate([error_example1, error_example2, error_example3, error_example4, error_example5, error_example6, error_example7], f"```\n{error_output}\n```\n\nPlease extract all the errors which include their source file path from the output and provide them as a list of objects as specified in the template. Only include errors that stem from the project, ignore non-trivial system errors and errors which stem from the build directory. Ensure your provided object matches the structure of example object precisely, if you forget or misname a key you will have failed your objective. If you can't find any return an empty array.", silent=False, force_local=force_local)
            returned_object_list: List[Dict[str, str]] = self.normalize_few_shot_object(returned_object)
            
            # filter objects
            returned_object_list = [obj for obj in returned_object_list if obj.get("file_path") and obj.get("line_number") and obj.get("error_message")] # all fields must be present and populated
            returned_object_list = [obj for obj in returned_object_list if os.path.exists(obj["file_path"])] # remove entries which file does not exist
            returned_object_list = [obj for obj in returned_object_list if 'autogen' not in obj["file_path"]] # remove entries which contain 'autogen' in the path
            
            if not returned_object_list:
                print(colored("Make Agent Execution Failure: No valid errors were found in the output.", "red"))
                return False
            
            error_dicts: List[Dict[str, str]] = self.process_make_errors(returned_object_list, force_local)
            
            self.handle_reimplementations(error_dicts, context_file_paths, force_local)
            
            print(colored("Make Agent Execution Run Completed, ", "green") + colored("some errors likely remain.", "yellow"))
            return True

        except subprocess.CalledProcessError as e:
            print(colored(f"Make Agent Execution failed with return code {e.returncode}", "red"))
            print(e.output)
            return False

    def normalize_few_shot_object(self, returned_object: Any) -> List[Dict[str, str]]:
        if isinstance(returned_object, dict):
            return returned_object.get("found_errors", [])
        elif isinstance(returned_object, list):
            if returned_object and isinstance(returned_object[0], dict):
                return returned_object[0].get("found_errors", returned_object)
        return returned_object

    def process_make_errors(self, returned_object_list: List[Dict[str, str]], force_local: bool = False) -> List[Dict[str, str]]:
        """
        Processes a list of make errors and extracts relevant code chunks
        containing the errors along with error information for each chunk.
        
        Args:
        returned_object_list (List[Dict[str, str]]): A list of dictionaries where each dictionary
        represents an error with keys such as "file_path", 'line', and 'errorMessage'.
        
        Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary contains:
        - "file_path": The path to the file containing errors.
        - 'chunk': A string representing the code chunk containing the error(s).
        - 'errors': A string with error information including line numbers and messages.
        """
        # Group errors by file using a defaultdict
        errors_by_file: DefaultDict[str, List[MakeError]] = defaultdict(list)
        for error in returned_object_list:
            # Convert each error dict to a MakeError object and group by file path
            errors_by_file[error["file_path"]].append(MakeError(**error))

        result: List[Dict[str, str]] = []

        # Process each file and its associated errors
        for file_path, errors in errors_by_file.items():
            with open(file_path, 'r') as file:
                file_content: str = file.read()

            # Use FewShotProvider to determine the appropriate chunking delimiter
            chunking_delimiter: str = FewShotProvider.few_shot_objectFromTemplate(
                [{"code_seperator": "def "}, {"code_seperator": "function "}],
                target_description=f"\n{file_content}\n\nA delimiter to split this code into small chunks like functions or methods. If the code is not a function or method, provide a single delimiter, formatted in json as shown in the examples, that would split the code into logical chunks. Respond only with a json object as shown in the examples.",
                preferred_models=["llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
                force_local=force_local
            )['code_seperator']
            
            # Optional: exclude license headers
            start_index = 0
            if "***************" in file_content:
                start_index = file_content.find('\n', file_content.rfind("***************")) + 1
            
            # Find all occurrences of the chunking delimiter in the file content
            chunk_start_indexes: List[int] = [start_index] + [match.start() for match in re.finditer(re.escape(chunking_delimiter), file_content)]
            chunk_start_indexes.append(len(file_content))  # Add end of file as last chunk boundary
            
            # Calculate absolute line numbers for each chunk start
            lineNumbers: List[int] = [1]  # First chunk always starts at line 1
            for i in range(1, len(chunk_start_indexes)):
                lineNumbers.append(file_content[:chunk_start_indexes[i]].count('\n') + 1)

            # Process each chunk separately
            for i in range(len(chunk_start_indexes) - 1):
                chunk_start_index = chunk_start_indexes[i]
                chunk_end_index = chunk_start_indexes[i + 1]
                chunk_start_line = lineNumbers[i]
                chunk_end_line = lineNumbers[i + 1] - 1

                # Extract the current code chunk
                chunk: str = file_content[chunk_start_index:chunk_end_index].strip()

                # Collect errors related to this chunk
                chunk_errors = [error for error in errors if chunk_start_line <= int(error.line_number) <= chunk_end_line]

                # If there are errors in this chunk, add it to the result
                if chunk_errors:
                    error_info: str = "\n\n".join([f'line_number: {error.line_number}, error_message: {error.error_message}' for error in chunk_errors])
                    if "has_is_blue" in error_info:
                        pass
                    result.append({
                        "file_path": file_path,
                        'chunk': chunk,
                        'errors': error_info,
                    })

        return result

    def handle_reimplementations(self, error_dicts: List[Dict[str, str]], context_file_paths: List[str], force_local: bool = False) -> None:
        diff_strings: List[str] = []
        for context_file_path in context_file_paths:
            diff_str = self.get_local_file_diff(context_file_path)
            diff_str += "\n```full_file\n" + open(context_file_path, 'r').read() + "\n```"
            diff_strings.append()
        
        
        diff_chat: Chat = Chat("You are a highly experienced enthusiastic software developer.")
        for diff_str in diff_strings:
            diff_chat.add_message(Role.USER, f"{diff_str}\n\nINSTRUCTION:\nPlease highlight what has changed and how. Afterwards make educated guesses what issues this is going to cause in dependent/related files and how to fix them accordingly.")
            response: str = LlmRouter.generate_completion(diff_chat, ["qwen2.5-coder:7b-instruct", "gpt-4o-mini"], use_reasoning=False, silent=True, force_local=force_local)
            diff_chat.add_message(Role.ASSISTANT, response)
        
        reimplementation_originals_pair: List[Dict[str, str]] = []
        
        for error_dict in error_dicts:
            chat: Chat = diff_chat.deep_copy()
            file_path: str = error_dict["file_path"]
            code_chunk: str = error_dict['chunk']
            error_messages: str = error_dict['errors']
            if not code_chunk or not error_messages or not file_path:
                pass
            
            block_to_reimplement: str = f"""```
{code_chunk}
```"""

            reimplement_prompt = f"{block_to_reimplement}\n\n{error_messages}\n\nINSTRUCTION: This snippet is now causing these errors, likely due to the discussed diffs. Please identify the issues and provide a fixed implementation. In your implementation provide the fixed error as comment above the modifed line. Please provide the implementation as a single block for drop-in replacement, pay attention to not leave any placeholders and do not remove or modify any existing comments. You must provide the replacement code block in full."
            chat.add_message(Role.USER, reimplement_prompt)
            response: str = LlmRouter.generate_completion(chat, ["qwen2.5-coder:7b-instruct", "gpt-4o-mini"], use_reasoning=True, silent_reasoning=False, force_local=force_local)
            chat.add_message(Role.ASSISTANT, response)
            
            # Guards
            contains_placeholders, is_drop_in_replacement = False, False
            contains_placeholders, _ = FewShotProvider.few_shot_YesNo(response + "\n\nQUESTION: \nDoes the response provide a incomplete implementation, eg. does it contain ANY placeholders like these: '// Further handling logic here...', '// Rest of the code remains unchanged' or '// Replace with actual method'", silent=False, force_local=force_local)
            if not contains_placeholders:
                is_drop_in_replacement, _ = FewShotProvider.few_shot_YesNo(response + "\n\nQUESTION: \nIs the response a valid drop-in replacement for the original code block?", silent=False, force_local=force_local)
            
            if contains_placeholders or (not is_drop_in_replacement):
                chat.add_message(Role.USER, "Your response was rejected by my automated system, please validate that your code does not contain any placeholders and starts + finishes exactly like the code block I provided does. It should only contain imports or includes if the original block also contained them. It must be an exact drop in replacement for the original block. Please provide a corrected version in full.")
                response = LlmRouter.generate_completion(chat, ["qwen2.5-coder:7b-instruct", "gpt-4o-mini"], use_reasoning=True, silent_reasoning=False, force_local=force_local)
                chat.add_message(Role.ASSISTANT, response)
            
            
            filetypes_blocks: List[Tuple[str, str]] = tooling.extract_blocks(response)
            longest_filetype_block: Tuple[str, str] = max(filetypes_blocks, key=lambda block: len(block[1])) if filetypes_blocks else ("", "")
            
            confidence: float = 1.0
            if abs(len(longest_filetype_block[1]) - len(block_to_reimplement)) > len(block_to_reimplement)*0.4:
                # print(colored("The reimplemented blocks length is significantly different from the original block. Please review the reimplemented block and ensure it is correct.", "yellow"))
                # chat.print_chat()
                # confidence = 0.5
                chat.add_message(Role.USER, "The reimplemented blocks length is significantly different from the original block. Please review the reimplemented block and ensure it is a correct and full, drop-in replacement.")
                response = LlmRouter.generate_completion(chat, ["qwen2.5-coder:7b-instruct", "gpt-4o-mini"], use_reasoning=True, silent_reasoning=False, force_local=force_local)
                chat.add_message(Role.ASSISTANT, response)
            
            # add contextful hq few shot example 
            if confidence == 1.0:
                diff_chat.add_message(Role.USER, reimplement_prompt)
                diff_chat.add_message(Role.ASSISTANT, response)
            
            reimplementation_originals_pair.append({"file_path": file_path, "reimplementation": longest_filetype_block[1], "original": code_chunk, "confidence": confidence})
        
        accepted_file_paths = self.confirm_and_apply_reimplementations(reimplementation_originals_pair)

    def confirm_and_apply_reimplementations(self, reimplementation_originals_pair: List[Dict[str, str]]) -> List[str]:
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
            print(f"{colored(f'Reimplementation {index}/{total_reimplementations}', 'cyan')} for {colored(reimplementation["file_path"], 'yellow')}:")
            print(f"{colored(reimplementation['reimplementation'], 'green')}\n")
            print(f"{colored('Implementation confidence:', 'cyan')} {reimplementation['confidence']}")
            user_input: str = input(f"{colored(f'ACCEPT OR DECLINE ({total_reimplementations - index} left) (Y/n):', 'white', 'on_blue')} ")
            if user_input.lower() in ['', 'y', 'yes']:
                confirmed_reimplementations.append(reimplementation)
            
            print("# " * 50 + "\n")
        
        print(f"\n{colored('Confirmed reimplementations:', 'cyan')} {len(confirmed_reimplementations)}/{total_reimplementations}")
        
        accepted_file_paths: List[str] = []
        for reimplementation in confirmed_reimplementations:
            original_file_content: str = open(reimplementation["file_path"], 'r').read()
            new_file_content: str = original_file_content.replace(reimplementation['original'], reimplementation['reimplementation'])
            open(reimplementation["file_path"], 'w').write(new_file_content)
            print(colored(f"Reimplementation applied to {reimplementation["file_path"]}", "green"))
            if reimplementation["file_path"] not in accepted_file_paths:
                accepted_file_paths.append(reimplementation["file_path"])
        return accepted_file_paths

    def get_local_file_diff(self, filePath: str) -> Optional[str]:
        try:
            repo_root: str = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                                cwd=os.path.dirname(filePath),
                                                text=True).strip()
            
            relative_path: str = os.path.relpath(filePath, repo_root)
            
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
        print("Usage: python make_agent.py <path_to_makefile_directory> <changed_filePath>")
        sys.exit(1)

    make_path: str = sys.argv[1]
    file_path: str = sys.argv[2]
    
    agent: MakeErrorCollectorAgent = MakeErrorCollectorAgent()
    success: bool = agent.execute(make_path, file_path)
    
    if success:
        print("Make command executed successfully.")
    else:
        print("Make command failed. Errors were processed and potential fixes were suggested.")