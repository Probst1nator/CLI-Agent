import os
import re
import json
from typing import List, Dict, Optional, Any, Tuple
from termcolor import colored
from py_classes.cls_util_base import UtilBase
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.enum_ai_strengths import AIStrengths

class FileMind(UtilBase):
    """
    An intelligent, multi-step utility that can understand high-level coding tasks,
    plan file interactions, and perform complex, multi-file edits.
    Uses branched async LLM calls to manage its internal workflow.
    """

    # Delimiters for robustly parsing file blocks from LLM output
    FILE_START_DELIMITER = ">>!>!>!_FILE_START_!<!<!<<"
    FILE_END_DELIMITER = ">>!>!>!_FILE_END_!<!<!<<"

    @staticmethod
    async def run(
        task_description: str = '',
        context_chat: Chat = None,
        relevant_files: Optional[List[str]] = None,
        **kwargs: Any  # Accept additional keyword arguments to be flexible
    ) -> str:
        """
        Main entry point for FileMind utility.
        """
        # Handle case where arguments might come through kwargs
        if not task_description and 'task_description' in kwargs:
            task_description = kwargs['task_description']
        if context_chat is None and 'context_chat' in kwargs:
            context_chat = kwargs['context_chat']  
        if relevant_files is None and 'relevant_files' in kwargs:
            relevant_files = kwargs['relevant_files']
            
        if context_chat is None:
            context_chat = Chat()

        try:
            # Discover relevant files if not provided
            if relevant_files is None:
                relevant_files = FileMind._discover_relevant_files()
                if not relevant_files:
                    return "FileMind Error: No relevant files found in the current directory."

            full_context = f"## Recent Conversation History:\n{context_chat.get_messages_as_string(-5)}\n\n## Current Task:\n{task_description}"

            plan = await FileMind._create_plan(full_context, "\n".join(relevant_files))
            if not plan.get('files_to_read') and not plan.get('files_to_edit'):
                return "FileMind Note: Planning phase determined no files needed to be read or edited."

            all_files_to_read = list(set(plan['files_to_read'] + plan['files_to_edit']))
            file_contents, failed_files = await FileMind._read_files_safely(all_files_to_read)
            
            if failed_files:
                return f"FileMind Error: Could not read required files:\n" + "\n".join(failed_files)

            if not plan['files_to_edit']:
                return "FileMind Note: No files were marked for editing in the plan."

            raw_edit_output = await FileMind._generate_edits(full_context, file_contents, plan['files_to_edit'])
            if not raw_edit_output.strip():
                return "FileMind Error: AI failed to generate edits."

            return FileMind._apply_edits(raw_edit_output, plan)
            
        except Exception as e:
            import traceback
            return f"FileMind Critical Error: {str(e)}\n{traceback.format_exc()}"

    @staticmethod
    async def _create_plan(task_description: str, file_list_str: str) -> Dict[str, List[str]]:
        """
        Phase 1: Create a plan of which files to read and edit.
        """
        print(colored("FileMind [Phase 1/3]: Creating execution plan...", "cyan"))
        instruction = """You are a senior software architect. Your task is to analyze a user's request and a file list
to determine which files to read for context and which to edit.

Respond with ONLY a single JSON object with two keys:
- "files_to_read": A list of file paths that provide necessary context.
- "files_to_edit": A list of file paths that will be directly modified.

Be concise. Only include files absolutely necessary for the task."""
        
        planning_chat = Chat(instruction, debug_title="FileMind-Planning")
        prompt = f"## User Task & Context:\n{task_description}\n\n## Available Project Files:\n{file_list_str}\n\n## JSON Plan:"
        planning_chat.add_message(Role.USER, prompt)

        try:
            response = await LlmRouter.generate_completion(
                planning_chat, strengths=[AIStrengths.REASONING], temperature=0.0, hidden_reason="FileMind Planning"
            )
            
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in the response.")
                
            plan: Dict[str, List[str]] = json.loads(match.group(0))
            plan.setdefault('files_to_read', [])
            plan.setdefault('files_to_edit', [])
            
            print(colored(f"  └─ Plan: Read {len(plan['files_to_read'])} files, Edit {len(plan['files_to_edit'])} files", "green"))
            return plan
        except Exception as e:
            print(colored(f"FileMind [Phase 1/3] Error: Could not create a valid plan. Error: {e}", "red"))
            return {"files_to_read": [], "files_to_edit": []}

    @staticmethod
    async def _generate_edits(task_description: str, file_contents: Dict[str, str], files_to_edit: List[str]) -> str:
        """
        Phase 2: Generate the complete, new content for all files that need editing.
        """
        print(colored("FileMind [Phase 2/3]: Generating code modifications...", "cyan"))
        instruction = f"""You are an expert AI programmer. Your task is to rewrite one or more files based on an instruction.

You MUST output the complete, new content for every file listed in the 'FILES_TO_EDIT' list.
If a file in 'FILES_TO_EDIT' requires no changes, output its original content verbatim.

Wrap each file's content with special delimiters EXACTLY as shown in the format below.
Do NOT add any commentary or explanations outside these blocks. Your output is parsed automatically.

FORMAT:
{FileMind.FILE_START_DELIMITER}
path/to/your/file.py
... the complete new content of file.py ...
{FileMind.FILE_END_DELIMITER}"""
        
        editing_chat = Chat(instruction, debug_title="FileMind-Editing")
        context_block = "\n\n".join(
            f"--- START OF FILE: {path} ---\n{content}\n--- END OF FILE: {path} ---" 
            for path, content in file_contents.items()
        )
        prompt = f"## User Task:\n{task_description}\n\n## Files to Edit:\n{', '.join(files_to_edit)}\n\n## Full Content of Relevant Files:\n{context_block}"
        editing_chat.add_message(Role.USER, prompt)

        return await LlmRouter.generate_completion(
            editing_chat, strengths=[AIStrengths.CODE, AIStrengths.STRONG], temperature=0.0, hidden_reason="FileMind Generating Edits"
        )

    @staticmethod
    def _apply_edits(raw_edit_output: str, plan: Dict[str, List[str]]) -> str:
        """
        Phase 3: Parse the LLM's output and write the changes to disk.
        """
        print(colored("FileMind [Phase 3/3]: Applying changes to disk...", "cyan"))
        try:
            pattern = re.compile(
                f"^{re.escape(FileMind.FILE_START_DELIMITER)}\n(.*?)\n(.*?)\n{re.escape(FileMind.FILE_END_DELIMITER)}$",
                re.DOTALL | re.MULTILINE
            )
            matches = pattern.findall(raw_edit_output)

            if not matches:
                return "Error: Could not parse edit output from the AI. No changes were applied."

            edited_files: Dict[str, str] = {path.strip(): content for path, content in matches}

            files_to_edit = plan['files_to_edit']
            for required_file in files_to_edit:
                if required_file not in edited_files:
                    return f"Error: AI output incomplete. Missing content for '{required_file}'."

            for path, content in edited_files.items():
                if path not in files_to_edit:
                    print(colored(f"  └─ Warning: Unplanned file '{path}' ignored.", "yellow"))
                    continue
                
                parent_dir = os.path.dirname(path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                    
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(colored(f"  └─ Wrote changes to: {os.path.abspath(path)}", "green"))

            return f"FileMind successfully completed the task. {len(files_to_edit)} file(s) were modified."
        except Exception as e:
            return f"FileMind [Phase 3/3] Error: Failed to apply edits. {e}"

    @staticmethod
    def _discover_relevant_files(base_path: str = '.', max_files: int = 50) -> List[str]:
        """
        Intelligently discover relevant files in the project.
        """
        ignore_dirs = {'.git', '__pycache__', '.venv', 'node_modules', '.idea', '.pytest_cache', 'dist', 'build'}
        relevant_extensions = {'.py', '.js', '.ts', '.html', '.css', '.md', '.txt', '.yaml', '.yml', '.toml', '.json'}
        
        relevant_files: List[str] = []
        
        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for name in files:
                if any(name.endswith(ext) for ext in relevant_extensions):
                    file_path = os.path.join(root, name)
                    relevant_files.append(file_path)
                    
                    if len(relevant_files) >= max_files:
                        break
            
            if len(relevant_files) >= max_files:
                break
        
        return relevant_files

    @staticmethod
    async def _read_files_safely(file_paths: List[str]) -> Tuple[Dict[str, str], List[str]]:
        """
        Safely read multiple files, handling errors gracefully.
        """
        file_contents: Dict[str, str] = {}
        failed_files: List[str] = []
        
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    file_contents[path] = f.read()
            except Exception as e:
                failed_files.append(f"Error reading {path}: {str(e)}")
        
        return file_contents, failed_files 