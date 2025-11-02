# tools/taskmanager.py
"""
This file implements the 'taskmanager' tool. It allows the agent to manage
background processes started with the bash tool, including listing, killing,
viewing their output, and sending input to them.

SECURITY NOTE: This tool is restricted to managing only processes that were
started by the bash tool using <bash><background>true</background>. It cannot
access or control system processes or processes started outside of the agent.
"""
import asyncio
import time
import xml.etree.ElementTree as ET
from typing import Any, Optional

from termcolor import colored

from core.globals import g
from tools.bash import bash  # Import to access the shared process list

# Import TextResourceManager for large output handling
try:
    from shared.utils.text_resource_manager import TextResourceManager
    RESOURCE_MANAGER_AVAILABLE = True

    def get_resource_manager() -> Any:
        return TextResourceManager._get_manager()

except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False

    def get_resource_manager() -> None:
        return None

class taskmanager:
    # Threshold for using resource manager for large outputs
    OUTPUT_THRESHOLD = 4096

    @staticmethod
    def get_delim() -> str:
        return 'taskmanager'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "taskmanager",
            "description": (
                "Manages background processes started with <bash>. Allows listing, killing, and inspecting processes. "
                "Large process outputs are automatically handled by the system - if content exceeds limits, "
                "a resource_id is provided and you can use the `searchresource` tool to inspect the output."
            ),
            "example": (
                "<taskmanager><list/></taskmanager>\n"
                "<taskmanager><kill><pid>bg_a1b2c3d4</pid></kill></taskmanager>\n"
                "<taskmanager><stdout><pid>bg_a1b2c3d4</pid></stdout></taskmanager>\n"
                "<taskmanager><input><pid>bg_a1b2c3d4</pid><data>y\\n</data></input></taskmanager> <!-- Use \\\\n for newlines in data -->"
            )
        }

    @staticmethod
    def _validate_bash_process(proc_info: dict) -> bool:
        """
        Validates that a process was started by the bash tool.
        This ensures we only manage processes within our security boundary.
        """
        return (proc_info.get('source') == 'bash' and
                'child' in proc_info and
                'command' in proc_info)

    @staticmethod
    def _update_process_status(proc_info: dict[str, Any]) -> None:
        """
        Helper to check if a running process has finished and update its status.
        pexpect's isalive() is the source of truth.
        Only updates processes that were started by the bash tool.
        """
        if not taskmanager._validate_bash_process(proc_info):
            return  # Skip non-bash processes

        if proc_info.get('status') == 'running' and not proc_info['child'].isalive():
            proc_info['status'] = 'finished'

    @staticmethod
    async def run(content: str) -> str:
        """
        Parses the XML command and dispatches to the appropriate handler using a robust XML parser.
        """
        # Initialize background processes tracking if it doesn't exist
        if not hasattr(bash, '_background_processes'):
            bash._background_processes = {}

        # Update status of all processes before any operation
        for pid, info in bash._background_processes.items():
            taskmanager._update_process_status(info)

        try:
            # Use a proper XML parser instead of brittle regex
            # Handle potential XML encoding issues with newlines and special chars
            content_clean = content.strip()

            # DEBUG: Log the exact input for troubleshooting
            from shared.utils.tool_logging import get_logger_for_tool_module
            logger = get_logger_for_tool_module(__name__)
            logger.debug(f"Taskmanager raw input: {repr(content)}")
            logger.debug(f"Taskmanager cleaned input: {repr(content_clean)}")

            # PREPROCESSING: Handle common XML issues
            # If content contains literal newlines in data tags, it will break XML parsing
            # Look for the pattern and escape newlines within data tags
            import re
            def escape_data_content(match: Any) -> str:
                data_content = match.group(1)
                # Escape problematic characters within data content
                escaped_content = data_content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                return f"<data>{escaped_content}</data>"

            # Apply escaping to data tags
            content_preprocessed = re.sub(r'<data>(.*?)</data>', escape_data_content, content_clean, flags=re.DOTALL)
            logger.debug(f"Taskmanager preprocessed input: {repr(content_preprocessed)}")

            root = ET.fromstring(f"<root>{content_preprocessed}</root>")
            command_element = root[0]  # Get the first child element
            command = command_element.tag.lower()

            if command == 'list':
                return taskmanager._list_processes()

            elif command == 'kill':
                pid_element = command_element.find('pid')
                if pid_element is not None and pid_element.text:
                    target = pid_element.text.strip()
                else:
                    target = command_element.text.strip() if command_element.text else ''
                if not target:
                    return "Error: <kill> requires <pid>process_id</pid> or 'all'."
                return taskmanager._kill_process(target)

            elif command == 'stdout':
                pid_element = command_element.find('pid')
                if pid_element is not None and pid_element.text:
                    pid = pid_element.text.strip()
                else:
                    pid = command_element.text.strip() if command_element.text else ''
                if not pid:
                    return "Error: <stdout> requires <pid>process_id</pid>."
                return taskmanager._read_stdout(pid)

            elif command == 'watch':
                pid_element = command_element.find('stdout')
                duration_element = command_element.find('duration')

                if pid_element is not None and duration_element is not None and pid_element.text and duration_element.text:
                    pid = pid_element.text.strip()
                    duration = int(duration_element.text.strip())
                    return await taskmanager._watch_process(pid, duration)
                else:
                    return "Error: <watch> requires both <stdout>pid</stdout> and <duration>seconds</duration> tags."

            elif command == 'input':
                pid_element = command_element.find('pid')
                data_element = command_element.find('data')

                if pid_element is not None and data_element is not None and pid_element.text:
                    pid = pid_element.text.strip()
                    # Data can be None or empty, and whitespace is significant, so we don't strip it.
                    # Handle both text content and CDATA sections
                    data = ""
                    if data_element.text is not None:
                        data = data_element.text
                    # Also check for CDATA or nested content
                    if not data and len(list(data_element)) == 0 and data_element.text is None:
                        # Element might be empty but valid
                        data = ""
                    return taskmanager._send_input(pid, data)
                else:
                    return "Error: <input> requires both <pid>some_pid</pid> and <data>keystrokes_to_send</data> tags."

            else:
                return f"Error: Unknown command '{command}' for taskmanager. Use <list/>, <kill>, <stdout>, <watch>, or <input>."

        except ET.ParseError as e:
            # Try to provide more helpful error message for common XML issues
            if "mismatched tag" in str(e):
                return (f"Error: XML parsing failed. This often happens when special characters "
                       f"like newlines are included in data tags. Try using \\n instead of literal newlines. "
                       f"Parser error: {e}")
            return f"Error: Malformed XML input for taskmanager. Parser error: {e}"
        except IndexError:
            return "Error: No command found in taskmanager input. Expected a command like <list/>, <kill>, <stdout>, <watch>, or <input>."
        except Exception as e:
            # Catch any other unexpected errors during parsing or execution
            return f"An unexpected error occurred in taskmanager: {e}"

    @staticmethod
    def _list_processes() -> str:
        procs = bash._background_processes
        if not procs:
            return "No background processes are currently managed."

        output = ["Process ID | Status     | Command"]
        output.append("-----------|------------|--------------------------------------------------")

        # Filter for only valid bash processes
        bash_procs = {pid: info for pid, info in procs.items() if taskmanager._validate_bash_process(info)}

        if not bash_procs:
            return "No active or recently finished background processes."

        for pid, info in sorted(bash_procs.items()):
            status = info.get('status', 'unknown')
            command = info['command']
            truncated_command = command[:48] + ('...' if len(command) > 48 else '')
            output.append(f"{pid:<10} | {status:<10} | {truncated_command}")

        return "\n".join(output)

    @staticmethod
    def _kill_process(target: str) -> str:
        if target.lower() == 'all':
            count = 0
            for pid, info in list(bash._background_processes.items()):
                if (taskmanager._validate_bash_process(info) and
                    info.get('status') == 'running'):
                    info['child'].close(force=True)
                    info['status'] = 'killed'
                    count += 1
            return f"Attempted to kill all {count} running background processes."

        proc_info = bash._background_processes.get(target)
        if not proc_info:
            return f"Error: Process with ID '{target}' not found."

        if not taskmanager._validate_bash_process(proc_info):
            return f"Error: Process '{target}' is not a bash-managed process."

        if proc_info.get('status') != 'running':
            return f"Process {target} is not running (status: {proc_info.get('status')})."

        proc_info['child'].close(force=True)
        proc_info['status'] = 'killed'
        return f"Process {target} killed."

    @staticmethod
    def _send_input(pid: str, data: str) -> str:
        """
        Sends a string of data (keystrokes) to the stdin of a running process.
        """
        proc_info = bash._background_processes.get(pid)
        if not proc_info:
            return f"Error: Process with ID '{pid}' not found."

        if not taskmanager._validate_bash_process(proc_info):
            return f"Error: Process '{pid}' is not a bash-managed process."

        if proc_info.get('status') != 'running':
            return f"Error: Process {pid} is not running (status: {proc_info.get('status')}). Cannot send input."

        try:
            child = proc_info['child']
            # Double-check liveness before sending, as status might be stale
            if not child.isalive():
                proc_info['status'] = 'finished'
                return f"Error: Process {pid} terminated before input could be sent."

            # The 'data' string is sent as-is. Convert common escape sequences.
            # Handle \n, \t, \r escape sequences
            processed_data = data.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
            child.send(processed_data)
            return f"Successfully sent {len(data)} byte(s) of input to process {pid}."
        except Exception as e:
            return f"Error sending input to process {pid}: {e}"

    @staticmethod
    def _read_stdout(pid: str, max_chars: Optional[int] = 4096) -> str:
        proc_info = bash._background_processes.get(pid)
        if not proc_info:
            return f"Error: Process with ID '{pid}' not found."

        if not taskmanager._validate_bash_process(proc_info):
            return f"Error: Process '{pid}' is not a bash-managed process."

        output = proc_info['output_buffer'].getvalue()
        status = proc_info.get('status', 'unknown')
        header = f"--- Output for PID {pid} (Status: {status}) [use the bash 'sleep' if you want to wait for it to finish] ---\n"

        # Return full output directly - main.py will create resource if too large
        if not output.strip():
            return header + "[No output captured yet]"

        return header + output

    @staticmethod
    async def _watch_process(pid: str, duration: int) -> str:
        """
        Streams the stdout of a background process for a given duration.
        """
        proc_info = bash._background_processes.get(pid)
        if not proc_info:
            return f"Error: Process with ID '{pid}' not found."

        if not taskmanager._validate_bash_process(proc_info):
            return f"Error: Process '{pid}' is not a bash-managed process."

        stdout_callback = getattr(g, 'stdout_callback', None) or (lambda text: print(text, end=''))

        header = colored(f"\n--- Watching output for PID {pid} for {duration}s (Status: {proc_info.get('status')}) ---", "cyan")
        stdout_callback(header + "\n")

        output_buffer = proc_info['output_buffer']
        child_process = proc_info['child']
        last_read_position = 0

        start_time = time.time()

        while time.time() - start_time < duration:
            if not child_process.isalive():
                stdout_callback(colored(f"--- Process {pid} finished. ---", "yellow") + "\n")
                break

            # The buffer's current write position is its "end"
            output_buffer.seek(0, 2) # Seek to the end of the stream
            current_size = output_buffer.tell()

            if current_size > last_read_position:
                output_buffer.seek(last_read_position)
                new_data = output_buffer.read(current_size - last_read_position)
                if new_data:
                    # Mimic bash tool's streaming output style
                    for line in new_data.strip().split('\n'):
                        if line:
                            stdout_callback(f"⚙️  {line}\n")
                last_read_position = current_size

            await asyncio.sleep(0.2) # Sleep to yield control to the event loop

        # Perform a final read to catch any output that arrived just before the loop ended
        output_buffer.seek(0, 2)
        current_size = output_buffer.tell()
        if current_size > last_read_position:
            output_buffer.seek(last_read_position)
            new_data = output_buffer.read(current_size - last_read_position)
            if new_data:
                for line in new_data.strip().split('\n'):
                    if line:
                        stdout_callback(f"⚙️  {line}\n")

        end_message = colored(f"--- Watch session for PID {pid} ended. ---", "cyan")
        stdout_callback(end_message + "\n")

        # Return full output directly - main.py will create resource if too large
        final_output = proc_info['output_buffer'].getvalue()
        header = f"--- Final Captured Output for PID {pid} ---\n"
        end_message = colored(f"--- Watch session for PID {pid} ended. ---", "cyan")
        (getattr(g, 'stdout_callback', print))(end_message + "\n") # Print end message to console

        return header + (final_output if final_output.strip() else "[No new output captured during watch]")