# tools/taskmanager.py
"""
This file implements the 'taskmanager' tool. It allows the agent to manage
background processes started with the bash tool, including listing, killing,
viewing their output, and sending input to them.

SECURITY NOTE: This tool is restricted to managing only processes that were
started by the bash tool using <bash><background>true</background>. It cannot
access or control system processes or processes started outside of the agent.
"""
import re
import time
import asyncio
from termcolor import colored
from core.globals import g
from tools.bash import bash  # Import to access the shared process list

# Import TextResourceManager for large output handling
try:
    from shared.utils.text_resource_manager import TextResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
    def get_resource_manager():
        return TextResourceManager._get_manager()
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
    def get_resource_manager():
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
                "IMPORTANT: If a process's output (`stdout` or `watch`) is large, it is NOT returned directly. "
                "Instead, a `resource_id` is returned. You MUST then use the `searchtext` tool to inspect the output."
            ),
            "example": (
                "<taskmanager><list/></taskmanager>\n"
                "<taskmanager><stdout>bg_a1b2c3d4</stdout></taskmanager> <!-- If output is large, this returns a resource_id -->"
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
    def _update_process_status(proc_info: dict):
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
        Parses the command and dispatches to the appropriate handler.
        """
        list_match = re.search(r'<list\s*/>', content, re.IGNORECASE)
        kill_match = re.search(r'<kill>(.*?)</kill>', content, re.DOTALL | re.IGNORECASE)
        stdout_match = re.search(r'<stdout>(.*?)</stdout>', content, re.DOTALL | re.IGNORECASE)
        watch_match = re.search(r'<watch>(.*?)</watch>', content, re.DOTALL | re.IGNORECASE)
        input_match = re.search(r'<input>(.*?)</input>', content, re.DOTALL | re.IGNORECASE)

        # Initialize background processes tracking if it doesn't exist
        if not hasattr(bash, '_background_processes'):
            bash._background_processes = {}
        
        # Update status of all processes before any operation
        for pid, info in bash._background_processes.items():
            taskmanager._update_process_status(info)

        if list_match:
            return taskmanager._list_processes()
        elif kill_match:
            target = kill_match.group(1).strip()
            return taskmanager._kill_process(target)
        elif stdout_match:
            pid = stdout_match.group(1).strip()
            return taskmanager._read_stdout(pid)
        elif watch_match:
            watch_content = watch_match.group(1)
            pid_match = re.search(r'<stdout>(.*?)</stdout>', watch_content, re.DOTALL | re.IGNORECASE)
            duration_match = re.search(r'<duration>(\d+)</duration>', watch_content, re.IGNORECASE)
            
            if pid_match and duration_match:
                pid = pid_match.group(1).strip()
                duration = int(duration_match.group(1).strip())
                return await taskmanager._watch_process(pid, duration)
            else:
                return "Error: <watch> requires both <stdout>pid</stdout> and <duration>seconds</duration> tags."
        elif input_match:
            input_content = input_match.group(1)
            pid_match = re.search(r'<pid>(.*?)</pid>', input_content, re.DOTALL | re.IGNORECASE)
            data_match = re.search(r'<data>(.*?)</data>', input_content, re.DOTALL | re.IGNORECASE)
            
            if pid_match and data_match:
                pid = pid_match.group(1).strip()
                data = data_match.group(1) # Do not strip data, whitespace is significant
                return taskmanager._send_input(pid, data)
            else:
                return "Error: <input> requires both <pid>some_pid</pid> and <data>keystrokes_to_send</data> tags."
        else:
            return "Error: Invalid command for taskmanager. Use <list/>, <kill>pid|all</kill>, <stdout>pid</stdout>, <watch>, or <input>."

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
            
            # The 'data' string is sent as-is. The user can include '\n' for newlines.
            child.send(data)
            return f"Successfully sent {len(data)} byte(s) of input to process {pid}."
        except Exception as e:
            return f"Error sending input to process {pid}: {e}"

    @staticmethod
    def _read_stdout(pid: str, max_chars: int = 4096) -> str:
        proc_info = bash._background_processes.get(pid)
        if not proc_info:
            return f"Error: Process with ID '{pid}' not found."
        
        if not taskmanager._validate_bash_process(proc_info):
            return f"Error: Process '{pid}' is not a bash-managed process."
        
        output = proc_info['output_buffer'].getvalue()
        status = proc_info.get('status', 'unknown')
        header = f"--- Output for PID {pid} (Status: {status}) ---\n"

        # *** CRITICAL WORKFLOW LOGIC ***
        if len(output) > taskmanager.OUTPUT_THRESHOLD:
            resource_manager = get_resource_manager()
            if resource_manager:
                resource = resource_manager.add_resource(
                    output, 
                    source_prefix=f"proc_{pid}",
                    metadata={'command': proc_info.get('command', 'unknown')}
                )
                return (f"Output for PID {pid} is too large to display directly.\n"
                        f"Loaded into searchable resource.\n"
                        f"Resource ID: {resource.resource_id}\n"
                        f"Total Lines: {resource.metadata['total_lines']}\n"
                        f"Use <searchtext resource_id=\"{resource.resource_id}\" pattern=\"...\"/> to inspect.")
        
        # Logic for small output
        if not output.strip():
            return header + "[No output captured yet]"

        if len(output) > max_chars:
            result = f"... (output is {len(output)} chars, showing last {max_chars})\n{output[-max_chars:]}"
        else:
            result = output

        return header + result

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
        
        # *** CRITICAL WORKFLOW LOGIC ***
        final_output = proc_info['output_buffer'].getvalue()
        if len(final_output) > taskmanager.OUTPUT_THRESHOLD:
            resource_manager = get_resource_manager()
            if resource_manager:
                resource = resource_manager.add_resource(
                    final_output, 
                    source_prefix=f"proc_{pid}_watch",
                    metadata={'command': proc_info.get('command', 'unknown')}
                )
                return (f"Watch session for PID {pid} completed.\n"
                        f"Captured {resource.metadata['total_lines']} lines ({len(final_output)} chars).\n"
                        f"Loaded output into searchable resource.\n"
                        f"Resource ID: {resource.resource_id}\n"
                        f"Use <searchtext resource_id=\"{resource.resource_id}\" pattern=\"...\"/> to inspect.")
        
        # Logic for small output
        header = f"--- Final Captured Output for PID {pid} ---\n"
        end_message = colored(f"--- Watch session for PID {pid} ended. ---", "cyan")
        (getattr(g, 'stdout_callback', print))(end_message + "\n") # Print end message to console
        
        return header + (final_output if final_output.strip() else "[No new output captured during watch]")