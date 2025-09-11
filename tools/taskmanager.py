# tools/taskmanager.py
"""
This file implements the 'taskmanager' tool. It allows the agent to manage
background processes started with the bash tool, including listing, killing,
and viewing their output.

SECURITY NOTE: This tool is restricted to managing only processes that were
started by the bash tool using <bash><background>true</background>. It cannot
access or control system processes or processes started outside of the agent.
"""
import re
from tools.bash import bash  # Import to access the shared process list

class taskmanager:
    @staticmethod
    def get_delim() -> str:
        return 'taskmanager'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "taskmanager",
            "description": "Manages background processes started with <bash><background>true</background>. Allows listing all processes, killing a specific process by ID (or all), and viewing the standard output of a process.",
            "example": "<taskmanager><list/></taskmanager>\n<taskmanager><kill>all</kill></taskmanager>\n<taskmanager><stdout>a2b4c6d8</stdout></taskmanager>"
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
    def run(content: str) -> str:
        """
        Parses the command and dispatches to the appropriate handler.
        """
        list_match = re.search(r'<list\s*/>', content, re.IGNORECASE)
        kill_match = re.search(r'<kill>(.*?)</kill>', content, re.DOTALL | re.IGNORECASE)
        stdout_match = re.search(r'<stdout>(.*?)</stdout>', content, re.DOTALL | re.IGNORECASE)

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
        else:
            return "Error: Invalid command for taskmanager. Use <list/>, <kill>pid|all</kill>, or <stdout>pid</stdout>."

    @staticmethod
    def _list_processes() -> str:
        procs = bash._background_processes
        if not procs:
            return "No background processes are currently managed."

        output = ["Process ID | Status     | Command"]
        output.append("-----------|------------|--------------------------------------------------")
        
        # We don't remove old processes so their output can be checked after they finish.
        # A more advanced implementation might have a TTL or manual cleanup command.
        if not procs:
            return "No active or recently finished background processes."

        for pid, info in sorted(procs.items()):
            # Only show processes that were started by the bash tool
            if not taskmanager._validate_bash_process(info):
                continue
                
            status = info.get('status', 'unknown')
            command = info['command']
            truncated_command = command[:48] + ('...' if len(command) > 48 else '')
            output.append(f"{pid:<10} | {status:<10} | {truncated_command}")
        
        return "\n".join(output)

    @staticmethod
    def _kill_process(target: str) -> str:
        if target.lower() == 'all':
            count = 0
            # Iterate over a copy of items since we modify the dict's status
            for pid, info in list(bash._background_processes.items()):
                # Only kill processes that were started by the bash tool
                if (taskmanager._validate_bash_process(info) and 
                    info.get('status') == 'running'):
                    info['child'].close(force=True)
                    info['status'] = 'killed'
                    count += 1
            return f"Attempted to kill all {count} running background processes."
        
        proc_info = bash._background_processes.get(target)
        if not proc_info:
            return f"Error: Process with ID '{target}' not found."
        
        # Validate this is a bash-managed process
        if not taskmanager._validate_bash_process(proc_info):
            return f"Error: Process '{target}' is not a bash-managed process."
        
        if proc_info.get('status') != 'running':
            return f"Process {target} is not running (status: {proc_info.get('status')})."

        proc_info['child'].close(force=True)
        proc_info['status'] = 'killed'
        return f"Process {target} killed."

    @staticmethod
    def _read_stdout(pid: str, max_chars: int = 4096) -> str:
        proc_info = bash._background_processes.get(pid)
        if not proc_info:
            return f"Error: Process with ID '{pid}' not found."
        
        # Validate this is a bash-managed process
        if not taskmanager._validate_bash_process(proc_info):
            return f"Error: Process '{pid}' is not a bash-managed process."
        
        output = proc_info['output_buffer'].getvalue()
        status = proc_info.get('status', 'unknown')
        
        header = f"--- Output for PID {pid} (Status: {status}) ---\n"
        
        if not output.strip():
            return header + "[No output captured yet]"

        if len(output) > max_chars:
            result = f"... (output is {len(output)} chars, showing last {max_chars})\n{output[-max_chars:]}"
        else:
            result = output

        return header + result