#!/usr/bin/env python3

import paramiko
import time
import logging
import re
import os
import subprocess
from typing import Tuple, Any, Optional, Callable, List
from py_classes.cls_python_sandbox import PythonSandbox
from py_classes.globals import g

logger = logging.getLogger(__name__)

class SSHSandbox(PythonSandbox):
    """
    A class to manage a persistent SSH connection to a remote server for code execution.
    This extends PythonSandbox to provide the same interface but execute code remotely.
    """
    
    def __init__(self, ssh_connection: str) -> None:
        """
        Initialize the SSH sandbox with a connection to a remote server.
        
        Args:
            ssh_connection: Connection string in format 'user@hostname[:port]'
        """
        self._initializing = True
        
        # Parse the SSH connection string
        pattern = r'^([^@]+)@([^:]+)(?::(\d+))?$'
        match = re.match(pattern, ssh_connection)
        if not match:
            raise ValueError("Invalid SSH connection string. Format should be 'user@hostname[:port]'")
        
        self.username = match.group(1)
        self.hostname = match.group(2)
        self.port = int(match.group(3)) if match.group(3) else 22
        self.ssh_connection_str = ssh_connection
        # Store the basic connection string without port for subprocess commands
        self.host_connection_str = f"{self.username}@{self.hostname}"
        
        # Check X11 forwarding availability
        self.x11_forwarding_available = self._check_x11_forwarding()
        
        # Initialize SSH client
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # Connect with X11 forwarding if available
            transport_params = {
                'hostname': self.hostname,
                'port': self.port,
                'username': self.username
            }
            if self.x11_forwarding_available:
                transport_params['x11_forwarding'] = True
            
            self.ssh_client.connect(**transport_params)
            self._use_subprocess_ssh = False
        except Exception:
            # Fall back to subprocess-based SSH
            self._use_subprocess_ssh = True
        
        # Create a sandbox directory on the remote machine
        self.remote_sandbox_dir = f"/tmp/cli-agent-sandbox-{self.username}"
        self._raw_execute(f"mkdir -p {self.remote_sandbox_dir}")
        self.cwd = self.remote_sandbox_dir
        
        # Execute initialization code
        self._execute_initialization_code()
        self._initializing = False
    
    def _check_x11_forwarding(self) -> bool:
        """Check if X11 forwarding is available."""
        try:
            if 'DISPLAY' not in os.environ:
                return False
            
            result = subprocess.run(
                ["xdpyinfo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
        
    def _raw_execute(self, command: str) -> Tuple[str, str, Any]:
        """
        Execute a shell command on the remote server.
        
        Args:
            command: The shell command to execute
            
        Returns:
            Tuple of (stdout, stderr, None)
        """
        try:
            if self._use_subprocess_ssh:
                # Use native ssh for better X11 support with proper port handling
                ssh_cmd = ["ssh", "-X", "-Y"]
                
                # Add port if not default
                if self.port != 22:
                    ssh_cmd.extend(["-p", str(self.port)])
                
                # Add host and command
                ssh_cmd.extend([self.host_connection_str, command])
                
                process = subprocess.Popen(
                    ssh_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout_str, stderr_str = process.communicate()
                return stdout_str, stderr_str, None
            else:
                # Use paramiko
                _, stdout, stderr = self.ssh_client.exec_command(command)
                stdout_str = stdout.read().decode('utf-8')
                stderr_str = stderr.read().decode('utf-8')
                return stdout_str, stderr_str, None
        except Exception as e:
            return "", str(e), None
    
    def _execute_initialization_code(self) -> None:
        """Execute initialization code for the SSH sandbox."""
        # Python version check
        self._raw_execute("python3 -c 'import sys; print(f\"Remote Python version: {sys.version}\")'")
        self._raw_execute(f"cd {self.remote_sandbox_dir}")
        
        # Setup script
        setup_script = """
import os, sys, json, io, traceback
print(f"Remote Python sandbox initialized with working directory: {os.getcwd()}")
"""
        
        # X11 test script (simplified)
        x11_test_script = """
import os, sys
try:
    import tkinter as tk
    root = tk.Tk()
    root.title("X11 Test")
    root.geometry("300x100")
    label = tk.Label(root, text="X11 Forwarding is working!")
    label.pack(pady=20)
    root.after(2000, root.destroy)
    root.mainloop()
    print("X11 test successful")
except Exception:
    print("X11 test failed")
"""
        
        # Write and upload scripts
        with open("/tmp/cli_agent_setup.py", "w") as f:
            f.write(setup_script)
        
        with open("/tmp/cli_agent_x11_test.py", "w") as f:
            f.write(x11_test_script)
        
        # Upload scripts
        if not self._use_subprocess_ssh:
            sftp = self.ssh_client.open_sftp()
            sftp.put("/tmp/cli_agent_setup.py", f"{self.remote_sandbox_dir}/cli_agent_setup.py")
            sftp.put("/tmp/cli_agent_x11_test.py", f"{self.remote_sandbox_dir}/cli_agent_x11_test.py")
            sftp.close()
        else:
            # Use scp for file transfer with proper port handling
            scp_cmd = ["scp"]
            if self.port != 22:
                scp_cmd.extend(["-P", str(self.port)])
            
            scp_cmd.extend([
                "/tmp/cli_agent_setup.py", 
                f"{self.host_connection_str}:{self.remote_sandbox_dir}/cli_agent_setup.py"
            ])
            
            subprocess.run(
                scp_cmd,
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            
            scp_cmd = ["scp"]
            if self.port != 22:
                scp_cmd.extend(["-P", str(self.port)])
                
            scp_cmd.extend([
                "/tmp/cli_agent_x11_test.py", 
                f"{self.host_connection_str}:{self.remote_sandbox_dir}/cli_agent_x11_test.py"
            ])
            
            subprocess.run(
                scp_cmd,
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        
        # Execute setup script
        self._raw_execute(f"cd {self.remote_sandbox_dir} && python3 cli_agent_setup.py")
    
    def test_x11_forwarding(self) -> bool:
        """Test if X11 forwarding is working correctly."""
        try:
            if self._use_subprocess_ssh:
                ssh_cmd = ["ssh", "-X", "-Y"]
                
                if self.port != 22:
                    ssh_cmd.extend(["-p", str(self.port)])
                    
                ssh_cmd.extend([
                    self.host_connection_str,
                    f"cd {self.remote_sandbox_dir} && python3 cli_agent_x11_test.py"
                ])
                
                process = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, _ = process.communicate()
            else:
                ssh_stdin, ssh_stdout, _ = self.ssh_client.exec_command(
                    f"cd {self.remote_sandbox_dir} && python3 cli_agent_x11_test.py"
                )
                stdout = ssh_stdout.read().decode('utf-8')
            
            return "X11 test successful" in stdout
        except Exception:
            return False
    
    def execute(self, code: str, stdout_callback: Optional[Callable[[str], None]] = None,
               stderr_callback: Optional[Callable[[str], None]] = None, timeout: Optional[float] = None) -> Tuple[str, str, Any]:
        """Execute code on the remote server."""
        # Handle multiple lines of code with potential Jupyter-style shell commands
        if '\n' in code.strip() and '!' in code:
            lines = code.strip().split('\n')
            # Check if all lines are shell commands (start with !)
            all_shell = all(line.strip().startswith('!') for line in lines if line.strip())
            
            if all_shell:
                # Process as a multiline shell script
                cleaned_code = '\n'.join(line.strip()[1:].strip() for line in lines if line.strip())
                # Execute as a shell script
                return self._execute_shell_command(cleaned_code, stdout_callback, stderr_callback)
        
        # Check if this is a shell command rather than Python code
        is_shell_command = False
        # Common shell command patterns
        shell_command_patterns = [
            'sudo', 'apt', 'systemctl', 'service', 'ls', 'cd', 'mkdir', 'rm', 'cp', 'mv',
            'cat', 'echo', 'grep', 'find', 'chmod', 'chown', 'tar', 'zip', 'unzip', 'wget',
            'curl', 'ssh', 'scp', 'rsync', 'ps', 'kill', 'pkill', 'top', 'df', 'du', 'mount'
        ]
        
        # Handle Jupyter magic style shell commands (starting with !)
        if code.strip().startswith('!'):
            is_shell_command = True
            code = code.strip()[1:].strip()  # Remove the leading '!'
        else:
            # Check if code appears to be a shell command
            stripped_code = code.strip()
            first_word = stripped_code.split()[0] if stripped_code else ""
            if (first_word in shell_command_patterns or 
                stripped_code.startswith('#!') or
                '|' in stripped_code or  # pipe symbol
                ';' in stripped_code or  # command separator
                '>' in stripped_code or  # redirection
                '<' in stripped_code):   # input redirection
                is_shell_command = True
        
        # Handle sudo commands with GUI prompts
        has_sudo = "sudo " in code and not "sudo -A " in code
        if has_sudo:
            if self.x11_forwarding_available:
                code = code.replace("sudo ", "DISPLAY=$DISPLAY sudo ")
            else:
                code = code.replace("sudo ", "sudo -A ")
        
        if is_shell_command:
            return self._execute_shell_command(code, stdout_callback, stderr_callback)
        else:
            return self._execute_python_code(code, stdout_callback, stderr_callback)
    
    def _execute_shell_command(self, command: str, stdout_callback: Optional[Callable[[str], None]] = None,
                            stderr_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, str, Any]:
        """Execute a shell command on the remote server."""
        stdout_buffer = ""
        stderr_buffer = ""
        
        # Handle sudo commands with GUI prompts
        has_sudo = "sudo " in command and not "sudo -A " in command
        if has_sudo:
            if self.x11_forwarding_available:
                command = command.replace("sudo ", "DISPLAY=$DISPLAY sudo ")
            else:
                command = command.replace("sudo ", "sudo -A ")
        
        try:
            if self._use_subprocess_ssh:
                # Use subprocess ssh for shell commands
                ssh_cmd = ["ssh"]
                
                # Add X11 forwarding if needed for sudo
                if has_sudo and self.x11_forwarding_available:
                    ssh_cmd.extend(["-X", "-Y"])
                
                # Add port if not default
                if self.port != 22:
                    ssh_cmd.extend(["-p", str(self.port)])
                
                # Add host and command
                ssh_cmd.extend([self.host_connection_str, command])
                
                process = subprocess.Popen(
                    ssh_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Process output in real-time
                for line in iter(process.stdout.readline, ''):
                    stdout_buffer += line
                    if stdout_callback:
                        stdout_callback(line)
                    else:
                        print(line, end="", flush=True)
                
                for line in iter(process.stderr.readline, ''):
                    stderr_buffer += line
                    if stderr_callback:
                        stderr_callback(line)
                    else:
                        print(line, end="", flush=True)
                
                process.wait()
            else:
                # Use paramiko for executing shell commands
                _, stdout, stderr = self.ssh_client.exec_command(command)
                
                # Process output
                stdout_str = stdout.read().decode('utf-8')
                stderr_str = stderr.read().decode('utf-8')
                
                # Use callbacks if provided
                if stdout_str:
                    stdout_buffer += stdout_str
                    if stdout_callback:
                        stdout_callback(stdout_str)
                    else:
                        print(stdout_str, end="", flush=True)
                
                if stderr_str:
                    stderr_buffer += stderr_str
                    if stderr_callback:
                        stderr_callback(stderr_str)
                    else:
                        print(stderr_str, end="", flush=True)
            
            return stdout_buffer, stderr_buffer, None
            
        except Exception as e:
            if stderr_callback:
                stderr_callback(str(e))
            return stdout_buffer, stderr_buffer + f"\nError: {str(e)}", None

    def _execute_python_code(self, code: str, stdout_callback: Optional[Callable[[str], None]] = None,
                         stderr_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, str, Any]:
        """Execute Python code on the remote server."""
        # Write the code to a temporary file
        tmp_file = f"/tmp/cli_agent_code_{int(time.time())}.py"
        with open(tmp_file, "w") as f:
            f.write(code)
        
        # Upload the code file
        remote_file = f"{self.remote_sandbox_dir}/cli_agent_code.py"
        if not self._use_subprocess_ssh:
            sftp = self.ssh_client.open_sftp()
            sftp.put(tmp_file, remote_file)
            sftp.close()
        else:
            scp_cmd = ["scp"]
            if self.port != 22:
                scp_cmd.extend(["-P", str(self.port)])
            
            scp_cmd.extend([tmp_file, f"{self.host_connection_str}:{remote_file}"])
            
            subprocess.run(
                scp_cmd,
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        
        # Execute the code
        command = f"cd {self.remote_sandbox_dir} && python3 cli_agent_code.py"
        stdout_buffer = ""
        stderr_buffer = ""
        
        # Determine if X11 is needed
        needs_x11 = any(x in code for x in ["import tkinter", "import gtk", "import PyQt", "import wx"])
        
        try:
            if self._use_subprocess_ssh or needs_x11:
                # Use subprocess for X11 forwarding and interactive prompts
                ssh_cmd = ["ssh"]
                if needs_x11 and self.x11_forwarding_available:
                    ssh_cmd.extend(["-X", "-Y"])
                
                # Add port if not default
                if self.port != 22:
                    ssh_cmd.extend(["-p", str(self.port)])
                    
                ssh_cmd.extend([self.host_connection_str, command])
                
                process = subprocess.Popen(
                    ssh_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Process output in real-time
                for line in iter(process.stdout.readline, ''):
                    stdout_buffer += line
                    if stdout_callback:
                        stdout_callback(line)
                    else:
                        print(line, end="", flush=True)
                
                for line in iter(process.stderr.readline, ''):
                    stderr_buffer += line
                    if stderr_callback:
                        stderr_callback(line)
                    else:
                        print(line, end="", flush=True)
                
                process.wait()
            else:
                # Use paramiko for standard non-GUI code
                ssh_stdin, ssh_stdout, ssh_stderr = self.ssh_client.exec_command(command)
                
                # Process output in real-time
                while not ssh_stdout.channel.exit_status_ready():
                    if ssh_stdout.channel.recv_ready():
                        data = ssh_stdout.channel.recv(1024).decode('utf-8')
                        stdout_buffer += data
                        if stdout_callback:
                            stdout_callback(data)
                        else:
                            print(data, end="", flush=True)
                    
                    if ssh_stderr.channel.recv_stderr_ready():
                        data = ssh_stderr.channel.recv_stderr(1024).decode('utf-8')
                        stderr_buffer += data
                        if stderr_callback:
                            stderr_callback(data)
                        else:
                            print(data, end="", flush=True)
                    
                    time.sleep(0.1)
                
                # Get any remaining output
                while ssh_stdout.channel.recv_ready():
                    data = ssh_stdout.channel.recv(1024).decode('utf-8')
                    stdout_buffer += data
                    if stdout_callback:
                        stdout_callback(data)
                    else:
                        print(data, end="", flush=True)
                
                while ssh_stderr.channel.recv_stderr_ready():
                    data = ssh_stderr.channel.recv_stderr(1024).decode('utf-8')
                    stderr_buffer += data
                    if stderr_callback:
                        stderr_callback(data)
                    else:
                        print(data, end="", flush=True)
            
            # Clean up
            os.unlink(tmp_file)
            
            return stdout_buffer, stderr_buffer, None
            
        except Exception as e:
            if stderr_callback:
                stderr_callback(str(e))
            return stdout_buffer, stderr_buffer + f"\nError: {str(e)}", None
    
    def restart(self) -> None:
        """Restart the SSH connection."""
        try:
            if not self._use_subprocess_ssh:
                self.ssh_client.close()
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Connect with X11 forwarding if available
                transport_params = {
                    'hostname': self.hostname,
                    'port': self.port,
                    'username': self.username
                }
                if self.x11_forwarding_available:
                    transport_params['x11_forwarding'] = True
                
                self.ssh_client.connect(**transport_params)
        except Exception:
            pass
    
    def shutdown(self) -> None:
        """Close the SSH connection."""
        try:
            if not self._use_subprocess_ssh:
                self.ssh_client.close()
        except Exception:
            pass
    
    def __del__(self) -> None:
        """Ensure the SSH connection is closed when the object is garbage collected."""
        try:
            self.shutdown()
        except:
            pass 