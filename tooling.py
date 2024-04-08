import subprocess
import sys
from typing import Any, Dict, List

from termcolor import colored


def run_command(command: str, verbose: bool = True) -> Dict[str, Any]:
    output_lines = []  # List to accumulate output lines

    try:
        with subprocess.Popen(command, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as process:
            if process.stdout is not None:
                if verbose:
                    for line in process.stdout:
                        print(line, end='')  # Print lines as they are received

            # Wait for the process to terminate and capture remaining output, if any
            remaining_output, error = process.communicate()

            # It's possible, though unlikely, that new output is generated between the last readline and communicate call
            if remaining_output:
                output_lines.append(remaining_output)

            # Combine all captured output lines into a single string
            final_output = ''.join(output_lines)

            result = {
                'output': final_output,
                'error': error,
                'exit_code': process.returncode
            }
            
            # Conditional checks on result can be implemented here as needed
            result_formatted = command
            if (result["output"]):
                result_formatted += f"\n```cmd_output\n{result['output']}```"
            if (result["error"] and result["exit_code"] != 0):
                result_formatted += f"\n```cmd_error\n{result['error']}```"
            if (not result["output"] and result["exit_code"] == 0):
                result_formatted += f"\n# Command executed successfully"

            return result_formatted
    except subprocess.CalledProcessError as e:
        # If a command fails, this block will be executed
        result = {
            'output': e.stdout,
            'error': e.stderr,
            'exit_code': e.returncode
        }
        # Conditional checks on result can be implemented here as needed
        result_formatted = command
        if (result["output"]):
            result_formatted += f"\n```cmd_output\n{result['output']}```"
        if (result["error"]):
            result_formatted += f"\n```cmd_error\n{result['error']}```"

        return result_formatted