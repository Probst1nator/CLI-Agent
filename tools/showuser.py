# tools/showuser.py
"""
This file implements the 'showuser' tool. It allows the agent to render
a self-contained block of HTML, CSS, and JavaScript in the user's default
web browser, providing a way to create simple, adaptive user interfaces.
"""
import webbrowser
import os
import datetime
from pathlib import Path
from core.globals import g

class showuser:
    """
    A tool to render HTML content in the user's default browser.
    Useful for displaying rich content, forms, or interactive elements.
    """
    @staticmethod
    def get_delim() -> str:
        return 'showuser'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "showuser",
            "description": "Renders a self-contained block of HTML, CSS, and JavaScript in the user's default web browser. Ideal for displaying complex information, interactive charts, or requesting user input via forms.",
            "example": "<showuser>\n<!DOCTYPE html>\n<html><body><h1>Hello!</h1></body></html>\n</showuser>"
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Takes a string of HTML, saves it to a temporary file, and opens it
        in the default web browser.
        """
        if not content.strip():
            return "Error: No HTML content provided to display."
            
        try:
            temp_dir = g.CLIAGENT_TEMP_STORAGE_PATH
            os.makedirs(temp_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"showuser_{timestamp}.html"
            filepath = os.path.join(temp_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            file_uri = Path(filepath).as_uri()
            
            webbrowser.open(file_uri)
            
            return f"Success: Content is being displayed in the default web browser. (File: {filepath})"

        except Exception as e:
            return f"Error executing showuser tool: {str(e)}"