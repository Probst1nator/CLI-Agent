# tools/getuserfiles.py
import asyncio
import websockets
from termcolor import colored

# Define the connection details as constants
WEBSOCKET_URI = "ws://localhost:8765"

class getuserfiles:
    """
    A tool to retrieve the current file selection and their contents
    from the File Copier GUI application.
    """

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'getuserfiles'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "getuserfiles",
            "description": "Gets the current selection of files and their formatted content from the user's active File Copier GUI session. Useful for getting project context directly from the user.",
            "example": "<getuserfiles></getuserfiles>"
        }

    @staticmethod
    async def _fetch_content_from_gui():
        """The core async logic to connect, receive, and close."""
        try:
            async with websockets.connect(WEBSOCKET_URI, open_timeout=3) as websocket:
                # The server sends the content immediately upon connection.
                # We just need to receive it once.
                content = await asyncio.wait_for(websocket.recv(), timeout=5)
                return content
        except (ConnectionRefusedError, asyncio.TimeoutError):
            return "Error: Could not connect to the File Copier GUI. Please ensure the GUI application is running."
        except websockets.exceptions.ConnectionClosed as e:
            return f"Error: Connection to the File Copier GUI was closed unexpectedly. Reason: {e.code} {e.reason}"
        except Exception as e:
            return f"An unexpected error occurred while communicating with the GUI: {e}"

    @staticmethod
    def run(content: str) -> str:
        """
        Executes the tool by connecting to the GUI's WebSocket server.

        Args:
            content: This tool does not take any arguments inside its tags.

        Returns:
            The string containing the formatted file paths and contents, or an error message.
        """
        print(colored(f"🖥️  Requesting file context from GUI via WebSocket ({WEBSOCKET_URI})...", "cyan"))
        
        # Run the asynchronous client logic
        result = asyncio.run(getuserfiles._fetch_content_from_gui())
        
        if result.startswith("Error:"):
            print(colored(f"   -> {result}", "red"))
        else:
            size_kb = len(result) / 1024
            print(colored(f"   -> Successfully received {size_kb:.1f} KB of context.", "green"))
            
        return result