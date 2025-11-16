#!/usr/bin/env python3
# main.py
"""
Minimal agent with LLM router and 4 tools: think, todos, readfile, grepsearch.
"""
import importlib.util
import os
import re
from pathlib import Path

from chat import Chat
from llm_router import LLMRouter


class MinimalAgent:
    """A minimal agent that can use tools via LLM."""

    def __init__(self):
        self.chat = Chat()
        self.llm_router = LLMRouter()
        self.tools = {}
        self._load_tools()
        self._initialize_system_prompt()

    def _load_tools(self):
        """Dynamically load all tools from the tools directory."""
        tools_dir = Path(__file__).parent / "tools"

        for tool_file in tools_dir.glob("*.py"):
            if tool_file.name.startswith("_"):
                continue

            module_name = tool_file.stem

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, tool_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for a class with the same name as the module
                if hasattr(module, module_name):
                    tool_class = getattr(module, module_name)
                    if hasattr(tool_class, 'get_delim') and hasattr(tool_class, 'run'):
                        delim = tool_class.get_delim()
                        self.tools[delim] = tool_class
                        print(f"✓ Loaded tool: {delim}")

    def _initialize_system_prompt(self):
        """Create system prompt with tool documentation."""
        tool_docs = []

        for delim, tool_class in self.tools.items():
            if hasattr(tool_class, 'get_tool_info'):
                info = tool_class.get_tool_info()
                tool_doc = f"""
Tool: {info['name']}
Description: {info['description']}
Example: {info['example']}
"""
                tool_docs.append(tool_doc)

        system_prompt = f"""You are a helpful AI assistant with access to the following tools:

{chr(10).join(tool_docs)}

IMPORTANT INSTRUCTIONS:
1. Use tools by wrapping them in XML tags like: <toolname>content</toolname>
2. You can use multiple tools in a single response
3. After using a tool, you will see its output and can continue the conversation
4. The 'think' tool is for your internal reasoning - it produces no output
5. Use 'todos' to track multi-step tasks
6. Use 'readfile' to read file contents with line numbers
7. Use 'grepsearch' to search for patterns in files

Always be helpful, accurate, and use tools when appropriate to accomplish tasks.
"""

        self.chat.add_system_message(system_prompt)

    def _parse_tool_calls(self, text: str) -> list[tuple[str, str, str]]:
        """
        Parse tool calls from LLM response.

        Returns:
            List of (tool_name, content, full_match) tuples
        """
        tool_calls = []

        # Find all tool invocations: <toolname>content</toolname>
        for tool_name in self.tools.keys():
            pattern = f'<{tool_name}>(.*?)</{tool_name}>'
            matches = re.finditer(pattern, text, re.DOTALL)

            for match in matches:
                content = match.group(1).strip()
                full_match = match.group(0)
                tool_calls.append((tool_name, content, full_match))

        return tool_calls

    def _execute_tool(self, tool_name: str, content: str) -> str:
        """Execute a tool and return its output."""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        tool_class = self.tools[tool_name]

        try:
            result = tool_class.run(content)

            # Handle tools that return None (like 'think')
            if result is None:
                return None

            return str(result)

        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def process_message(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.

        This handles the full loop of:
        1. Send message to LLM
        2. Parse tool calls
        3. Execute tools
        4. Get final response
        """
        # Add user message to chat
        self.chat.add_user_message(user_message)

        # Agent loop - continue until no more tool calls
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get LLM response
            messages = self.chat.to_dict_list()

            try:
                response = self.llm_router.generate_completion(messages, stream=False)
            except Exception as e:
                return f"Error calling LLM: {str(e)}"

            # Add assistant response to chat
            self.chat.add_assistant_message(response)

            # Parse and execute tool calls
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # No tool calls - return response to user
                return response

            # Execute all tool calls
            tool_results = []

            for tool_name, content, full_match in tool_calls:
                print(f"\n🔧 Executing: {tool_name}")

                result = self._execute_tool(tool_name, content)

                if result is not None:
                    tool_results.append(f"Result of {tool_name}:\n{result}")

            # If there are no visible results, continue loop
            if not tool_results:
                continue

            # Add tool results as user message
            results_message = "\n\n".join(tool_results)
            self.chat.add_user_message(results_message)

        return "Error: Maximum iterations reached. Agent may be stuck in a loop."

    def run(self):
        """Run the interactive agent loop."""
        print("=" * 60)
        print("Minimal Agent")
        print("=" * 60)
        print(f"\nLoaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")

        providers = self.llm_router.list_available_providers()
        if providers:
            print(f"Available providers: {', '.join(providers)}")
        else:
            print("\n⚠️  No providers configured!")
            print("Configure at least one (in order of preference):")
            print("  1. GOOGLE_API_KEY for Gemini models (recommended)")
            print("  2. Install Ollama for local models (https://ollama.ai/)")
            print("  3. ANTHROPIC_API_KEY for Claude models")
            return

        print("\nType your message and press Enter. Use 'quit' or 'exit' to stop.")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                # Process the message
                print("\n🤖 Agent: ", end="", flush=True)
                response = self.process_message(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def main():
    """Entry point."""
    # Load environment variables from .env file if it exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()

    agent = MinimalAgent()
    agent.run()


if __name__ == "__main__":
    main()
