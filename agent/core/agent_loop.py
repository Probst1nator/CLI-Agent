"""
Main agent execution loop for CLI-Agent.

This module extracts the main conversation loop from main.py into a
clean, modular component with proper error handling and typing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from termcolor import colored

from agent.commands.base_command import CommandResult
from agent.commands.command_handler import CommandHandler

if TYPE_CHECKING:
    import argparse

    from agent.core.app_context import AppContext

class AgentLoop:
    """
    Main conversation loop for CLI-Agent.

    This class manages the primary REPL (Read-Eval-Print Loop) that was
    previously embedded in the main() function.
    """

    def __init__(self, context: AppContext) -> None:
        self.context = context
        self.logger = logging.getLogger("agent_loop")
        self.command_handler = CommandHandler(context)
        self._running = False

    async def run(
        self,
        args: argparse.Namespace,
        input_event: Optional[asyncio.Event] = None,
        input_lock: Optional[threading.Lock] = None,
        shared_input: Optional[dict[str, Any]] = None,
        tui_callback: Optional[Callable] = None
    ) -> None:
        """
        Run the main agent conversation loop.

        Args:
            args: Parsed command line arguments
            input_event: Event for TUI mode input coordination
            input_lock: Lock for TUI mode thread safety
            shared_input: Shared input dictionary for TUI mode
            tui_callback: Callback for TUI updates
        """
        self._running = True
        context_chat = None

        try:
            # Import here to reduce startup time
            from core.chat import Chat
            context_chat = Chat()
            self.context.current_chat = context_chat

            self.logger.info("Agent loop started")

            while self._running:
                try:
                    # Get user input
                    user_input, base64_images = await self._get_user_input(
                        args, context_chat, input_event, input_lock, shared_input
                    )

                    # Handle empty input or exit conditions
                    if not user_input and not base64_images:
                        continue

                    # Handle commands
                    if user_input and user_input.startswith('/'):
                        command_response = await self.command_handler.handle_command(
                            user_input,
                            args,
                            context_chat=context_chat,
                            command_handler=self.command_handler
                        )

                        # Process command results
                        if command_response.result == CommandResult.EXIT:
                            break
                        elif command_response.result == CommandResult.REGENERATE:
                            user_input = ""  # Trigger regeneration
                        elif command_response.result == CommandResult.SUCCESS:
                            if command_response.base64_images:
                                base64_images = command_response.base64_images
                            if command_response.user_input:
                                user_input = command_response.user_input
                            if not user_input and not base64_images:
                                continue  # Command handled, get next input
                        elif command_response.result == CommandResult.CONTINUE:
                            continue  # Get next input
                        elif command_response.result == CommandResult.ERROR:
                            print(colored(f"Error: {command_response.message}", "red"))
                            continue

                    # Process AI interaction
                    if user_input or base64_images:
                        await self._process_ai_interaction(
                            user_input, base64_images, context_chat, args
                        )

                except KeyboardInterrupt:
                    print(colored("\n⚠️  Operation interrupted by user (Ctrl+C)", "yellow", attrs=['bold']))
                    continue
                except Exception as e:
                    self.logger.exception("Error in agent loop")
                    print(colored(f"Error: {str(e)}", "red"))
                    continue

        except Exception as e:
            self.logger.exception("Fatal error in agent loop")
            print(colored(f"Fatal error: {str(e)}", "red"))
        finally:
            self._running = False
            self.logger.info("Agent loop stopped")

    async def _get_user_input(
        self,
        args: argparse.Namespace,
        context_chat,
        input_event: Optional[asyncio.Event] = None,
        input_lock: Optional[threading.Lock] = None,
        shared_input: Optional[dict[str, Any]] = None
    ) -> tuple[str, list[str]]:
        """Get user input using the enhanced input system."""
        try:
            # Import here to reduce startup time
            from main import get_user_input_with_bindings

            return await get_user_input_with_bindings(
                args=args,
                context_chat=context_chat,
                input_event=input_event,
                input_lock=input_lock,
                shared_input=shared_input
            )
        except Exception as e:
            self.logger.error(f"Error getting user input: {e}")
            return "", []

    async def _process_ai_interaction(
        self,
        user_input: str,
        base64_images: list[str],
        context_chat,
        args: argparse.Namespace
    ) -> None:
        """Process AI model interaction."""
        try:
            # Import here to reduce startup time

            # Add user message to chat
            if user_input:
                from core.chat import Role
                context_chat.add_message(Role.USER, user_input, base64_images)

            # Process with AI models
            await self._process_single_model_interaction(context_chat, args)

        except Exception as e:
            self.logger.exception("Error in AI interaction")
            print(colored(f"AI interaction error: {str(e)}", "red"))


    async def _process_single_model_interaction(
        self,
        context_chat,
        args: argparse.Namespace
    ) -> None:
        """Process interaction using single model."""
        try:
            # Use LLM router for single model interaction
            llm_router = self.context.get_llm_router()

            # Simple single model response
            response = await llm_router.chat_with_retry(
                chat=context_chat,
                model_key=self.context.llm_config.model_key,
                temperature=self.context.llm_config.temperature,
                stream=self.context.llm_config.stream
            )

            if response:
                from core.chat import Role
                context_chat.add_message(Role.ASSISTANT, response)
                print(colored(f"Response: {response}", "green"))
            else:
                print(colored("No response generated", "yellow"))

        except Exception as e:
            self.logger.exception("Single model processing failed")
            print(colored(f"Single model error: {str(e)}", "red"))


    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        self.logger.info("Agent loop stop requested")