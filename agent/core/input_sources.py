"""
Concrete input source implementations for CLI-Agent.

This module contains specific implementations of input sources for
CLI, GUI, Web, and Voice interfaces.
"""

from __future__ import annotations
import asyncio
import time
import re
import math
import threading
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from agent.core.input_manager import (
    InputSource_ABC, InputEvent, InputEventType, InputSource
)

if TYPE_CHECKING:
    from agent.core.app_context import AppContext
    from core.chat import Chat
    import argparse


class CLIInputSource(InputSource_ABC):
    """CLI input source using prompt_toolkit."""

    def __init__(self, context: AppContext):
        super().__init__(InputSource.CLI, context)
        self._session = None

    async def get_input(
        self,
        prompt: str = "",
        timeout: Optional[float] = None,
        context_chat: Optional[Chat] = None,
        **kwargs: Any
    ) -> InputEvent:
        """Get input from CLI using prompt_toolkit."""
        try:
            # Import here to avoid circular imports
            from agent.core.command_system import get_prompt_session
            from prompt_toolkit import HTML

            session = get_prompt_session()

            # Clean prompt and format for HTML
            clean_prompt = re.sub(r'\x1b\[[0-9;]*m', '', prompt)

            if context_chat:
                token_info = f"\n[Tokens: {math.ceil(len(context_chat.__str__())*3/4)} | Messages: {len(context_chat.messages)}] "
                html_prompt = HTML(token_info + f'<dim>{clean_prompt}</dim>')
            else:
                html_prompt = HTML(f'<dim>{clean_prompt}</dim>')

            # Get user input
            if timeout:
                user_input = await asyncio.wait_for(
                    session.prompt_async(html_prompt),
                    timeout=timeout
                )
            else:
                user_input = await session.prompt_async(html_prompt)

            # Determine event type
            event_type = InputEventType.COMMAND if user_input.startswith('/') else InputEventType.TEXT

            return InputEvent(
                event_type=event_type,
                source=InputSource.CLI,
                content=user_input,
                metadata={
                    'prompt': prompt,
                    'has_context': context_chat is not None
                },
                timestamp=time.time()
            )

        except ImportError:
            # Fallback to regular input
            user_input = input(prompt)
            event_type = InputEventType.COMMAND if user_input.startswith('/') else InputEventType.TEXT

            return InputEvent(
                event_type=event_type,
                source=InputSource.CLI,
                content=user_input,
                metadata={'prompt': prompt, 'fallback': True},
                timestamp=time.time()
            )

        except KeyboardInterrupt:
            return InputEvent(
                event_type=InputEventType.INTERRUPT,
                source=InputSource.CLI,
                content="",
                metadata={'reason': 'keyboard_interrupt'},
                timestamp=time.time()
            )

    async def cleanup(self) -> None:
        """Clean up CLI input source."""
        self._session = None


class GUIInputSource(InputSource_ABC):
    """GUI input source for prompt_toolkit GUI interface."""

    def __init__(self, context: AppContext):
        super().__init__(InputSource.GUI, context)
        self._input_event: Optional[asyncio.Event] = None
        self._input_lock: Optional[threading.Lock] = None
        self._shared_input: Optional[Dict[str, Any]] = None

    def set_gui_handlers(
        self,
        input_event: asyncio.Event,
        input_lock: threading.Lock,
        shared_input: Dict[str, Any]
    ) -> None:
        """Set GUI event handlers for coordinated input."""
        self._input_event = input_event
        self._input_lock = input_lock
        self._shared_input = shared_input

    async def get_input(
        self,
        prompt: str = "",
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> InputEvent:
        """Get input from GUI interface."""
        if not all([self._input_event, self._input_lock, self._shared_input]):
            raise RuntimeError("GUI handlers not configured")

        self.logger.debug("Waiting for GUI input event...")

        # Wait for GUI input
        if timeout:
            await asyncio.wait_for(self._input_event.wait(), timeout=timeout)
        else:
            await self._input_event.wait()

        # Get the input value
        with self._input_lock:
            user_input = self._shared_input["value"]
            self.logger.debug(f"Retrieved GUI input: '{user_input}'")

        # Clear event for next input
        self._input_event.clear()

        # Determine event type
        event_type = InputEventType.COMMAND if user_input.startswith('/') else InputEventType.TEXT

        return InputEvent(
            event_type=event_type,
            source=InputSource.GUI,
            content=user_input,
            metadata={'prompt': prompt},
            timestamp=time.time()
        )

    async def cleanup(self) -> None:
        """Clean up GUI input source."""
        self._input_event = None
        self._input_lock = None
        self._shared_input = None


class WebInputSource(InputSource_ABC):
    """Web input source for browser-based interface."""

    def __init__(self, context: AppContext):
        super().__init__(InputSource.WEB, context)
        self._pending_inputs: asyncio.Queue = asyncio.Queue()

    async def get_input(
        self,
        prompt: str = "",
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> InputEvent:
        """Get input from web interface."""
        try:
            # HIGHEST PRIORITY: Check the legacy args.message queue from main.py.
            # This bridges the gap between the background InputCollector and the new unified system.
            args = kwargs.get('args')
            if args and args.message:
                from main import _get_next_priority_message
                from termcolor import colored

                user_input = _get_next_priority_message(args.message)
                if user_input:
                    args.message.remove(user_input)
                    print(colored(f"💬 Processing message: {user_input}" + (f" (⏳ {len(args.message)} more queued)" if args.message else ""), 'blue', attrs=['bold']), flush=True)

                    # Send user message to Visual Response System if in GUI mode
                    try:
                        from core.globals import g
                        if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
                            print(f"[DEBUG] GUI mode detected, sending user message: {user_input}")
                            from tools.visualresponse import visualresponse
                            success = visualresponse.send_user_message(user_input)
                            if success:
                                print(f"[DEBUG] ✅ Sent user message to Visual Response System: {user_input[:50]}...")
                            else:
                                print("[DEBUG] ❌ Failed to send user message to Visual Response System")
                        else:
                            print("[DEBUG] GUI mode not active, not sending user message")
                    except Exception as e:
                        print(f"[DEBUG] ❌ Error sending user message to Visual Response System: {e}")
                        import traceback
                        traceback.print_exc()

                    event_type = InputEventType.COMMAND if user_input.startswith('/') else InputEventType.TEXT

                    return InputEvent(
                        event_type=event_type,
                        source=InputSource.WEB,
                        content=user_input.strip(),
                        metadata={'prompt': prompt, 'source_queue': 'args.message'},
                        timestamp=time.time()
                    )

            # If no legacy messages, proceed with polling the new visual_input_queue.
            from core.globals import g
            from queue import Empty

            if not hasattr(g, 'visual_input_queue'):
                raise RuntimeError("Visual input queue not available")

            self.logger.info("🌐 Waiting for browser input...")
            start_time = time.time()

            while True:
                # Check for background input collector commands during the wait
                try:
                    from main import get_input_collector
                    input_collector = get_input_collector()
                    if input_collector and input_collector.has_pending_commands():
                        # Return special event to signal checking args.message
                        return InputEvent(
                            event_type=InputEventType.SYSTEM,
                            source=InputSource.WEB,
                            content="",
                            metadata={'source_queue': 'input_collector', 'action': 'check_args_message'},
                            timestamp=time.time()
                        )
                except ImportError:
                    pass  # Input collector not available

                try:
                    # Wait for input from the browser with a timeout to remain responsive
                    raw_input = g.visual_input_queue.get(timeout=0.1)

                    # Handle structured interrupt commands
                    if isinstance(raw_input, dict) and raw_input.get('type') == 'interrupt_command':
                        self.logger.info("🛑 Received interrupt command from web UI")
                        try:
                            from main import _interrupt_flag
                            _interrupt_flag.set()
                        except ImportError:
                            pass  # Interrupt flag not available

                        # Force exit from autonomous mode
                        if hasattr(g, 'autonomous_iteration_count'):
                            g.autonomous_iteration_count = 999

                        print("🛑 INTERRUPT SIGNAL RECEIVED - Cancelling AI generation...")
                        print("  → Provider cancellation initiated")
                        print("  → Breaking out of autonomous mode and returning to user input...")

                        return InputEvent(
                            event_type=InputEventType.INTERRUPT,
                            source=InputSource.WEB,
                            content="interrupt",
                            metadata={'type': 'interrupt_command', 'source': 'web_ui'},
                            timestamp=time.time()
                        )

                    # Handle LLM selection updates from web UI
                    if isinstance(raw_input, dict) and raw_input.get('type') == 'llm_selection_update':
                        self.logger.info("🎯 Received LLM selection update from web UI")
                        self.logger.info(f"🔍 Raw input: {raw_input}")
                        selection_data = raw_input.get('data', {})
                        self.logger.info(f"🔍 Selection data type: {type(selection_data)}, content: {selection_data}")


                        try:
                            # Ensure selection_data is a dictionary
                            if not isinstance(selection_data, dict):
                                # Try to parse as JSON if it's a string
                                if isinstance(selection_data, str):
                                    try:
                                        import json
                                        selection_data = json.loads(selection_data)
                                        self.logger.info("🔄 Successfully parsed selection_data from JSON string")
                                    except json.JSONDecodeError as e:
                                        self.logger.error(f"❌ Failed to parse selection_data as JSON: {e}")
                                        continue
                                else:
                                    self.logger.error(f"❌ Expected selection_data to be dict, got {type(selection_data)}: {selection_data}")
                                    continue

                            # Verify it's now a dictionary
                            if not isinstance(selection_data, dict):
                                self.logger.error(f"❌ selection_data is still not a dict after processing: {type(selection_data)}")
                                continue

                            # Update the system configuration
                            new_fallback_models = []
                            for key in ['primary', 'fallback1', 'fallback2']:
                                if selection_data.get(key):
                                    new_fallback_models.append(selection_data[key])

                            # Update g.SELECTED_LLMS to match the UI selection
                            if new_fallback_models:
                                # Keep specialized models and add UI fallback models
                                llm_config = g.get_llm_config()
                                self.logger.info(f"🔍 LLM config structure: {type(llm_config)} with {len(llm_config)} items")
                                specialized_models = []

                                # Find specialized models to preserve
                                for model_key, config in llm_config.items():
                                    # Skip special keys that aren't model configs
                                    if model_key.startswith('__') and model_key.endswith('__'):
                                        continue

                                    # Debug: Check if config is actually a dict
                                    if not isinstance(config, dict):
                                        self.logger.warning(f"⚠️ Model {model_key} has non-dict config: {type(config)} = {config}")
                                        continue

                                    # No specialized roles in simplified system

                                # Create new selection list: specialized models + UI fallback models
                                new_selected_llms = specialized_models + new_fallback_models

                                # Remove duplicates while preserving order
                                unique_selected = []
                                seen = set()
                                for model in new_selected_llms:
                                    if model not in seen:
                                        unique_selected.append(model)
                                        seen.add(model)

                                # Update global selection
                                old_selection = g.SELECTED_LLMS.copy()
                                g.SELECTED_LLMS = unique_selected

                                self.logger.info("🔄 Updated LLM selection from UI:")
                                self.logger.info(f"  → Old fallback models: {len(old_selection)} total")
                                self.logger.info(f"  → New fallback models: {new_fallback_models}")
                                self.logger.info(f"  → Total selected: {len(unique_selected)} models")

                                # Always update the persistent configuration
                                try:
                                    # Update the LLM config to match the new selection
                                    updated_config = g.get_llm_config(force_reread=True)

                                    # Mark all models as unselected first
                                    for model_key in updated_config:
                                        # Skip special keys that aren't model configs
                                        if model_key.startswith('__') and model_key.endswith('__'):
                                            continue
                                        # Ensure the config is a dictionary
                                        if not isinstance(updated_config[model_key], dict):
                                            self.logger.warning(f"⚠️ Converting non-dict config for {model_key} to dict")
                                            updated_config[model_key] = {"selected": False}
                                        else:
                                            updated_config[model_key]['selected'] = False

                                    # Mark the selected models as selected and save their order
                                    for i, model in enumerate(unique_selected):
                                        if model in updated_config:
                                            updated_config[model]['selected'] = True
                                            updated_config[model]['order'] = i
                                        else:
                                            # New model, add as selected fallback with order
                                            updated_config[model] = {"selected": True, "order": i}

                                    # Save the updated configuration
                                    g.save_llm_config(updated_config)
                                    self.logger.info("💾 Saved updated LLM configuration to disk")

                                except Exception as e:
                                    self.logger.error(f"❌ Failed to save LLM configuration: {e}")

                        except Exception as e:
                            import traceback
                            self.logger.error(f"Error processing LLM selection update: {e}")
                            self.logger.error(f"Full traceback: {traceback.format_exc()}")

                        # Continue without treating this as user input
                        continue

                    # Handle string inputs
                    if isinstance(raw_input, str):
                        user_input = raw_input.strip()

                        if user_input == "__AGENT_RESET__":
                            self.logger.info("🔄 Received agent reset request from browser")
                            g.AGENT_RESET_REQUESTED = True
                            return InputEvent(
                                event_type=InputEventType.SYSTEM,
                                source=InputSource.WEB,
                                content="reset",
                                metadata={'type': 'agent_reset', 'source': 'web_ui'},
                                timestamp=time.time()
                            )

                        if user_input:
                            self.logger.info(f"🌐 Received browser input: {user_input[:50]}...")
                            event_type = InputEventType.COMMAND if user_input.startswith('/') else InputEventType.TEXT
                            return InputEvent(
                                event_type=event_type,
                                source=InputSource.WEB,
                                content=user_input,
                                metadata={'prompt': prompt},
                                timestamp=time.time()
                            )
                    else:
                        # Handle non-string inputs (like dicts)
                        self.logger.warning(f"Received non-string input: {type(raw_input)} - {raw_input}")
                        continue

                except Empty:
                    # Timeout occurred, loop again. This allows us to re-check for legacy messages
                    # if they arrive while waiting.
                    if timeout and (time.time() - start_time) >= timeout:
                        raise asyncio.TimeoutError("Web input timeout")
                    # Before continuing, check the legacy queue again in case a message arrived.
                    if args and args.message:
                        return await self.get_input(prompt, timeout, **kwargs)
                    continue
                except Exception as e:
                    self.logger.error(f"Error checking visual input queue: {e}")
                    await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error getting web input: {e}")
            raise RuntimeError(f"Web input error: {e}")

    async def submit_web_input(self, input_data: str) -> None:
        """Submit input from web interface."""
        await self._pending_inputs.put(input_data)

    async def cleanup(self) -> None:
        """Clean up web input source."""
        # Clear any pending inputs
        while not self._pending_inputs.empty():
            try:
                self._pending_inputs.get_nowait()
            except asyncio.QueueEmpty:
                break


class VoiceInputSource(InputSource_ABC):
    """Voice input source with speech recognition."""

    def __init__(self, context: AppContext):
        super().__init__(InputSource.VOICE, context)

    async def get_input(
        self,
        prompt: str = "",
        timeout: Optional[float] = None,
        args: Optional[argparse.Namespace] = None,
        **kwargs: Any
    ) -> InputEvent:
        """Get input via voice recognition."""
        try:
            # Import voice functions
            from main import get_listen_microphone, text_to_speech

            listen_microphone = get_listen_microphone()

            # Clean prompt for TTS
            clean_prompt = re.sub(r'\x1b\[[0-9;]*m', '', prompt).strip()

            if clean_prompt == ">":
                clean_prompt = "Please state your choice."
            elif clean_prompt.endswith(":"):
                clean_prompt = clean_prompt[:-1]

            # Handle common prompt patterns
            if '(y/n)' in clean_prompt:
                clean_prompt = clean_prompt.replace('(y/n)', 'Please say yes or no.')
            elif '(y=execute, n=abort, c=refine)' in clean_prompt:
                clean_prompt = clean_prompt.replace(
                    '(y=execute, n=abort, c=refine)',
                    'Please say execute, abort, or refine.'
                )
            elif 'Permissions>' in clean_prompt:
                clean_prompt = "Permissions manager. What would you like to do?"

            self.logger.info(f"🎤 Asking: \"{clean_prompt}\"")

            # Speak the prompt
            text_to_speech(text=clean_prompt, speed=0.8, volume=0.7, use_piper=True)

            # Listen for response
            transcription, _, _ = listen_microphone(
                private_remote_wake_detection=getattr(args, 'private_remote_wake_detection', False),
                remote_voice_enabled=getattr(args, 'online_voice', False)
            )

            self.logger.info(f"🎤 User replied: \"{transcription}\"")

            # Map voice keywords to commands
            mapped_command = self._map_voice_keywords(transcription)

            return InputEvent(
                event_type=InputEventType.TEXT,
                source=InputSource.VOICE,
                content=mapped_command,
                metadata={
                    'prompt': prompt,
                    'original_transcription': transcription,
                    'mapped': mapped_command != transcription
                },
                timestamp=time.time()
            )

        except ImportError as e:
            raise RuntimeError(f"Voice input not available: {e}")

    def _map_voice_keywords(self, transcription: str) -> str:
        """Map voice keywords to command shortcuts."""
        keyword_map = {
            'y': ['yes', 'yeah', 'yep', 'confirm', 'konfirm', 'allow', 'okay', 'ok', 'go', 'execute', 'proceed'],
            'n': ['no', 'nope', 'stop', 'abort', 'cancel', 'deny'],
            'a': ['always', 'always allow', 'always permit'],
            'd': ['always deny'],
            'm': ['manage', 'manager', 'manage permissions'],
            '?': ['help', 'options', 'what are the options'],
            'c': ['refine', 'change', 'correct', 'custom', 'custom pattern'],
            '1': ['exact', 'exact command', 'one'],
            '2': ['code', 'code execution', 'two'],
            '3': ['files', 'file management', 'three'],
            '4': ['shell', 'bash', 'shell operations', 'four'],
            '5': ['pattern', 'five'],
            'list': ['list', 'show'],
            'remove': ['remove', 'delete'],
            'clear': ['clear', 'reset'],
            'exit': ['exit', 'quit', 'back'],
            'x': ['cancel', 'exit', 'back']
        }

        user_reply = transcription.lower().strip()
        for command, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in user_reply:
                    self.logger.info(f"🎤 Interpreted \"{transcription}\" as command: '{command}'")
                    return command

        # No mapping found, return original
        return transcription

    async def cleanup(self) -> None:
        """Clean up voice input source."""
        pass


class QueueInputSource(InputSource_ABC):
    """Queued/programmatic input source for automated inputs."""

    def __init__(self, context: AppContext):
        super().__init__(InputSource.QUEUE, context)
        self._message_queue: List[str] = []

    def add_messages(self, messages: List[str]) -> None:
        """Add messages to the queue."""
        self._message_queue.extend(messages)
        self.logger.debug(f"Added {len(messages)} messages to queue")

    def has_messages(self) -> bool:
        """Check if there are queued messages."""
        return len(self._message_queue) > 0

    async def get_input(
        self,
        prompt: str = "",
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> InputEvent:
        """Get input from message queue."""
        if not self._message_queue:
            raise RuntimeError("No queued messages available")

        # Get next message
        message = self._message_queue.pop(0)
        remaining = len(self._message_queue)

        self.logger.info(f"📋 Processing queued command: {message}" +
                        (f" (⏳ {remaining} more queued)" if remaining else ""))

        # Send user message to Visual Response System if in GUI mode (but skip system messages)
        try:
            from core.globals import g
            if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
                # Check if this is a system message that shouldn't be displayed as user input
                is_system_message = False
                try:
                    import ast
                    import json
                    # Try to parse as Python dict or JSON to check for system message types
                    if message.strip().startswith('{'):
                        parsed = None
                        try:
                            # First try JSON
                            parsed = json.loads(message)
                        except json.JSONDecodeError:
                            # Then try Python dict syntax
                            parsed = ast.literal_eval(message)

                        if isinstance(parsed, dict) and parsed.get('type') in ['llm_selection_update']:
                            is_system_message = True
                            print(f"[DEBUG] Detected system message type '{parsed.get('type')}', not sending to visual response")
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    # Not a valid dict/JSON, treat as regular user message
                    pass

                if not is_system_message:
                    print(f"[DEBUG] GUI mode detected, sending queued user message: {message}")
                    from tools.visualresponse import visualresponse
                    success = visualresponse.send_user_message(message)
                    if success:
                        print(f"[DEBUG] ✅ Sent queued user message to Visual Response System: {message[:50]}...")
                    else:
                        print("[DEBUG] ❌ Failed to send queued user message to Visual Response System")
            else:
                print("[DEBUG] GUI mode not active, not sending queued user message")
        except Exception as e:
            print(f"[DEBUG] ❌ Error sending queued user message to Visual Response System: {e}")
            import traceback
            traceback.print_exc()

        # Determine event type
        event_type = InputEventType.COMMAND if message.startswith('/') else InputEventType.TEXT

        return InputEvent(
            event_type=event_type,
            source=InputSource.QUEUE,
            content=message,
            metadata={
                'remaining_count': remaining,
                'total_processed': len(self._message_queue) + 1
            },
            timestamp=time.time()
        )

    async def cleanup(self) -> None:
        """Clean up queue input source."""
        self._message_queue.clear()