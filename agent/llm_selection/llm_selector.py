# py_classes/cls_llm_selection.py
"""
LLM Selection interface for the CLI-Agent application.
Provides a full-screen, interactive, multi-column UI for selecting and configuring
language models with status indicators, persistence, and live editing.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# --- prompt_toolkit imports ---
from prompt_toolkit import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import (HTML, FormattedText,
                                            to_formatted_text)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Label, TextArea


class LlmSelector:
    """
    Interactive LLM selection interface with multi-selection and multi-column display.
    Handles LLM status checking, selection persistence, and UI for choosing models.
    """

    def __init__(self):
        """Initialize the LLM selector."""
        self.config_file = None
        try:
            from core.globals import g
            # Use the consistent config path from globals
            self.config_file = Path(g.LLM_CONFIG_PATH)
        except (ImportError, AttributeError):
            # Fallback for standalone execution
            self.config_file = Path.home() / ".cli-agent" / "llm_config.json"
            logging.info("Running in standalone mode, using default config path.")

        self.data = []
        self.filtered_data = []  # Data after applying search filter
        self.current_row = 0
        self.current_col = 0  # 0: Selector, 1: Host, 2: Branches, 3: Eval, 4: Guard
        self.editing = False
        self.scroll_offset = 0  # For viewport scrolling
        self.edit_buffer = ""
        self.force_local_mode = False
        
        # Search functionality
        self.search_box = None  # Will be initialized when UI is created
        
        # Selection order tracking
        self.selection_order = {}  # model_key -> order_index
        self.next_selection_order = 1  # Next order index to assign
        
        
        # Focus management
        self.current_focus = "table"  # "table" or "search"
        
        # Cache for Ollama status
        self.ollama_status = {}
        
        # Discovered models cache path
        self.discovered_models_file = None
        try:
            from core.globals import g
            self.discovered_models_file = Path(g.LLM_CONFIG_PATH).parent / "discovered_models.json"
        except (ImportError, AttributeError):
            self.discovered_models_file = Path.home() / ".cli-agent" / "discovered_models.json"

    def _apply_search_filter(self):
        """Apply search filter to data and update filtered_data."""
        search_text = self.search_box.text.strip() if self.search_box else ""
        
        if not search_text:
            # No search term, show all data
            self.filtered_data = self.data.copy()
        else:
            # Filter by model name containing search term (case insensitive)
            search_term = search_text.lower().strip()
            self.filtered_data = []
            for item in self.data:
                model_key = item.get('model_key', '').lower()
                provider = item.get('provider', '').lower()
                
                # Primary match: model name contains search term
                if search_term in model_key:
                    self.filtered_data.append(item)
                # Secondary match: provider name (but exclude generic matches like "client")
                elif (search_term in provider and 
                      search_term not in ['client', 'llama'] and  # Avoid matching "OllamaClient" when searching "llama"
                      len(search_term) > 2):  # Avoid very short provider matches
                    self.filtered_data.append(item)
                # Specific provider searches
                elif ((search_term == 'ollama' and 'ollama' in provider) or
                      (search_term == 'openai' and 'openai' in provider) or
                      (search_term == 'anthropic' and 'anthropic' in provider)):
                    self.filtered_data.append(item)
        
        # Reset cursor to top of filtered results
        if self.filtered_data:
            self.current_row = min(self.current_row, len(self.filtered_data) - 1)
        else:
            self.current_row = 0
        self.scroll_offset = 0  # Reset scroll when filter changes

    def _get_usable_terminal_lines(self) -> int:
        """Calculate how many lines are available for displaying model rows."""
        try:
            import os
            terminal_height = os.get_terminal_size().lines
        except:
            terminal_height = 25  # Fallback if we can't get terminal size
        
        # Reserve space for: header(1) + status(1) + column header(1) + separator(1) + search box(1) + separator(1) + help(1) = 7 lines
        reserved_lines = 7  # Total UI chrome
        return max(10, terminal_height - reserved_lines)  # Minimum 10 lines for models

    def _get_dynamic_column_widths(self) -> dict:
        """Calculate dynamic column widths based on terminal width."""
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except:
            terminal_width = 120  # Fallback if we can't get terminal size
        
        # Base column widths (minimum required)
        base_widths = {
            'cursor': 1,           # "▶"
            'active': 6,           # "[✓] "
            'provider': 15,        # "Provider"
            'model': 35,           # "Model" (base)
            'host': 20,           # "Host" (base)
            'branches': 8,         # "Branches"
            'eval': 6,            # "Eval"
            'guard': 6,           # "Guard"
            'pricing': 9,        # "Pricing"
            'vision': 6,          # "Vision"
            'context': 10,        # "Context"
            'status': 15,         # "Status" (estimated)
        }
        
        # Calculate used space with base widths and gaps
        base_used = sum(base_widths.values())
        base_gaps = 11  # Number of spaces between columns
        total_base = base_used + base_gaps
        
        # Calculate extra space available
        extra_space = max(0, terminal_width - total_base)
        
        # Distribute extra space to Model and Host columns (70% to Model, 30% to Host)
        model_extra = int(extra_space * 0.7)
        host_extra = int(extra_space * 0.3)
        
        # Apply the extra space
        widths = base_widths.copy()
        widths['model'] += model_extra
        widths['host'] += host_extra
        
        return widths

    def _adjust_scroll_for_cursor(self):
        """Adjust scroll offset to keep cursor visible and allow access to header."""
        usable_lines = self._get_usable_terminal_lines()
        header_lines = 4  # Just the table header portion
        
        # If cursor is above the current scroll window, scroll up
        if self.current_row < self.scroll_offset:
            self.scroll_offset = self.current_row
        
        # If cursor is below the current scroll window, scroll down
        elif self.current_row >= self.scroll_offset + usable_lines:
            self.scroll_offset = self.current_row - usable_lines + 1
            
        # Always allow scrolling back to see the header (scroll_offset can be negative)
        # But don't scroll too far up beyond what's useful
        self.scroll_offset = max(-header_lines, self.scroll_offset)

    

    def _update_activation_status(self, row_index: int):
        """Activates a model if its beams, eval, or guard count is positive."""
        row_data = self.data[row_index]
        if row_data.get('beams', 0) > 0 or row_data.get('eval', 0) > 0 or row_data.get('guard', 0) > 0:
            row_data['selected'] = True

    def _reset_deselected_model(self, row_index: int):
        """Resets all counts for a model if it's deselected."""
        row_data = self.data[row_index]
        if not row_data.get('selected'):
            row_data['beams'] = 0
            row_data['eval'] = 0
            row_data['guard'] = 0
    
    def _check_vision_capability(self, model_key: str) -> bool:
        """
        Check if a model supports vision/multimodal capabilities.
        
        Args:
            model_key (str): The model identifier to check
            
        Returns:
            bool: True if the model supports vision, False otherwise
        """
        try:
            # Check if this is an Ollama model first
            if hasattr(self, 'data'):
                for row in self.data:
                    if row.get('model_key') == model_key and row.get('provider') != 'OllamaClient':
                        # For non-Ollama models, check known vision models
                        model_lower = model_key.lower()
                        vision_keywords = ['vision', 'gpt-4o', 'gpt-4-turbo', 'claude-3', 'gemini-pro-vision']
                        return any(keyword in model_lower for keyword in vision_keywords)
            
            # Try to get model info from Ollama
            try:
                import ollama
                from core.globals import g
                
                # Try each reachable Ollama host
                for host_url in g.DEFAULT_OLLAMA_HOSTS:
                    try:
                        if "://" in host_url:
                            host = host_url.split("://")[1].split("/")[0]
                        else:
                            host = host_url
                            
                        if not self.check_host_reachability(host.split(':')[0]):
                            continue
                            
                        client = ollama.Client(host=f"http://{host}")
                        model_info = client.show(model_key)
                        
                        # Check various fields that might indicate vision capability
                        if hasattr(model_info, 'modelfile') and model_info.modelfile:
                            modelfile_lower = str(model_info.modelfile).lower()
                            if any(keyword in modelfile_lower for keyword in ['vision', 'multimodal', 'image']):
                                return True
                        
                        if hasattr(model_info, 'template') and model_info.template:
                            template_lower = str(model_info.template).lower()
                            if any(keyword in template_lower for keyword in ['vision', 'multimodal', 'image']):
                                return True
                                
                        if hasattr(model_info, 'parameters') and model_info.parameters:
                            params = model_info.parameters
                            for key, value in params.items():
                                if any(keyword in str(key).lower() or keyword in str(value).lower() 
                                      for keyword in ['vision', 'multimodal', 'image']):
                                    return True
                        
                        # Check model name for vision indicators
                        model_lower = model_key.lower()
                        vision_keywords = [
                            'vision', 'visual', 'multimodal', 'llava', 'minicpm-v', 'qwen2-vl', 
                            'internvl', 'cogvlm', 'blip', 'instructblip', 'bakllava'
                        ]
                        if any(keyword in model_lower for keyword in vision_keywords):
                            return True
                        
                        # Successfully checked one host, break
                        break
                        
                    except Exception:
                        continue  # Try next host
                        
            except Exception:
                pass  # Fallback to name-based detection
                
            # Fallback: Check model name for common vision model patterns
            model_lower = model_key.lower()
            vision_keywords = [
                'vision', 'visual', 'multimodal', 'llava', 'minicpm-v', 'qwen2-vl', 
                'internvl', 'cogvlm', 'blip', 'instructblip', 'bakllava'
            ]
            return any(keyword in model_lower for keyword in vision_keywords)
            
        except Exception:
            return False

    def _load_discovered_models(self) -> Dict[str, Any]:
        """Load previously discovered models from cache."""
        if not self.discovered_models_file or not self.discovered_models_file.exists():
            return {}
        
        try:
            with open(self.discovered_models_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load discovered models: {e}")
            return {}

    def _save_discovered_models(self, discovered_models: Dict[str, Any]) -> None:
        """Save discovered models to cache with timestamp, replacing previous cache completely."""
        if not self.discovered_models_file:
            return
            
        try:
            # Ensure directory exists
            self.discovered_models_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata - this completely replaces the previous cache
            data_to_save = {
                'timestamp': datetime.now().isoformat(),
                'models': discovered_models
            }
            
            # Atomic write to prevent corruption during concurrent access
            temp_file = self.discovered_models_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            # Replace the old file with the new one atomically
            temp_file.replace(self.discovered_models_file)
                
        except Exception as e:
            logging.warning(f"Failed to save discovered models: {e}")

    async def _update_models_with_progress(self) -> bool:
        """
        Show progress interface and discover new models.
        Returns True if models were updated, False if cancelled or failed.
        """
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.layout import Layout
        from prompt_toolkit.formatted_text import HTML, to_formatted_text
        from core.llm import Llm
        import asyncio
        
        # Progress tracking state
        progress_state = {
            'providers': {
                'Google': {'status': 'waiting', 'models': 0},
                'Groq': {'status': 'waiting', 'models': 0},
                'Ollama': {'status': 'waiting', 'models': 0}
            },
            'completed': False,
            'cancelled': False,
            'results': None
        }
        
        def get_progress_display():
            """Generate the progress display."""
            fragments = []
            
            # Header
            fragments.extend(to_formatted_text(HTML("<b><ansicyan>🔄 Discovering Models</ansicyan></b>")))
            fragments.append(('', '\n\n'))
            
            # Provider status
            for provider, info in progress_state['providers'].items():
                status = info['status']
                models = info['models']
                
                if status == 'waiting':
                    icon = "⏳"
                    color = "ansibrightblack"
                    status_text = "Waiting..."
                elif status == 'starting':
                    icon = "🔍"
                    color = "ansiyellow"
                    status_text = "Discovering..."
                elif status == 'success':
                    icon = "✅"
                    color = "ansigreen"
                    # Show both new and total counts
                    total_models = info.get('total', models)  # fallback to models count if total not available
                    if models == 0 and total_models > 0:
                        status_text = f"No new models ({total_models} total)"
                    elif models == total_models:
                        status_text = f"Found {models} new models"
                    else:
                        status_text = f"Found {models} new, {total_models} total"
                elif status == 'error':
                    icon = "❌"
                    color = "ansired"
                    status_text = "Discovery failed"
                elif status == 'skipped':
                    icon = "⏭️"
                    color = "ansibrightblack"
                    status_text = "Skipped (no API key)"
                else:
                    icon = "?"
                    color = "ansibrightblack"
                    status_text = status
                
                fragments.extend(to_formatted_text(HTML(f"<{color}>{icon} {provider:<8} {status_text}</{color}>")))
                fragments.append(('', '\n'))
            
            fragments.append(('', '\n'))
            
            # Instructions
            if progress_state['completed']:
                if progress_state['results']:
                    total = progress_state['results'].get('total_discovered', 0)
                    if total > 0:
                        fragments.extend(to_formatted_text(HTML(f"<ansigreen><b>🎉 Discovered {total} new models!</b></ansigreen>")))
                    else:
                        fragments.extend(to_formatted_text(HTML("<ansiyellow><b>ℹ️  No new models found (all may be cached)</b></ansiyellow>")))
                else:
                    fragments.extend(to_formatted_text(HTML("<ansired><b>❌ Discovery failed</b></ansired>")))
                
                fragments.append(('', '\n\n'))
                fragments.extend(to_formatted_text(HTML("<ansibrightblack>Press any key to continue...</ansibrightblack>")))
            else:
                fragments.extend(to_formatted_text(HTML("<ansibrightblack>Discovering models from APIs...</ansibrightblack>")))
                fragments.append(('', '\n'))
                fragments.extend(to_formatted_text(HTML("<ansibrightblack>Press Esc to cancel</ansibrightblack>")))
            
            return fragments
        
        # Progress callback for discovery
        def progress_callback(provider: str, status: str, models_found: int, total_models: int = None):
            progress_state['providers'][provider]['status'] = status
            progress_state['providers'][provider]['models'] = models_found
            if total_models is not None:
                progress_state['providers'][provider]['total'] = total_models
        
        # Key bindings for progress interface
        bindings = KeyBindings()
        
        @bindings.add('escape')
        def cancel_discovery(event):
            if not progress_state['completed']:
                progress_state['cancelled'] = True
            event.app.exit(result=False)
        
        @bindings.add('<any>')
        def any_key(event):
            if progress_state['completed']:
                event.app.exit(result=True)
        
        # Create progress interface
        control = FormattedTextControl(text=get_progress_display, key_bindings=bindings, focusable=True)
        window = Window(content=control, wrap_lines=False)
        layout = Layout(window)
        app = Application(layout=layout, key_bindings=bindings, full_screen=True, refresh_interval=0.1)
        
        # Start discovery in background
        async def run_discovery():
            await asyncio.sleep(0.5)  # Brief delay for UI to show
            
            if not progress_state['cancelled']:
                try:
                    results = Llm.discover_models_with_progress(progress_callback)
                    progress_state['results'] = results
                    progress_state['completed'] = True
                    
                    # Always save discovered models to update the cache with current state
                    # This ensures the cache reflects what's actually available now
                    self._save_discovered_models(results)
                    
                except Exception as e:
                    logging.error(f"Model discovery failed: {e}")
                    progress_state['completed'] = True
        
        # Run discovery and UI concurrently
        discovery_task = asyncio.create_task(run_discovery())
        
        try:
            result = await app.run_async()
            discovery_task.cancel()
            # Return True if discovery completed successfully (regardless of whether new models were found)
            # This ensures the cache is always updated with the current remote state
            return result and progress_state['completed'] and progress_state['results'] is not None
        except Exception as e:
            discovery_task.cancel()
            logging.error(f"Progress interface failed: {e}")
            return False

    def _add_discovered_models_to_data(self):
        """Add discovered models from cache to the current data list."""
        discovered_cache = self._load_discovered_models()
        if not discovered_cache or 'models' not in discovered_cache:
            logging.debug("No discovered models cache found or invalid format")
            return
            
        logging.debug(f"Loading discovered models from cache: {len(discovered_cache.get('models', {}).get('providers', {}))} providers")
        
        discovered_models = discovered_cache['models']
        saved_config = self._load_config()
        
        # Get existing model keys to avoid duplicates
        existing_models = {row['model_key'] for row in self.data}
        
        # Add discovered Google models
        google_results = discovered_models.get('providers', {}).get('Google', {})
        if google_results.get('status') == 'success':
            for model_info in google_results.get('models', []):
                model_key = model_info['name']
                if model_key not in existing_models:
                    model_config = saved_config.get(model_key, {})
                    
                    self.data.append({
                        "selected": model_config.get("selected", False),
                        "beams": model_config.get("beams", 0),
                        "eval": model_config.get("eval", 0),
                        "guard": model_config.get("guard", 0),
                        "provider": "GoogleAPI",
                        "model_key": model_key,
                        "pricing_str": "Free",  # Google pricing varies, simplified
                        "has_vision": 'VISION' in model_info.get('strengths', []),
                        "context_window": model_info['context_window'],
                        "download_status_text": "",
                        "assigned_host": "cloud",
                    })
                    existing_models.add(model_key)
                    logging.debug(f"Added discovered Google model: {model_key}")
        
        # Add discovered Groq models
        groq_results = discovered_models.get('providers', {}).get('Groq', {})
        if groq_results.get('status') == 'success':
            for model_info in groq_results.get('models', []):
                model_key = model_info['name']
                if model_key not in existing_models:
                    model_config = saved_config.get(model_key, {})
                    
                    self.data.append({
                        "selected": model_config.get("selected", False),
                        "beams": model_config.get("beams", 0),
                        "eval": model_config.get("eval", 0),
                        "guard": model_config.get("guard", 0),
                        "provider": "GroqAPI",
                        "model_key": model_key,
                        "pricing_str": "Free",
                        "has_vision": 'VISION' in model_info.get('strengths', []),
                        "context_window": model_info['context_window'],
                        "download_status_text": "",
                        "assigned_host": "cloud",
                    })
                    existing_models.add(model_key)
                    logging.debug(f"Added discovered Groq model: {model_key}")
            
    @classmethod
    def check_host_reachability(cls, host: str) -> bool:
        """
        Simple host reachability check for internal use.
        """
        try:
            import socket
            hostname, port_str = host.split(':') if ':' in host else (host, '11434')
            port = int(port_str)
            with socket.create_connection((hostname, port), timeout=1):
                return True
        except:
            return False
    
    def _estimate_context_window(self, model_key: str) -> int:
        """
        Estimate context window length based on model name patterns.
        Returns reasonable estimates for common Ollama models.
        """
        model_lower = model_key.lower()
        
        # Context window estimates based on model families
        if any(x in model_lower for x in ['llama3.3', 'llama3.2', 'llama3.1']):
            return 128000  # Most Llama 3.x models support 128k context
        elif 'llama3' in model_lower:
            return 8192    # Base Llama 3 models
        elif any(x in model_lower for x in ['llama2', 'llama']):
            return 4096    # Llama 2 and earlier
            
        elif any(x in model_lower for x in ['qwen2.5', 'qwen3']):
            return 128000  # Qwen 2.5+ supports long context
        elif 'qwen2' in model_lower:
            return 32768   # Qwen 2
        elif 'qwen' in model_lower:
            return 8192    # Earlier Qwen models
            
        elif any(x in model_lower for x in ['mistral', 'mixtral']):
            if 'mixtral' in model_lower:
                return 32768  # Mixtral MoE models
            else:
                return 32768  # Mistral models
                
        elif any(x in model_lower for x in ['phi3.5', 'phi-3.5']):
            return 128000  # Phi 3.5 long context
        elif any(x in model_lower for x in ['phi3', 'phi-3']):
            return 4096    # Phi 3 base models
        elif 'phi' in model_lower:
            return 2048    # Earlier Phi models
            
        elif any(x in model_lower for x in ['gemma2', 'gemma-2']):
            return 8192    # Gemma 2
        elif 'gemma' in model_lower:
            return 8192    # Gemma models
            
        elif any(x in model_lower for x in ['codellama', 'code-llama']):
            return 16384   # CodeLlama models
        elif any(x in model_lower for x in ['deepseek-coder', 'deepseek']):
            return 16384   # DeepSeek Coder
            
        elif any(x in model_lower for x in ['embed', 'embedding']):
            return 512     # Embedding models typically have smaller context
            
        else:
            return 8192    # Conservative default for unknown models

    def _get_display_lines(self) -> FormattedText:
        """
        Generate the formatted text for the interactive table.
        This must return a list of (style, text) tuples.
        """
        all_fragments = []
        
        app_header = "CLI-Agent - model selection"
        # Center the header for better presentation
        all_fragments.extend(to_formatted_text(HTML(f"<b><i>{app_header.center(130)}</i></b>")))
        all_fragments.append(('', '\n'))
        
        
        # Add status line
        search_text = self.search_box.text.strip() if self.search_box else ""
        if search_text:
            match_count = len(self.filtered_data)
            total_count = len(self.data)
            focus_indicator = " | 🎯 Search focused" if self.current_focus == "search" else ""
            search_line = f"🔍 Filter: '{search_text}' ({match_count}/{total_count} models){focus_indicator}"
            all_fragments.extend(to_formatted_text(HTML(f"<b><ansigreen>{search_line}</ansigreen></b>")))
        else:
            focus_indicator = "🎯 Search focused | " if self.current_focus == "search" else ""
            search_line = f"🔍 Search: {focus_indicator}F: Focus search | Tab: Switch focus"
            color = "ansigreen" if self.current_focus == "search" else "ansibrightblack"
            all_fragments.extend(to_formatted_text(HTML(f"<{color}>{search_line}</{color}>")))
        all_fragments.append(('', '\n'))
        
        # Add a column header with dynamic widths that exactly matches data row structure
        widths = self._get_dynamic_column_widths()
        header_str = (
            f"{' ':{widths['cursor']}} {'Active':<{widths['active']}} "
            f"{'Provider':<{widths['provider']}} {'Model':<{widths['model']}} "
            f"{'Host':<{widths['host']}} {'Branches':<{widths['branches']}} {'Eval':<{widths['eval']}} {'Guard':<{widths['guard']}} "
            f"{'Pricing':<{widths['pricing']}} {'Vision':<{widths['vision']}} {'Context':<{widths['context']}} {'Status':<{widths['status']}}   "
        )
        all_fragments.extend(to_formatted_text(HTML(f"<b>{header_str}</b>")))
        all_fragments.append(('', '\n'))
        all_fragments.extend(to_formatted_text(HTML(f"<b>{'─' * (len(header_str) + 5)}</b>")))
        all_fragments.append(('', '\n'))

        # Apply scroll offset - only show visible rows that fit in terminal
        max_visible_rows = self._get_usable_terminal_lines()
        
        # Get dynamic column widths for consistent formatting
        widths = self._get_dynamic_column_widths()
        
        visible_start = max(0, self.scroll_offset)
        visible_end = min(len(self.filtered_data), visible_start + max_visible_rows)
        visible_rows = self.filtered_data[visible_start:visible_end]
        
        for i, row in enumerate(visible_rows):
            # Calculate the actual row index in the full data
            actual_row_index = visible_start + i
            # Find the original index in self.data for updates
            original_index = next((idx for idx, original_row in enumerate(self.data) if original_row is row), i)
            
            # Ensure models with any count > 0 are marked as active before display
            self._update_activation_status(original_index)
            
            cursor = "▶" if actual_row_index == self.current_row else " "
            selected_char = "✓" if row.get('selected') else " "
            
            # Format the selection box part to align with the new 'Active' header
            selection_box = f"[{selected_char}]"

            beams_text_base = f"{row.get('beams', 0)}"
            eval_text_base = f"{row.get('eval', 0)}"
            guard_text_base = f"{row.get('guard', 0)}"

            if self.editing and actual_row_index == self.current_row:
                edit_display = f"[{self.edit_buffer}_]"
                if self.current_col == 2:
                    beams_text_base = edit_display
                elif self.current_col == 3:
                    eval_text_base = edit_display
                elif self.current_col == 4:
                    guard_text_base = edit_display

            style_map = {
                'provider': 'provider', 'model': 'model', 'beams': 'beams',
                'eval': 'eval', 'guard': 'guard', 'pricing': 'pricing', 'vision': 'vision', 'context': 'context'
            }
            # This block was removed to prevent non-local LLMs from being grayed out.
            # The force_local_mode flag still correctly auto-selects local models,
            # but no longer affects the display style of other models.

            # Use dynamic column widths
            provider_text = f"<{style_map['provider']}>{row.get('provider', ''):<{widths['provider']}}</{style_map['provider']}>"
            model_text = f"<{style_map['model']}>{row.get('model_key', ''):<{widths['model']}}</{style_map['model']}>"
            
            # Color host text: green for Ollama hosts, different color for cloud
            host_value = row.get('assigned_host', 'localhost:11434')
            if host_value == 'cloud':
                host_text = f"<{style_map['provider']}>{host_value:<{widths['host']}}</{style_map['provider']}>"
            else:
                host_text = f"<ansigreen>{host_value:<{widths['host']}}</ansigreen>"
            beams_text = f"<{style_map['beams']}>{beams_text_base:<{widths['branches']}}</{style_map['beams']}>"
            eval_text = f"<{style_map['eval']}>{eval_text_base:<{widths['eval']}}</{style_map['eval']}>"
            guard_text = f"<{style_map['guard']}>{guard_text_base:<{widths['guard']}}</{style_map['guard']}>"
            pricing_text = f"<{style_map['pricing']}>{row.get('pricing_str', ''):<{widths['pricing']}}</{style_map['pricing']}>"
            
            # Check vision capability and display
            has_vision = row.get('has_vision', False)
            vision_text = f"<{style_map['vision']}>{'✓' if has_vision else ' ':<{widths['vision']}}</{style_map['vision']}>"
            
            context_text = f"<{style_map['context']}>{str(row.get('context_window', 'N/A')):<{widths['context']}}</{style_map['context']}>"
            
            status_text_content = row.get('download_status_text', '')
            if 'Downloaded' in status_text_content:
                status_text = f"<downloaded>{status_text_content:<{widths['status']}}</downloaded>"
            elif 'Downloadable' in status_text_content:
                status_text = f"<notdownloaded>{status_text_content:<{widths['status']}}</notdownloaded>"
            else:
                status_text = f"<{style_map['provider']}>{status_text_content:<{widths['status']}}</{style_map['provider']}>"

            if actual_row_index == self.current_row and not self.editing:
                if self.current_col == 0:
                    model_text = f"<reverse>{row.get('model_key', ''):<{widths['model']}}</reverse>"
                elif self.current_col == 1:
                    host_text = f"<reverse>{row.get('assigned_host', 'localhost:11434'):<{widths['host']}}</reverse>"
                elif self.current_col == 2:
                    beams_text = f"<reverse>{beams_text_base:<{widths['branches']}}</reverse>"
                elif self.current_col == 3:
                    eval_text = f"<reverse>{eval_text_base:<{widths['eval']}}</reverse>"
                elif self.current_col == 4:
                    guard_text = f"<reverse>{guard_text_base:<{widths['guard']}}</reverse>"

            line_str = (
                f"{cursor} {selection_box:<{widths['active']}} "
                f"{provider_text} {model_text} "
                f"{host_text} {beams_text} {eval_text} {guard_text} "
                f"{pricing_text} {vision_text} {context_text} {status_text}   "
            )
            all_fragments.extend(to_formatted_text(HTML(line_str)))
            all_fragments.append(('', '\n'))
            
        return all_fragments

    def _get_cursor_position(self) -> Point:
        # Calculate cursor position relative to the visible area
        # Account for header lines and scroll offset
        visible_row = self.current_row - max(0, self.scroll_offset)
        y_pos = visible_row + 4  # header + status line + column header + separator
        x_pos = 0
        return Point(x=x_pos, y=y_pos)

    async def get_selection(self, preselected_llms: Optional[List[str]] = None,
                             force_local: bool = False,
                             save_selection: bool = True) -> Dict[str, Any]:
        
        self.force_local_mode = force_local
        
        try:
            from core.llm import Llm
            from core.globals import g
            from core.providers.cls_ollama_interface import OllamaClient

            saved_config = self._load_config()
            
            # --- New Multi-Host Discovery Logic ---

            self.data = []
            processed_models = set() # Tracks 'model_key@host' to prevent duplicates

            # 1. Add non-Ollama models and 'Any but local'
            if not force_local:
                self.data.append({
                    "selected": saved_config.get("any_local", {}).get("selected", False), "beams": 0, "eval": 0, "guard": 0,
                    "provider": "Any", "model_key": "Any but local",
                    "pricing_str": "Automatic", "has_vision": False, "context_window": "N/A", "download_status_text": "",
                    "assigned_host": "cloud",
                })
            
            # Add other AI Providers (OpenAI, Anthropic, etc.)
            all_llms = Llm.get_available_llms(exclude_guards=True, include_dynamic=False) # No dynamic for non-Ollama
            for llm in all_llms:
                provider_name = llm.provider.__class__.__name__
                if provider_name != "OllamaClient":
                    model_key = llm.model_key
                    model_config = saved_config.get(model_key, {})
                    is_selected = model_config.get("selected", False)

                    self.data.append({
                        "selected": is_selected, "beams": model_config.get("beams", 0),
                        "eval": model_config.get("eval", 0), "guard": model_config.get("guard", 0),
                        "provider": provider_name, "model_key": model_key,
                        "pricing_str": f"${llm.pricing_in_dollar_per_1M_tokens}/1M" if llm.pricing_in_dollar_per_1M_tokens else "Free",
                        "has_vision": self._check_vision_capability(model_key), "context_window": llm.context_window,
                        "download_status_text": "", "assigned_host": "cloud",
                    })
                    processed_models.add(f"{model_key}@cloud")

            # 2. Get comprehensive model data (cache or live)
            ollama_status = {}
            try:
                if g.is_model_cache_stale(max_age_minutes=30): g.refresh_model_discovery()
                cached_data = g.get_cached_model_discovery()
                if cached_data:
                    ollama_status = cached_data
                else:
                    if g.wait_for_model_discovery(timeout=3.0):
                        ollama_status = g.get_cached_model_discovery() or {}
                    else:
                        ollama_hosts_list = [h.split(':')[0] for h in g.DEFAULT_OLLAMA_HOSTS]
                        ollama_status = OllamaClient.get_comprehensive_downloadable_models(ollama_hosts_list)
            except Exception as e:
                logging.warning(f"Could not get comprehensive Ollama models: {e}")
            self.ollama_status = ollama_status # Store for use in other methods

            # 3. Discover all downloaded Ollama models from all hosts
            for host_url in g.DEFAULT_OLLAMA_HOSTS:
                hostname = host_url.split(':')[0]
                if not OllamaClient.check_host_reachability(hostname):
                    continue

                downloaded_models = OllamaClient.get_downloaded_models(hostname)
                for model_info in downloaded_models:
                    model_key = model_info['name']
                    identifier = f"{model_key}@{host_url}"
                    if identifier in processed_models:
                        continue

                    model_config = saved_config.get(model_key, {})
                    size_bytes = model_info.get('size', 0)
                    status_text = "✓ Downloaded"
                    if size_bytes > 0:
                        size_gb = size_bytes / (1024 * 1024 * 1024)
                        status_text += f" ({size_gb:.1f}GB)"

                    self.data.append({
                        "selected": model_config.get("selected", False),
                        "beams": model_config.get("beams", 0), "eval": model_config.get("eval", 0), "guard": model_config.get("guard", 0),
                        "provider": "OllamaClient", "model_key": model_key, "pricing_str": "Free",
                        "has_vision": self._check_vision_capability(model_key),
                        "context_window": self._estimate_context_window(model_key),
                        "download_status_text": status_text, "assigned_host": host_url,
                    })
                    processed_models.add(identifier)
            
            # 4. Add all DOWNLOADABLE models not already present on any host
            default_host = g.DEFAULT_OLLAMA_HOSTS[0] if g.DEFAULT_OLLAMA_HOSTS else "localhost:11434"
            if ollama_status:
                for model_base, status_info_val in ollama_status.items():
                    for variant_name, variant_info in status_info_val.get('variants', {}).items():
                        is_already_present = any(f"{variant_name}@" in p for p in processed_models)
                        if is_already_present:
                            continue

                        model_config = saved_config.get(variant_name, {})
                        context_window = variant_info.get('context_window') or self._estimate_context_window(variant_name)
                        size_str = variant_info.get('size_str', '')
                        
                        # Determine assigned host and check if model is downloaded there
                        assigned_host = model_config.get("assigned_host", default_host)
                        identifier = f"{variant_name}@{assigned_host}"
                        
                        # Check if this specific model is downloaded on the assigned host
                        is_downloaded = variant_info.get('downloaded', False)
                        if is_downloaded:
                            status_text = "✓ Downloaded"
                            if size_str and any(unit in size_str.upper() for unit in ['GB', 'MB', 'TB']):
                                status_text += f" ({size_str})"
                        else:
                            status_text = "⬇ Downloadable"
                            if size_str and any(unit in size_str.upper() for unit in ['GB', 'MB', 'TB']):
                                status_text += f" ({size_str})"

                        if identifier not in processed_models:
                            self.data.append({
                                "selected": model_config.get("selected", False),
                                "beams": model_config.get("beams", 0), "eval": model_config.get("eval", 0), "guard": model_config.get("guard", 0),
                                "provider": "OllamaClient", "model_key": variant_name, "pricing_str": "Free",
                                "has_vision": self._check_vision_capability(variant_name), "context_window": context_window,
                                "download_status_text": status_text, "assigned_host": assigned_host,
                            })
                            processed_models.add(identifier)

            # 5. Handle force_local mode by selecting all Ollama models
            if force_local:
                for row in self.data:
                    if row.get('provider') == 'OllamaClient':
                        row['selected'] = True
            
            # --- End of New Discovery Logic ---

            # Get modification dates for sorting
            model_dates = {}
            for host_url in g.DEFAULT_OLLAMA_HOSTS:
                hostname = host_url.split(':')[0]
                if OllamaClient.check_host_reachability(hostname):
                    for model_info in OllamaClient.get_downloaded_models(hostname):
                        if model_info.get('name') and model_info.get('modified_at'):
                            model_dates[model_info['name']] = model_info['modified_at']

            for i in range(len(self.data)):
                self._update_activation_status(i)

            def sort_key(item):
                # ... (sorting logic remains the same)
                import re
                from datetime import datetime
                is_active = item.get('selected', False)
                active_priority = 0 if is_active else 1
                model_key = item.get('model_key', '')
                provider = item.get('provider', '')
                status_text = item.get('download_status_text', '')
                if provider != 'OllamaClient': group_priority = 0
                elif 'Downloaded' in status_text: group_priority = 1
                else: group_priority = 2
                date_sort_key, param_sort_key = float('inf'), float('inf')
                alpha_sort_key = model_key.lower()
                if group_priority == 1:
                    mod_date = model_dates.get(model_key)
                    if mod_date:
                        try:
                            ts = mod_date.timestamp() if hasattr(mod_date, 'timestamp') else datetime.fromisoformat(mod_date.replace('Z', '+00:00')).timestamp()
                            if ts > 0: date_sort_key = -ts
                        except: pass
                elif group_priority == 2:
                    status_info = self.ollama_status.get(model_key.split(':')[0], {})
                    variant_info = status_info.get('variants', {}).get(model_key, {})
                    date_str = variant_info.get('variant_modified_at') or variant_info.get('api_modified_at') or status_info.get('api_modified_at', '')
                    if date_str:
                        try:
                            parsed_ts = 0
                            if 'ago' in date_str:
                                from core.providers.cls_ollama_interface import parse_relative_date
                                parsed_dt = parse_relative_date(date_str)
                                if parsed_dt: parsed_ts = parsed_dt.timestamp()
                            else:
                                parsed_ts = datetime.fromisoformat(date_str.replace('Z', '+00:00')).timestamp()
                            if parsed_ts: date_sort_key = -parsed_ts
                        except: pass
                    tag = model_key.split(':')[-1]
                    numeric_match = re.search(r'(\d+(?:\.\d+)?)', tag)
                    if numeric_match: param_sort_key = float(numeric_match.group(1))
                return (active_priority, group_priority, date_sort_key, param_sort_key, alpha_sort_key)
            
            any_local_item = None
            regular_items = []
            for item in self.data:
                if item.get('model_key') == 'Any but local': any_local_item = item
                else: regular_items.append(item)
            regular_items.sort(key=sort_key)
            self.data = []
            if any_local_item: self.data.append(any_local_item)
            self.data.extend(regular_items)
            
            # Add any previously discovered models from cache
            self._add_discovered_models_to_data()
            
            # Initialize selection order for models that are already selected
            for row in self.data:
                if row.get('selected', False):
                    model_key = row.get('model_key', '')
                    if model_key and model_key not in self.selection_order:
                        self.selection_order[model_key] = self.next_selection_order
                        self.next_selection_order += 1
            
            self._apply_search_filter()
            
            self.search_box = TextArea(text="", multiline=False, height=1, wrap_lines=False, accept_handler=lambda buf: self._apply_search_filter())
            def on_text_changed(buf): self._apply_search_filter()
            self.search_box.buffer.on_text_changed += on_text_changed
            
            bindings = KeyBindings()
            @Condition
            def table_focused(): return self.current_focus == "table"
            @Condition 
            def search_focused(): return self.current_focus == "search"
            
            app_style = Style.from_dict({
                'provider': 'ansibrightblue', 'model': 'ansicyan', 'beams': 'ansiyellow',
                'eval': 'ansimagenta', 'guard': 'ansired', 'pricing': 'ansigreen',
                'vision': 'ansibrightgreen', 'context': 'ansiblue', 'downloaded': 'ansigreen', 'notdownloaded': 'ansiyellow',
                'reverse': 'reverse', 'separator': 'ansibrightblack',
            })

            @bindings.add("up", filter=table_focused)
            def _(event): 
                if self.current_row > 0:
                    self.current_row -= 1
                    self._adjust_scroll_for_cursor()
            
            @bindings.add("down", filter=table_focused)
            def _(event):
                if self.current_row < len(self.filtered_data) - 1: 
                    self.current_row += 1
                    self._adjust_scroll_for_cursor()
                else: 
                    self.current_focus = "search"; event.app.layout.focus(self.search_box)

            @bindings.add("left", filter=table_focused)
            def _(event):
                if not self.editing: self.current_col = max(0, self.current_col - 1)
            
            @bindings.add("right", filter=table_focused)
            def _(event):
                if not self.editing: self.current_col = min(4, self.current_col + 1)
            
            @bindings.add(" ", filter=table_focused)
            def _(event):
                if not self.editing and self.filtered_data:
                    row_data = self.filtered_data[self.current_row]
                    original_index = next((idx for idx, original_row in enumerate(self.data) if original_row is row_data), self.current_row)
                    model_key = row_data.get('model_key', '')
                    
                    # Toggle selection
                    was_selected = row_data.get('selected', False)
                    row_data['selected'] = not was_selected
                    
                    # Track selection order
                    if not was_selected:  # Now selecting
                        self.selection_order[model_key] = self.next_selection_order
                        self.next_selection_order += 1
                    else:  # Now deselecting
                        if model_key in self.selection_order:
                            del self.selection_order[model_key]
                    
                    self._reset_deselected_model(original_index)
            
            def adjust_value(increment: int):
                if not self.editing and self.current_col in [2, 3, 4] and self.filtered_data:
                    keys = {2: 'beams', 3: 'eval', 4: 'guard'}
                    col_key = keys[self.current_col]
                    row_data = self.filtered_data[self.current_row]
                    original_index = next((idx for idx, original_row in enumerate(self.data) if original_row is row_data), self.current_row)
                    current_value = row_data.get(col_key, 0)
                    row_data[col_key] = max(0, current_value + increment)
                    self._update_activation_status(original_index)
            
            @bindings.add("+", filter=table_focused)
            def _(event): adjust_value(1)
            
            @bindings.add("-", filter=table_focused)
            def _(event): adjust_value(-1)
            
            @bindings.add("enter", filter=table_focused)
            async def _(event):
                if self.editing:
                    try:
                        new_value = int(self.edit_buffer) if self.edit_buffer else 0
                        keys = {2: 'beams', 3: 'eval', 4: 'guard'}
                        col_key = keys[self.current_col]
                        if self.filtered_data:
                            row_data = self.filtered_data[self.current_row]
                            original_index = next((idx for idx, original_row in enumerate(self.data) if original_row is row_data), self.current_row)
                            row_data[col_key] = new_value
                            self._update_activation_status(original_index)
                    except ValueError: pass
                    self.editing = False; self.edit_buffer = ""
                elif self.current_col == 1 and self.filtered_data:
                    row_data = self.filtered_data[self.current_row]
                    if row_data.get('provider') == 'OllamaClient':
                        try:
                            await self._configure_host_for_model(row_data)
                            event.app.renderer.reset()
                        except Exception as e:
                            logging.error(f"Failed to open host editor: {e}", exc_info=True)
                elif self.current_col in [2, 3, 4] and self.filtered_data:
                    self.editing = True
                    keys = {2: 'beams', 3: 'eval', 4: 'guard'}
                    col_key = keys[self.current_col]
                    row_data = self.filtered_data[self.current_row]
                    self.edit_buffer = str(row_data[col_key])
            
            @bindings.add("escape", filter=table_focused)
            def _(event):
                if self.editing: self.editing = False; self.edit_buffer = ""
                else: event.app.exit(result=None)
            
            @bindings.add("c", filter=table_focused)
            def _(event): 
                if not self.editing: event.app.exit(result=self.data)
            
            @bindings.add("u", filter=table_focused)
            async def _(event):
                if not self.editing:
                    # Run model discovery with progress interface
                    discovery_completed = await self._update_models_with_progress()
                    if discovery_completed:
                        # Rebuild the model list with new discoveries
                        # Save current selections first
                        current_selections = {}
                        for row in self.data:
                            model_key = row.get('model_key')
                            if model_key:
                                current_selections[model_key] = {
                                    'selected': row.get('selected', False),
                                    'beams': row.get('beams', 0),
                                    'eval': row.get('eval', 0),
                                    'guard': row.get('guard', 0)
                                }
                        
                        # Rebuild data from scratch
                        try:
                            # Re-run the model discovery logic from get_selection
                            from core.llm import Llm
                            all_llms = Llm.get_available_llms(exclude_guards=True, include_dynamic=False) # Only static models
                            saved_config = self._load_config()
                            
                            # Rebuild data list (similar to original logic but simpler)
                            new_data = []
                            
                            # Add "Any but local" if not in force local mode
                            if not self.force_local_mode:
                                any_local_config = current_selections.get("Any but local", saved_config.get("any_local", {}))
                                new_data.append({
                                    "selected": any_local_config.get("selected", False), 
                                    "beams": 0, "eval": 0, "guard": 0,
                                    "provider": "Any", "model_key": "Any but local",
                                    "pricing_str": "Automatic", "has_vision": False, 
                                    "context_window": "N/A", "download_status_text": "",
                                    "assigned_host": "cloud",
                                })
                            
                            # Add regular LLMs
                            for llm in all_llms:
                                model_key = llm.model_key
                                current_config = current_selections.get(model_key, saved_config.get(model_key, {}))
                                provider_name = llm.provider.__class__.__name__
                                
                                is_selected = current_config.get("selected", False)
                                if self.force_local_mode and provider_name == "OllamaClient":
                                    is_selected = True
                                
                                assigned_host = "cloud"
                                if provider_name == "OllamaClient":
                                    assigned_host = current_config.get("assigned_host", "localhost:11434")
                                
                                new_data.append({
                                    "selected": is_selected,
                                    "beams": current_config.get("beams", 0),
                                    "eval": current_config.get("eval", 0),
                                    "guard": current_config.get("guard", 0),
                                    "provider": provider_name,
                                    "model_key": model_key,
                                    "pricing_str": f"${llm.pricing_in_dollar_per_1M_tokens}/1M" if llm.pricing_in_dollar_per_1M_tokens else "Free",
                                    "has_vision": self._check_vision_capability(model_key),
                                    "context_window": llm.context_window,
                                    "download_status_text": "",
                                    "assigned_host": assigned_host,
                                })
                            
                            # Update self.data with new data
                            self.data = new_data
                            
                            # Add discovered models
                            self._add_discovered_models_to_data()
                            
                            # Save the updated model configuration to persist discovered models
                            self._save_config({row['model_key']: row for row in self.data})
                            
                            # Apply search filter and reset cursor
                            self._apply_search_filter()
                            self.current_row = 0
                            
                        except Exception as e:
                            logging.error(f"Failed to rebuild model list after discovery: {e}")
                    
                    # Force UI refresh
                    event.app.renderer.reset()
            
            @bindings.add("f", filter=table_focused)
            def _(event):
                if not self.editing: self.current_focus = "search"; event.app.layout.focus(self.search_box)
            
            @bindings.add("tab")
            def _(event):
                if not self.editing:
                    if self.current_focus == "table": self.current_focus = "search"; event.app.layout.focus(self.search_box)
                    else: self.current_focus = "table"; event.app.layout.focus(control)
            
            @bindings.add("up", filter=search_focused)
            def _(event): self.current_focus = "table"; event.app.layout.focus(control)
            
            @bindings.add("enter", filter=search_focused)
            def _(event): self.current_focus = "table"; event.app.layout.focus(control)
            
            @bindings.add("escape", filter=search_focused)
            def _(event):
                if self.search_box.text: self.search_box.text = ""; self._apply_search_filter()
                self.current_focus = "table"; event.app.layout.focus(control)
            
            @bindings.add("c-c", filter=table_focused)
            def _(event): event.app.exit(result=None)
            
            @bindings.add("backspace", filter=table_focused)
            def _(event):
                if self.editing: self.edit_buffer = self.edit_buffer[:-1]
            
            for digit in "0123456789":
                @bindings.add(digit, filter=table_focused)
                def _(event, d=digit):
                    if self.editing: self.edit_buffer += d

            control = FormattedTextControl(text=self._get_display_lines, get_cursor_position=self._get_cursor_position, key_bindings=bindings, focusable=True)
            window = Window(content=control, right_margins=[ScrollbarMargin(display_arrows=True)], wrap_lines=False, always_hide_cursor=False)
            help_label = Label(text="Arrows: Navigate | F/Tab: Search | Space: Toggle | +/-: Adjust | Enter: Edit | u: Update Models | c: Confirm | Esc: Abort")
            root_container = HSplit([window, Window(height=1, char='─', style='class:separator'), self.search_box, Window(height=1, char='─', style='class:separator'), help_label])
            layout = Layout(root_container)
            app = Application(layout=layout, key_bindings=bindings, style=app_style, full_screen=True)
            app.layout.focus(control)
            final_data = await app.run_async()
            
            if not final_data: return {"status": "Cancelled", "selected_configs": [], "message": "Selection cancelled."}
            await self._auto_download_selected_models(final_data)
            if save_selection: self._save_config({row['model_key']: row for row in final_data})
            # Sort selected configs by the order they were checked
            selected_configs = [row for row in final_data if row.get('selected')]
            selected_configs.sort(key=lambda row: self.selection_order.get(row.get('model_key', ''), float('inf')))
            if any(row['model_key'] == 'Any but local' and row.get('selected') for row in final_data):
                return {"status": "Success", "selection_type": "any_local", "selected_configs": [], "message": "'Any local' selected."}
            else:
                return {"status": "Success", "selection_type": "specific_models", "selected_configs": selected_configs, "message": f"Selected {len(selected_configs)} model(s)."}

        except Exception as e:
            logging.error(f"Failed to run LLM selection: {e}", exc_info=True)
            return {"status": "Error", "selected_configs": [], "message": f"Failed to run LLM selection: {e}"}

    def _save_config(self, config: Dict[str, Any]) -> None:
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            if 'Any but local' in config:
                config['any_local'] = config.pop('Any but local')

            config_to_save = {
                model: {k: v for k, v in data.items() if k not in ['provider']} 
                for model, data in config.items() 
                if data.get('selected') or any(data.get(k, 0) > 0 for k in ['beams', 'eval', 'guard'])
            }
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save LLM configuration: {e}")

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_file or not self.config_file.exists():
            return {}
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Could not load LLM configuration: {e}")
            return {}

    async def _auto_download_selected_models(self, final_data: List[Dict[str, Any]]) -> None:
        """
        Automatically download selected Ollama models that are not yet downloaded.
        Shows progress for each individual model download.
        """
        import ollama
        from termcolor import colored
        
        # Find selected Ollama models that need downloading
        models_to_download = []
        for row in final_data:
            if (row.get('selected', False) and 
                row.get('provider') == 'OllamaClient' and 
                row.get('download_status_text', '').startswith('⬇ Downloadable')):
                models_to_download.append((row['model_key'], row.get('assigned_host', 'localhost:11434')))
        
        if not models_to_download:
            return
        
        # Get available Ollama hosts for downloading
        try:
            from core.providers.cls_ollama_interface import OllamaClient
            
            # Get hosts from environment or use defaults
            import os
            ollama_host_env = os.getenv("OLLAMA_HOST", "")
            if ollama_host_env:
                ollama_hosts = ollama_host_env.split(",")
            else:
                ollama_hosts = ["localhost"]
            
            print(f"\n{colored('🔄 Auto-downloading selected models...', 'cyan', attrs=['bold'])}")
            print(f"Models to download: {', '.join([f'{model}@{host}' for model, host in models_to_download])}\n")
            
            downloaded_count = 0
            failed_models = []
            
            for i, (model_key, assigned_host) in enumerate(models_to_download, 1):
                print(f"{colored(f'[{i}/{len(models_to_download)}]', 'yellow')} Downloading {colored(model_key, 'cyan')} on {colored(assigned_host, 'green')}...")
                
                download_success = False
                
                # Use the assigned host for this model
                host = assigned_host.split(':')[0]  # Extract hostname without port
                # Check if host is reachable
                if not OllamaClient.check_host_reachability(host):
                    print(f"  ❌ {colored(f'Host {assigned_host} is not reachable', 'red')}")
                else:
                    try:
                        client = ollama.Client(host=f'http://{assigned_host}')  # Use full assigned_host with port
                        
                        print(f"  📡 Using host: {colored(assigned_host, 'green')}")
                        
                        # Start download with progress
                        def bytes_to_mb(bytes_value):
                            return bytes_value / (1024 * 1024)
                        
                        last_status = ""
                        download_complete = False
                        
                        for response in client.pull(model_key, stream=True):
                            if "status" in response:
                                if response["status"] == "pulling manifest":
                                    status = colored("  📋 Pulling manifest...", "yellow")
                                elif response["status"].startswith("pulling"):
                                    digest = response.get("digest", "")[:12]  # Shorter digest for cleaner output
                                    total_bytes = response.get("total") or 0
                                    completed_bytes = response.get("completed") or 0
                                    total = bytes_to_mb(total_bytes)
                                    completed = bytes_to_mb(completed_bytes)
                                    if total > 0:
                                        percentage = (completed / total) * 100
                                        status = colored(f"  📥 {digest}: {completed:.1f}/{total:.1f} MB ({percentage:.1f}%)", "yellow")
                                    else:
                                        status = colored(f"  📥 {digest}: {completed:.1f} MB", "yellow")
                                elif response["status"] == "verifying sha256 digest":
                                    status = colored("  🔍 Verifying integrity...", "yellow")
                                elif response["status"] == "writing manifest":
                                    status = colored("  📝 Writing manifest...", "yellow")
                                elif response["status"] == "removing any unused layers":
                                    status = colored("  🧹 Cleaning up...", "yellow")
                                elif response["status"] == "success":
                                    status = colored(f"  ✅ {model_key} downloaded successfully!", "green")
                                    download_complete = True
                                else:
                                    continue
                            
                                # Use carriage return for progress updates, newline for final status
                                if response["status"].startswith("pulling") and response.get("digest"):
                                    # Overwrite previous line for progress updates
                                    print(f'\r{status}', end='', flush=True)
                                else:
                                    # New line for other statuses
                                    if last_status and last_status.startswith("  📥"):
                                        print()  # Complete the progress line
                                    print(status)
                                
                                last_status = status
                    
                        # Only print success message if not already printed
                        if not download_complete:
                            print(f"  ✅ {colored(f'{model_key} downloaded successfully!', 'green')}")
                        downloaded_count += 1
                        download_success = True
                        
                    except Exception as e:
                        print(f"  ❌ {colored(f'Download failed on {assigned_host}: {str(e)}', 'red')}")
                        continue
                
                if not download_success:
                    failed_models.append(f"{model_key}@{assigned_host}")
                    print(f"  ❌ {colored(f'Failed to download {model_key} from {assigned_host}', 'red')}")
                
                # Add spacing between models
                if i < len(models_to_download):
                    print()
            
            # Summary
            print(f"\n{colored('📊 Download Summary:', 'cyan', attrs=['bold'])}")
            print(f"  ✅ Successfully downloaded: {colored(str(downloaded_count), 'green')} models")
            if failed_models:
                print(f"  ❌ Failed downloads: {colored(str(len(failed_models)), 'red')} models")
                print(f"     {', '.join(failed_models)}")
            print()
            
        except Exception as e:
            print(f"\n{colored(f'Error during auto-download: {str(e)}', 'red')}")
            logging.error(f"Auto-download failed: {e}", exc_info=True)
    
    async def _configure_host_for_model(self, row_data: Dict[str, Any]) -> None:
        """
        Configure the Ollama host for a specific model.
        Shows an interactive menu to select from available hosts or add new ones.
        """
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.layout import Layout
        from prompt_toolkit.formatted_text import HTML, to_formatted_text
        
        try:
            model_key = row_data.get('model_key', '')
            current_host = row_data.get('assigned_host', 'localhost:11434')
            
            # Always start with default host
            available_hosts = ["localhost:11434"]
            
            # Load previously saved custom hosts
            saved_hosts = self._load_saved_hosts()
            for host in saved_hosts:
                if host not in available_hosts:
                    available_hosts.append(host)
            
            selected_index = 0
            if current_host in available_hosts:
                selected_index = available_hosts.index(current_host)
            
            custom_host_input = ""
            input_mode = False  # False = host selection, True = custom input
            edit_mode = False  # True when editing an existing host
            
            def get_display_text():
                fragments = []
                
                # Clean minimal header
                header = f"Configure Host for {model_key}"
                fragments.extend(to_formatted_text(HTML(f"<ansicyan><b>{header}</b></ansicyan>")))
                fragments.append(('', '\n'))
                fragments.extend(to_formatted_text(HTML(f"<ansibrightblack>{'─' * len(header)}</ansibrightblack>")))
                fragments.append(('', '\n\n'))
                
                # Current host info
                fragments.extend(to_formatted_text(HTML(f"<ansiyellow>Current:</ansiyellow> <b>{current_host}</b>")))
                fragments.append(('', '\n\n'))
                
                if not input_mode:
                    # Host selection mode - cleaner layout
                    for i, host in enumerate(available_hosts):
                        # Check host reachability
                        hostname_only = host.split(':')[0] if ':' in host else host
                        is_reachable = self._check_host_reachability_simple(hostname_only)
                        status_indicator = "●" if is_reachable else "○"
                        
                        if i == selected_index:
                            # Selected host with arrow
                            fragments.extend(to_formatted_text(HTML(f"<reverse> ▶ <ansigreen>{status_indicator}</ansigreen> <b>{host}</b> </reverse>")))
                        else:
                            # Regular host entry
                            status_color = "ansigreen" if is_reachable else "ansired"
                            fragments.extend(to_formatted_text(HTML(f"   <{status_color}>{status_indicator}</{status_color}> {host}")))
                        fragments.append(('', '\n'))
                    
                    fragments.append(('', '\n'))
                    
                    # Dynamic help - cleaner format
                    selected_host = available_hosts[selected_index] if available_hosts else ""
                    # Allow deletion of any host if there are multiple hosts
                    can_delete = len(available_hosts) > 1
                    
                    # Build help text dynamically
                    help_parts = ["<ansibrightblack>"]
                    help_parts.append("↑↓:Navigate")
                    help_parts.append("Enter:Select") 
                    help_parts.append("N:Add")
                    help_parts.append("E:Edit")  # Always show edit
                    if can_delete:
                        help_parts.append("D:Delete")
                    help_parts.append("Esc:Cancel")
                    help_parts.append("</ansibrightblack>")
                    
                    fragments.extend(to_formatted_text(HTML(" • ".join(help_parts))))
                    
                else:
                    # Input mode - minimal and clean
                    mode_text = "Edit Host" if edit_mode else "Add New Host"
                    fragments.extend(to_formatted_text(HTML(f"<ansicyan><b>{mode_text}</b></ansicyan>")))
                    fragments.append(('', '\n\n'))
                    
                    # Input field with disappearing example
                    if custom_host_input:
                        # User has typed something, show their input with cursor
                        fragments.extend(to_formatted_text(HTML(f"<ansiyellow>Host:</ansiyellow> <b>{custom_host_input}</b><ansigreen>_</ansigreen>")))
                    else:
                        # No input yet, show example that will disappear
                        fragments.extend(to_formatted_text(HTML("<ansiyellow>Host:</ansiyellow> <ansibrightblack>(ex: localhost:11434)</ansibrightblack>")))
                    
                    fragments.append(('', '\n\n'))
                    
                    # Clean command help
                    if edit_mode:
                        fragments.extend(to_formatted_text(HTML("<ansibrightblack>Enter:Save • Esc:Cancel</ansibrightblack>")))
                    else:
                        fragments.extend(to_formatted_text(HTML("<ansibrightblack>Enter:Add • Esc:Back</ansibrightblack>")))
                
                return fragments
            
            bindings = KeyBindings()
            
            # Create conditions for proper filter evaluation
            @Condition
            def not_in_input_mode():
                return not input_mode
            
            @Condition 
            def in_input_mode():
                return input_mode
            
            @bindings.add("up")
            def _(event):
                nonlocal selected_index
                if not input_mode and selected_index > 0:
                    selected_index -= 1
            
            @bindings.add("down")
            def _(event):
                nonlocal selected_index
                if not input_mode and selected_index < len(available_hosts) - 1:
                    selected_index += 1
            
            @bindings.add("enter")
            def _(event):
                nonlocal input_mode, custom_host_input, selected_index, edit_mode
                if input_mode:
                    # Add or edit custom host
                    new_host = custom_host_input.strip()
                    if new_host:
                        # No automatic port addition - user must specify explicitly
                        if edit_mode:
                            # Replace the existing host
                            old_host = available_hosts[selected_index]
                            available_hosts[selected_index] = new_host
                            self._remove_custom_host(old_host)
                            self._save_custom_host(new_host)
                            edit_mode = False
                        else:
                            # Add new host
                            if new_host not in available_hosts:
                                available_hosts.append(new_host)
                                self._save_custom_host(new_host)
                                selected_index = available_hosts.index(new_host)
                        
                        input_mode = False
                        custom_host_input = ""
                else:
                    # Select host and exit (applies to all hosts)
                    selected_host = available_hosts[selected_index]
                    row_data['assigned_host'] = selected_host
                    
                    # Update status after host change
                    self._update_model_status_for_host(row_data, selected_host)
                    
                    event.app.exit(result=True)
            
            # Add both lowercase and uppercase variants with proper Condition filters
            @bindings.add("n", filter=not_in_input_mode)
            def add_new_host_n(event):
                nonlocal input_mode, edit_mode
                input_mode = True
                edit_mode = False
                    
            @bindings.add("N", filter=not_in_input_mode) 
            def add_new_host_N(event):
                nonlocal input_mode, edit_mode
                input_mode = True
                edit_mode = False
                    
            @bindings.add("e", filter=not_in_input_mode)
            def edit_host_e(event):
                nonlocal input_mode, edit_mode, custom_host_input, selected_index
                if available_hosts:
                    selected_host = available_hosts[selected_index]
                    # Allow editing any host - will add as custom if not already custom
                    input_mode = True
                    edit_mode = True
                    custom_host_input = selected_host
                        
            @bindings.add("E", filter=not_in_input_mode)
            def edit_host_E(event):
                nonlocal input_mode, edit_mode, custom_host_input, selected_index
                if available_hosts:
                    selected_host = available_hosts[selected_index]
                    # Allow editing any host - will add as custom if not already custom
                    input_mode = True
                    edit_mode = True
                    custom_host_input = selected_host
                    
            @bindings.add("d", filter=not_in_input_mode)
            def delete_host_d(event):
                nonlocal selected_index
                if len(available_hosts) > 1:
                    selected_host = available_hosts[selected_index]
                    
                    # Remove any host if there are multiple hosts available
                    available_hosts.pop(selected_index)
                    
                    # Remove from custom hosts if it was saved there
                    self._remove_custom_host(selected_host)
                    
                    # Adjust selected index if necessary
                    if selected_index >= len(available_hosts):
                        selected_index = len(available_hosts) - 1
                    elif selected_index < 0:
                        selected_index = 0
                            
            @bindings.add("D", filter=not_in_input_mode)
            def delete_host_D(event):
                nonlocal selected_index
                if len(available_hosts) > 1:
                    selected_host = available_hosts[selected_index]
                    
                    # Remove any host if there are multiple hosts available
                    available_hosts.pop(selected_index)
                    
                    # Remove from custom hosts if it was saved there
                    self._remove_custom_host(selected_host)
                    
                    # Adjust selected index if necessary
                    if selected_index >= len(available_hosts):
                        selected_index = len(available_hosts) - 1
                    elif selected_index < 0:
                        selected_index = 0
            
            @bindings.add("escape")
            def _(event):
                nonlocal input_mode, edit_mode, custom_host_input
                if input_mode:
                    input_mode = False
                    edit_mode = False
                    custom_host_input = ""
                else:
                    event.app.exit(result=False)
            
            # Handle character input in custom input mode (exclude hotkey characters)
            # Exclude n, e, d and their uppercase variants to avoid conflicts with hotkeys
            input_chars = "abcfghijklmopqrstuvwxyzABCFGHIJKLMOPQRSTUVWXYZ0123456789.-:"
            for c in input_chars:
                @bindings.add(c)
                def _(event, char=c):
                    nonlocal custom_host_input
                    if input_mode:
                        custom_host_input += char
                        
            # Special handlers for n, e, d in input mode only
            @bindings.add("n", filter=in_input_mode)
            def input_n(event):
                nonlocal custom_host_input
                custom_host_input += "n"
                
            @bindings.add("N", filter=in_input_mode) 
            def input_N(event):
                nonlocal custom_host_input
                custom_host_input += "N"
                
            @bindings.add("e", filter=in_input_mode)
            def input_e(event):
                nonlocal custom_host_input
                custom_host_input += "e"
                
            @bindings.add("E", filter=in_input_mode)
            def input_E(event):
                nonlocal custom_host_input
                custom_host_input += "E"
                
            @bindings.add("d", filter=in_input_mode)
            def input_d(event):
                nonlocal custom_host_input
                custom_host_input += "d"
                
            @bindings.add("D", filter=in_input_mode)
            def input_D(event):
                nonlocal custom_host_input
                custom_host_input += "D"
                        
            @bindings.add("backspace")
            def _(event):
                nonlocal custom_host_input
                if input_mode and custom_host_input:
                    custom_host_input = custom_host_input[:-1]
            
            control = FormattedTextControl(text=get_display_text, key_bindings=bindings, focusable=True)
            window = Window(content=control, wrap_lines=False)
            layout = Layout(window)
            
            app = Application(layout=layout, key_bindings=bindings, full_screen=True)
            await app.run_async()
            
        except Exception as e:
            logging.error(f"Failed to configure host for model {row_data.get('model_key', 'unknown')}: {e}", exc_info=True)
    
    def _check_host_reachability_simple(self, hostname: str) -> bool:
        """Simple host reachability check for the host configuration menu."""
        try:
            import socket
            port = 11434  # Default Ollama port
            with socket.create_connection((hostname, port), timeout=1):
                return True
        except:
            return False
    
    def _load_saved_hosts(self) -> List[str]:
        """Load previously saved custom hosts from configuration."""
        try:
            config = self._load_config()
            return config.get('custom_hosts', [])
        except:
            return []
    
    def _save_custom_host(self, host: str) -> None:
        """Save a custom host to configuration."""
        try:
            config = self._load_config()
            custom_hosts = config.get('custom_hosts', [])
            if host not in custom_hosts:
                custom_hosts.append(host)
                config['custom_hosts'] = custom_hosts
                # Save the updated config
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_file, 'w') as f:
                    import json
                    json.dump(config, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save custom host {host}: {e}")
    
    def _remove_custom_host(self, host: str) -> None:
        """Remove a custom host from configuration."""
        try:
            config = self._load_config()
            custom_hosts = config.get('custom_hosts', [])
            if host in custom_hosts:
                custom_hosts.remove(host)
                config['custom_hosts'] = custom_hosts
                # Save the updated config
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_file, 'w') as f:
                    import json
                    json.dump(config, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to remove custom host {host}: {e}")
    
    def _update_model_status_for_host(self, row_data: Dict[str, Any], host: str) -> None:
        """Update model download status for the specified host."""
        try:
            from core.providers.cls_ollama_interface import OllamaClient
            
            model_key = row_data.get('model_key', '')
            hostname = host.split(':')[0] if ':' in host else host
            
            if row_data.get('provider') == 'OllamaClient' and model_key:
                # Check if model is downloaded on this specific host
                downloaded_models = OllamaClient.get_downloaded_models(hostname)
                model_info = next((m for m in downloaded_models if model_key == m.get('name')), None)
                
                if model_info:
                    # Model is downloaded on the new host
                    size_bytes = model_info.get('size', 0)
                    if size_bytes > 0:
                        size_gb = size_bytes / (1024 * 1024 * 1024)
                        row_data['download_status_text'] = f"✓ Downloaded ({size_gb:.1f}GB)"
                    else:
                        row_data['download_status_text'] = "✓ Downloaded"
                else:
                    # Model is not downloaded, show downloadable status with size if available
                    status_text = "⬇ Downloadable"
                    if hasattr(self, 'ollama_status') and self.ollama_status:
                        base_name = model_key.split(':')[0]
                        status_info = self.ollama_status.get(base_name, {})
                        variant_info = status_info.get('variants', {}).get(model_key, {})
                        size_str = variant_info.get('size_str', '')
                        if size_str and any(unit in size_str.upper() for unit in ['GB', 'MB', 'TB']):
                            status_text += f" ({size_str})"
                    row_data['download_status_text'] = status_text
                    
        except Exception as e:
            # If the host is unreachable or another error occurs, mark as downloadable
            row_data['download_status_text'] = "⬇ Downloadable"
            logging.debug(f"Failed to update model status for {row_data.get('model_key', 'unknown')} on {host}: {e}")

async def main_test():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load environment variables when running standalone
    try:
        from dotenv import load_dotenv
        import os
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        load_dotenv(env_file)
        print(f"✅ Loaded environment from: {env_file}")
    except ImportError:
        print("⚠️  python-dotenv not available, API-based model discovery may not work")
    except Exception as e:
        print(f"⚠️  Failed to load .env file: {e}")
    
    selector = LlmSelector()
    
    print("="*60 + "\n   Interactive LLM Selection Test\n" + "="*60)
    input("Press Enter to launch...")
    result = await selector.get_selection()
    
    print("\nFinal Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_test())