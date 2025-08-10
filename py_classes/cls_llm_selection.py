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

# --- prompt_toolkit imports ---
from prompt_toolkit import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import (HTML, FormattedText,
                                            to_formatted_text)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Label


class LlmSelector:
    """
    Interactive LLM selection interface with multi-selection and multi-column display.
    Handles LLM status checking, selection persistence, and UI for choosing models.
    """

    def __init__(self):
        """Initialize the LLM selector."""
        self.config_file = None
        try:
            from py_classes.globals import g
            # Use the consistent config path from globals
            self.config_file = Path(g.LLM_CONFIG_PATH)
        except (ImportError, AttributeError):
            # Fallback for standalone execution
            self.config_file = Path.home() / ".cli-agent" / "llm_config.json"
            logging.info("Running in standalone mode, using default config path.")

        self.data = []
        self.current_row = 0
        self.current_col = 0  # 0: Selector, 1: Branches, 2: Eval, 3: Guard
        self.editing = False
        self.edit_buffer = ""
        self.force_local_mode = False

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
        
        # Add a column header for the selection checkbox ('Active') and the new 'Guard' column
        header_str = (
            f"{' ':>1} {'Active':<5} {'Provider':<15} {'Model':<35} {'Branches':<6} {'Eval':<6} {'Guard':<6} "
            f"{'Pricing':<18} {'Context':<10} {'Status'}"
        )
        all_fragments.extend(to_formatted_text(HTML(f"<b>{header_str}</b>")))
        all_fragments.append(('', '\n'))
        all_fragments.extend(to_formatted_text(HTML(f"<b>{'─' * (len(header_str) + 5)}</b>")))
        all_fragments.append(('', '\n'))

        for i, row in enumerate(self.data):
            cursor = "▶" if i == self.current_row else " "
            selected_char = "✓" if row.get('selected') else " "
            
            # Format the selection box part to align with the new 'Active' header
            selection_box = f"[{selected_char}]"

            beams_text_base = f"{row.get('beams', 0)}"
            eval_text_base = f"{row.get('eval', 0)}"
            guard_text_base = f"{row.get('guard', 0)}"

            if self.editing and i == self.current_row:
                edit_display = f"[{self.edit_buffer}_]"
                if self.current_col == 1:
                    beams_text_base = edit_display
                elif self.current_col == 2:
                    eval_text_base = edit_display
                elif self.current_col == 3:
                    guard_text_base = edit_display

            style_map = {
                'provider': 'provider', 'model': 'model', 'beams': 'beams',
                'eval': 'eval', 'guard': 'guard', 'pricing': 'pricing', 'context': 'context'
            }
            # This block was removed to prevent non-local LLMs from being grayed out.
            # The force_local_mode flag still correctly auto-selects local models,
            # but no longer affects the display style of other models.

            provider_text = f"<{style_map['provider']}>{row.get('provider', ''):<15}</{style_map['provider']}>"
            model_text = f"<{style_map['model']}>{row.get('model_key', ''):<35}</{style_map['model']}>"
            beams_text = f"<{style_map['beams']}>{beams_text_base:<6}</{style_map['beams']}>"
            eval_text = f"<{style_map['eval']}>{eval_text_base:<6}</{style_map['eval']}>"
            guard_text = f"<{style_map['guard']}>{guard_text_base:<6}</{style_map['guard']}>"
            pricing_text = f"<{style_map['pricing']}>{row.get('pricing_str', ''):<18}</{style_map['pricing']}>"
            context_text = f"<{style_map['context']}>{str(row.get('context_window', 'N/A')):<10}</{style_map['context']}>"
            
            status_text_content = row.get('download_status_text', '')
            status_text = f"<{style_map['provider']}>{status_text_content}</{style_map['provider']}>"
            if 'Downloaded' in status_text_content:
                status_text = f"<downloaded>{status_text_content}</downloaded>"
            elif 'Downloadable' in status_text_content:
                status_text = f"<notdownloaded>{status_text_content}</notdownloaded>"

            if i == self.current_row and not self.editing:
                if self.current_col == 0:
                    model_text = f"<reverse>{row.get('model_key', ''):<35}</reverse>"
                elif self.current_col == 1:
                    beams_text = f"<reverse>{beams_text_base:<6}</reverse>"
                elif self.current_col == 2:
                    eval_text = f"<reverse>{eval_text_base:<6}</reverse>"
                elif self.current_col == 3:
                    guard_text = f"<reverse>{guard_text_base:<6}</reverse>"

            line_str = (
                f"{cursor} {selection_box:<5} {provider_text} {model_text} {beams_text} {eval_text} {guard_text} "
                f"{pricing_text} {context_text} {status_text}"
            )
            all_fragments.extend(to_formatted_text(HTML(line_str)))
            all_fragments.append(('', '\n'))
            
        return all_fragments

    def _get_cursor_position(self) -> Point:
        # Adjust y_pos to account for the new header line and separator
        y_pos = self.current_row + 3 
        x_pos = 0
        return Point(x=x_pos, y=y_pos)

    async def get_selection(self, preselected_llms: Optional[List[str]] = None,
                             force_local: bool = False,
                             save_selection: bool = True) -> Dict[str, Any]:
        
        self.force_local_mode = force_local
        
        try:
            from py_classes.cls_llm_router import Llm
            from py_classes.globals import g
            from py_classes.ai_providers.cls_ollama_interface import OllamaClient

            available_llms = Llm.get_available_llms(exclude_guards=True, include_dynamic=True)
            saved_config = self._load_config()
            
            # Get both downloaded and discoverable models for Ollama
            ollama_status = {}
            discoverable_models = set()
            try:
                # Get downloaded models status - extract hostnames from URLs (without port)
                ollama_hosts = []
                for host_url in g.DEFAULT_OLLAMA_HOSTS:
                    if "://" in host_url:
                        host = host_url.split("://")[1].split("/")[0]
                    else:
                        host = host_url
                    
                    # Remove port from hostname for get_downloaded_models (it adds :11434 automatically)
                    if ":" in host:
                        host = host.split(":")[0]
                    ollama_hosts.append(host)
                ollama_status = OllamaClient.get_comprehensive_model_status(ollama_hosts)
                
                # Get discoverable models from Ollama library/registry
                # This is a more comprehensive list that includes models available for download
                for host_url in g.DEFAULT_OLLAMA_HOSTS:
                    try:
                        # Extract hostname from URL format (e.g., "http://localhost:11434" -> "localhost")
                        if "://" in host_url:
                            host = host_url.split("://")[1].split("/")[0]
                        else:
                            host = host_url
                        
                        # For reachability check, keep port; for model listing, remove port
                        host_for_reachability = host
                        host_for_models = host.split(":")[0] if ":" in host else host
                        
                        if OllamaClient.check_host_reachability(host_for_reachability):
                            # Try to get a list of popular/available models from Ollama
                            # We'll add some common models that are typically available
                            common_models = [
                                'llama3.3', 'llama3.2', 'llama3.1', 'llama3', 'llama2',
                                'mistral', 'mixtral', 'qwen2.5', 'qwen2', 'phi3.5', 'phi3', 
                                'codellama', 'deepseek-coder', 'gemma2', 'gemma', 
                                'nomic-embed-text', 'mxbai-embed-large', 'bge-m3'
                            ]
                            for model_base in common_models:
                                if model_base not in ollama_status:
                                    ollama_status[model_base] = {'downloaded': False, 'hosts': []}
                                    discoverable_models.add(model_base)
                            break  # Only need one reachable host
                    except Exception:
                        continue
            except Exception as e:
                logging.warning(f"Could not get Ollama status: {e}")

            self.data = []
            if not force_local:
                self.data.append({
                    "selected": saved_config.get("any_local", {}).get("selected", False), "beams": 0, "eval": 0, "guard": 0,
                    "provider": "Any", "model_key": "Any but local",
                    "pricing_str": "Automatic", "context_window": "N/A", "download_status_text": "",
                })

            for llm in available_llms:
                model_key = llm.model_key
                model_config = saved_config.get(model_key, {})
                provider_name = llm.provider.__class__.__name__

                is_selected = False
                if force_local and provider_name == "OllamaClient":
                    is_selected = True
                elif preselected_llms is not None:
                    is_selected = model_key in preselected_llms
                else:
                    is_selected = model_config.get("selected", False)
                
                status_text = ""
                if provider_name == "OllamaClient":
                    model_base_name = llm.model_key.split(':')[0]
                    status_info = ollama_status.get(model_base_name, {})
                    status_text = '✓ Downloaded' if status_info.get('downloaded') else '⬇ Downloadable'
                
                self.data.append({
                    "selected": is_selected,
                    "beams": model_config.get("beams", 0),
                    "eval": model_config.get("eval", 0),
                    "guard": model_config.get("guard", 0),
                    "provider": provider_name,
                    "model_key": model_key,
                    "pricing_str": f"${llm.pricing_in_dollar_per_1M_tokens}/1M" if llm.pricing_in_dollar_per_1M_tokens else "Free",
                    "context_window": llm.context_window,
                    "download_status_text": status_text,
                })
            
            # Add discoverable but not downloaded Ollama models
            existing_models = {row['model_key'] for row in self.data}
            for model_base in discoverable_models:
                if model_base not in existing_models:
                    # Create a generic model entry for downloadable models
                    model_key = f"{model_base}:latest"
                    if model_key not in existing_models:
                        model_config = saved_config.get(model_key, {})
                        
                        # Estimate context window based on model name
                        context_window = self._estimate_context_window(model_key)
                        
                        self.data.append({
                            "selected": model_config.get("selected", False),
                            "beams": model_config.get("beams", 0),
                            "eval": model_config.get("eval", 0),
                            "guard": model_config.get("guard", 0),
                            "provider": "OllamaClient",
                            "model_key": model_key,
                            "pricing_str": "Free",
                            "context_window": context_window,
                            "download_status_text": "⬇ Downloadable",
                        })
            
            bindings = KeyBindings()
            
            app_style = Style.from_dict({
                'provider': 'ansibrightblue', 'model': 'ansicyan', 'beams': 'ansiyellow',
                'eval': 'ansimagenta', 'guard': 'ansired', 'pricing': 'ansigreen',
                'context': 'ansiblue', 'downloaded': 'ansigreen', 'notdownloaded': 'ansiyellow',
                'reverse': 'reverse', 'separator': 'ansibrightblack',
            })

            @bindings.add("up")
            def _(event): self.current_row = max(0, self.current_row - 1)

            @bindings.add("down")
            def _(event): self.current_row = min(len(self.data) - 1, self.current_row + 1)

            @bindings.add("left")
            def _(event):
                if not self.editing: self.current_col = max(0, self.current_col - 1)

            @bindings.add("right")
            def _(event):
                if not self.editing: self.current_col = min(3, self.current_col + 1)

            @bindings.add(" ")
            def _(event):
                if not self.editing:
                    row_data = self.data[self.current_row]
                    row_data['selected'] = not row_data.get('selected', False)
                    self._reset_deselected_model(self.current_row)

            def adjust_value(increment: int):
                if not self.editing and self.current_col in [1, 2, 3]:
                    keys = {1: 'beams', 2: 'eval', 3: 'guard'}
                    col_key = keys[self.current_col]
                    current_value = self.data[self.current_row].get(col_key, 0)
                    new_value = max(0, current_value + increment)
                    self.data[self.current_row][col_key] = new_value
                    self._update_activation_status(self.current_row)

            @bindings.add("+")
            def _(event): adjust_value(1)

            @bindings.add("-")
            def _(event): adjust_value(-1)

            @bindings.add("enter")
            def _(event):
                if self.editing:
                    try:
                        new_value = int(self.edit_buffer) if self.edit_buffer else 0
                        keys = {1: 'beams', 2: 'eval', 3: 'guard'}
                        col_key = keys[self.current_col]
                        self.data[self.current_row][col_key] = new_value
                        self._update_activation_status(self.current_row)
                    except ValueError: pass
                    self.editing = False
                    self.edit_buffer = ""
                elif self.current_col in [1, 2, 3]:
                    self.editing = True
                    keys = {1: 'beams', 2: 'eval', 3: 'guard'}
                    col_key = keys[self.current_col]
                    self.edit_buffer = str(self.data[self.current_row][col_key])
                else:
                    event.app.exit(result=self.data)

            @bindings.add("c-c", "q", "escape")
            def _(event):
                if self.editing:
                    self.editing = False
                    self.edit_buffer = ""
                else:
                    event.app.exit(result=None)
            
            @bindings.add("c")
            def _(event): event.app.exit(result=self.data)
            
            @bindings.add("a")
            def _(event): event.app.exit(result=None)

            @bindings.add("backspace")
            def _(event):
                if self.editing: self.edit_buffer = self.edit_buffer[:-1]
            
            @bindings.add("<any>")
            def _(event):
                if self.editing and event.data.isdigit():
                    self.edit_buffer += event.data

            control = FormattedTextControl(
                text=self._get_display_lines,
                get_cursor_position=self._get_cursor_position,
                key_bindings=bindings,
                focusable=True
            )
            window = Window(
                content=control,
                right_margins=[ScrollbarMargin(display_arrows=True)],
                wrap_lines=False
            )
            help_label = Label(text="Arrows: Navigate | Space: Toggle | +/-: Adjust | Enter: Edit/Confirm | c: Confirm | a/q/Esc: Abort")
            root_container = HSplit([
                window,
                Window(height=1, char='─', style='class:separator'),
                help_label,
                Window(height=1, char='─', style='class:separator'),
            ])
            layout = Layout(root_container)
            app = Application(layout=layout, key_bindings=bindings, style=app_style, full_screen=True)
            
            final_data = await app.run_async()

            if not final_data:
                return {"status": "Cancelled", "selected_configs": [], "message": "Selection cancelled."}

            if save_selection:
                full_config = {row['model_key']: row for row in final_data}
                self._save_config(full_config)
            
            selected_configs = [row for row in final_data if row.get('selected')]
            
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
                model: {k: v for k, v in data.items() if k != 'provider'} 
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

async def main_test():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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