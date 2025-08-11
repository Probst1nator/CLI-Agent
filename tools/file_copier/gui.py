import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from typing import Dict, List, Optional
import threading
import asyncio
import fnmatch
import signal
import re
from datetime import datetime
from dotenv import load_dotenv

# Setup CLI-Agent imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.path_resolver import setup_cli_agent_imports
setup_cli_agent_imports("file_copier")

# Load environment variables
load_dotenv()

from smart_paster import apply_changes_to_files, IGNORE_DIRS, build_clipboard_content, process_smart_request

try:
    import pyperclip
except ImportError:
    pyperclip = None

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except (ImportError, AttributeError):
    pass

DEFAULT_PRESET_NAME = "default"
CONFIG_FILENAME = ".file_copier_config.json"

DARK_BG, DARK_FG, DARK_SELECT_BG = "#2b2b2b", "#ffffff", "#404040"
DARK_ENTRY_BG, DARK_BUTTON_BG, DARK_TREE_BG = "#3c3c3c", "#404040", "#2b2b2b"

def is_text_file(filepath: str) -> bool:
    try:
        with open(filepath, 'rb') as f:
            return b'\0' not in f.read(1024)
    except (IOError, PermissionError):
        return False

def is_includable_file(filepath: str) -> bool:
    if os.path.splitext(filepath)[1].lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
        return True
    return is_text_file(filepath)

def get_script_directory() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

class FileCopierApp:
    def __init__(self, root: tk.Tk, directory: str):
        self.root = root
        self.directory = os.path.abspath(directory)
        self.config_file_path = os.path.join(get_script_directory(), CONFIG_FILENAME)
        self._initialize_state()
        self._setup_styles()
        self._create_widgets()
        self._bind_events()
        self._setup_interrupt_handler()
        self._log_message("Initializing...")
        self.load_project_config()
        self.root.after(100, self.start_async_project_load)

    def _initialize_state(self):
        self.selected_files_map: Dict[str, bool] = {}
        self.preview_visible = False
        self.all_text_files: List[str] = []
        self._search_job: Optional[str] = None
        self._auto_save_job: Optional[str] = None
        self.full_config: Dict[str, Dict] = {}
        self.presets: Dict[str, Dict] = {}
        self.drag_start_index: Optional[int] = None

    def _setup_styles(self):
        self.root.title(f"File Copier - {os.path.basename(self.directory)}")
        self.root.geometry("1400x900")
        self.root.configure(bg=DARK_BG)
        style = ttk.Style()
        base_font = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 11)
        style.theme_use('clam')
        style.configure('.', font=base_font, background=DARK_BG, foreground=DARK_FG)
        style.configure("TFrame", background=DARK_BG)
        style.configure("TLabel", background=DARK_BG, foreground=DARK_FG)
        style.configure("TCombobox", fieldbackground=DARK_ENTRY_BG, background=DARK_ENTRY_BG, foreground=DARK_FG, bordercolor=DARK_SELECT_BG, insertcolor=DARK_FG, arrowcolor=DARK_FG)
        style.configure("TEntry", fieldbackground=DARK_ENTRY_BG, background=DARK_ENTRY_BG, foreground=DARK_FG, bordercolor=DARK_SELECT_BG, insertcolor=DARK_FG)
        style.configure("TButton", background=DARK_BUTTON_BG, foreground=DARK_FG, bordercolor=DARK_SELECT_BG, padding=5)
        style.configure("Treeview", background=DARK_TREE_BG, foreground=DARK_FG, fieldbackground=DARK_TREE_BG, rowheight=25)
        style.map("Treeview", background=[('selected', DARK_SELECT_BG)])
        style.configure("TCheckbutton", background=DARK_BG, foreground=DARK_FG)
        style.configure('Accent.TButton', font=(base_font[0], base_font[1], "bold"), background="#0078d4", foreground=DARK_FG)
        style.map('Accent.TButton', background=[('active', '#106ebe')])
        style.configure("TNotebook", background=DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=DARK_BUTTON_BG, foreground=DARK_FG, padding=[8, 4])
        style.map("TNotebook.Tab", background=[("selected", DARK_SELECT_BG)], expand=[("selected", [1, 1, 1, 0])])

    def _create_widgets(self):
        self.main_container = ttk.Frame(self.root, padding=10)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        vertical_pane = ttk.PanedWindow(self.main_container, orient=tk.VERTICAL)
        vertical_pane.pack(fill=tk.BOTH, expand=True)
        self.top_pane = ttk.PanedWindow(vertical_pane, orient=tk.HORIZONTAL)
        vertical_pane.add(self.top_pane, weight=4)
        bottom_pane_container = ttk.Frame(vertical_pane)
        vertical_pane.add(bottom_pane_container, weight=2)
        
        self._create_tree_pane()
        self._create_selection_pane()
        self._create_bottom_notebook(bottom_pane_container)

        bottom_controls_frame = ttk.Frame(self.main_container)
        bottom_controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        self.btn_toggle_preview = ttk.Button(bottom_controls_frame, text="Show Preview", command=self.toggle_preview)
        self.btn_toggle_preview.pack(side=tk.LEFT)
        self.btn_copy = ttk.Button(bottom_controls_frame, text="Copy to Clipboard", command=self.copy_to_clipboard, style='Accent.TButton')
        self.btn_copy.pack(side=tk.RIGHT)

        self.preview_frame = ttk.Frame(self.main_container)
        self._create_preview_widgets()

    def _create_tree_pane(self, *args):
        tree_frame = ttk.Frame(self.top_pane, padding=(0, 0, 5, 0))
        search_frame = ttk.Frame(tree_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_frame, text="Filter:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        exclusion_main_frame = ttk.Frame(tree_frame)
        exclusion_main_frame.pack(fill=tk.X, pady=(0, 10))
        self.advanced_exclude_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(exclusion_main_frame, text="Advanced Exclusions (Regex)", variable=self.advanced_exclude_var, command=self._toggle_exclude_mode).pack(anchor='w')
        self.simple_exclude_frame = ttk.Frame(exclusion_main_frame)
        ttk.Label(self.simple_exclude_frame, text="Exclude Dirs:").grid(row=0, column=0, sticky='w', pady=(5, 0))
        self.exclude_dirs_var = tk.StringVar(value=" ".join(sorted(list(IGNORE_DIRS))))
        self.exclude_dirs_entry = ttk.Entry(self.simple_exclude_frame, textvariable=self.exclude_dirs_var)
        self.exclude_dirs_entry.grid(row=0, column=1, sticky='ew', pady=(5, 0), padx=(5, 0))
        ttk.Label(self.simple_exclude_frame, text="Exclude Files:").grid(row=1, column=0, sticky='w')
        self.exclude_patterns_var = tk.StringVar(value="*.log *.json *.csv *.env .DS_Store .gitignore")
        self.exclude_patterns_entry = ttk.Entry(self.simple_exclude_frame, textvariable=self.exclude_patterns_var)
        self.exclude_patterns_entry.grid(row=1, column=1, sticky='ew', padx=(5, 0))
        self.simple_exclude_frame.grid_columnconfigure(1, weight=1)
        self.advanced_exclude_frame = ttk.Frame(exclusion_main_frame)
        ttk.Label(self.advanced_exclude_frame, text="Exclude (regex):").pack(side=tk.LEFT)
        self.exclusion_var = tk.StringVar(value=r"venv/|\.git/|\.idea/|\.vscode/|__pycache__|/node_modules/|/build/|/dist/|.*\.log$")
        self.exclusion_entry = ttk.Entry(self.advanced_exclude_frame, textvariable=self.exclusion_var)
        self.exclusion_entry.pack(fill=tk.X, expand=True)
        tree_controls = ttk.Frame(tree_frame)
        tree_controls.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(tree_controls, text="Add Folder", command=self.add_selected_folder).pack(side=tk.LEFT)
        ttk.Button(tree_controls, text="Add All Visible", command=self.add_all_visible_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(tree_controls, text="Expand All", command=self.expand_all_tree_items).pack(side=tk.LEFT)
        ttk.Button(tree_controls, text="Collapse All", command=self.collapse_all_tree_items).pack(side=tk.LEFT, padx=5)
        self.tree = ttk.Treeview(tree_frame, show="tree headings")
        self.tree.heading("#0", text="Project Structure", anchor='w')
        ysb = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        xsb = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.top_pane.add(tree_frame, weight=2)
        self.tree.insert("", "end", text="Scanning project...", tags=('info',))
        self.tree.tag_configure('info', foreground='#888888')

    def _create_selection_pane(self, *args):
        selection_frame = ttk.Frame(self.top_pane, padding=(5, 0, 0, 0))
        preset_frame = ttk.Frame(selection_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar()
        self.preset_combobox = ttk.Combobox(preset_frame, textvariable=self.preset_var, state="readonly")
        self.preset_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(preset_frame, text="Save As...", command=self.save_current_as_preset, width=10).pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="Remove", command=self.remove_selected_preset, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(selection_frame, text="Selected Files (Drag to Reorder)", font=("Segoe UI", 10, "bold")).pack(pady=(0, 5), anchor='w')
        listbox_frame = ttk.Frame(selection_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE, bg=DARK_TREE_BG, fg=DARK_FG, selectbackground=DARK_SELECT_BG, font=("Segoe UI", 10), highlightthickness=0, borderwidth=0)
        listbox_scrollbar = ttk.Scrollbar(listbox_frame, orient='vertical', command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=listbox_scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls_frame = ttk.Frame(selection_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        ttk.Button(controls_frame, text="Remove", command=self.remove_selected).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        self.selected_count_var = tk.StringVar(value="0 files selected")
        ttk.Label(controls_frame, textvariable=self.selected_count_var).pack(side=tk.RIGHT)
        self.top_pane.add(selection_frame, weight=3)

    def _create_bottom_notebook(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        tools_tab = ttk.Frame(notebook, padding=5)
        notebook.add(tools_tab, text="Tools")
        self._create_tools_pane(tools_tab)
        log_tab = ttk.Frame(notebook, padding=5)
        notebook.add(log_tab, text="Log")
        self._create_log_pane(log_tab)

    def _create_tools_pane(self, parent):
        tools_container = ttk.Frame(parent)
        tools_container.pack(fill=tk.BOTH, expand=True)

        # Frame for "Apply Changes" on the left
        apply_changes_frame = ttk.Frame(tools_container)
        apply_changes_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(apply_changes_frame, text="Apply Changes to Files", font=("Segoe UI", 10, "bold")).pack(anchor='w')
        self.apply_changes_text = scrolledtext.ScrolledText(apply_changes_frame, height=4, wrap=tk.WORD, bg=DARK_ENTRY_BG, fg=DARK_FG, insertbackground=DARK_FG, font=("Segoe UI", 10), borderwidth=0, highlightthickness=1)
        self.apply_changes_text.pack(fill=tk.BOTH, expand=True, pady=5)
        apply_changes_controls = ttk.Frame(apply_changes_frame)
        apply_changes_controls.pack(fill=tk.X)
        ttk.Button(apply_changes_controls, text="Apply to Files", command=self._initiate_apply_changes, style='Accent.TButton').pack(side=tk.LEFT)

        # Frame for "Smart Paster" on the right
        smart_paster_frame = ttk.Frame(tools_container)
        smart_paster_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(smart_paster_frame, text="Smart Paster (File Discovery)", font=("Segoe UI", 10, "bold")).pack(anchor='w')
        self.smart_paste_text = scrolledtext.ScrolledText(smart_paster_frame, height=4, wrap=tk.WORD, bg=DARK_ENTRY_BG, fg=DARK_FG, insertbackground=DARK_FG, font=("Segoe UI", 10), borderwidth=0, highlightthickness=1)
        self.smart_paste_text.pack(fill=tk.BOTH, expand=True, pady=5)
        smart_paster_controls = ttk.Frame(smart_paster_frame)
        smart_paster_controls.pack(fill=tk.X)
        ttk.Button(smart_paster_controls, text="Find & Select Files", command=self._initiate_smart_paste, style='Accent.TButton').pack(side=tk.LEFT)

    def _create_log_pane(self, parent):
        ttk.Label(parent, text="Global Log", font=("Segoe UI", 10, "bold")).pack(anchor='w', pady=(5, 2))
        self.log_text = scrolledtext.ScrolledText(parent, height=5, wrap=tk.WORD, bg=DARK_ENTRY_BG, fg=DARK_FG, font=("Consolas", 9), borderwidth=0, highlightthickness=1)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.tag_config("success", foreground="#4CAF50")
        self.log_text.tag_config("error", foreground="#F44336")
        self.log_text.tag_config("info", foreground="#FFFFFF")
        self.log_text.tag_config("warning", foreground="#FFC107")
        self.log_text.config(state=tk.DISABLED)
        
    def _create_preview_widgets(self):
        preview_header_frame = ttk.Frame(self.preview_frame)
        preview_header_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(preview_header_frame, text="Preview", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.preview_stats_var = tk.StringVar()
        ttk.Label(preview_header_frame, textvariable=self.preview_stats_var, foreground="#aaaaaa").pack(side=tk.RIGHT)
        self.preview_text = scrolledtext.ScrolledText(self.preview_frame, height=10, wrap=tk.WORD, bg=DARK_ENTRY_BG, fg=DARK_FG, font=("Consolas", 10), borderwidth=0, highlightthickness=1)
        self.preview_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    def _bind_events(self):
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        self.tree.bind('<<TreeviewOpen>>', self.on_tree_expand)
        self.listbox.bind("<Double-1>", lambda e: self.remove_selected())
        self.listbox.bind("<Button-1>", self.on_drag_start)
        self.listbox.bind("<B1-Motion>", self.on_drag_motion)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.update_preview())
        self.preset_combobox.bind("<<ComboboxSelected>>", self.on_preset_selected)
        self.search_var.trace_add("write", self._debounce_search)
        self.exclude_dirs_var.trace_add("write", self._debounce_search)
        self.exclude_patterns_var.trace_add("write", self._debounce_search)
        self.exclusion_var.trace_add("write", self._debounce_search)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        for widget in [self.search_entry, self.exclusion_entry, self.exclude_dirs_entry, self.exclude_patterns_entry, self.preview_text, self.apply_changes_text]:
            self._bind_select_all(widget)
            
    def _log_message(self, message: str, level: str = 'info'):
        def update_log():
            self.log_text.config(state=tk.NORMAL)
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", (level,))
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, update_log)
    
    def _initiate_apply_changes(self):
        content = self.apply_changes_text.get("1.0", tk.END).strip()
        if not content:
            self._log_message("Apply Changes: Textbox is empty.", 'warning')
            return
        self._log_message("Apply Changes: Starting file writing operation...")
        threading.Thread(target=self._apply_changes_worker, args=(content,), daemon=True).start()

    def _apply_changes_worker(self, content: str):
        results = apply_changes_to_files(content, self.directory)
        for file_path in results["success"]:
            self._log_message(f"  ✓ Applied changes to {file_path}", 'success')
        for error_msg in results["errors"]:
            self._log_message(f"  ✗ {error_msg}", 'error')
        summary = f"Apply Changes: Finished. {len(results['success'])} success, {len(results['errors'])} errors."
        self._log_message(summary, 'error' if results['errors'] else 'success')
        if results['success']:
            self.root.after(0, self._perform_filter)

    def _initiate_smart_paste(self):
        content = self.smart_paste_text.get("1.0", tk.END).strip()
        if not content:
            self._log_message("Smart Paster: Textbox is empty.", 'warning')
            return
        self._log_message("Smart Paster: Starting file discovery...")
        self.smart_paste_text.config(state=tk.DISABLED)
        threading.Thread(target=self._smart_paste_worker, args=(content,), daemon=True).start()

    def _smart_paste_worker(self, content: str):
        try:
            # Run the async orchestrator
            found_rel_paths = asyncio.run(process_smart_request(content, self.directory))
            # Schedule the UI update on the main thread
            self.root.after(0, self._update_ui_with_smart_results, found_rel_paths)
        except Exception as e:
            self._log_message(f"Smart Paster Error: {e}", "error")
        finally:
            # Re-enable the text box on the main thread
            self.root.after(0, lambda: self.smart_paste_text.config(state=tk.NORMAL))

    def _update_ui_with_smart_results(self, rel_paths: List[str]):
        if not rel_paths:
            self._log_message("Smart Paster: No files found.", "warning")
            return
        
        added_count = 0
        for rel_path in rel_paths:
            # Ensure file exists and is not already selected
            abs_path = os.path.join(self.directory, os.path.normpath(rel_path))
            if os.path.exists(abs_path) and rel_path not in self.selected_files_map:
                self.listbox.insert(tk.END, rel_path)
                self.selected_files_map[rel_path] = True
                added_count += 1
        
        if added_count > 0:
            self._update_ui_state() # This updates count and preview
            self._log_message(f"Smart Paster: Added {added_count} new file(s) to selection.", "success")
        else:
            self._log_message("Smart Paster: All found files were already selected.", "info")

    def start_async_project_load(self):
        self._log_message("Scanning project files...")
        threading.Thread(target=self._project_load_worker, daemon=True).start()

    def _project_load_worker(self):
        self._scan_and_cache_all_files()
        self.root.after(0, self.finish_project_load)

    def finish_project_load(self):
        self.load_preset_into_ui()

    def _update_ui_state(self, auto_save: bool = True):
        self.update_selected_count()
        self.update_preview()
        if auto_save:
            self._debounce_auto_save()

    def on_tree_double_click(self, event: tk.Event):
        item_id = self.tree.identify_row(event.y)
        if item_id:
            item = self.tree.item(item_id)
            if 'file' in item.get('tags', []) and item['values']:
                fp = item['values'][0]
                if fp not in self.selected_files_map:
                    self.listbox.insert(tk.END, fp)
                    self.selected_files_map[fp] = True
                    self._update_ui_state()
            elif 'folder' in item.get('tags', []):
                self.tree.selection_set(item_id)
                self.add_selected_folder()

    def repopulate_tree(self, files_to_display: Optional[List[str]] = None):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if files_to_display is None:
            self.tree.bind('<<TreeviewOpen>>', self.on_tree_expand)
            self.process_directory("", self.directory)
            return
        self.tree.unbind('<<TreeviewOpen>>')
        if not files_to_display:
            self.tree.insert("", "end", text="No matching files found.", tags=('info',))
            return
        nodes: Dict[str, str] = {"": ""}
        tree_insert = self.tree.insert
        for file_path in sorted(files_to_display):
            path_parts = file_path.split('/')
            parent_path = ""
            for i, part in enumerate(path_parts[:-1]):
                current_path = os.path.join(*path_parts[:i+1])
                if current_path not in nodes:
                    nodes[current_path] = tree_insert(nodes.get(parent_path, ""), 'end', text=f"📁 {part}", values=[current_path], tags=('folder',), open=True)
                parent_path = current_path
            tree_insert(nodes.get(parent_path, ""), 'end', text=f"📄 {path_parts[-1]}", values=[file_path], tags=('file',))
        self.tree.tag_configure('file', foreground='#87CEEB')
        self.tree.tag_configure('folder', foreground='#DDA0DD')

    def add_all_visible_files(self):
        visible_files, added_count = [], 0
        def _collect_files(parent_id):
            for child_id in self.tree.get_children(parent_id):
                item = self.tree.item(child_id)
                if 'file' in item.get('tags', []) and item['values']:
                    visible_files.append(item['values'][0])
                elif 'folder' in item.get('tags', []):
                    _collect_files(child_id)
        _collect_files("")
        for fp in visible_files:
            if fp not in self.selected_files_map:
                self.listbox.insert(tk.END, fp)
                self.selected_files_map[fp] = True
                added_count += 1
        if added_count > 0:
            self._update_ui_state()
        self._log_message(f"Added {added_count} visible file(s).")

    def collapse_all_tree_items(self):
        for item in self.tree.get_children():
            self.tree.item(item, open=False)

    def load_preset_into_ui(self):
        name = self.preset_var.get()
        if not name or name not in self.presets:
            return
        self._log_message(f"Loading preset '{name}'...")
        data = self.presets[name]
        self.search_var.set(data.get("filter_text", ""))
        self.advanced_exclude_var.set(data.get("advanced_exclude_mode", False))
        self.exclude_dirs_var.set(data.get("exclude_dirs", " ".join(sorted(list(IGNORE_DIRS)))))
        self.exclude_patterns_var.set(data.get("exclude_patterns", "*.log *.json"))
        self.exclusion_var.set(data.get("exclusion_regex", r"venv/|\.git/"))
        self._toggle_exclude_mode()
        self._perform_filter(from_preset_load=True)
        self.clear_all(auto_save=False)
        added = 0
        for fp in data.get("selected_files", []):
            if os.path.exists(os.path.join(self.directory, os.path.normpath(fp))) and fp not in self.selected_files_map:
                self.listbox.insert(tk.END, fp)
                self.selected_files_map[fp] = True
                added += 1
        self._update_ui_state(auto_save=False)
        self._log_message(f"Loaded preset '{name}'. ({added}/{len(data.get('selected_files', []))} files).")

    def auto_save_current_preset(self):
        name = self.preset_var.get()
        if not name:
            return
        data = {"selected_files": list(self.listbox.get(0, tk.END)), "filter_text": self.search_var.get(), "advanced_exclude_mode": self.advanced_exclude_var.get(), "exclude_dirs": self.exclude_dirs_var.get(), "exclude_patterns": self.exclude_patterns_var.get(), "exclusion_regex": self.exclusion_var.get()}
        if self.presets.get(name) != data:
            self.presets[name] = data
            self.save_config()

    def _toggle_exclude_mode(self):
        if self.advanced_exclude_var.get():
            self.simple_exclude_frame.pack_forget()
            self.advanced_exclude_frame.pack(fill=tk.X, expand=True)
        else:
            self.advanced_exclude_frame.pack_forget()
            self.simple_exclude_frame.pack(fill=tk.X, expand=True)
        self._debounce_search()

    def _get_exclusion_regex(self) -> Optional[re.Pattern]:
        try:
            if self.advanced_exclude_var.get():
                if exclusion_str := self.exclusion_var.get():
                    return re.compile(exclusion_str, re.IGNORECASE)
                return None
            
            dirs = [d for d in self.exclude_dirs_var.get().split() if d]
            files = [p for p in self.exclude_patterns_var.get().split() if p]
            parts = []
            if dirs:
                sep = re.escape(os.path.sep)
                dir_alternations = "|".join(re.escape(d) for d in dirs)
                dir_pattern = f"(^|{sep})({dir_alternations})({sep}|$)"
                parts.append(dir_pattern)
            if files:
                file_patterns = [fnmatch.translate(p) for p in files]
                parts.extend(file_patterns)
            
            if not parts:
                return None
            
            final_regex_str = "|".join(f"({p})" for p in parts)
            return re.compile(final_regex_str, re.IGNORECASE)
        except re.error as e:
            self._log_message(f"Regex Error: {e}", 'error')
            return None

    def _scan_and_cache_all_files(self):
        all_files_list, regex = [], self._get_exclusion_regex()
        for root, dirs, files in os.walk(self.directory, topdown=True):
            rel_root = os.path.relpath(root, self.directory).replace(os.path.sep, '/')
            dirs[:] = [d for d in dirs if not (regex and regex.search(f"{rel_root}/{d}/".replace('./', '')))]
            for filename in files:
                rel_path = os.path.relpath(os.path.join(root, filename), self.directory).replace(os.path.sep, '/')
                if not (regex and regex.search(rel_path)) and is_includable_file(os.path.join(root, filename)):
                    all_files_list.append(rel_path)
        self.all_text_files = sorted(all_files_list, key=str.lower)

    def _perform_filter(self, from_preset_load: bool = False):
        search_term = self.search_var.get().lower()
        current_exclusion_state = (self.advanced_exclude_var.get(), self.exclude_dirs_var.get(), self.exclude_patterns_var.get(), self.exclusion_var.get())
        if not hasattr(self, '_last_exclusion_state') or self._last_exclusion_state != current_exclusion_state:
            self._last_exclusion_state = current_exclusion_state
            threading.Thread(target=self._scan_and_repopulate, args=(search_term, from_preset_load), daemon=True).start()
        else:
            files_to_display = [f for f in self.all_text_files if search_term in os.path.basename(f).lower()] if search_term else None
            self.repopulate_tree(files_to_display)
            if not from_preset_load:
                self._debounce_auto_save()

    def _scan_and_repopulate(self, search_term: str, from_preset_load: bool):
        self._scan_and_cache_all_files()
        def callback():
            files_to_display = [f for f in self.all_text_files if search_term in os.path.basename(f).lower()] if search_term else None
            self.repopulate_tree(files_to_display)
            if not from_preset_load:
                self._debounce_auto_save()
        self.root.after(0, callback)

    def on_closing(self):
        self.auto_save_current_preset()
        self.root.destroy()
    def _debounce_search(self, *args):
        if self._search_job:
            self.root.after_cancel(self._search_job)
        self._search_job = self.root.after(300, self._perform_filter)

    def load_project_config(self):
        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    self.full_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            self.full_config = {}
        if self.directory not in self.full_config:
            self.full_config[self.directory] = {"presets": {DEFAULT_PRESET_NAME: {}}, "last_active_preset": DEFAULT_PRESET_NAME}
        self.project_data = self.full_config[self.directory]
        self.presets = self.project_data.get('presets', {})
        if DEFAULT_PRESET_NAME not in self.presets:
            self.presets[DEFAULT_PRESET_NAME] = {}
        self.update_preset_combobox()
        self.preset_var.set(self.project_data.get("last_active_preset", DEFAULT_PRESET_NAME))

    def save_config(self, quiet: bool = True):
        self.project_data['last_active_preset'] = self.preset_var.get()
        self.project_data['presets'] = self.presets
        self.full_config[self.directory] = self.project_data
        try:
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.full_config, f, indent=4)
            if not quiet:
                self._log_message(f"Preset '{self.preset_var.get()}' saved.", 'success')
        except IOError as e:
            messagebox.showerror("Config Error", f"Could not save config: {e}")

    def _debounce_auto_save(self, *args):
        if self._auto_save_job:
            self.root.after_cancel(self._auto_save_job)
        self._auto_save_job = self.root.after(1500, self.auto_save_current_preset)

    def update_preset_combobox(self):
        self.preset_combobox['values'] = [DEFAULT_PRESET_NAME] + sorted([p for p in self.presets.keys() if p != DEFAULT_PRESET_NAME], key=str.lower)

    def save_current_as_preset(self):
        name = simpledialog.askstring("Save New Preset", "Enter a name:", parent=self.root)
        if not (name and name.strip()):
            return
        name = name.strip()
        if name in self.presets and not messagebox.askyesno("Confirm Overwrite", f"Preset '{name}' exists. Overwrite?", parent=self.root):
            return
        self.auto_save_current_preset()
        current_name = self.preset_var.get()
        if current_name in self.presets:
            self.presets[name] = self.presets[current_name]
        self.update_preset_combobox()
        self.preset_var.set(name)
        self.save_config(quiet=False)

    def on_preset_selected(self, event=None):
        self.load_preset_into_ui()
        self._debounce_auto_save()

    def remove_selected_preset(self):
        name = self.preset_var.get()
        if name == DEFAULT_PRESET_NAME:
            messagebox.showerror("Action Denied", "Default preset cannot be removed.")
            return
        if messagebox.askyesno("Confirm Deletion", f"Delete preset '{name}'?", parent=self.root):
            if name in self.presets:
                del self.presets[name]
                self.update_preset_combobox()
                self.preset_var.set(DEFAULT_PRESET_NAME)
                self.load_preset_into_ui()
                self.save_config()
                self._log_message(f"Preset '{name}' removed.", 'info')

    def _bind_select_all(self, w: tk.Widget):
        def sa(e=None):
            if isinstance(w, (ttk.Entry, tk.Entry)):
                w.select_range(0, 'end')
            elif isinstance(w, (scrolledtext.ScrolledText, tk.Text)):
                w.tag_add('sel', '1.0', 'end')
            return "break"
        w.bind("<Control-a>", sa)
        w.bind("<Command-a>", sa)

    def _setup_interrupt_handler(self):
        self.interrupted = False
        try:
            signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'interrupted', True))
        except (ValueError, TypeError):
            pass
        self.root.after(250, self._check_for_interrupt)

    def _check_for_interrupt(self):
        if self.interrupted:
            self.on_closing()
        else:
            self.root.after(250, self._check_for_interrupt)

    def process_directory(self, parent_id: str, path: str):
        try:
            items = sorted(os.listdir(path), key=str.lower)
        except (OSError, PermissionError):
            return
        regex = self._get_exclusion_regex()
        for cid in self.tree.get_children(parent_id):
            if self.tree.item(cid, "values") == ("dummy",):
                self.tree.delete(cid)
        for name in items:
            full_path = os.path.join(path, name)
            rel_path = os.path.relpath(full_path, self.directory).replace(os.path.sep, '/')
            is_dir = os.path.isdir(full_path)
            check_path = rel_path + '/' if is_dir else rel_path
            if regex and regex.search(check_path):
                continue
            if is_dir:
                did = self.tree.insert(parent_id, 'end', text=f"📁 {name}", values=[rel_path], tags=('folder',))
                self.tree.insert(did, 'end', text='...', values=['dummy'])
            elif is_includable_file(full_path):
                self.tree.insert(parent_id, 'end', text=f"📄 {name}", values=[rel_path], tags=('file',))
        self.tree.tag_configure('file', foreground='#87CEEB')
        self.tree.tag_configure('folder', foreground='#DDA0DD')

    def on_tree_expand(self, event: Optional[tk.Event]):
        item_id = self.tree.focus()
        self._populate_tree_node(item_id)
    def _populate_tree_node(self, item_id: str):
        if not item_id or not (children := self.tree.get_children(item_id)) or self.tree.item(children[0], "values") != ("dummy",):
            return
        if full_path_parts := self.tree.item(item_id, "values"):
            self.process_directory(item_id, os.path.join(self.directory, full_path_parts[0].replace('/', os.path.sep)))
    def expand_all_tree_items(self):
        for item in self.tree.get_children():
            self._expand_tree_item_recursive(item)
    def _expand_tree_item_recursive(self, item_id: str):
        self._populate_tree_node(item_id)
        if self.tree.get_children(item_id):
            self.tree.item(item_id, open=True)
            for child in self.tree.get_children(item_id):
                self._expand_tree_item_recursive(child)

    def get_all_files_in_folder(self, path: str) -> List[str]:
        return [f for f in self.all_text_files if f.replace(os.path.sep, '/').startswith(path.replace(os.path.sep, '/') + '/')]

    def add_selected_folder(self):
        if not self.tree.selection():
            return
        item = self.tree.item(self.tree.selection()[0])
        if 'folder' not in item['tags']:
            return
        path, files, count = item['values'][0], self.get_all_files_in_folder(item['values'][0]), 0
        for fp in files:
            if fp not in self.selected_files_map:
                self.listbox.insert(tk.END, fp)
                self.selected_files_map[fp] = True
                count += 1
        self._update_ui_state()
        self._log_message(f"Added {count} file(s) from {os.path.basename(path)}.")

    def remove_selected(self):
        if self.listbox.curselection():
            idx = self.listbox.curselection()[0]
            fp = self.listbox.get(idx)
            self.listbox.delete(idx)
            if fp in self.selected_files_map:
                del self.selected_files_map[fp]
            self._update_ui_state()

    def clear_all(self, auto_save: bool = True):
        self.listbox.delete(0, tk.END)
        self.selected_files_map.clear()
        self._update_ui_state(auto_save=auto_save)

    def update_selected_count(self):
        c = self.listbox.size()
        self.selected_count_var.set(f"{c} file{'s' if c != 1 else ''} selected")

    def toggle_preview(self):
        self.preview_visible = not self.preview_visible
        if self.preview_visible:
            self.preview_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0), in_=self.main_container)
            self.update_preview()
        else:
            self.preview_frame.pack_forget()
        self.btn_toggle_preview.configure(text="Hide Preview" if self.preview_visible else "Show Preview")

    def update_preview(self):
        if not self.preview_visible:
            return
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        selected_files = list(self.listbox.get(0, tk.END))
        if not selected_files:
            self.preview_text.insert(1.0, "No files selected.")
            self.preview_stats_var.set("L: 0 | C: 0")
            return
        out = build_clipboard_content([os.path.join(self.directory, f) for f in selected_files], self.directory, 200000)
        self.preview_stats_var.set(f"L: {len(out.splitlines()):,} | C: {len(out):,}")
        self.preview_text.insert(1.0, out)
        self.preview_text.see(1.0)
        self.preview_text.config(state=tk.DISABLED)

    def copy_to_clipboard(self):
        if pyperclip is None:
            messagebox.showerror("Error", "Install pyperclip: pip install pyperclip")
            return
        selected = list(self.listbox.get(0, tk.END))
        if not selected:
            self._log_message("Copy: No files selected.", 'warning')
            return
        self._log_message("Copy: Processing files...")
        self.root.update_idletasks()
        out = build_clipboard_content([os.path.join(self.directory, f) for f in selected], self.directory)
        pyperclip.copy(out)
        size_kb = len(out) / 1024
        self._log_message(f"Copied {len(selected)} file(s) to clipboard! ({size_kb:.1f} KB)", 'success')

    def on_drag_start(self, event: tk.Event):
        self.drag_start_index = event.widget.nearest(event.y)
    def on_drag_motion(self, event: tk.Event):
        if self.drag_start_index is not None and (ci := event.widget.nearest(event.y)) != -1 and ci != self.drag_start_index:
            item = self.listbox.get(self.drag_start_index)
            self.listbox.delete(self.drag_start_index)
            self.listbox.insert(ci, item)
            self.drag_start_index = ci
            self._update_ui_state()