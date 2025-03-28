#!/usr/bin/env python3
"""
Directory Structurizer

This script opens a folder picker dialog and lists all files and folders within the selected directory.
The script then creates a folder named "structured_files" in the selected directory and one by one scans the files using a mcdp agent to copy them in a structured way into this folder.
"""

import os
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional


def select_directory() -> Optional[str]:
    """
    Opens a directory selection dialog.
    
    Returns:
        str or None: Path to the selected directory, or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    directory = filedialog.askdirectory(
        title="Select a directory to scan",
        mustexist=True
    )
    
    root.destroy()
    return directory if directory else None


def get_all_paths(directory: str, recursive: bool = True) -> List[str]:
    """
    Gets all file and directory paths within the specified directory.
    
    Args:
        directory (str): Path to the directory to scan
        recursive (bool): Whether to scan subdirectories recursively (default: True)
        
    Returns:
        List[str]: List of all paths (files and directories) found
    """
    all_paths = []
    
    if recursive:
        # Walk through the directory tree recursively
        for root, dirs, files in os.walk(directory):
            # Add directories
            for dir_name in dirs:
                path = os.path.join(root, dir_name)
                all_paths.append(path)
            
            # Add files
            for file_name in files:
                path = os.path.join(root, file_name)
                all_paths.append(path)
    else:
        # Non-recursive mode: only get items in the top directory
        # Add directories
        for dir_name in os.listdir(directory):
            path = os.path.join(directory, dir_name)
            if os.path.isdir(path):
                all_paths.append(path)
        
        # Add files
        for file_name in os.listdir(directory):
            path = os.path.join(directory, file_name)
            if os.path.isfile(path):
                all_paths.append(path)
    
    return all_paths




def main() -> None:
    """Main function to run the directory structurizer."""
    print("Directory Structurizer")
    print("---------------------")
    
    # Open the folder picker dialog
    selected_dir = select_directory()
    
    if not selected_dir:
        print("No directory selected. Exiting.")
        return
    
    print(f"Selected directory: {selected_dir}")
    print("Scanning for files and folders...")
    
    # Get all paths
    paths = get_all_paths(selected_dir, recursive=False)
    
    # Print results
    print(f"\nFound {len(paths)} items:")
    for i, path in enumerate(paths, 1):
        rel_path = os.path.relpath(path, selected_dir)
        is_dir = os.path.isdir(path)
        type_indicator = "[DIR]" if is_dir else "[FILE]"
        print(f"{i}. {type_indicator} {rel_path}")
    
    print("\nComplete!")


if __name__ == "__main__":
    main()
