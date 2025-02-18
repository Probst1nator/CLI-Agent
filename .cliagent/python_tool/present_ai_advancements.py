#!/usr/bin/env python3
"""
present_ai_advancements.py

A script to present the latest AI advancements in a formatted presentation.
Uses FPDF for PDF creation and tkinter for display.
"""

from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk, messagebox
import requests
from PIL import Image
import matplotlib.pyplot as plt
import io


class AIAdvancementsPresentation:
    """Class to handle AI advancements presentation creation and display."""

    def __init__(self) -> None:
        """Initialize the presentation class with default values."""
        self.output_file: str = 'latest_ai_advancements.pdf'
        self.ai_data: Dict = {}
        self.pdf: Optional[FPDF] = None

    def fetch_ai_advancements(self) -> bool:
        """
        Fetch latest AI advancements from a mock API.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Mock data - in real implementation, this would be an API call
            self.ai_data = {
                "advancements": [
                    {
                        "title": "Large Language Models",
                        "description": "Recent breakthroughs in transformer-based architectures",
                        "impact_score": 95
                    },
                    {
                        "title": "Computer Vision",
                        "description": "Advances in object detection and recognition",
                        "impact_score": 88
                    },
                    {
                        "title": "Reinforcement Learning",
                        "description": "Progress in algorithms that learn through trial and error",
                        "impact_score": 90
                    }
                ],
            }
            return True
        except Exception as e:
            print(f"Error fetching AI advancements: {str(e)}")
            return False

    def create_pdf(self) -> bool:
        """
        Create a PDF presentation with formatted content.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.pdf = FPDF()
            self.pdf.add_page()
            self.pdf.set_font("Arial", size=12)
            
            self.pdf.cell(200, 10, txt="Latest AI Advancements", ln=True, align='C')
            for advancement in self.ai_data['advancements']:
                self.pdf.cell(200, 10, txt=f"{advancement['title']}: {advancement['description']} (Impact Score: {advancement['impact_score']})", ln=True)
            
            self.pdf.output(self.output_file)
            return True
        except Exception as e:
            print(f"Error creating PDF: {str(e)}")
            return False

    def display_presentation(self) -> None:
        """Display the PDF presentation in a tkinter window."""
        try:
            root = tk.Tk()
            root.title("AI Advancements Presentation")
            root.geometry("800x600")

            # Create main frame
            main_frame = ttk.Frame(root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Add success message
            ttk.Label(
                main_frame,
                text="Presentation has been generated successfully!",
                font=("Arial", 14)
            ).pack(pady=20)

            # Add file location
            ttk.Label(
                main_frame,
                text=f"File saved as: {os.path.abspath(self.output_file)}",
                font=("Arial", 12)
            ).pack(pady=10)

            # Add view button
            def open_pdf():
                """Open the PDF file with default system viewer."""
                try:
                    os.startfile(self.output_file) if os.name == 'nt' else \
                    os.system(f'xdg-open {self.output_file}')
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open PDF: {str(e)}")

            ttk.Button(
                main_frame,
                text="View Presentation",
                command=open_pdf
            ).pack(pady=20)

            root.mainloop()
        except Exception as e:
            print(f"Error displaying presentation: {str(e)}")
            messagebox.showerror("Error", f"Error displaying presentation: {str(e)}")


def main() -> None:
    """Main function to run the presentation creation and display."""
    try:
        presentation = AIAdvancementsPresentation()
        
        if not presentation.fetch_ai_advancements():
            raise Exception("Failed to fetch AI advancements data")
        
        if not presentation.create_pdf():
            raise Exception("Failed to create PDF presentation")
        
        presentation.display_presentation()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        messagebox.showerror("Error", f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()