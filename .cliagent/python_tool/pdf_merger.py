import tkinter as tk
from tkinter import filedialog, ttk
from typing import List, Optional, Tuple
from PyPDF2 import PdfMerger
from PIL import Image
import os
import io

class PDFMergerApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("PDF & Image Merger")
        self.root.geometry("600x400")
        
        # List to store file paths
        self.files: List[Tuple[str, str]] = []  # (path, type) tuples
        
        # Create and configure main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self) -> None:
        # Create listbox to display selected files
        self.file_listbox = tk.Listbox(self.main_frame, width=70, height=15)
        self.file_listbox.grid(row=0, column=0, columnspan=3, pady=5)
        
        # Create scrollbar for listbox
        scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.grid(row=0, column=3, sticky=(tk.N, tk.S))
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Create buttons
        ttk.Button(self.main_frame, text="Add PDFs", command=self.add_pdfs).grid(row=1, column=0, pady=5)
        ttk.Button(self.main_frame, text="Add Images", command=self.add_images).grid(row=1, column=1, pady=5)
        ttk.Button(self.main_frame, text="Remove Selected", command=self.remove_selected).grid(row=1, column=2, pady=5)
        
        # Create clear and merge buttons
        ttk.Button(self.main_frame, text="Clear All", command=self.clear_all).grid(row=2, column=0, pady=5)
        ttk.Button(self.main_frame, text="Merge Files", command=self.merge_files).grid(row=2, column=1, columnspan=2, pady=5)
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.grid(row=3, column=0, columnspan=3)
        
    def add_pdfs(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select PDF files",
            filetypes=[("PDF files", "*.pdf")]
        )
        for file in files:
            if file not in [f[0] for f in self.files]:
                self.files.append((file, "pdf"))
                self.file_listbox.insert(tk.END, f"[PDF] {os.path.basename(file)}")
                
    def add_images(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select Image files",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")
            ]
        )
        for file in files:
            if file not in [f[0] for f in self.files]:
                self.files.append((file, "image"))
                self.file_listbox.insert(tk.END, f"[IMG] {os.path.basename(file)}")
                
    def remove_selected(self) -> None:
        selection = self.file_listbox.curselection()
        for index in reversed(selection):
            self.file_listbox.delete(index)
            self.files.pop(index)
            
    def clear_all(self) -> None:
        self.file_listbox.delete(0, tk.END)
        self.files.clear()
        
    def image_to_pdf(self, image_path: str) -> io.BytesIO:
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        pdf_bytes = io.BytesIO()
        image.save(pdf_bytes, format='PDF')
        pdf_bytes.seek(0)
        return pdf_bytes
        
    def merge_files(self) -> None:
        if len(self.files) < 1:
            self.status_label.config(text="Please select at least 1 file to merge!")
            return
            
        output_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save merged PDF as"
        )
        
        if output_path:
            try:
                merger = PdfMerger()
                for file_path, file_type in self.files:
                    if file_type == "pdf":
                        merger.append(file_path)
                    else:  # image
                        pdf_bytes = self.image_to_pdf(file_path)
                        merger.append(pdf_bytes)
                        
                merger.write(output_path)
                merger.close()
                
                self.status_label.config(text=f"Files successfully merged to: {os.path.basename(output_path)}")
            except Exception as e:
                self.status_label.config(text=f"Error merging files: {str(e)}")
                
    def run(self) -> None:
        self.root.mainloop()

def main() -> None:
    app = PDFMergerApp()
    app.run()

if __name__ == "__main__":
    main()
