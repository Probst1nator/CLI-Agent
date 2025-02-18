import tkinter as tk
from typing import Optional

class SimpleWindow:
    def __init__(self) -> None:
        self.window: tk.Tk = tk.Tk()
        self.window.title("Simple GUI")
        self.window.geometry("300x200")
        
        # Create and pack the label
        self.label: tk.Label = tk.Label(
            self.window, 
            text="hi", 
            font=("Arial", 24)
        )
        self.label.pack(expand=True)
    
    def run(self) -> None:
        self.window.mainloop()

if __name__ == "__main__":
    app: SimpleWindow = SimpleWindow()
    app.run()
