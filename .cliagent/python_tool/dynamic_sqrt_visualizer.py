import tkinter as tk
from typing import Optional, List
from decimal import Decimal, getcontext
import threading
from time import sleep

class SqrtVisualizer:
    def __init__(self) -> None:
        self.root: tk.Tk = tk.Tk()
        self.root.title("√10 Calculator")
        self.root.geometry("800x400")
        
        self.digits: int = 0
        self.text: tk.Text = tk.Text(self.root, wrap=tk.WORD, font=('Courier', 12))
        self.text.pack(fill=tk.BOTH, expand=True)
        
        self.running: bool = True
        self.calculation_thread: Optional[threading.Thread] = None
        
    def calculate_sqrt(self) -> None:
        while self.running:
            self.digits += 10
            getcontext().prec = self.digits
            result: Decimal = Decimal('10').sqrt()
            
            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END, f"√10 to {self.digits} digits:\n\n{result}")
            sleep(1)  # Update every second
            
    def start(self) -> None:
        self.calculation_thread = threading.Thread(target=self.calculate_sqrt)
        self.calculation_thread.daemon = True
        self.calculation_thread.start()
        self.root.mainloop()
        
    def cleanup(self) -> None:
        self.running = False
        if self.calculation_thread:
            self.calculation_thread.join()
        self.root.destroy()

if __name__ == "__main__":
    visualizer: SqrtVisualizer = SqrtVisualizer()
    try:
        visualizer.start()
    finally:
        visualizer.cleanup()
