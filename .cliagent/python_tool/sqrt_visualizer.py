import tkinter as tk
from decimal import Decimal, getcontext
from typing import Optional

class SqrtCalculator:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Square Root Calculator")
        
        # Set precision for decimal calculations
        getcontext().prec = 100
        
        # Create and pack widgets
        self.label = tk.Label(root, text="Square Root of 10:")
        self.label.pack(pady=10)
        
        self.result_text = tk.Text(root, height=10, width=50)
        self.result_text.pack(pady=10, padx=10)
        
        # Calculate and display immediately
        self.calculate_sqrt()
        
    def calculate_sqrt(self) -> None:
        result: Decimal = Decimal('10').sqrt()
        formatted_result: str = f"√10 = {result}"
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', formatted_result)

def main() -> None:
    root: tk.Tk = tk.Tk()
    app: SqrtCalculator = SqrtCalculator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
