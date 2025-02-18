import tkinter as tk
from decimal import Decimal, getcontext
from typing import Optional

class SqrtCalculator(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Square Root of 10 Calculator")
        self.geometry("600x400")
        
        # Initialize variables
        self.current_precision: int = 1
        self.previous_result: Optional[str] = None
        
        # Create display
        self.result_text = tk.Text(self, height=20, width=70)
        self.result_text.pack(pady=20)
        
        # Start calculation
        self.calculate_next_digit()
    
    def calculate_next_digit(self) -> None:
        getcontext().prec = self.current_precision + 1
        result = str(Decimal('10').sqrt())
        
        if self.previous_result != result:
            self.result_text.insert('1.0', f"Precision {self.current_precision}: {result}\n")
            self.previous_result = result
        
        self.current_precision += 1
        self.after(100, self.calculate_next_digit)

if __name__ == "__main__":
    app = SqrtCalculator()
    app.mainloop()
