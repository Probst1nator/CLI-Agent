import tkinter as tk
from typing import Optional, Tuple
from decimal import Decimal, getcontext
import time

class SqrtVisualizerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Live √10 Calculator")
        self.root.geometry("600x400")
        
        # Configure display
        self.display = tk.Text(root, height=20, width=60, font=('Courier', 12))
        self.display.pack(pady=10)
        
        # Control buttons
        self.start_button = tk.Button(root, text="Start Calculation", command=self.start_calculation)
        self.start_button.pack(pady=5)
        
        # State variables
        self.running: bool = False
        self.current_precision: int = 1
        self.previous_result: Optional[Decimal] = None
        
    def newton_sqrt_step(self, n: Decimal, x0: Decimal, precision: int) -> Decimal:
        """Single step of Newton's method for square root"""
        getcontext().prec = precision
        return (x0 + n / x0) / 2

    def format_decimal_diff(self, num: Decimal) -> Tuple[str, str]:
        """Format decimal showing new digits in bold"""
        str_num = str(num)
        if self.previous_result is None:
            return str_num, "First calculation"
        
        prev_str = str(self.previous_result)
        diff_pos = 0
        for i, (a, b) in enumerate(zip(str_num, prev_str)):
            if a != b:
                diff_pos = i
                break
                
        return (str_num[:diff_pos] + ">" + str_num[diff_pos:], 
                f"New digits from position {diff_pos}")

    def start_calculation(self) -> None:
        if not self.running:
            self.running = True
            self.start_button.config(text="Stop Calculation")
            self.display.delete(1.0, tk.END)
            self.calculate_step()
        else:
            self.running = False
            self.start_button.config(text="Start Calculation")

    def calculate_step(self) -> None:
        if not self.running:
            return
            
        # Increase precision and calculate
        getcontext().prec = self.current_precision
        n = Decimal(10)
        x = Decimal(3) if self.previous_result is None else self.previous_result
        result = self.newton_sqrt_step(n, x, self.current_precision)
        
        # Format and display result
        formatted_result, info = self.format_decimal_diff(result)
        self.display.insert(tk.END, f"Precision {self.current_precision}:\n{formatted_result}\n{info}\n\n")
        self.display.see(tk.END)
        
        # Update state
        self.previous_result = result
        self.current_precision += 1
        
        # Schedule next calculation
        if self.running:
            self.root.after(100, self.calculate_step)

def main() -> None:
    root = tk.Tk()
    app = SqrtVisualizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
