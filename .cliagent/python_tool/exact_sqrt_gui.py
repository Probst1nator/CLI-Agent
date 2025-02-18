import tkinter as tk
from decimal import Decimal, getcontext
from typing import List, Optional

class SqrtVisualizer:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Square Root of 10 Visualizer")
        self.root.geometry("600x400")
        
        # Set decimal precision
        getcontext().prec = 1000
        
        # Initialize variables
        self.current_digits: List[str] = []
        self.digit_index: int = 0
        self.sqrt_10: Optional[Decimal] = None
        
        # Create GUI elements
        self.display = tk.Label(
            self.root,
            text="√10 = ",
            font=("Courier", 14),
            wraplength=580,
            justify="left"
        )
        self.display.pack(pady=20)
        
        self.next_button = tk.Button(
            self.root,
            text="Show Next Digit",
            command=self.show_next_digit
        )
        self.next_button.pack(pady=10)
        
        # Calculate sqrt(10)
        self.sqrt_10 = Decimal(10).sqrt()
        self.current_digits = list(str(self.sqrt_10))
        
    def show_next_digit(self) -> None:
        if self.digit_index < len(self.current_digits):
            display_text = "√10 = "
            for i in range(self.digit_index + 1):
                if i == 1:  # Add decimal point
                    display_text += "."
                display_text += self.current_digits[i]
            self.display.config(text=display_text)
            self.digit_index += 1
            
    def run(self) -> None:
        self.root.mainloop()

if __name__ == "__main__":
    visualizer = SqrtVisualizer()
    visualizer.run()
