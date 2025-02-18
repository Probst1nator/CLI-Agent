import tkinter as tk
from typing import List, Set, Tuple
import random

class GameOfLife:
    def __init__(self, width: int = 50, height: int = 50, cell_size: int = 15):
        self.width: int = width
        self.height: int = height
        self.cell_size: int = cell_size
        self.alive_cells: Set[Tuple[int, int]] = set()

        # Initialize GUI
        self.root: tk.Tk = tk.Tk()
        self.root.title("Conway's Game of Life")
        
        # Create canvas
        self.canvas: tk.Canvas = tk.Canvas(
            self.root, 
            width=width * cell_size, 
            height=height * cell_size,
            bg='white'
        )
        self.canvas.pack()

        # Create buttons
        self.control_frame: tk.Frame = tk.Frame(self.root)
        self.control_frame.pack(pady=5)
        
        tk.Button(self.control_frame, text="Start", command=self.start).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Random", command=self.randomize).pack(side=tk.LEFT, padx=5)

        # Bind mouse click
        self.canvas.bind("<Button-1>", self.toggle_cell)
        
        self.running: bool = False

    def toggle_cell(self, event: tk.Event) -> None:
        x: int = event.x // self.cell_size
        y: int = event.y // self.cell_size
        
        if (x, y) in self.alive_cells:
            self.alive_cells.remove((x, y))
        else:
            self.alive_cells.add((x, y))
        self.draw_grid()

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors: List[Tuple[int, int]] = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                neighbors.append((nx, ny))
        return neighbors

    def update(self) -> None:
        new_cells: Set[Tuple[int, int]] = set()
        candidates: Set[Tuple[int, int]] = set()

        # Add all alive cells and their neighbors to candidates
        for cell in self.alive_cells:
            candidates.add(cell)
            candidates.update(self.get_neighbors(*cell))

        # Check each candidate
        for cell in candidates:
            count: int = sum(1 for n in self.get_neighbors(*cell) if n in self.alive_cells)
            if cell in self.alive_cells:
                if count in [2, 3]:
                    new_cells.add(cell)
            else:
                if count == 3:
                    new_cells.add(cell)

        self.alive_cells = new_cells
        self.draw_grid()
        if self.running:
            self.root.after(100, self.update)

    def draw_grid(self) -> None:
        self.canvas.delete("all")
        
        # Draw cells
        for x, y in self.alive_cells:
            self.canvas.create_rectangle(
                x * self.cell_size, 
                y * self.cell_size,
                (x + 1) * self.cell_size, 
                (y + 1) * self.cell_size,
                fill="black"
            )

        # Draw grid lines
        for i in range(self.width + 1):
            self.canvas.create_line(
                i * self.cell_size, 0,
                i * self.cell_size, self.height * self.cell_size,
                fill="gray"
            )
        for i in range(self.height + 1):
            self.canvas.create_line(
                0, i * self.cell_size,
                self.width * self.cell_size, i * self.cell_size,
                fill="gray"
            )

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.update()

    def stop(self) -> None:
        self.running = False

    def clear(self) -> None:
        self.alive_cells.clear()
        self.draw_grid()

    def randomize(self) -> None:
        self.alive_cells = {
            (x, y) 
            for x in range(self.width) 
            for y in range(self.height) 
            if random.random() < 0.2
        }
        self.draw_grid()

    def run(self) -> None:
        self.root.mainloop()

if __name__ == "__main__":
    game: GameOfLife = GameOfLife()
    game.run()
