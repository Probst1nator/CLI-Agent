import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict, Optional
import random
from dataclasses import dataclass

@dataclass
class GridWorldEnv:
    size: int = 5
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    obstacles: List[Tuple[int, int]] = None

    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = [(1, 1), (2, 2), (3, 1)]
        self.q_table: Dict[Tuple[int, int], List[float]] = {}
        self.initialize_q_table()

    def initialize_q_table(self) -> None:
        for i in range(self.size):
            for j in range(self.size):
                self.q_table[(i, j)] = [0.0] * 4  # Up, Right, Down, Left

class QLearningVisualizer:
    def __init__(
        self,
        env: GridWorldEnv,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1
    ):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.current_pos = env.start
        
        # Setup visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.setup_plot()
        
    def setup_plot(self) -> None:
        self.ax.grid(True)
        self.ax.set_xticks(range(self.env.size))
        self.ax.set_yticks(range(self.env.size))
        
    def get_action(self, state: Tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.env.q_table[state])
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        x, y = state
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Right
            y = min(self.env.size - 1, y + 1)
        elif action == 2:  # Down
            x = min(self.env.size - 1, x + 1)
        else:  # Left
            y = max(0, y - 1)
        
        return (x, y) if (x, y) not in self.env.obstacles else state
    
    def get_reward(self, state: Tuple[int, int]) -> float:
        if state == self.env.goal:
            return 100.0
        elif state in self.env.obstacles:
            return -10.0
        return -1.0
    
    def update(self, frame: int) -> None:
        self.ax.clear()
        self.setup_plot()
        
        # Draw grid
        for i in range(self.env.size):
            for j in range(self.env.size):
                q_values = self.env.q_table[(i, j)]
                max_q = max(q_values)
                color = plt.cm.viridis(max_q / 100.0 if max_q > 0 else 0)
                self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
        
        # Draw obstacles
        for obs in self.env.obstacles:
            self.ax.add_patch(plt.Rectangle((obs[1], obs[0]), 1, 1, color='red'))
        
        # Draw goal
        self.ax.add_patch(plt.Rectangle((self.env.goal[1], self.env.goal[0]), 1, 1, color='green'))
        
        # Draw current position
        self.ax.plot(self.current_pos[1] + 0.5, self.current_pos[0] + 0.5, 'bo')
        
        # Q-learning update
        action = self.get_action(self.current_pos)
        next_state = self.get_next_state(self.current_pos, action)
        reward = self.get_reward(next_state)
        
        # Update Q-table
        current_q = self.env.q_table[self.current_pos][action]
        next_max_q = max(self.env.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.env.q_table[self.current_pos][action] = new_q
        
        self.current_pos = next_state
        if self.current_pos == self.env.goal:
            self.current_pos = self.env.start
            
        self.ax.set_title(f'Episode Frame: {frame}')

def main():
    env = GridWorldEnv()
    visualizer = QLearningVisualizer(env)
    anim = FuncAnimation(visualizer.fig, visualizer.update, frames=200, interval=100)
    plt.show()

if __name__ == "__main__":
    main()
