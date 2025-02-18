import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import random
from matplotlib.animation import FuncAnimation
import numpy as np

@dataclass
class MultiwayState:
    """Represents a state in the multiway system"""
    id: int
    value: str
    level: int

class MultiwaySystem:
    def __init__(self, initial_state: str, rules: Dict[str, List[str]]):
        self.states: Dict[int, MultiwayState] = {}
        self.edges: Set[Tuple[int, int]] = set()
        self.rules = rules
        self.next_id = 0
        
        # Initialize with first state
        self.add_state(initial_state, 0)
        
    def add_state(self, value: str, level: int) -> int:
        state_id = self.next_id
        self.states[state_id] = MultiwayState(state_id, value, level)
        self.next_id += 1
        return state_id
    
    def evolve(self, steps: int = 1) -> None:
        for _ in range(steps):
            current_states = [s for s in self.states.values() if s.level == max(s.level for s in self.states.values())]
            
            for state in current_states:
                for pattern, replacements in self.rules.items():
                    if pattern in state.value:
                        for replacement in replacements:
                            new_value = state.value.replace(pattern, replacement, 1)
                            new_id = self.add_state(new_value, state.level + 1)
                            self.edges.add((state.id, new_id))

    def visualize(self, frame: int) -> None:
        plt.clf()
        G = nx.DiGraph()
        
        # Add nodes and edges up to current frame
        current_states = [s for s in self.states.values() if s.level <= frame]
        current_edges = [(u, v) for (u, v) in self.edges 
                        if self.states[u].level < frame and self.states[v].level <= frame]
        
        for state in current_states:
            G.add_node(state.id, label=state.value)
        G.add_edges_from(current_edges)
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1000, arrowsize=20, 
                labels={node: G.nodes[node]['label'] for node in G.nodes()})
        plt.title(f'Step {frame}')

def main() -> None:
    # Define simple string rewriting rules
    rules: Dict[str, List[str]] = {
        'A': ['AA', 'B'],
        'B': ['AB', 'BA']
    }
    
    # Create multiway system with more steps
    system = MultiwaySystem('A', rules)
    
    # Create animation with longer duration
    fig = plt.figure(figsize=(12, 10))
    steps = 6  # Increased from 3 to 6
    system.evolve(steps)
    
    anim = FuncAnimation(fig, lambda frame: system.visualize(frame), 
                        frames=steps+1, interval=2000,  # Increased interval from 1000 to 2000
                        repeat=True)  # Changed to True for continuous loop
    
    plt.show()

if __name__ == "__main__":
    main()
