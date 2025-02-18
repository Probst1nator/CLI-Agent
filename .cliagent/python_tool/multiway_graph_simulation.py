import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class MultiwaySystem:
    initial_state: str
    rules: Dict[str, List[str]]
    max_steps: int
    
    def evolve(self) -> nx.DiGraph:
        G = nx.DiGraph()
        states: Set[str] = {self.initial_state}
        state_by_level: Dict[int, Set[str]] = defaultdict(set)
        state_by_level[0].add(self.initial_state)
        
        # Generate states for each step
        for step in range(self.max_steps):
            current_states = state_by_level[step]
            for state in current_states:
                for i in range(len(state)):
                    # Check each possible rule application
                    for pattern, replacements in self.rules.items():
                        if i + len(pattern) <= len(state) and state[i:i+len(pattern)] == pattern:
                            for replacement in replacements:
                                new_state = state[:i] + replacement + state[i+len(pattern):]
                                state_by_level[step + 1].add(new_state)
                                G.add_edge(state, new_state)
        
        return G

    def visualize(self, G: nx.DiGraph) -> None:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=1000,
                             alpha=0.6)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, 
                              font_size=8,
                              font_family='sans-serif')
        
        plt.title("Multiway System Evolution")
        plt.axis('off')
        plt.show()

def main() -> None:
    # Define a simple string rewriting system
    rules = {
        'A': ['AA', 'B'],
        'B': ['A', 'BB']
    }
    
    # Create and evolve the system
    system = MultiwaySystem(
        initial_state='A',
        rules=rules,
        max_steps=3
    )
    
    # Generate the multiway graph
    graph = system.evolve()
    
    # Visualize the result
    system.visualize(graph)

if __name__ == "__main__":
    main()
