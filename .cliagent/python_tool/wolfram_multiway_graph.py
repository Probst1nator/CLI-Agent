import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict

@dataclass
class MultiwaySystem:
    """Represents a multiway system with evolving timesteps."""
    
    nodes: Set[int]
    rules: Dict[int, List[int]]
    timesteps: int
    graph: nx.DiGraph
    
    def __init__(self, initial_state: int, rules: Dict[int, List[int]], timesteps: int):
        self.nodes = {initial_state}
        self.rules = rules
        self.timesteps = timesteps
        self.graph = nx.DiGraph()
        self.graph.add_node(initial_state, time=0)
    
    def evolve(self) -> None:
        """Evolves the system through specified timesteps."""
        for t in range(self.timesteps):
            current_nodes = {n for n in self.graph.nodes() 
                           if self.graph.nodes[n]['time'] == t}
            
            for node in current_nodes:
                if node in self.rules:
                    for next_state in self.rules[node]:
                        self.graph.add_node(next_state, time=t+1)
                        self.graph.add_edge(node, next_state)
                        self.nodes.add(next_state)
    
    def visualize(self) -> None:
        """Visualizes the multiway graph."""
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(10, 8))
        
        # Draw nodes with different colors based on timestep
        colors = [self.graph.nodes[node]['time'] for node in self.graph.nodes()]
        nx.draw(self.graph, pos, node_color=colors, 
                with_labels=True, node_size=500,
                cmap=plt.cm.viridis, arrows=True)
        
        plt.title("Multiway Graph Evolution")
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), 
                    label="Timestep")
        plt.show()
    
    def get_causal_graph(self) -> nx.DiGraph:
        """Returns the causal relationships between nodes."""
        return self.graph
    
    def get_branchial_graph(self) -> nx.Graph:
        """Creates a branchial graph showing parallel evolution paths."""
        branchial = nx.Graph()
        
        for t in range(self.timesteps):
            nodes_at_time = [n for n in self.graph.nodes() 
                           if self.graph.nodes[n]['time'] == t]
            
            # Connect all nodes at the same timestep
            for i in range(len(nodes_at_time)):
                for j in range(i + 1, len(nodes_at_time)):
                    branchial.add_edge(nodes_at_time[i], nodes_at_time[j])
        
        return branchial
    
    def get_states_graph(self) -> Dict[int, Set[int]]:
        """Returns a mapping of timesteps to possible states."""
        states: Dict[int, Set[int]] = defaultdict(set)
        
        for node in self.graph.nodes():
            time = self.graph.nodes[node]['time']
            states[time].add(node)
            
        return states

def main() -> None:
    # Example usage
    # Rules: each number can evolve to number+1 and number*2
    initial_state = 1
    rules = {
        1: [2, 3],
        2: [3, 4],
        3: [4, 6],
        4: [5, 8],
        5: [6, 10],
        6: [7, 12]
    }
    
    # Create and evolve system
    system = MultiwaySystem(initial_state, rules, timesteps=3)
    system.evolve()
    
    # Visualize the evolution
    system.visualize()
    
    # Print states at each timestep
    states = system.get_states_graph()
    for time, state_set in states.items():
        print(f"Time {time}: States {state_set}")

if __name__ == "__main__":
    main()
