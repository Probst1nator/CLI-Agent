import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass

@dataclass
class Node:
    id: int
    position: np.ndarray
    connections: Set[int]
    time_step: int

class WolframMultiwayGraph:
    def __init__(self, num_initial_nodes: int = 10, branching_factor: int = 3):
        self.nodes: Dict[int, Node] = {}
        self.next_id: int = 0
        self.branching_factor: int = branching_factor
        self.time_step: int = 0
        
        # Initialize with random nodes
        for _ in range(num_initial_nodes):
            position = np.random.rand(3) * 10
            self.add_node(position)
    
    def add_node(self, position: np.ndarray) -> int:
        node_id = self.next_id
        self.nodes[node_id] = Node(
            id=node_id,
            position=position,
            connections=set(),
            time_step=self.time_step
        )
        self.next_id += 1
        return node_id
    
    def add_connection(self, node1_id: int, node2_id: int) -> None:
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id].connections.add(node2_id)
            self.nodes[node2_id].connections.add(node1_id)
    
    def evolve(self) -> None:
        self.time_step += 1
        new_nodes: List[Tuple[np.ndarray, List[int]]] = []
        
        # Generate new nodes based on current configuration
        for node in self.nodes.values():
            for _ in range(self.branching_factor):
                # Create new position with small random displacement
                displacement = np.random.randn(3) * 0.5
                new_position = node.position + displacement
                new_nodes.append((new_position, [node.id]))
        
        # Add new nodes and connections
        for position, connected_to in new_nodes:
            new_id = self.add_node(position)
            for parent_id in connected_to:
                self.add_connection(new_id, parent_id)
    
    def visualize(self) -> None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes
        positions = np.array([node.position for node in self.nodes.values()])
        colors = [node.time_step for node in self.nodes.values()]
        
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=colors,
            cmap='viridis',
            s=50
        )
        
        # Plot edges
        for node in self.nodes.values():
            for connected_id in node.connections:
                connected_node = self.nodes[connected_id]
                ax.plot(
                    [node.position[0], connected_node.position[0]],
                    [node.position[1], connected_node.position[1]],
                    [node.position[2], connected_node.position[2]],
                    'gray',
                    alpha=0.2
                )
        
        plt.colorbar(scatter, label='Time Step')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Wolfram Multiway Graph (t={self.time_step})')
        plt.show()

def main() -> None:
    # Create and evolve the graph
    graph = WolframMultiwayGraph(num_initial_nodes=5, branching_factor=2)
    
    # Evolve for several time steps
    for _ in range(3):
        graph.evolve()
        graph.visualize()
        print(f"Time step: {graph.time_step}")
        print(f"Number of nodes: {len(graph.nodes)}")
        print(f"Average connections per node: {sum(len(node.connections) for node in graph.nodes.values()) / len(graph.nodes):.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
