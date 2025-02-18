import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
import random
import time
from dataclasses import dataclass
import numpy as np

@dataclass
class Node:
    id: int
    position: Tuple[float, float, float]
    connections: Set[int]

class WolframMultiwayGraph:
    def __init__(self, initial_nodes: int = 10):
        self.nodes: Dict[int, Node] = {}
        self.next_id: int = 0
        self.time_step: int = 0
        
        # Initialize with random nodes
        for _ in range(initial_nodes):
            self.add_node()
    
    def add_node(self) -> int:
        position = (
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10)
        )
        node = Node(self.next_id, position, set())
        self.nodes[self.next_id] = node
        self.next_id += 1
        return node.id
    
    def evolve(self) -> None:
        # Add new nodes
        new_nodes = max(2, len(self.nodes) // 3)
        for _ in range(new_nodes):
            new_id = self.add_node()
            # Connect to random existing nodes
            num_connections = random.randint(1, 3)
            possible_connections = list(self.nodes.keys())
            if possible_connections:
                connections = random.sample(
                    possible_connections,
                    min(num_connections, len(possible_connections))
                )
                for conn in connections:
                    self.nodes[new_id].connections.add(conn)
                    self.nodes[conn].connections.add(new_id)
        
        # Evolution rules
        for node in list(self.nodes.values()):
            # Randomly modify some connections
            if random.random() < 0.1:
                possible_new_connections = set(self.nodes.keys()) - {node.id} - node.connections
                if possible_new_connections:
                    new_conn = random.choice(list(possible_new_connections))
                    node.connections.add(new_conn)
                    self.nodes[new_conn].connections.add(node.id)
        
        self.time_step += 1
        
    def print_stats(self) -> None:
        avg_connections = sum(len(node.connections) for node in self.nodes.values()) / len(self.nodes)
        print(f"Time step: {self.time_step}")
        print(f"Number of nodes: {len(self.nodes)}")
        print(f"Average connections per node: {avg_connections:.2f}")
        print("-" * 50)

def main() -> None:
    graph = WolframMultiwayGraph()
    try:
        while True:
            graph.evolve()
            graph.print_stats()
            time.sleep(0.1)  # Small delay to prevent overwhelming the system
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

if __name__ == "__main__":
    main()
