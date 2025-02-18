import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import itertools
import random
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Node:
    id: int
    position: Tuple[float, float, float]
    connections: Set[int]
    state: int

class MultiwayGraphSimulation:
    def __init__(self, 
                 size: int = 1000, 
                 dimensions: int = 3, 
                 max_connections: int = 5,
                 mutation_rate: float = 0.01):
        self.size = size
        self.dimensions = dimensions
        self.max_connections = max_connections
        self.mutation_rate = mutation_rate
        self.nodes: Dict[int, Node] = {}
        self.time_step: int = 0
        self.graph = nx.Graph()
        self.initialize_nodes()

    def initialize_nodes(self) -> None:
        """Initialize nodes with random positions in 3D space"""
        for i in range(self.size):
            position = tuple(random.uniform(-1, 1) for _ in range(self.dimensions))
            self.nodes[i] = Node(
                id=i,
                position=position,
                connections=set(),
                state=random.randint(0, 1)
            )
            self.graph.add_node(i, pos=position)

        # Initialize connections based on proximity
        self._initialize_connections()

    def _initialize_connections(self) -> None:
        """Create initial connections between nearby nodes"""
        positions = np.array([node.position for node in self.nodes.values()])
        for i in range(self.size):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            nearest = np.argsort(distances)[1:self.max_connections+1]
            self.nodes[i].connections.update(nearest)
            for j in nearest:
                self.nodes[j].connections.add(i)
                self.graph.add_edge(i, j)

    def _update_node_state(self, node_id: int) -> None:
        """Update state of a single node based on its connections"""
        node = self.nodes[node_id]
        connected_states = [self.nodes[conn].state for conn in node.connections]
        if connected_states:
            # Rule: Node takes majority state of connections
            new_state = 1 if sum(connected_states) > len(connected_states)/2 else 0
            # Apply mutation with probability
            if random.random() < self.mutation_rate:
                new_state = 1 - new_state
            node.state = new_state

    def _update_node_position(self, node_id: int) -> None:
        """Update position of a node based on connected nodes"""
        node = self.nodes[node_id]
        if node.connections:
            # Calculate center of mass of connected nodes
            connected_positions = np.array([self.nodes[conn].position 
                                         for conn in node.connections])
            center = tuple(np.mean(connected_positions, axis=0))
            # Move slightly toward center
            current = np.array(node.position)
            target = np.array(center)
            new_pos = tuple(current + 0.1 * (target - current))
            node.position = new_pos
            self.graph.nodes[node_id]['pos'] = new_pos

    def step(self) -> None:
        """Perform one time step of the simulation"""
        self.time_step += 1
        
        # Parallel update of nodes
        with ThreadPoolExecutor() as executor:
            # Update states
            executor.map(self._update_node_state, self.nodes.keys())
            # Update positions
            executor.map(self._update_node_position, self.nodes.keys())

        # Periodically regenerate connections
        if self.time_step % 10 == 0:
            self._initialize_connections()

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return current state of the simulation"""
        positions = np.array([node.position for node in self.nodes.values()])
        states = np.array([node.state for node in self.nodes.values()])
        edges = np.array(list(self.graph.edges()))
        return {
            'positions': positions,
            'states': states,
            'edges': edges,
            'time_step': self.time_step
        }

    def run(self, steps: int = 1000) -> List[Dict[str, np.ndarray]]:
        """Run simulation for specified number of steps"""
        history = []
        for _ in range(steps):
            self.step()
            history.append(self.get_state())
        return history

if __name__ == "__main__":
    # Initialize and run simulation
    simulation = MultiwayGraphSimulation(size=1000)
    history = simulation.run(steps=100)
    
    # Example of accessing results
    print(f"Final time step: {history[-1]['time_step']}")
    print(f"Number of nodes: {len(history[-1]['positions'])}")
    print(f"Number of edges: {len(history[-1]['edges'])}")
