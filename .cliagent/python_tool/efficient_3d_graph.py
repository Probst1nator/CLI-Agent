import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import random
from collections import defaultdict

@dataclass
class Node:
    position: Tuple[float, float, float]
    connections: Set[int]
    state: str

class MultiwayGraph:
    def __init__(self, window_size: Tuple[int, int] = (800, 600)):
        self.nodes: List[Node] = []
        self.window_size = window_size
        self.init_pygame()
        self.init_opengl()
        self.rotation: float = 0.0
        
    def init_pygame(self) -> None:
        pygame.init()
        pygame.display.set_mode(self.window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        
    def init_opengl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 1, 0))
        
        gluPerspective(45, (self.window_size[0]/self.window_size[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)
        
    def add_node(self, state: str) -> None:
        position = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2))
        self.nodes.append(Node(position=position, connections=set(), state=state))
        
    def add_connection(self, node1: int, node2: int) -> None:
        self.nodes[node1].connections.add(node2)
        self.nodes[node2].connections.add(node1)
        
    def draw_sphere(self, position: Tuple[float, float, float], radius: float = 0.05) -> None:
        glPushMatrix()
        glTranslatef(*position)
        quad = gluNewQuadric()
        gluSphere(quad, radius, 32, 32)
        glPopMatrix()
        
    def draw_connection(self, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> None:
        glBegin(GL_LINES)
        glVertex3f(*start)
        glVertex3f(*end)
        glEnd()
        
    def update(self) -> None:
        self.rotation += 1.0
        
    def render(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5)
        glRotatef(self.rotation, 0, 1, 0)
        
        # Draw nodes
        glColor3f(1, 1, 1)
        for node in self.nodes:
            self.draw_sphere(node.position)
            
        # Draw connections
        glColor3f(0.5, 0.5, 0.5)
        for i, node in enumerate(self.nodes):
            for connection in node.connections:
                self.draw_connection(node.position, self.nodes[connection].position)
                
    def run(self) -> None:
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            self.update()
            self.render()
            pygame.display.flip()
            clock.tick(60)
            
        pygame.quit()

def main() -> None:
    graph = MultiwayGraph()
    
    # Create sample nodes and connections
    for _ in range(20):
        graph.add_node("state")
    
    # Add random connections
    for i in range(len(graph.nodes)):
        for _ in range(random.randint(1, 3)):
            j = random.randint(0, len(graph.nodes)-1)
            if i != j:
                graph.add_connection(i, j)
    
    graph.run()

if __name__ == "__main__":
    main()
