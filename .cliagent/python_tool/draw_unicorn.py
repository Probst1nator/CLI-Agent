#!/usr/bin/env python3

import turtle
from typing import Tuple, Optional
import sys
from PIL import Image

class UnicornDrawer:
    """A class to draw a unicorn using Python's turtle graphics."""

    def __init__(self) -> None:
        """Initialize the UnicornDrawer with turtle setup."""
        try:
            self.screen = turtle.Screen()
            self.screen.title("Magical Unicorn")
            self.screen.setup(800, 600)
            self.screen.bgcolor("lightblue")
            self.t = turtle.Turtle()
            self.t.speed(0)
        except turtle.TurtleGraphicsError as e:
            sys.exit(f"Failed to initialize turtle graphics: {e}")

    def set_position(self, x: float, y: float) -> None:
        """Move turtle to position without drawing."""
        self.t.penup()
        self.t.goto(x, y)
        self.t.pendown()

    def draw_circle(self, radius: float, color: str) -> None:
        """Draw a filled circle with given radius and color."""
        self.t.fillcolor(color)
        self.t.begin_fill()
        self.t.circle(radius)
        self.t.end_fill()

    def draw_horn(self) -> None:
        """Draw the unicorn's horn."""
        self.set_position(-20, 150)
        self.t.setheading(60)
        self.t.fillcolor("gold")
        self.t.begin_fill()
        for _ in range(3):
            self.t.forward(60)
            self.t.right(120)
        self.t.end_fill()

    def draw_head(self) -> None:
        """Draw the unicorn's head."""
        self.set_position(-50, 100)
        self.draw_circle(40, "white")

    def draw_body(self) -> None:
        """Draw the unicorn's body."""
        self.set_position(-20, 0)
        self.t.setheading(0)
        self.t.fillcolor("white")
        self.t.begin_fill()
        self.t.forward(150)
        self.t.right(90)
        self.t.forward(60)
        self.t.right(90)
        self.t.forward(150)
        self.t.right(90)
        self.t.forward(60)
        self.t.end_fill()

    def draw_legs(self) -> None:
        """Draw the unicorn's legs."""
        leg_positions = [(-10, 0), (30, 0), (90, 0), (130, 0)]
        for x, y in leg_positions:
            self.set_position(x, y)
            self.t.setheading(270)
            self.t.fillcolor("white")
            self.t.begin_fill()
            self.t.forward(80)
            self.t.right(90)
            self.t.forward(20)
            self.t.right(90)
            self.t.forward(80)
            self.t.right(90)
            self.t.forward(20)
            self.t.end_fill()

    def draw_face(self) -> None:
        """Draw the unicorn's face features."""
        # Eyes
        self.set_position(-35, 150)
        self.draw_circle(5, "black")
        self.set_position(-15, 150)
        self.draw_circle(5, "black")
        
        # Mouth
        self.set_position(-35, 120)
        self.t.setheading(0)
        self.t.color("pink")
        self.t.forward(30)

    def draw_mane(self) -> None:
        """Draw the unicorn's flowing mane."""
        colors = ["purple", "pink", "blue", "violet"]
        start_x, start_y = -60, 140
        
        for i, color in enumerate(colors):
            self.set_position(start_x - i*5, start_y - i*10)
            self.t.setheading(270)
            self.t.color(color)
            self.t.fillcolor(color)
            self.t.begin_fill()
            for _ in range(20):
                self.t.forward(5)
                self.t.right(10)
            self.t.end_fill()

    def save_drawing(self, filename: str = "unicorn.png") -> None:
        """Save the drawing as a PNG file."""
        try:
            canvas = self.screen.getcanvas()
            canvas.postscript(file="temp.eps")
            Image.open("temp.eps").save(filename)
        except Exception as e:
            print(f"Failed to save drawing: {e}")

    def draw_unicorn(self) -> None:
        """Draw the complete unicorn."""
        try:
            self.t.hideturtle()
            self.draw_body()
            self.draw_legs()
            self.draw_head()
            self.draw_horn()
            self.draw_face()
            self.draw_mane()
            self.save_drawing()
            self.screen.exitonclick()
        except turtle.TurtleGraphicsError as e:
            print(f"Error drawing unicorn: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

def main() -> None:
    """Main function to create and draw the unicorn."""
    try:
        unicorn = UnicornDrawer()
        unicorn.draw_unicorn()
    except Exception as e:
        print(f"Failed to create unicorn: {e}")

if __name__ == "__main__":
    main()