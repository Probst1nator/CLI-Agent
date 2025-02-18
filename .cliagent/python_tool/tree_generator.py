import typing as t

def draw_tree(height: int) -> None:
    """Draws a tree of specified height."""
    
    def recursive_draw(current_height: int, current_indent: str) -> None:
        if current_height > 0:
            print(f"{' ' * (len(current_indent) + 4)}|")
            for _ in range(current_height):
                print(f"{' ' * (len(current_indent) + 2)}O")
            recursive_draw(current_height - 1, current_indent + "    ")
    
    # Start drawing the tree
    recursive_draw(height, "")

def main() -> None:
    height = int(input("Enter the tree's height: "))
    draw_tree(height)

if __name__ == "__main__":
    main()
