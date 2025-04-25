from termcolor import colored

class TextStreamPainter:
    """
    A class used to apply custom coloring to strings in a streaming context.
    
    Attributes:
    ----------
    saved_block_delimiters : str
        Saved block delimiters.
    color_red : bool
        Flag to indicate whether to color the text red.
    backtick_buffer : str
        Buffer to collect backticks as they arrive in the stream.
    in_code_block : bool
        Flag to track if we're currently inside a code block.
    """
    
    def __init__(self):
        """Initialize the TextStreamPainter with default values."""
        self.saved_block_delimiters = ""
        self.color_red = False
        self.backtick_buffer = ""
        self.in_code_block = False
    
    def apply_color(self, string: str, return_remaining: bool = False) -> str:
        """
        Applies custom coloring to streamed input characters.
        Preserves ALL characters including backticks.
        
        Parameters:
        ----------
        string : str
            The input string to be colored.
        return_remaining : bool, optional
            Flag to indicate whether to return the remaining string (default is False).
            
        Returns:
        -------
        str
            The colored string with all characters preserved.
        """
        result = ""
        
        # Process current chunk of characters
        for char in string:
            if char == "`":
                # Add backtick to buffer for detection
                self.backtick_buffer += char
                
                # Check if we have three consecutive backticks
                if len(self.backtick_buffer) == 3:
                    # Toggle code block state
                    self.in_code_block = not self.in_code_block
                    
                    # Apply the appropriate color to the triple backticks
                    if self.in_code_block:
                        result += colored(self.backtick_buffer, "light_red")
                    else:
                        result += colored(self.backtick_buffer, "light_red")
                    
                    # Reset buffer after processing
                    self.backtick_buffer = ""
                
            else:
                # If we have backticks in buffer but not yet 3
                if self.backtick_buffer:
                    # Apply the color based on our current state
                    if self.in_code_block:
                        result += colored(self.backtick_buffer, "light_red")
                    else:
                        result += colored(self.backtick_buffer, "magenta")
                    
                    # Reset buffer
                    self.backtick_buffer = ""
                
                # Add the current character with appropriate color
                if self.in_code_block:
                    result += colored(char, "light_red")
                else:
                    result += colored(char, "magenta")
        
        # Handle any remaining backticks in buffer when returning
        if return_remaining and self.backtick_buffer:
            if self.in_code_block:
                result += colored(self.backtick_buffer, "light_red")
            else:
                result += colored(self.backtick_buffer, "magenta")
            self.backtick_buffer = ""
        
        return result