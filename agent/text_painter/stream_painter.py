from termcolor import colored
import re
from typing import Optional

class TextStreamPainter:
    """
    A class used to apply custom coloring to strings in a streaming context.
    Supports both code block coloring and XML tag coloring with nesting depth.
    
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
    xml_tag_stack : list
        Stack to track nested XML tags and their depths.
    xml_buffer : str
        Buffer to collect XML tag characters as they arrive.
    in_xml_tag : bool
        Flag to track if we're currently parsing an XML tag.
    """
    
    # Define color schemes for different XML tag types and nesting depths
    TAG_TYPE_COLORS = {
        'python': 'green',           # Python code - green
        'bash': 'blue',              # Bash commands - blue 
        'think': 'yellow',           # Thinking blocks - yellow
        'todos': 'cyan',             # Todo lists - cyan
        'context': 'magenta',        # Context blocks - magenta
        'tool': 'white',             # Generic tools - white
        'editfile': 'white',         # File editing - white
        'writefile': 'white',        # File writing - white
        'readfile': 'white',         # File reading - white
        'viewfiles': 'white',        # File viewing - white
        'default': 'white'           # Default tags - white
    }
    
    DEPTH_COLORS = [
        'red',           # depth 0
        'green',         # depth 1  
        'yellow',        # depth 2
        'blue',          # depth 3
        'magenta',       # depth 4
        'cyan',          # depth 5
        'white'          # depth 6+
    ]
    
    def __init__(self):
        """Initialize the TextStreamPainter with default values."""
        self.saved_block_delimiters = ""
        self.color_red = False
        self.backtick_buffer = ""
        self.in_code_block = False
        
        # XML tag handling
        self.xml_tag_stack = []  # Stack of (tag_name, depth) tuples
        self.xml_buffer = ""
        self.in_xml_tag = False
    
    def _get_color_for_tag_type(self, tag_name: str, depth_override: Optional[int] = None) -> str:
        """
        Get the color for a specific tag type. Used by both tag and content coloring.
        
        Parameters:
        ----------
        tag_name : str
            The name of the XML tag
        depth_override : Optional[int]
            Override the depth calculation (used for content vs tag differences)
            
        Returns:
        -------
        str
            The color name
        """
        # Priority tags always use their defined colors
        priority_tags = {'python', 'bash', 'think', 'todos', 'editfile', 'writefile', 'readfile', 'viewfiles', 'tool'}
        if tag_name.lower() in priority_tags:
            return self.TAG_TYPE_COLORS.get(tag_name.lower(), self.TAG_TYPE_COLORS['default'])
        
        # Non-priority tags use depth-based colors
        if depth_override is not None:
            current_depth = depth_override
        else:
            current_depth = len(self.xml_tag_stack)
            
        depth_color = self.DEPTH_COLORS[min(current_depth, len(self.DEPTH_COLORS) - 1)]
        return depth_color
    
    def _get_tag_color(self, tag_name: str, is_closing: bool = False) -> str:
        """
        Get the appropriate color for a tag based on its type and current nesting depth.
        
        Parameters:
        ----------
        tag_name : str
            The name of the XML tag
        is_closing : bool
            Whether this is a closing tag
            
        Returns:
        -------
        str
            The color name for the tag
        """
        # Calculate depth for tag (different for opening vs closing)
        current_depth = len(self.xml_tag_stack)
        if is_closing and current_depth > 0:
            current_depth -= 1
            
        return self._get_color_for_tag_type(tag_name, current_depth)
    
    def _process_xml_tag(self, tag_content: str) -> str:
        """
        Process a complete XML tag and update the tag stack.
        
        Parameters:
        ----------
        tag_content : str
            The complete tag content including < and >
            
        Returns:
        -------
        str
            The colored tag
        """
        # Parse tag to extract name and determine if it's closing
        tag_match = re.match(r'<(/?)([a-zA-Z][a-zA-Z0-9]*)', tag_content)
        if not tag_match:
            # Not a valid tag, color as default
            return colored(tag_content, self.TAG_TYPE_COLORS['default'])
        
        is_closing = bool(tag_match.group(1))  # True if starts with </
        tag_name = tag_match.group(2)
        
        # Get appropriate color
        tag_color = self._get_tag_color(tag_name, is_closing)
        
        # Update tag stack
        if is_closing:
            # Pop from stack if it matches
            if self.xml_tag_stack and self.xml_tag_stack[-1][0] == tag_name:
                self.xml_tag_stack.pop()
        else:
            # Push to stack (opening tag)
            depth = len(self.xml_tag_stack)
            self.xml_tag_stack.append((tag_name, depth))
        
        return colored(tag_content, tag_color)
    
    def _get_content_color(self) -> str:
        """
        Get the color for content based on current context.
        Uses the exact same helper as _get_tag_color() to ensure perfect consistency.
        
        Returns:
        -------
        str
            The color name for content
        """
        if self.in_code_block:
            return "light_red"
        elif self.xml_tag_stack:
            # Color content based on innermost tag using exact same logic as tags
            current_tag = self.xml_tag_stack[-1][0]
            # Content uses current depth (same as tag would use when inside this context)
            current_depth = len(self.xml_tag_stack) - 1  
            return self._get_color_for_tag_type(current_tag, current_depth)
        else:
            # Use default/normal color for regular text
            return "white"

    def apply_color(self, string: str, return_remaining: bool = False) -> str:
        """
        Applies custom coloring to streamed input characters.
        Supports both code blocks (```) and XML tags with nesting depth.
        Preserves ALL characters including backticks and XML.
        
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
            # Handle backtick logic for code blocks
            if char == "`":
                # If we're in an XML tag, treat backtick as part of tag
                if self.in_xml_tag:
                    self.xml_buffer += char
                    continue
                    
                # Add backtick to buffer for detection
                self.backtick_buffer += char
                
                # Check if we have three consecutive backticks
                if len(self.backtick_buffer) == 3:
                    # Toggle code block state
                    self.in_code_block = not self.in_code_block
                    
                    # Apply the appropriate color to the triple backticks
                    result += colored(self.backtick_buffer, "light_red", attrs=['bold'])
                    
                    # Reset buffer after processing
                    self.backtick_buffer = ""
                
            # Handle XML tag logic
            elif char == "<":
                # Flush any pending backticks
                if self.backtick_buffer:
                    content_color = self._get_content_color()
                    result += colored(self.backtick_buffer, content_color)
                    self.backtick_buffer = ""
                
                # Start XML tag parsing
                self.in_xml_tag = True
                self.xml_buffer = char
                
            elif char == ">" and self.in_xml_tag:
                # Complete XML tag
                self.xml_buffer += char
                result += self._process_xml_tag(self.xml_buffer)
                
                # Reset XML parsing state
                self.in_xml_tag = False
                self.xml_buffer = ""
                
            elif self.in_xml_tag:
                # Continue collecting XML tag content
                self.xml_buffer += char
                
            else:
                # Regular character processing
                # First, handle any pending backticks
                if self.backtick_buffer:
                    # Apply the color based on our current state
                    content_color = self._get_content_color()
                    result += colored(self.backtick_buffer, content_color)
                    # Reset buffer
                    self.backtick_buffer = ""
                
                # Add the current character with appropriate color
                content_color = self._get_content_color()
                result += colored(char, content_color)
        
        # Handle any remaining buffers when returning
        if return_remaining:
            if self.backtick_buffer:
                content_color = self._get_content_color()
                result += colored(self.backtick_buffer, content_color)
                self.backtick_buffer = ""
            if self.xml_buffer and self.in_xml_tag:
                # Incomplete XML tag, color as default
                result += colored(self.xml_buffer, self.TAG_TYPE_COLORS['default'])
                self.xml_buffer = ""
                self.in_xml_tag = False
        
        return result