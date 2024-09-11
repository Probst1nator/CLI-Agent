from termcolor import colored


class CustomColoring:
    """
    A class used to apply custom coloring to strings.

    Attributes:
    ----------
    saved_block_delimiters : str
        Saved block delimiters.
    color_red : bool
        Flag to indicate whether to color the text red.
    """

    saved_block_delimiters: str = ""
    color_red: bool = False

    def apply_color(self, string: str, return_remaining: bool = False) -> str:
        """
        Applies custom coloring to the input string.

        Parameters:
        ----------
        string : str
            The input string to be colored.
        return_remaining : bool, optional
            Flag to indicate whether to return the remaining string (default is False).

        Returns:
        -------
        str
            The colored string.
        """
        last_red: bool = False
        if "`" in string:
            # If the string contains a backtick, save it as a block delimiter
            self.saved_block_delimiters += string
            string = ""
            if self.saved_block_delimiters.count("`") == 3:
                # If three block delimiters are found, toggle the color_red flag
                self.color_red = not self.color_red
                string = self.saved_block_delimiters
                self.saved_block_delimiters = ""
                last_red = True
        else:
            # If the string does not contain a backtick, append it to the saved block delimiters
            string = self.saved_block_delimiters + string
            self.saved_block_delimiters = ""
        # elif len(self.saved_block_delimiters) >= 3:
        #     string = self.saved_block_delimiters
        #     self.saved_block_delimiters = ""
        if return_remaining:
            # If return_remaining is True, return the remaining string
            string = self.saved_block_delimiters + string
            self.saved_block_delimiters = ""
        if self.color_red or last_red:
            # If color_red or last_red is True, color the string red
            string = colored(string, "light_red")
        else:
            # Otherwise, color the string magenta
            string = colored(string, "magenta")
        return string