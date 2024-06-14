
from termcolor import colored


class CustomColoring:
    saved_block_delimiters: str = ""
    color_red: bool = False

    def apply_color(self, string: str, return_remaining: bool = False):
        last_red: bool = False
        if "`" in string:
            self.saved_block_delimiters += string
            string = ""
            if self.saved_block_delimiters.count("`") == 3:
                self.color_red = not self.color_red
                string = self.saved_block_delimiters
                self.saved_block_delimiters = ""
                last_red = True
        else:
            string = self.saved_block_delimiters + string
            self.saved_block_delimiters = ""
        # elif len(self.saved_block_delimiters) >= 3:
        #     string = self.saved_block_delimiters
        #     self.saved_block_delimiters = ""
        if (return_remaining):
            string = self.saved_block_delimiters + string
            self.saved_block_delimiters = ""
        if self.color_red or last_red:
            string = colored(string, "light_red")
        else:
            string = colored(string, "magenta")
        return string