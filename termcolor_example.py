from termcolor import colored

colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white',
          'light_grey', 'dark_grey', 'light_red', 'light_green', 'light_yellow',
          'light_blue', 'light_magenta', 'light_cyan']

highlights = ['on_black', 'on_red', 'on_green', 'on_yellow', 'on_blue', 'on_magenta',
              'on_cyan', 'on_white', 'on_light_grey', 'on_dark_grey', 'on_light_red',
              'on_light_green', 'on_light_yellow', 'on_light_blue', 'on_light_magenta',
              'on_light_cyan']

attributes = ['bold', 'dark', 'underline', 'blink', 'reverse', 'concealed']

print("Available text colors:")
for color in colors:
    print(colored(f"This is {color} text", color))

print("\nAvailable highlights:")
for highlight in highlights:
    print(colored(f"This is text with {highlight} highlight", on_color=highlight))

print("\nAvailable attributes:")
for attribute in attributes:
    print(colored(f"This is text with {attribute} attribute", attrs=[attribute]))

print("\nCombination examples:")
print(colored('Hello, World!', 'red', 'on_black', ['bold', 'blink']))
print(colored('Hello, World!', 'green'))