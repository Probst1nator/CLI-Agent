#!/bin/bash
# shellcheck disable=SC1090
source "$(grep PYTHON_ENV_PATH ~/repos/CLI-Agent/.env | cut -d '=' -f2)"

# Run the CLI-Agent with the screen grabber
python3 ~/repos/CLI-Agent/main.py --online -img --yolo -m "Observe the contents of the attached image and explain what you see in detail.", "Create a dense blogpost about the contents of the image, you may present the image with a short text or research key aspects for scientific grounding or additional information using the websearch tool. Present the blogpost via a html using the visualresponse tool, ensure to style it and include the image in the final blogpost."
# python3 ~/repos/CLI-Agent/main.py --online --yolo --sandbox -m "You have access to homeassistant, interview the system and gather interesting information, to prepare a markdown home overview dashboard", "Create a blogpost about the topic and your exploration of it in a html file inside a 'blogposts' folder, style it using css. You can create the file by coding it inside a multiline python string and saving it to a html file. Please ensure the absolute path of the html file is printed at the end of the script. Because of repeated issues please do not use latex at all.", "Please open the file in firefox, run it such that it isnt blocking your environment."

# Print a message to the console
echo "Script execution completed."
echo "Press Enter to close this window..."
read
