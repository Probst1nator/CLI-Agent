#!/bin/bash
# shellcheck disable=SC1090
source "$(grep PYTHON_ENV_PATH /home/prob/repos/CLI-Agent/.env | cut -d '=' -f2)"

# Run the CLI-Agent with the screen grabber
python3 /home/prob/repos/CLI-Agent/main.py --online -img -a --sandbox --minimized -m "Deepdive into the topic(s) of the screenshot, consider different perspectives and approaches to enhance insights. Afterwards summarize your exploration into a introduction to the matter, if any mathematical or other problems are presented in the screenshot, try to solve them from first principles. Do not make up topics, only explore what is presented and closely related to the screenshot.", "Create a dense educational blogpost about the topic and your exploration of it in a html file, style it using css and include the screenshot as an image. You can create the file by coding it inside a multiline python string and saving it to a html file. Please ensure the absolute path of the html file is printed at the end of the script. Because of repeated issues please do not use latex at all.", "Lastly, please author some tooltips to add to the existing html file, use them to add educational information, definitions or descriptions. You can implement search and replace operations to interact with the html files contents more seamlessly than re-writing the whole file."

# Print a message to the console
echo "Script execution completed."
echo "Press Enter to close this window..."
read
