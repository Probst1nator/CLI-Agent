#!/bin/bash
# Find all Python files containing g.debug_log calls with chat parameter
for file in $(grep -l "g.debug_log.*chat=" /home/prob/repos/CLI-Agent/py_classes/ai_providers/*.py /home/prob/repos/CLI-Agent/py_classes/*.py); do
  echo "Processing $file"
  # Create a backup of the file
  cp "$file" "${file}.bak"
  # Process the file with sed
  # Find lines with g.debug_log and chat= parameters and create a temp file with modified contents
  sed -i -E "s/(g\\.debug_log\\(.*(chat=)([^,)]+)(.*))/prefix = \\3.get_debug_title_prefix() if hasattr(\\3, \"get_debug_title_prefix\") else \"\"\n\\1/" "$file"
  # Now replace chat= with prefix=
  sed -i -E "s/chat=([^,)]+)/prefix=prefix/g" "$file"
done
echo "Replacement complete!"
