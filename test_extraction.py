#!/usr/bin/env python3

from utils.extractcontents import ExtractContents
import logging
import traceback
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://example.com"  # Default test URL
    
    print(f"Testing content extraction for: {url}")
    print("-" * 50)
    
    try:
        # Run the extraction and get formatted output
        result = ExtractContents.run(url, headless=True)
        
        # Print the result
        print(result)
        
        return 0
    except Exception as e:
        print(f"Error extracting content: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 