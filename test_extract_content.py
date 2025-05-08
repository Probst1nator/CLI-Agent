from utils.extractcontents import ExtractContents
import asyncio
import json

async def main():
    video_url = "https://www.youtube.com/watch?v=V71AJoYAtBQ"
    print(f"Attempting to extract content from: {video_url}")

    try:
        # Get the result from the ExtractContents.run method
        result = ExtractContents.run(url=video_url, headless=True, timeout=45)
        
        # If the result is a coroutine or future, await it
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            extracted_data = await result
        else:
            extracted_data = result
            
        print("Extraction successful.")
        # Print relevant parts of the extracted data
        print(f"Title: {extracted_data.get('title', 'N/A')}")
        print(f"URL: {extracted_data.get('url', 'N/A')}")
        
        # Print a snippet of the text content, as it might contain description or other info
        text_snippet = extracted_data.get('text', 'N/A')
        if text_snippet:
            print(f"Text snippet (first 500 chars): {text_snippet[:500]}...")
        else:
            print("No text content extracted.")
        
        # Print error if any
        if 'error' in extracted_data:
            print(f"Error reported: {extracted_data['error']}")
            
    except Exception as e:
        print(f"An error occurred during extraction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 