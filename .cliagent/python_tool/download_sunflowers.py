import os
from typing import List, Optional
import requests
from pathlib import Path

def download_sunflower_images(download_path: str = "~/Downloads", num_images: int = 3) -> List[str]:
    """
    Downloads sunflower images from Unsplash and saves them to the specified path.
    
    Args:
        download_path (str): Path to save the images
        num_images (int): Number of images to download
    
    Returns:
        List[str]: List of saved image paths
    """
    # Expand user path (~ to actual home directory)
    download_dir = os.path.expanduser(download_path)
    saved_paths: List[str] = []
    
    # Create directory if it doesn't exist
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        try:
            # Get random sunflower image from Unsplash
            response = requests.get(
                "https://source.unsplash.com/1600x900/?random,sunflowers",
                allow_redirects=True
            )
            
            if response.status_code == 200:
                # Create file path
                file_path = os.path.join(download_dir, f"sunflower_{i+1}.jpg")
                
                # Save image
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                saved_paths.append(file_path)
                print(f"Downloaded: {file_path}")
            else:
                print(f"Failed to download image {i+1}: Status code {response.status_code}")
                
        except Exception as e:
            print(f"Error downloading image {i+1}: {str(e)}")
    
    return saved_paths

if __name__ == "__main__":
    downloaded_files = download_sunflower_images()
    print(f"\nSuccessfully downloaded {len(downloaded_files)} images")
