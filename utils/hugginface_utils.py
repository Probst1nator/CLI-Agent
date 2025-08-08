# utils/hugginface_utils.py
import json
import requests
import os
import subprocess
from py_classes.cls_util_base import UtilBase
from typing import Literal, Optional, Dict, Any

# --- Hugging Face Interaction Utilities ---

class HuggingFaceSearch(UtilBase):
    """
    A utility to search for models on Hugging Face based on various criteria.
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["huggingface", "search models", "find llm", "model hub", "latest models", "agent models", "language models", "under 30gb", "model search", "discover models", "browse models"],
            "use_cases": [
                "Find the latest agent LLMs under 30GB on HuggingFace.",
                "Search for code generation models on the model hub.",
                "Look for recently updated language models.",
                "Find models by specific authors or organizations."
            ],
            "arguments": {
                "query": "Search term for finding models (e.g., 'agent', 'code', 'chat').",
                "limit": "Maximum number of results to return.",
                "sort_by": "Sort results by 'downloads', 'likes', or 'updated'.",
                "filter_size_gb": "Only show models under this size in GB."
            }
        }
    
    @staticmethod
    def run(query: str,
            limit: int = 10,
            sort_by: Literal['downloads', 'likes', 'updated'] = 'downloads',
            filter_size_gb: Optional[float] = None) -> str:
        """
        Searches the Hugging Face Hub for models.

        Args:
            query (str): The search term for models.
            limit (int): The maximum number of results to return. Defaults to 10.
            sort_by (Literal['downloads', 'likes', 'updated']): The field to sort results by. Defaults to 'downloads'.
            filter_size_gb (Optional[float]): Filters models to those with an inferred size less than or equal to this value in GB.
                                              Note: Model size is often not directly available and inferred from weights.
                                              Defaults to None (no size filtering).

        Returns:
            str: A JSON string containing search results or an error message.
        """
        api_url = "https://huggingface.co/api/models"
        params = {
            "search": query,
            "limit": limit,
            "sort": sort_by,
        }

        try:
            print(f"Searching Hugging Face with query: '{query}', sort: '{sort_by}', limit: {limit}")
            response = requests.get(api_url, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            models_data = response.json()

            filtered_models = []
            if filter_size_gb is not None:
                print(f"Applying size filter: <= {filter_size_gb} GB")
                for model in models_data:
                    inferred_size_bytes = 0
                    has_size_info = False
                    
                    # Check if model has sibling files listed and attempt to sum their sizes
                    if 'siblings' in model and model['siblings']:
                        for file_info in model['siblings']:
                            # Look for common model weight file extensions
                            if any(ext in file_info.get('rfilename', '').lower() for ext in ['.safetensors', '.bin', '.pt', '.gguf', '.pth', '.h5']):
                                if 'size' in file_info and file_info['size'] is not None:
                                    inferred_size_bytes += file_info['size']
                                    has_size_info = True
                                else:
                                    # If any relevant file has no size, we can't reliably filter this model.
                                    # Mark it as potentially too large to be safe.
                                    inferred_size_bytes = filter_size_gb + 1 # Ensure it exceeds the limit
                                    print(f"Warning: Size information missing for file {file_info.get('rfilename')} in model {model.get('id')}. Excluding from size filter.")
                                    break # No need to check other files if one is missing size

                    # Only consider models for which we could gather some size information
                    if has_size_info:
                        if inferred_size_bytes <= filter_size_gb * 1024 * 1024 * 1024: # Convert GB to Bytes
                            filtered_models.append(model)
                    else:
                        # If no files with size info were found, we skip this model for size filtering.
                        print(f"Info: No size information found for relevant files in model {model.get('id')}. Excluding from size-filtered results.")
            else:
                # No size filter applied, use all models from the search
                filtered_models = models_data

            # Format output for clarity
            formatted_results = []
            for model in filtered_models:
                # Attempt to get a direct download link for a common file, or the repo URL
                repo_url = f"https://huggingface.co/{model.get('id', '')}"
                download_target = repo_url # Default to repo URL
                
                if 'siblings' in model and model['siblings']:
                    # Try to find a common, significant file to suggest downloading
                    potential_files = sorted(model['siblings'], key=lambda x: x.get('size', 0), reverse=True)
                    for f in potential_files:
                        if any(ext in f.get('rfilename', '').lower() for ext in ['.safetensors', '.bin', '.gguf']):
                            download_target = f"{repo_url}/resolve/main/{f.get('rfilename')}"
                            break
                        elif '.json' in f.get('rfilename', '').lower(): # fallback to config.json if no weights found
                             download_target = f"{repo_url}/resolve/main/{f.get('rfilename')}"

                formatted_results.append({
                    "modelId": model.get("id", "N/A"),
                    "author": model.get("author", "N/A"),
                    "downloads": model.get("downloads", 0),
                    "likes": model.get("likes", 0),
                    "lastModified": model.get("lastModified", "N/A"),
                    "pipeline_tag": model.get("pipeline_tag", "N/A"),
                    "tags": model.get("tags", []),
                    "repo_url": repo_url,
                    "suggested_download_file_url": download_target
                })

            print(f"Found {len(formatted_results)} models matching criteria.")
            return json.dumps({"result": formatted_results}, indent=2)

        except requests.exceptions.HTTPError as e:
            return json.dumps({"error": f"HTTP error occurred: {e.response.status_code} - {e.response.text}"}, indent=2)
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"Request failed: {e}"}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred during Hugging Face search: {e}"}, indent=2)

class HuggingFaceDownloader(UtilBase):
    """
    A utility to download a specific file from a Hugging Face model repository.
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["download model", "huggingface download", "get model files", "model weights", "download llm", "fetch model", "pull model", "model download", "save model locally"],
            "use_cases": [
                "Download a specific model from HuggingFace to local storage.",
                "Get model weight files for local deployment.",
                "Download configuration files from a model repository.",
                "Fetch model files for use with Ollama or other local systems."
            ],
            "arguments": {
                "model_id": "The full model ID from HuggingFace (e.g., 'meta-llama/Llama-2-7b-chat-hf').",
                "filename": "Specific file to download (e.g., 'config.json', 'pytorch_model.bin').",
                "download_path": "Local directory to save the file."
            }
        }
    
    @staticmethod
    def run(model_id: str, filename: str, download_path: str = ".") -> str:
        """
        Downloads a specific file from a Hugging Face model repository.

        Args:
            model_id (str): The full model ID (e.g., 'meta-llama/Llama-2-7b-chat-hf').
            filename (str): The name of the file to download (e.g., 'config.json', 'pytorch_model.bin').
            download_path (str): The local directory to save the downloaded file. Defaults to current directory.

        Returns:
            str: A JSON string indicating success or failure and the downloaded file path.
        """
        if not model_id or not filename:
            return json.dumps({"error": "Both model_id and filename are required for download."}, indent=2)

        download_url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
        local_filepath = os.path.join(download_path, filename.split('/')[-1]) # Use only the filename part for local path

        print(f"Attempting to download '{filename}' from '{model_id}' to '{local_filepath}'...")

        try:
            # Ensure download directory exists
            os.makedirs(download_path, exist_ok=True)

            # Use requests to stream the download
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()  # Check for download errors
                
                # total_size = int(r.headers.get('content-length', 0)) # Optional: for progress bar
                
                with open(local_filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        # downloaded_size += len(chunk) # Optional: for progress bar
                        # print(f"Downloaded: {downloaded_size}/{total_size} bytes", end='') # Optional: for progress bar
                
            print(f"Successfully downloaded '{filename}' to '{os.path.abspath(local_filepath)}'")
            return json.dumps({"result": {"status": "Success", "downloaded_file": os.path.abspath(local_filepath)}}, indent=2)

        except requests.exceptions.HTTPError as e:
            return json.dumps({"error": f"HTTP error occurred during download: {e.response.status_code} - {e.response.text}"}, indent=2)
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"Request failed during download: {e}"}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred during download: {e}"}, indent=2)

# --- Ollama Interaction Utilities ---

class OllamaManager(UtilBase):
    """
    A utility to interact with the Ollama service (assuming it's running locally).
    """
    @staticmethod
    def _map_hf_to_ollama(hf_model_id: str) -> str:
        """
        Attempts to map a Hugging Face model ID to a plausible Ollama model tag.
        This is a heuristic and might need manual adjustment for specific models.
        """
        # Example mappings:
        if hf_model_id.lower().startswith("meta-llama/llama-2-"):
            parts = hf_model_id.split('-')
            if len(parts) >= 3:
                # e.g., meta-llama/llama-2-7b-chat-hf -> llama2:7b-chat
                ollama_tag = f"llama2:{parts[2]}"
                if "chat" in hf_model_id.lower():
                    ollama_tag += "-chat"
                return ollama_tag.lower()
        elif hf_model_id.lower().startswith("mistralai/mistral-7b-"):
            return "mistral" # Mistral models are often just tagged as 'mistral' in Ollama
        elif hf_model_id.lower().startswith("google/gemini-"):
            # Gemini models are not directly available on Ollama typically, but if they were...
            return hf_model_id.lower().replace("google/", "") # e.g., google/gemini-pro -> gemini-pro
        
        # Generic fallback: take the last part of the repo name and convert to lowercase
        # Remove common suffixes like '-hf', '-onnx', etc.
        fallback_tag = hf_model_id.split('/')[-1].lower()
        fallback_tag = fallback_tag.replace('-hf', '').replace('-onnx', '').replace('-cpp', '').replace('-gguf', '')
        return fallback_tag

    @staticmethod
    def run(action: Literal['pull', 'list', 'remove', 'serve'], model_name_or_tag: Optional[str] = None, hf_model_id: Optional[str] = None) -> str:
        """
        Manages models in Ollama or starts the Ollama server.

        Args:
            action (Literal['pull', 'list', 'remove', 'serve']): The action to perform.
            model_name_or_tag (Optional[str]): The Ollama model name or tag for 'pull' or 'remove' actions (e.g., 'llama2:7b').
            hf_model_id (Optional[str]): The Hugging Face model ID, used to automatically determine the Ollama tag for 'pull'.

        Returns:
            str: A JSON string indicating the result of the Ollama operation.
        """
        if action == 'pull':
            if not model_name_or_tag and not hf_model_id:
                return json.dumps({"error": "Either model_name_or_tag or hf_model_id is required for 'pull' action."}, indent=2)
            
            ollama_tag = model_name_or_tag if model_name_or_tag else OllamaManager._map_hf_to_ollama(hf_model_id)
            
            if not ollama_tag:
                 return json.dumps({"error": f"Could not determine Ollama tag from hf_model_id: {hf_model_id}. Please provide model_name_or_tag directly."}, indent=2)
            
            print(f"Attempting to pull Ollama model: '{ollama_tag}' (from HF ID: {hf_model_id})")
            command = f"ollama pull {ollama_tag}"
            
        elif action == 'list':
            print("Listing Ollama models...")
            command = "ollama list"
        elif action == 'remove':
            if not model_name_or_tag:
                return json.dumps({"error": "model_name_or_tag is required for 'remove' action."}, indent=2)
            print(f"Removing Ollama model: '{model_name_or_tag}'")
            command = f"ollama rm {model_name_or_tag}"
        elif action == 'serve':
            print("Starting Ollama server...")
            # Note: Running 'ollama serve' directly might block. In a real agent, you'd want to run this in a background process or separate thread.
            # For simplicity here, we'll just print the command. A more robust implementation would use subprocess.Popen.
            return json.dumps({"info": "To start Ollama server, run: ollama serve", "command_to_run": "ollama serve"}, indent=2)
        else:
            return json.dumps({"error": f"Unsupported action: {action}"}, indent=2)

        try:
            # Execute shell command
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            if action == 'pull':
                if "success" in output.lower() or "already exists" in output.lower():
                    return json.dumps({"result": {"status": "Success", "message": f"Ollama pull command executed for '{ollama_tag}'. Output: {output}"}}, indent=2)
                else:
                    return json.dumps({"result": {"status": "Partial Success / Warning", "message": output}}, indent=2)
            elif action == 'list':
                if output:
                    return json.dumps({"result": {"status": "Success", "models": output}}, indent=2)
                else:
                    return json.dumps({"result": {"status": "Success", "message": "No models found in Ollama."}}, indent=2)
            elif action == 'remove':
                return json.dumps({"result": {"status": "Success", "message": f"Ollama remove command executed for '{model_name_or_tag}'. Output: {output}"}}, indent=2)
                
        except subprocess.CalledProcessError as e:
            return json.dumps({"error": f"Ollama command failed: '{command}'
Exit Code: {e.returncode}
Stderr: {e.stderr.strip()}"}, indent=2)
        except FileNotFoundError:
            return json.dumps({"error": "Ollama command not found. Is Ollama installed and in your PATH?"}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred with Ollama command '{command}': {e}"}, indent=2)

