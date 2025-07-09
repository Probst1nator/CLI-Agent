# CLI-Agent: A multipurpose hackable agent accessible from the commandline

**CLI-Agent** is a powerful, multi-modal, and extensible AI assistant that lives in your terminal. It leverages multiple LLM backends, executes code, sees your screen, listens to your voice, and uses custom tools to automate complex tasks directly from the command line.

![cli-agent-demo](https://user-images.githubusercontent.com/1392817/233072559-9f7a7524-ac6e-443b-82d2-5a9e3f739669.gif)

## ✨ Key Features

*   **🤖 Multi-LLM Routing:** Intelligently routes requests to the best-suited LLM from a wide range of providers (Ollama, OpenAI, Google, Anthropic, Groq, NVIDIA). Supports local-first or cloud-first preferences with a multi-select TUI.
*   **💻 Code Execution Notebook:** A persistent computational environment that can execute `bash` and `python` code to interact with your system, install packages, and perform complex scripting tasks.
*   **👁️ Vision Enabled:** Can take and analyze screenshots. Use the `-img` flag or hotkey to have the agent "see" your screen and answer questions about it.
*   **🗣️ Voice Interface:** Full voice-in, voice-out capabilities. Use the `--voice` flag to talk to your agent, with local wake-word detection and high-quality TTS for responses.
*   **🛠️ Extensible Tool System:** Easily add new capabilities by dropping Python files into a `utils` directory. Comes with pre-built tools for web search, file authoring, image generation, Home Assistant control, and more.
*   **🧠 Advanced Agentic Features:**
    *   **MCT Mode:** Explores multiple response branches from different LLMs to select the best path forward when multiple models are selected.
    *   **Dynamic Hyperparameters:** Automatically adjusts LLM thinking parameters like creativity and reasoning budget based on the user's query.
    *   **Auto-Execution Guard:** A safety layer that uses an LLM to review code for safety and completeness before running it automatically.
*   **🚀 Remote Host Service:** Offload demanding services like high-quality Whisper transcription and wake-word detection to a separate server (including ARM/Jetson devices) using a provided Docker setup.
*   **💾 Persistent Context:** Remembers conversations (`-c`), LLM selections, and tool usage history, allowing you to continue where you left off.
*   **🌐 Optional Web UI:** Launch a simple web interface with `--gui` to monitor the agent's full conversation context in real-time.

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.10+
*   Git & build essentials (e.g., `build-essential` on Debian/Ubuntu)
*   Optionally, [Docker](https://www.docker.com/get-started) for the Remote Host Service.

### 2. Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/cli-agent.git
    cd cli-agent
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    The project uses a wide range of libraries. While a `requirements.txt` file is recommended, you can get started by installing the key packages:
    ```bash
    # Core
    pip install python-dotenv pyfiglet termcolor pyperclip prompt-toolkit pexpect
    
    # AI Providers
    pip install openai groq "google-generativeai" anthropic ollama
    
    # Voice & Audio
    pip install SpeechRecognition PyAudio pynput vosk faster-whisper sounddevice soundfile
    
    # Tools & Vision
    pip install requests beautifulsoup4 brave-search chromadb-client sentence-transformers Pillow homeassistant-api
    ```

4.  **Configure API Keys**
    Create a `.env` file by copying the example and add your keys.
    ```bash
    cp .env.example .env
    nano .env
    ```
    ```ini
    # .env Example Content
    
    # --- AI Provider API Keys ---
    ANTHROPIC_API_KEY="sk-ant-..."
    GEMINI_API_KEY="AIzaSy..."
    OPENAI_API_KEY="sk-..."
    GROQ_API_KEY="gsk_..."
    NVIDIA_API_KEY="nvai-..."
    BRAVE_API_KEY="..." # For SearchWeb tool
    
    # --- Ollama Configuration (Optional) ---
    OLLAMA_HOST="localhost"
    
    # --- Home Assistant Tool (Optional) ---
    HASS_URL="http://your-home-assistant-ip:8123"
    HASS_TOKEN="your-long-lived-access-token"
    ```

## 🖥️ Usage

### Running the Agent
Simply run `main.py` from your terminal:
```bash
python main.py
```
The agent will start and display a help menu. You can then type your requests.

### Command-Line Arguments
Launch the agent with flags to configure its behavior from the start.

| Flag | Shorthand | Description |
|---|---|---|
| `--help` | `-h` | Display help and exit. |
| `--auto` | `-a` | Automatically execute safe commands. |
| `--continue` | `-c` | Continue the last conversation. |
| `--local` | `-l` | Use only local Ollama models. |
| `--message` | `-m` | Provide a message to process. Can be used multiple times. |
| `--regenerate` | `-r` | Regenerate the last response from the previous session. |
| `--voice` | `-v` | Enable microphone input and TTS output (enables `-a`). |
| `--speak` | `-s` | Enable text-to-speech output only. |
| `--fast` | `-f` | Use only fast LLMs. |
| `--strong`| | Use strong, slower LLMs. |
| `--image` | `-img` | Take a screenshot for analysis before the first prompt. |
| `--mct`| | Enable Monte Carlo Tree Search for exploring responses. |
| `--sandbox` | `-sbx` | Use a sandboxed Python execution environment. |
| `--online` | `-o` | Force the use of cloud-based AI models. |
| `--llm [model_key]`| | Specify an LLM, or use without a value to open a selection menu. |
| `--gui` | | Open a web interface to monitor the chat. |
| `--dyn` | | Dynamically adjust LLM thinking parameters based on the query. |
| `--exit` | `-e` | Exit after all automatic messages have been processed. |

### Interactive Hotkeys

While the agent is running, you can use these hotkeys to change settings on the fly.

| Hotkey | Description |
|---|---|
| `-h` | Show the help message. |
| `-r` | Regenerate the last response. |
| `-l` | Toggle local mode (Ollama only). |
| `-llm`| Open the multi-select menu to choose different LLMs. |
| `-a` | Toggle automatic code execution. |
| `-f` | Toggle using only fast LLMs. |
| `-v` | Toggle voice mode (mic in, speech out). |
| `-speak`| Toggle text-to-speech output. |
| `-strong`| Toggle using only strong LLMs. |
| `-img` | Take a new screenshot for context. |
| `-mct` | Toggle Monte Carlo Tree Search (multi-response evaluation). |
| `-m` | Enter multiline input mode. |
| `-p` | Print the raw chat history. |
| `-dyn` | Toggle dynamic LLM parameter adjustment. |
| `-e` | Exit the application. |

## 🏛️ Core Concepts

### Extensible Tools
You can easily add new functionality to the agent by creating tools.

1.  Create a new Python file in the `utils/` directory (e.g., `my_tool.py`).
2.  Inside the file, create a class that inherits from `UtilBase`.
3.  Implement a static `run` method. The LLM will call this method with the necessary arguments.
4.  The agent will automatically discover and load your new tool.

**Example: `utils/writefile.py`**
```python
# utils/writefile.py
import os
import json
from py_classes.cls_util_base import UtilBase
from typing import Literal

class WriteFile(UtilBase):
    """A utility for creating, overwriting, or appending to files."""
    @staticmethod
    def run(path: str, content: str, mode: Literal['w', 'a'] = 'w') -> str:
        try:
            # ... implementation ...
            return json.dumps({"result": {"status": "Success", "path": os.path.abspath(path)}})
        except Exception as e:
            return json.dumps({"error": f"Could not write to file. Reason: {e}"})
```

### Remote Host Service
For better performance, you can offload demanding services like voice transcription to a dedicated server (e.g., a home server with a GPU or a Jetson device).

1.  On the **server machine**, clone the repository.
2.  Run the appropriate setup script from the `shell_scripts/` directory:
    *   `setup_remote_host_docker.sh`: For standard Linux/Windows with Docker and NVIDIA GPUs.
    *   `setup_remote_host_jetson.sh` or `setup_remote_host_arm.sh`: For ARM-based devices like NVIDIA Jetson.
3.  On your **client machine**, set `CLI_AGENT_REMOTE_HOST` in your `.env` file to the server's IP address (e.g., `http://192.168.1.101:5000`).
4.  When you use `--voice` on the client, the agent will automatically use the remote services for transcription.