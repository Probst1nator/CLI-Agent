# CLI-Agent: Intelligent Command-Line Assistant

CLI-Agent is a powerful AI-powered command-line assistant that enables natural language interaction with your computer. It provides a hackable playground inside a langchain-like framework, leveraging multiple AI model providers including Groq, Ollama, Anthropic, Google, and OpenAI.

## Features

- **Natural Language Understanding**: Communicate with your computer in plain English
- **Intelligent Code Execution**: Write and run Python code to solve your problems
- **Multi-Model Support**: Use local models via Ollama or cloud models (OpenAI, Groq, Anthropic, Google)
- **Voice Interaction**: Speak to your assistant and hear responses with text-to-speech
- **Screenshot Analysis**: Capture and analyze screen content
- **Web Search**: Find information online without leaving your terminal
- **Context Awareness**: Maintains conversation history for coherent interactions
- **Extensible Utilities**: Easily add new capabilities with the utility system
- **Command Suggestions**: The Agent suggests commands and, upon your confirmation, executes them with analysis of the output
- **Podcast Generator**: Create AI-powered podcasts on any topic with customizable speakers and styles

## Getting Started

### Prerequisites

- **Operating System:** Currently, only Ubuntu is supported
- **Python**: Version 3.9+ required
- **Optional**: Ollama for local model support

### Installation

1. **Using the Installer Script**

   To add the CLI-Agent to your system, execute the `installer.sh` script:

   ```bash
   ./installer.sh
   ```

2. **Running Manually**

   Alternatively, you can run the Agent manually by executing `main.py`:

   ```bash
   python3 ./main.py
   ```

3. **Environment Setup**

   The system uses environment variables configured in the `.env` file:

   ```
   # Copy the example and edit with your API keys
   cp .env.example .env
   ```

   Key environment variables:
   ```
   # Model providers
   OLLAMA_HOST_1=localhost        # Local Ollama endpoint
   GROQ_API_KEY=your_key_here     # Groq API key
   ANTHROPIC_API_KEY=your_key_here # Anthropic API key
   OPENAI_API_KEY=your_key_here   # OpenAI API key
   ```

## Usage

### Basic Usage

```bash
# Start CLI-Agent
python main.py
```

### Command-Line Options

```bash
# Use local models only
python main.py --local

# Enable voice input and text-to-speech output
python main.py --voice

# Use faster models
python main.py --fast

# Automatically execute safe code
python main.py --auto

# Continue previous conversation
python main.py -c

# Process a message and exit
python main.py --message "Write a script to list all PNG files in the current directory"

# Take a screenshot for analysis
python main.py --image

# Enable debug mode
python main.py --debug
```

### Key Bindings

During an interactive session, you can use these key bindings:

- `-r`: Regenerate the last response
- `-l`: Show LLM selection menu
- `-a`: Toggle automatic code execution
- `-f`: Toggle fast LLM mode
- `-v`: Toggle voice mode
- `-speak`: Toggle text-to-speech
- `-img`: Take a screenshot
- `-mct`: Toggle Monte Carlo Tree Search (branching)
- `-p`: Print chat history
- `-m`: Enter multiline input
- `-e`: Exit
- `-h`: Show help

## CLI-Agent Remote Host

The CLI-Agent supports running certain services remotely, which is useful for:

- Offloading CPU/GPU-intensive tasks to more powerful servers
- Creating distributed setups with specialized servers for specific tasks
- Centralizing services that might be used by multiple clients

Currently, the following services are supported remotely:
- Wake word detection (more services will be added in the future)

### Running the Remote Host Server

1. Start the CLI-Agent Remote Host server:

   ```bash
   python py_classes/remote_host/run_server.py
   ```

   This will start a Flask server that listens for service requests.

2. Or using Docker:

   ```bash
   docker build -f py_classes/remote_host/Dockerfile -t cli-agent-remote-host .
   docker run -p 5000:5000 cli-agent-remote-host
   ```

3. **For ARM devices (like Jetson Orin):**

   We provide a specialized Docker setup for ARM devices:

   ```bash
   # Use our helper script that automatically detects ARM architecture
   ./py_classes/remote_host/build_and_run_docker.sh
   ```

   Or manually:

   ```bash
   # Use the ARM-specific Dockerfile
   docker build -f py_classes/remote_host/Dockerfile.arm -t cli-agent-remote-host .
   docker run -p 5000:5000 cli-agent-remote-host
   ```

### Using Remote Services

When using the PyAiHost class, you can enable remote services by setting the `use_remote` parameter:

```python
from py_classes.ai_providers.cls_pyaihost_interface import PyAiHost

# Set the server URL (optional, defaults to environment variable or localhost:5000)
import os
os.environ["CLI_AGENT_REMOTE_HOST"] = "http://your-server:5000"

# Use remote wake word detection
wake_word = PyAiHost.wait_for_wake_word(use_remote=True)

# Or when recording audio
audio_data, sample_rate = PyAiHost.record_audio(use_remote=True)
```

You can test the remote functionality with:

```bash
python py_classes/remote_host/test_remote_host.py --remote --server-url http://your-server:5000
```

### Extending with New Services

The remote host is designed to be easily extended with new services. To add a new service:

1. Create a new service module in the `py_classes/remote_host/services/` directory
2. Update the remote host server to include the new service
3. Update the client to support the new service

## Extending CLI-Agent

### Adding Custom Utilities

1. Create a new Python file in the `utils/` directory
2. Implement a class that inherits from `UtilBase`
3. Implement a static `run()` method
4. CLI-Agent will automatically discover and load your utility

Example custom utility:

```python
# utils/my_custom_util.py
from py_classes.cls_util_base import UtilBase

class MyCustomUtil(UtilBase):
    """
    A custom utility for [purpose].
    """
    
    @staticmethod
    def run(param1: str, param2: int = 42) -> str:
        """
        Does something awesome.
        
        Args:
            param1: Description of param1
            param2: Description of param2, default is 42
            
        Returns:
            A string result
        """
        result = f"Processed {param1} with value {param2}"
        return result
```

## Examples

### General Questions

```
💬 What's the capital of France?
```

### Running Commands

```
💬 Show me the latest log files in /var/log and summarize their content
```

### Creating Scripts

```
💬 Write a script that monitors CPU usage and sends an alert if it exceeds 90% for more than 5 minutes
```

### Web Search

```
💬 Search for the latest developments in quantum computing
```

### Image Analysis

```
💬 Take a screenshot of my browser and extract all links from the webpage
```

## Podcast Generator

CLI-Agent includes a podcast generator that can create AI-generated podcasts on any topic. It uses LLM models to generate engaging scripts and Parler TTS for high-quality speech synthesis.

### Generating Podcasts

```bash
# Generate a podcast about AI
python generate_podcast.py --topic "Artificial Intelligence trends" --duration 3

# Use the shell script wrapper for easier usage
./shell_scripts/podcast_generator.sh -t "Climate change solutions" -d 2
```

For more details, see the [Podcast Generator Documentation](./docs/podcast_generator.md).

## Troubleshooting

### Common Issues

- **API Key Errors**: Ensure your API keys are correctly configured in `.env`
- **Model Availability**: Check that Ollama is running for local models
- **Voice Recognition**: Test your microphone with `arecord -l` and ensure it's properly configured

## Architecture

For a deeper understanding of CLI-Agent's architecture, please see the [ARCHITECTURE.md](ARCHITECTURE.md) file.

## Contributing

Your thoughts, feedback, and contributions are highly appreciated. Whether you're proposing changes, committing updates, or sharing ideas, every form of engagement enriches the project. Feel free to get involved.

## License

This project is open source and available under an as-is license. You are free to use, modify, and adapt it according to your needs.









# CLI-Agent Architecture

## Overview

CLI-Agent is an intelligent command-line agent built with Python, leveraging various LLM (Large Language Model) providers to assist users through natural language interaction. The system acts as a powerful assistant that can understand requests, execute code, perform tasks, and interact with the user's environment through a CLI interface.

## Key Components

### Core System

```
main.py                  # Main entry point and command loop
py_classes/              # Core classes for the system
py_methods/              # Utility methods and functions
utils/                   # Custom utility modules
```

### Core Architecture

The application follows a modular architecture with several key components:

1. **CLI Interface** (`main.py`): Handles user interaction, command parsing, and the main execution loop
2. **Chat System** (`cls_chat.py`): Manages conversation contexts and message history
3. **LLM Router** (`cls_llm_router.py`): Routes requests to appropriate AI providers based on capabilities and availability
4. **Python Sandbox** (`cls_python_sandbox.py`): Provides isolated code execution environment
5. **Utilities Manager** (`cls_util_manager.py`): Dynamically loads and manages utility modules
6. **AI Provider Interfaces**: Adapters for various LLM services

## Component Details

### 1. CLI Interface (`main.py`)

The main entry point serves as the command loop that:
- Parses command-line arguments
- Handles user input with key bindings
- Manages the conversation flow
- Controls the agent's action cycle
- Executes code in the Python sandbox

Key functions:
- `parse_cli_args()`: Processes command-line arguments
- `get_user_input_with_bindings()`: Handles user input with special key bindings
- `handle_screenshot_capture()`: Manages screenshot functionality
- `confirm_code_execution()`: Validates code safety before execution
- Main loop with agentic inner loop for handling conversations and actions

Key features:
- **Monte Carlo Tree Search (MCT)**: Implementation of a branching decision-making algorithm that explores multiple possible responses and selects the best one
- **Automatic code execution**: Ability to safely execute code with user confirmation
- **Key bindings system**: Special commands prefixed with "-" that toggle various features

### 2. Chat System (`cls_chat.py`)

The Chat class manages conversation contexts and history:
- Stores messages with role information (system, user, assistant)
- Provides serialization/deserialization to JSON
- Supports debug visualization
- Converts chat history to various LLM provider formats

Key methods:
- `add_message()`: Adds a message to the chat history
- `save_to_json()` / `load_from_json()`: Persistence methods
- Conversion methods for different provider formats:
  - `to_ollama()`
  - `to_openai()`
  - `to_groq()`
  - `to_gemini()`

Specialized features:
- Debug window visualization with auto-updating UI
- Token counting and context management
- Conversation state persistence

### 3. LLM Router (`cls_llm_router.py`)

Manages routing requests to appropriate LLMs:
- Singleton pattern for global state management
- Model selection based on capabilities and constraints
- Response caching and error handling
- Support for streaming responses

Key components:
- `Llm` class: Represents a language model with its properties
- `LlmRouter` class: Handles model selection and response generation
- Model capability checks and filtering

Key methods:
- `generate_completion()`: Central method for generating completions from LLMs
- `get_model()`: Selects the best model for a given request
- Caching and handling functions

Advanced features:
- Dynamic model strength attributes (LOCAL, VISION, CODE, etc.)
- Automatic fallback mechanisms when models fail
- Response streaming with real-time display
- Special handling for multi-modal (text+image) inputs

### 4. Python Sandbox (`cls_python_sandbox.py`)

Provides a safe, isolated environment for executing Python code:
- Uses Jupyter kernels for isolated execution
- Maintains state between executions
- Supports streaming output and callbacks
- Handles timeouts and errors

Key methods:
- `execute()`: Runs Python code in the sandbox
- `restart()`: Restarts the kernel if needed
- Timeout and error handling

Implementation details:
- Uses Jupyter kernels for process isolation
- Maintains state across multiple code executions
- Real-time output streaming with callbacks
- Idle time detection to prevent infinite loops

### 5. Utilities Manager (`cls_util_manager.py`)

Dynamically loads and manages utility modules:
- Scans the utils directory for Python modules
- Loads classes that inherit from UtilBase
- Provides access to utilities by name

Key methods:
- `_load_utils()`: Dynamically loads utilities
- `get_util()`: Retrieves a utility by name
- `get_available_utils_info()`: Generates documentation

Special capabilities:
- Dynamic utility discovery at runtime
- Introspection of utility methods and documentation
- Automatic help text generation

### 6. AI Provider Interfaces

Adapters for various LLM services:
- `cls_ollama_interface.py`: For local Ollama models
- `cls_groq_interface.py`: For Groq API
- `cls_anthropic_interface.py`: For Anthropic Claude models
- `cls_google_interface.py`: For Google Gemini models
- `cls_openai_interface.py`: For OpenAI models

Each implements a common interface for:
- Generating completions
- Handling authentication
- Managing rate limits and errors
- Supporting model-specific features

## Utility Modules

The system includes several utility modules located in the `utils/` directory:

### Text and Web Utilities
- `searchweb.py`: Web search capabilities with relevance checking and query refinement
- `authorfile.py`: File authoring utilities

### Image and Media Utilities
- `generateimage.py`: Image generation capabilities
- `imagetotext.py`: Image-to-text conversion using vision-capable LLMs

### General Utilities
- `tobool.py`: Boolean conversion utilities
- Custom utilities following the `UtilBase` class pattern

## Audio and Speech Capabilities

The CLI-Agent includes significant audio processing capabilities:

### Audio Input
- Wake word detection using Vosk
- Microphone input with automatic calibration
- Speech recognition and transcription

### Audio Output
- Text-to-speech synthesis
- Notification sounds
- Audio playback functions

Implementation details:
- Uses various libraries (sounddevice, whisper, vosk)
- Handles device management and audio format conversion
- Supports both local and remote wake word detection

## Screenshot and Image Processing

The system provides robust screen capture capabilities:
- Interactive region selection
- Application window capture
- Full-screen capture
- Image analysis and OCR using vision-capable LLMs

Implementation details:
- Uses a combination of libraries (PIL, tkinter)
- Multiple fallback methods for different environments
- Image preprocessing for better LLM analysis

## Data Flow

1. User provides input through the CLI
2. Input is processed and added to the Chat context
3. The LLM Router selects an appropriate model
4. The model generates a response
5. If the response includes code, the Python Sandbox executes it
6. Results are returned to the user
7. The loop continues

## Agentic Loop System

The CLI-Agent implements an intelligent agent loop that:
1. **Processes user input**: Handles text, voice, or screenshot inputs
2. **Generates responses**: Uses the LLM Router to get AI responses
3. **Extracts executable code**: Identifies Python code blocks in responses
4. **Validates code safety**: Uses a separate LLM to check code before execution
5. **Executes code safely**: Runs code in the Python Sandbox
6. **Processes results**: Captures and formats execution output
7. **Updates context**: Adds results to conversation history for continuity

Special features:
- Multi-branch response generation with Monte Carlo Tree Search
- Code execution safety checks
- Adaptive loop that can run multiple iterations within a single user turn

## Configuration

The system uses several configuration mechanisms:
- Environment variables (via `.env` file)
- Command-line arguments
- Global state in `globals.py`

Key environment variables:
- LLM API keys (GROQ_API_KEY, ANTHROPIC_API_KEY, etc.)
- Infrastructure settings (OLLAMA_HOST_1, etc.)
- Remote host configuration

## Remote Host Support

The system optionally supports running services remotely:
- Wake word detection
- Potentially other CPU/GPU intensive tasks
- Configured via environment variables and command-line flags

Implementation details:
- Uses Flask server for the remote host
- Provides Docker support, including ARM-specific builds
- Configurable via environment variables

## Global State Management

Global state is managed through the `globals.py` module:
- Constants and configuration values
- Runtime flags
- Path definitions
- State variables accessed via the `g` singleton

Important global settings:
- `FORCE_LOCAL`: Forces use of local models only
- `FORCE_FAST`: Prioritizes faster models over more capable ones
- `DEBUG_CHATS`: Enables debug windows for chat contexts
- Various path definitions for persistent storage

## Error Handling and Resilience

The system includes several error handling mechanisms:
- Exception handling for LLM API errors
- Fallbacks when models are unavailable
- Retry mechanisms for transient failures
- Input validation and sanitization

Implementation details:
- Multi-level error handling (function, module, global)
- Graceful degradation when services are unavailable
- User feedback for error conditions
- Debugging aids for development

## Best Practices

The code demonstrates several best practices:
- Type hints throughout the codebase
- Comprehensive error handling
- Modular design with clear separation of concerns
- Singleton patterns for shared resources
- Persistent state management
- Streaming output for better UX

## Advanced Features

### RAG (Retrieval-Augmented Generation)
- PDF and document processing with extraction and embedding
- Vector database integration for semantic search
- Query enhancement and context injection

### Command History Integration
- Integration with Atuin for enhanced command history
- Embeddings for semantic similarity search of past commands

### Multi-modal Capabilities
- Image input processing via screenshots
- Text-to-speech and speech-to-text conversions
- Web search and information retrieval

## Conclusion

CLI-Agent presents a well-structured, modular architecture that combines multiple AI capabilities with a user-friendly CLI interface. The system's design allows for easy extension and customization, making it a powerful tool for developers and users who need AI assistance directly in their terminal environment. 