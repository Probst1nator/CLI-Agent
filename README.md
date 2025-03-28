# A custom general CLI-Agent solution, providing a hackable playground inside a langchain-like framework

This project introduces a CLI (Command Line Interface) Agent leveraging the Groq-API and Ollama backends, based on the Local Language Model Toolkit Project I've developed.

## Getting Started

### Prerequisites

- **Operating System:** Currently, only Ubuntu is supported.

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

### Features

- **Command Suggestions:** The Agent suggests commands and, upon your confirmation, executes them. It will then discuss and act based on the output of these commands.

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

## Contributing

Your thoughts, feedback, and contributions are highly appreciated. Whether you're proposing changes, committing updates, or sharing ideas, every form of engagement enriches the project. Feel free to get involved.

## License

This project is open source and available under an as-is license. You are free to use, modify, and adapt it according to your needs.
