# Simple CLI Agent Using Groq-API and Ollama Backends

This project introduces a CLI (Command Line Interface) Agent that leverages the Groq-API and Ollama backends. It is part of the Local Language Model Toolkit Project that I have developed.

## Getting Started

### Prerequisites

- **Operating System:** Currently, this project only supports Ubuntu.

### Installation

There are two ways to install and run the CLI Agent: using Docker or running it manually.

#### Using Docker (Optional)

1. Run the Ollama backend in a Docker container:

   ```bash
   sudo docker run -d --name ollama --restart always -p 5000:11434 ollama/ollama:latest
   ```

   Alternatively, you can configure your Groq API details in the `.env` file.

#### Using the Installer Script

1. To add the CLI Agent to your command line, execute the `installer.sh` script:

   ```bash
   ./installer.sh
   ```

#### Running Manually

1. Alternatively, you can run the Agent manually by executing `main.py`:

   ```bash
   python3 ./main.py
   ```

### Features

- **Command Suggestions:** The Agent suggests commands and, upon your confirmation, executes them. It discusses and acts based on the output of these commands.

## Contributing

Your thoughts, feedback, and contributions are highly appreciated. Whether you're proposing changes, committing updates, or sharing ideas, every form of engagement enriches the project. Feel free to get involved.

## License

This project is open source and available under an as-is license. You are free to use, modify, and adapt it according to your needs.