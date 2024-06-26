# CLI Agent Using Groq-API, Ollama, and OpenAI Backends

This project introduces a CLI (Command Line Interface) Agent that leverages the Groq-API, Ollama, and OpenAI backends. It is part of the Local Language Model Toolkit Project that I have developed.

## Getting Started

#### Docker/ Groq

The large language model needs to be hosted somewhere. I recommend the Groq-API for quick and powerful model responses, but Ollama and OpenAI are also supported.

Add your `GROQ_API_KEY` and `OPENAI_API_KEY` to the `.env` file, you can get one here:

- [Groq-API Key](https://example.com/groq-api-key)
- [OpenAI API Key](https://beta.openai.com/signup/)

Run the Ollama backend in a Docker container:
   ```bash
   sudo docker run -d --name ollama --restart always -p 5000:11434 ollama/ollama:latest
   ```

#### Run from script

1. The easiest way to use the Agent is like this:

   ```bash
   chmod +x ./easy_runner.sh
   ./easy_runner.sh
   ```

#### Run from cli

1. To add the CLI Agent to your command line, execute the `add_alias_to_bash.sh` script:

   ```bash
   chmod +x ./installer.sh
   ./installer.sh
   ```

### Features

- **Command Suggestions:** The Agent suggests commands and, upon your confirmation, executes them. It discusses and acts based on the output of these commands.

### Release Information

#### Latest Release: v1.1.0

The latest release of CLI-Agent includes the following features:

- Initial integration with Groq-API, Ollama, and OpenAI backends
- Command suggestion and execution functionality
- Support for Ubuntu operating system
- Installation options via Docker or manual setup

You can download the release assets, including the source code and installation scripts, from the [GitHub Releases page](https://github.com/Probst1nator/CLI-Agent/releases).

## Contributing

Your thoughts, feedback, and contributions are highly appreciated. Whether you're proposing changes, committing updates, or sharing ideas, every form of engagement enriches the project. Feel free to get involved.

## License

This project is open source and available under an as-is license. You are free to use, modify, and adapt it according to your needs.
