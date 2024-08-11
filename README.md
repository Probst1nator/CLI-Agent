
# CLI Agent Using Groq-API, Ollama, OpenAI, and Anthropic Backends

This project introduces a CLI (Command Line Interface) Agent that utilizes the Groq-API, Ollama, OpenAI, and Anthropic backends.

## Getting Started

### Requirements

```bash
sudo apt-get install libportaudio2 portaudio19-dev python3-tk
pip install -r requirements.txt
```

#### Docker/ Groq

The large language model requires hosting. I recommend using the Groq-API for fast and robust model responses. However, Ollama, OpenAI, and Anthropic are also viable options.

Please add your `GROQ_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` to the `.env` file. Keys can be obtained here:

- [Groq-API Key](https://console.groq.com/keys)
- [OpenAI API Key](https://beta.openai.com/signup/)
- [Anthropic API Key](https://console.anthropic.com/settings/keys)

To run the Ollama backend in a Docker container, use the following command:
   ```bash
   sudo docker run -d --name ollama --restart always -p 5000:11434 ollama/ollama:latest
   ```

#### Run from script

1. For straightforward usage of the Agent, execute the following:

   ```bash
   chmod +x ./easy_runner.sh
   ./easy_runner.sh
   ```

#### Run from cli

1. To integrate the CLI Agent into your command line environment, run the `add_alias_to_bash.sh` script:

   ```bash
   chmod +x ./installer.sh
   ./installer.sh
   ```

### Features

- **Command Suggestions:** The Agent recommends commands and, upon your approval, carries them out. It responds and adapts based on the output from these commands.

### Release Information

#### Latest Release: v1.1.0

The latest release of the CLI-Agent incorporates the following features:

- Integration with Groq-API, Ollama, OpenAI, and Anthropic backends
- Command suggestion and execution capabilities
- Compatibility with the Ubuntu operating system

You can download the release assets, such as source code and installation scripts, from the [GitHub Releases page](https://github.com/Probst1nator/CLI-Agent/releases).

## Contributing

Feel free to comment, develop or follow.

## License

This project is open source and released under an as-is license. You are free to use, modify, and customize it as per your requirements.
