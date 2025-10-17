# Isolated Mode - Quick Reference

## TL;DR

```bash
python3 main.py --isolated           # Run in Docker container
```

First run: ~2-3 min (builds image)
Next runs: ~2-5 sec (instant)

## Common Commands

```bash
# Basic usage
python3 main.py --isolated
python3 main.py --isolated -m "Your message"
python3 main.py --isolated -c         # Continue conversation

# GUI mode
python3 main.py --isolated --gui      # Web UI at localhost:7860

# Voice mode
python3 main.py --isolated -v         # Voice input/output
python3 main.py --isolated -s         # TTS only

# Debug mode
python3 main.py --isolated --debug    # Verbose logging

# Specific LLM
python3 main.py --isolated --llm gemini-2.5-flash
```

## How It Works

1. Detects `--isolated` flag early in main.py
2. Checks for cached Docker image `cli-agent:isolated-cache`
3. Builds if missing (~2-3 min first time)
4. Copies latest source code into container
5. Executes `python3 main.py [args]` inside container
6. Streams output to your terminal

## What's Mounted

- `~/.cli-agent` → `/home/cliagent/.cli-agent` (config, chat history, cache)
- Current working directory → `/workspace` (access your files)
- `.env` file automatically loaded (API keys)

## Troubleshooting

```bash
# Docker not found
sudo apt install docker.io
sudo usermod -aG docker $USER
newgrp docker

# Rebuild image
docker rmi cli-agent:isolated-cache
python3 main.py --isolated

# Clean up containers
docker rm -f $(docker ps -aq --filter name=cli-agent-isolated)

# Check running containers
docker ps | grep cli-agent-isolated
```

## Files Created

- `docker_files/Dockerfile.isolated-exec` - Docker image definition
- `shared/utils/docker_runner.py` - Container orchestration
- `main.py` - Added `--isolated` flag (line 578) and early detection (line 87)

## Full Documentation

See [ISOLATED_MODE.md](ISOLATED_MODE.md) for complete documentation.
