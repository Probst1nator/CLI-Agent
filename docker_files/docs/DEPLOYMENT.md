# CLI-Agent Deployment Guide

## Deployment Modes

### 1. Local (No Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e core/ -e shared/
cp .env.example .env   # add your API keys

python main.py         # CLI mode
python main.py --gui   # Web interface (localhost:8765)
python main.py --api   # API server only
```

### 2. Isolated Mode (Recommended for untrusted input)

```bash
# First run builds image (~2-3 min), subsequent runs ~2-5 sec
python main.py --isolated
python main.py --isolated --gui
```

Each run gets a fresh isolated Docker container. See [ISOLATED_MODE.md](ISOLATED_MODE.md) for details.

### 3. Production (Single-User, Always-On)

Run as a systemd service with Nginx as a reverse proxy for SSL and basic auth.

**systemd unit (`/etc/systemd/system/cli-agent.service`):**

```ini
[Unit]
Description=CLI-Agent API Server
After=network.target

[Service]
User=<YOUR_USERNAME>
WorkingDirectory=/path/to/CLI-Agent
EnvironmentFile=/path/to/CLI-Agent/.env
ExecStart=/path/to/CLI-Agent/.venv/bin/python main.py --api
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable cli-agent
sudo systemctl start cli-agent
```

**Nginx config snippet:**

```nginx
location / {
    proxy_pass http://127.0.0.1:8765;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

Add SSL via Certbot and restrict access with `auth_basic` or firewall rules.

---

## Docker Images

| Dockerfile | Purpose |
|------------|---------|
| `Dockerfile.backend` | Full backend image |
| `Dockerfile.frontend` | Frontend-only image |
| `Dockerfile.isolated-exec` | Used by `--isolated` mode |

Build manually:
```bash
cd docker_files
docker build -f Dockerfile.backend -t cli-agent-backend:latest ..
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | One of these | Claude API |
| `OPENAI_API_KEY` | One of these | OpenAI API |
| `GOOGLE_API_KEY` | One of these | Gemini API |
| `GROQ_API_KEY` | One of these | Groq API |
| `BRAVE_API_KEY` | Optional | Web search |
| `HF_TOKEN` | Optional | HuggingFace local models |
| `OLLAMA_HOSTS` | Optional | Comma-separated Ollama URLs |

See `.env.example` for the full list.
