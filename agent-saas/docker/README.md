# Agent Docker Images - Build Guide

This directory contains Dockerfiles and build scripts for converting CLI-based agents into containerized WebSocket/HTTP services.

## Overview

The Agent SaaS infrastructure uses Docker containers to run individual agent sessions. Each agent type (minimal_agent, developer_agent) gets its own Docker image.

### Architecture

```
cli-agent-base:latest
  ↓
  ├─→ cli-agent-minimal_agent:latest
  └─→ cli-agent-developer_agent:latest
```

**Base Image** (`cli-agent-base:latest`):
- Python 3.11-slim
- Common dependencies (anthropic, google-generativeai, fastapi, etc.)
- Agent server framework
- ~400MB

**Agent Images**:
- Based on base image
- Contains agent code + tools
- Agent server wrapper
- ~420-450MB each

---

## Files

```
docker/
├── Dockerfile.agent-base          # Base image with shared dependencies
├── Dockerfile.minimal-agent        # Minimal agent image
├── Dockerfile.developer-agent      # Developer agent image
├── agent_server.py                 # WebSocket/HTTP server wrapper
├── build-all.sh                    # Build all images
├── build-agent-image.sh            # Build single agent image
├── test-agent-container.sh         # Test running container
└── README.md                       # This file
```

---

## Quick Start

### 1. Build All Images

```bash
cd /path/to/CLI-Agent_DEV
./agent-saas/docker/build-all.sh
```

**Output**:
```
Step 1/3: Building base image (cli-agent-base:latest)
✓ Base image built successfully

Step 2/3: Building minimal_agent image
✓ minimal_agent image built successfully

Step 3/3: Building developer_agent image
✓ developer_agent image built successfully

Build Complete!
```

**Time**: ~5-10 minutes (first build), ~30 seconds (subsequent builds with cache)

### 2. Verify Images

```bash
docker images | grep cli-agent
```

Expected output:
```
cli-agent-developer_agent   latest   abc123def456   10 minutes ago   450MB
cli-agent-minimal_agent     latest   def456abc123   10 minutes ago   420MB
cli-agent-base              latest   789xyz456abc   10 minutes ago   400MB
```

### 3. Test an Image

```bash
# Start minimal_agent container
docker run --rm -p 8000:8000 cli-agent-minimal_agent:latest

# In another terminal, test it
./agent-saas/docker/test-agent-container.sh 8000
```

---

## Build Options

### Build Single Agent

```bash
./agent-saas/docker/build-agent-image.sh minimal_agent
```

### Rebuild with No Cache

```bash
docker build --no-cache -f agent-saas/docker/Dockerfile.agent-base -t cli-agent-base:latest .
```

### Build for Specific Platform

```bash
# For ARM (M1/M2 Mac)
docker build --platform linux/arm64 -f agent-saas/docker/Dockerfile.agent-base -t cli-agent-base:latest .

# For x86_64
docker build --platform linux/amd64 -f agent-saas/docker/Dockerfile.agent-base -t cli-agent-base:latest .
```

---

## Testing

### Manual Test (HTTP)

```bash
# Start container
docker run --rm -p 8000:8000 --name test-agent cli-agent-minimal_agent:latest

# Health check
curl http://localhost:8000/health

# Get agent info
curl http://localhost:8000/api/info

# Send message
curl -X POST http://localhost:8000/api/chat/send \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello! What tools do you have?"}'

# Get chat history
curl http://localhost:8000/api/chat/history

# Clear history
curl -X POST http://localhost:8000/api/chat/clear
```

### Manual Test (WebSocket)

```bash
# Install wscat
npm install -g wscat

# Start container
docker run --rm -p 8000:8000 cli-agent-minimal_agent:latest

# Connect via WebSocket
wscat -c ws://localhost:8000/ws

# Send messages
{"type": "chat_message", "content": "Hello agent!"}
{"type": "chat_message", "content": "What files are in /agent?"}
{"type": "get_history"}
{"type": "clear_history"}
```

### Automated Test

```bash
# Start container
docker run --rm -p 8000:8000 cli-agent-minimal_agent:latest

# Run test script
./agent-saas/docker/test-agent-container.sh 8000
```

**Expected Output**:
```
Test 1: Health Check
✓ Health check successful

Test 2: Agent Info
✓ Agent info retrieved

Test 3: Send Message (HTTP)
✓ Message sent successfully

Test 4: Get Chat History
✓ Chat history retrieved

Test 5: Clear Chat History
✓ Chat history cleared

All Tests Passed! ✓
```

---

## Agent Server API

Each agent container exposes the following HTTP and WebSocket endpoints:

### HTTP Endpoints

#### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "agent_class": "MinimalAgent",
  "tools_loaded": 4
}
```

#### `GET /api/info`
Get agent information.

**Response**:
```json
{
  "status": "success",
  "agent_class": "MinimalAgent",
  "tools": ["think", "todos", "readfile", "grepsearch"],
  "tool_count": 4
}
```

#### `POST /api/chat/send`
Send message to agent (non-streaming).

**Request**:
```json
{
  "content": "What tools do you have?"
}
```

**Response**:
```json
{
  "status": "success",
  "response": "I have 4 tools: think, todos, readfile, and grepsearch..."
}
```

#### `GET /api/chat/history`
Get chat history.

**Response**:
```json
{
  "status": "success",
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant..."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

#### `POST /api/chat/clear`
Clear chat history.

**Response**:
```json
{
  "status": "success",
  "message": "Chat history cleared"
}
```

### WebSocket Endpoint

#### `WebSocket /ws`
Real-time chat with streaming responses.

**Client → Server**:
```json
{
  "type": "chat_message",
  "content": "User message here"
}
```

**Server → Client (streaming)**:
```json
{
  "type": "llm_chunk",
  "content": "Let me ",
  "is_complete": false
}

{
  "type": "tool_execution",
  "tool_name": "readfile",
  "content": "Executing readfile...",
  "is_complete": false
}

{
  "type": "tool_result",
  "tool_name": "readfile",
  "content": "File contents...",
  "is_complete": false
}

{
  "type": "final_response",
  "content": "Complete response here",
  "is_complete": true
}
```

**Message Types**:
- `chat_message`: User message
- `get_history`: Request chat history
- `clear_history`: Clear chat history

**Response Types**:
- `llm_chunk`: Streaming LLM response
- `tool_execution`: Tool about to execute
- `tool_result`: Tool execution result
- `final_response`: Complete response (no more tool calls)
- `error`: Error occurred
- `chat_history`: Response to get_history request
- `history_cleared`: Response to clear_history request

---

## Environment Variables

### In Dockerfile

```dockerfile
ENV AGENT_CLASS_NAME="MinimalAgent"    # Agent class name
ENV AGENT_MODULE_NAME="main"           # Python module containing agent
ENV AGENT_TYPE="minimal_agent"         # Agent type identifier
```

### At Runtime

```bash
docker run -e GOOGLE_API_KEY=your_key \
           -e ANTHROPIC_API_KEY=your_key \
           -p 8000:8000 \
           cli-agent-minimal_agent:latest
```

**Available**:
- `GOOGLE_API_KEY` / `GEMINI_API_KEY`: Google Gemini
- `ANTHROPIC_API_KEY`: Anthropic Claude
- `OPENAI_API_KEY`: OpenAI
- `GROQ_API_KEY`: Groq
- `OLLAMA_HOST`: Ollama (e.g., `http://host.docker.internal:11434`)
- `AGENT_PORT`: Server port (default: 8000)

---

## Volumes

Mount volumes for persistent data:

```bash
docker run --rm \
  -p 8000:8000 \
  -v $(pwd)/agent-data:/workspace/data \
  -v $(pwd)/agent-logs:/workspace/logs \
  -v $(pwd)/agent-config:/workspace/config \
  cli-agent-minimal_agent:latest
```

**Directories**:
- `/workspace/data`: Chat history, session data
- `/workspace/logs`: Agent logs
- `/workspace/config`: Configuration files

---

## Troubleshooting

### Build Fails: "tools directory not found"

**Problem**: Agent's tools directory missing.

**Solution**:
```bash
# Verify agent structure
ls -la agents/minimal_agent/
# Should show: main.py, chat.py, llm_router.py, tools/

# Rebuild if needed
./agent-saas/docker/build-agent-image.sh minimal_agent
```

### Container Starts but Health Check Fails

**Problem**: Agent not loading correctly.

**Solution**:
```bash
# Check logs
docker logs <container-id>

# Common issues:
# 1. Missing API keys (GOOGLE_API_KEY, etc.)
# 2. Import errors (missing dependencies)
# 3. Port already in use
```

### "Module not found" Error

**Problem**: Agent server can't import agent code.

**Solution**:
```bash
# Verify agent code is in image
docker run --rm cli-agent-minimal_agent:latest ls -la /agent/

# Should show:
# main.py, chat.py, llm_router.py, tools/, agent_server.py
```

### WebSocket Connection Refused

**Problem**: Port not exposed or mapped incorrectly.

**Solution**:
```bash
# Ensure port is mapped
docker run --rm -p 8000:8000 cli-agent-minimal_agent:latest

# Test connectivity
curl http://localhost:8000/health
```

---

## Development

### Adding a New Agent

1. Create agent directory: `agents/my_agent/`
2. Implement agent class: `class MyAgent` in `main.py`
3. Create Dockerfile:

```dockerfile
# agent-saas/docker/Dockerfile.my-agent
FROM cli-agent-base:latest

ENV AGENT_CLASS_NAME="MyAgent" \
    AGENT_MODULE_NAME="main" \
    AGENT_TYPE="my_agent"

COPY agents/my_agent/ /agent/
COPY agent-saas/docker/agent_server.py /agent/agent_server.py

RUN python -c "from main import MyAgent; agent = MyAgent(); print(f'Loaded {len(agent.tools)} tools')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "agent_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

4. Build:
```bash
./agent-saas/docker/build-agent-image.sh my_agent
```

### Modifying agent_server.py

After modifying `agent_server.py`, rebuild all agent images:

```bash
./agent-saas/docker/build-all.sh
```

---

## Production Deployment

### With Agent Container Manager

The agent container manager automatically:
- Builds/pulls images
- Creates containers on-demand
- Manages lifecycle
- Handles networking

**No manual docker run needed** - the container manager does it all.

### Resource Limits

When running manually (for testing), add resource limits:

```bash
docker run --rm \
  -p 8000:8000 \
  --memory="1g" \
  --cpus="1.0" \
  --pids-limit=50 \
  cli-agent-minimal_agent:latest
```

### Security

For production containers:
```bash
docker run --rm \
  -p 8000:8000 \
  --memory="1g" \
  --cpus="1.0" \
  --pids-limit=50 \
  --security-opt="no-new-privileges:true" \
  --cap-drop=ALL \
  --cap-add=CHOWN \
  --cap-add=SETUID \
  --cap-add=SETGID \
  cli-agent-minimal_agent:latest
```

---

## Image Sizes

Expected sizes (approximate):

| Image | Size | Contents |
|-------|------|----------|
| `cli-agent-base:latest` | ~400MB | Python 3.11 + common deps |
| `cli-agent-minimal_agent:latest` | ~420MB | Base + minimal agent |
| `cli-agent-developer_agent:latest` | ~450MB | Base + developer agent |

**Optimization tips**:
- Use multi-stage builds (future enhancement)
- Remove unnecessary dependencies
- Use `.dockerignore` to exclude files

---

## Next Steps

After building images:

1. **Test locally**: Run containers and verify functionality
2. **Start container manager**: `python agent-saas/agent_container_manager.py`
3. **Create session**: Use API to create test session
4. **Build frontend**: Implement UI for session management

See `agent-saas/README.md` for complete setup guide.
