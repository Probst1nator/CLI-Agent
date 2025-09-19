# CLI-Agent Production Docker Setup

This guide explains how to run CLI-Agent in a production Docker environment with the visual response interface exposed.

## Quick Start

### 1. Build the Production Image

```bash
# Build the production container
docker build -f Dockerfile.prod -t cli-agent:prod .
```

### 2. Run with Docker Compose (Recommended)

```bash
# Create environment file for API keys
cp .env.example .env
# Edit .env with your API keys

# Start the container
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop the container
docker-compose -f docker-compose.prod.yml down
```

### 3. Access the Interface

Open your browser to:
- **Primary**: http://localhost:7860
- **Fallback**: http://localhost:7861 or http://localhost:7862

## Environment Configuration

### Required API Keys

Set these in your `.env` file or as environment variables:

```bash
# OpenAI API (for GPT models)
OPENAI_API_KEY=sk-your-openai-key-here

# Anthropic API (for Claude models)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Google API (for Gemini models)
GOOGLE_API_KEY=your-google-api-key-here

# Groq API (for fast inference)
GROQ_API_KEY=gsk_your-groq-key-here
```

### Optional Configuration

```bash
# CLI-Agent specific settings
CLI_AGENT_GUI_MODE=true
CLI_AGENT_HOST=0.0.0.0
CLI_AGENT_DATA_DIR=/cli-agent/data
CLI_AGENT_LOG_DIR=/cli-agent/logs

# Resource limits
MEMORY_LIMIT=2G
CPU_LIMIT=1.0
```

## Manual Docker Run

If you prefer to run without docker-compose:

```bash
docker run -d \
  --name cli-agent-prod \
  -p 7860:7860 \
  -p 7861:7861 \
  -p 7862:7862 \
  -e OPENAI_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  -v cli-agent-data:/cli-agent/data \
  -v cli-agent-logs:/cli-agent/logs \
  --restart unless-stopped \
  cli-agent:prod
```

## Data Persistence

The container uses Docker volumes for data persistence:

- **cli-agent-data**: Chat histories and user data
- **cli-agent-logs**: Application logs and debug info
- **cli-agent-temp**: Temporary files and uploads

### Backup Data

```bash
# Backup data volume
docker run --rm -v cli-agent-data:/data -v $(pwd):/backup ubuntu tar czf /backup/cli-agent-backup.tar.gz -C /data .

# Restore data volume
docker run --rm -v cli-agent-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/cli-agent-backup.tar.gz -C /data
```

## Monitoring and Health

### Health Checks

The container includes built-in health checks:

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# View health check logs
docker inspect cli-agent-prod | jq '.[0].State.Health'
```

### Viewing Logs

```bash
# Real-time logs
docker logs -f cli-agent-prod

# Last 100 lines
docker logs --tail 100 cli-agent-prod

# Logs since specific time
docker logs --since 30m cli-agent-prod
```

### Resource Monitoring

```bash
# Resource usage
docker stats cli-agent-prod

# Detailed container info
docker inspect cli-agent-prod
```

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check container logs
docker logs cli-agent-prod

# Check if ports are available
netstat -tulpn | grep :7860
```

**Visual interface not accessible:**
```bash
# Check if container is healthy
docker ps --filter "name=cli-agent-prod"

# Test internal health check
docker exec cli-agent-prod curl -f http://localhost:7860/
```

**Out of memory errors:**
```bash
# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G  # Increase from 2G
```

### Debug Mode

Run with debug logging:

```bash
docker run -it --rm \
  -p 7860:7860 \
  -e OPENAI_API_KEY=your-key \
  -e PYTHONPATH=/cli-agent \
  -e DEBUG=1 \
  cli-agent:prod \
  python3 main.py --gui --debug
```

## Security Considerations

### Production Security

1. **API Keys**: Never hardcode API keys in the image
2. **Network**: Use reverse proxy (nginx/traefik) for HTTPS
3. **User**: Container runs as non-root user `cliagent`
4. **Volumes**: Set appropriate volume permissions
5. **Updates**: Regularly update base image and dependencies

### Recommended Production Setup

```yaml
# docker-compose.prod-secure.yml
version: '3.8'
services:
  cli-agent:
    image: cli-agent:prod
    restart: unless-stopped
    environment:
      - CLI_AGENT_HOST=127.0.0.1  # Only local access
    volumes:
      - ./data:/cli-agent/data:rw
      - ./logs:/cli-agent/logs:rw
    networks:
      - internal

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - cli-agent
    networks:
      - internal

networks:
  internal:
    driver: bridge
```

## Multi-Agent Setup (Future)

The current setup provides the foundation for running multiple agent instances:

```yaml
# Example multi-agent configuration
services:
  cli-agent-1:
    extends:
      service: cli-agent
    container_name: cli-agent-1
    ports:
      - "7860:7860"

  cli-agent-2:
    extends:
      service: cli-agent
    container_name: cli-agent-2
    ports:
      - "7863:7860"
```

## Performance Optimization

### Resource Tuning

```yaml
# Optimized resource allocation
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 512M
      cpus: '0.25'
```

### Image Optimization

```bash
# Multi-stage build for smaller images
# See Dockerfile.prod-optimized (future enhancement)

# Remove unnecessary dependencies
pip install --no-deps package-name
```

## Support

For issues related to the Docker setup:

1. Check the logs: `docker logs cli-agent-prod`
2. Verify health: `docker ps --format "table {{.Names}}\t{{.Status}}"`
3. Test manually: `docker exec -it cli-agent-prod python3 main.py --gui`
4. Report issues with logs and configuration details

## Version History

- **v1.0**: Initial production Docker setup
- **v1.1**: Added health checks and volume persistence
- **v1.2**: Enhanced security and monitoring features