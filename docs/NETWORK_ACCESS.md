# Network Access and Container Security

This document details the network access patterns, security restrictions, and connectivity configuration for CLI-Agent containers.

## Overview

CLI-Agent containers run in a sandboxed environment with carefully controlled network access. This ensures security while maintaining necessary connectivity to AI services, local resources, and other container instances.

## Container Network Architecture

### Docker Network Configuration
- **Network Type**: Bridge network (`docker_files_cli-agent-multi-net`)
- **Subnet**: `172.18.0.0/16`
- **Gateway**: `172.18.0.1`
- **Internal**: `false` (allows internet access)

### Container IP Assignments
```
172.18.0.2 - cli-agent-instance-1
172.18.0.3 - cli-agent-instance-3  
172.18.0.4 - cli-agent-instance-2
172.18.0.5 - cli-agent-session-manager
```

## Network Access Matrix

### ✅ Accessible Ports (Outbound)

#### Internet Services
| Port | Protocol | Service Type | Status |
|------|----------|--------------|---------|
| 80   | HTTP     | Web services | ✅ Open |
| 443  | HTTPS    | Secure web   | ✅ Open |
| 8080 | HTTP-Alt | Web services | ✅ Open |
| 8443 | HTTPS-Alt| Secure web   | ✅ Open |

#### AI Service APIs
| Service | Endpoint | Port | Status |
|---------|----------|------|---------|
| OpenAI | `api.openai.com` | 443 | ✅ Accessible |
| Anthropic | `api.anthropic.com` | 443 | ✅ Accessible |
| Google AI | `generativelanguage.googleapis.com` | 443 | ✅ Accessible |
| Groq | API endpoints | 443 | ✅ Accessible |

#### Local Network Access
| Target | Port | Status | Notes |
|--------|------|--------|-------|
| Host HTTP | 80 | ✅ Open | General web access |
| Host HTTPS | 443 | ✅ Open | Secure web access |
| CLI-Agent Instance 1 | 7861 | ✅ Open | Direct instance access |
| Ollama (`192.168.178.37`) | 11434 | ✅ Open | **33 models available** |

#### Inter-Container Communication
| Target | IP | Port | Status |
|--------|----|------|--------|
| Session Manager | `172.18.0.5` | 8000 | ✅ Open |
| Other Instances | `172.18.0.x` | 7860 | ✅ Open |

### ❌ Blocked/Filtered Ports

#### System Services (Security Restriction)
| Port | Service | Status | Reason |
|------|---------|--------|---------|
| 22 | SSH | ❌ Blocked | Security |
| 21 | FTP | ❌ Blocked | Security |
| 23 | Telnet | ❌ Blocked | Security |

#### Database Services
| Port | Service | Status | Reason |
|------|---------|--------|---------|
| 3306 | MySQL | ❌ Blocked | Unnecessary access |
| 5432 | PostgreSQL | ❌ Blocked | Unnecessary access |
| 6379 | Redis | ❌ Blocked | Unnecessary access |
| 27017 | MongoDB | ❌ Blocked | Unnecessary access |

#### Mail Services
| Port | Service | Status | Reason |
|------|---------|--------|---------|
| 25 | SMTP | ❌ Blocked | Security |
| 587 | SMTP-TLS | ❌ Blocked | Security |
| 993 | IMAPS | ❌ Blocked | Security |
| 995 | POP3S | ❌ Blocked | Security |

## Ollama Configuration

### Environment Configuration
The Ollama connectivity is configured via environment variables in `.env`:

```bash
OLLAMA_HOST=192.168.178.37,192.168.178.25,localhost
OLLAMA_HOST_AUTO_DOWNLOAD_MODELS=192.168.178.37,192.168.178.25,localhost
```

### Host Status
| Host | Port | Status | Models Available |
|------|------|--------|------------------|
| `192.168.178.37` | 11434 | ✅ **Active** | **33 models** |
| `192.168.178.25` | 11434 | ❌ Unreachable | N/A |
| `localhost` | 11434 | ❌ Not configured | N/A |

### Available Models on Active Host
The primary Ollama host (`192.168.178.37:11434`) provides access to 33 models including:
- **Code Generation**: `qwen3-coder:latest`, `devstral:latest`, `qwen2.5-coder:7b`
- **General Purpose**: `gemma3:27b`, `mistral-small3.2:latest`, `qwen3:30b`
- **Vision Models**: `qwen2.5vl:3b`
- **Embeddings**: `nomic-embed-text:latest`, `bge-m3:latest`
- **Specialized**: `magistral:latest`, `cogito:32b`

## Security Considerations

### Sandboxing Benefits
1. **Limited Attack Surface**: Only necessary ports are accessible
2. **Service Isolation**: Database and system service ports are blocked
3. **Controlled Internet Access**: Only HTTP/HTTPS protocols allowed
4. **Container Isolation**: Each instance has isolated network namespace

### Risk Mitigation
1. **No SSH Access**: Prevents unauthorized shell access
2. **No Database Access**: Prevents data exfiltration attempts  
3. **Protocol Restrictions**: Only web protocols allowed outbound
4. **Container Network**: Isolated from host network stack

## Configuration Examples

### Testing Network Access
From within a container, you can test connectivity:

```bash
# Test internet access
curl -s --connect-timeout 3 http://google.com

# Test AI service APIs
curl -s --connect-timeout 2 https://api.openai.com/v1/models

# Test Ollama connectivity
curl -s http://192.168.178.37:11434/api/tags

# Test inter-container communication
curl -s http://172.18.0.5:8000/health
```

### Environment Variables
Key environment variables that affect network access:

```bash
# Ollama host configuration
OLLAMA_HOST=192.168.178.37,192.168.178.25,localhost
OLLAMA_HOST_AUTO_DOWNLOAD_MODELS=192.168.178.37,192.168.178.25,localhost

# API Keys (configured in .env)
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
```

## Troubleshooting

### Common Issues

#### Ollama Connectivity
```bash
# Check Ollama host reachability
timeout 3 curl -s http://192.168.178.37:11434/api/tags

# Expected: JSON response with model list
# If timeout: Host is unreachable or Ollama service is down
```

#### AI Service Access
```bash
# Test OpenAI API access (expect 401 without API key)
curl -s --connect-timeout 2 https://api.openai.com/v1/models
# Expected: 401 Unauthorized (confirms connectivity)
```

#### Inter-Container Communication
```bash
# From one instance, test another
curl -s http://172.18.0.4:7860/api/llms
# Expected: JSON response with LLM configuration
```

### Network Diagnostics
From within a container:

```bash
# Check container IP and routing
ip addr show eth0
ip route show

# Test DNS resolution
nslookup api.openai.com

# Test port connectivity (replace HOST:PORT)
timeout 2 bash -c "echo > /dev/tcp/HOST/PORT" && echo "Open" || echo "Blocked"
```

## Security Best Practices

1. **Minimal Access**: Only grant necessary network permissions
2. **Environment Isolation**: Use separate .env files for different environments
3. **API Key Security**: Never log or expose API keys in container output
4. **Network Monitoring**: Monitor unusual outbound connections
5. **Regular Updates**: Keep container images and dependencies updated

## Related Documentation

- [Docker Multi-Container Setup](DOCKER_MULTI_CONTAINER.md) - Container orchestration
- [Architecture Overview](ARCHITECTURE.md) - System architecture
- [Ollama Integration](OLLAMA_INFO.md) - Local model configuration
- [Permission System](PERMISSION_SYSTEM.md) - File system security