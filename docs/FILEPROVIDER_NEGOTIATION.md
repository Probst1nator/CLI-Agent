# File Provider Negotiation Architecture

## Problem Statement

Currently, the `getuserfiles.py` tool uses a hardcoded WebSocket URI (`ws://localhost:8765`) to communicate with the File Copier GUI. This creates several issues:

- **Port conflicts**: If port 8765 is unavailable, the entire tool fails
- **Inflexibility**: No way to run multiple instances or use different ports  
- **Connection failures**: Tool gets excluded from documentation when server isn't reachable
- **Deployment issues**: Different environments may require different port configurations

## Current Implementation

```python
# tools/getuserfiles.py - Line 7
WEBSOCKET_URI = "ws://localhost:8765"
```

The `_check_getuserfiles_available()` function in `main.py` tests this hardcoded connection and excludes the tool if it fails.

## Proposed Dynamic Negotiation Architecture

### Multi-tier Discovery Approach

1. **Environment Variable** (Highest Priority)
   ```bash
   export GETUSERFILES_PORT=8766
   export GETUSERFILES_HOST=192.168.1.100  # Optional, defaults to localhost
   ```

2. **Configuration File** (Medium Priority)
   ```json
   # ~/.cli-agent/getuserfiles.json
   {
     "port": 8766,
     "host": "localhost",
     "protocol": "ws",
     "timeout": 5
   }
   ```

3. **Port Range Scanning** (Fallback)
   - Try ports 8765-8769 sequentially
   - Test connection + basic handshake
   - Cache successful port for session

4. **Service Discovery** (Advanced Option)
   - mDNS/Bonjour service advertisement
   - Service name: `_getuserfiles._tcp.local.`

### Implementation Strategy

```python
async def discover_websocket_endpoint():
    """Discover WebSocket endpoint using multiple strategies"""
    
    # Strategy 1: Environment variables
    if port := os.getenv('GETUSERFILES_PORT'):
        host = os.getenv('GETUSERFILES_HOST', 'localhost')
        return f"ws://{host}:{port}"
    
    # Strategy 2: Config file
    config_path = Path.home() / '.cli-agent' / 'getuserfiles.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return f"ws://{config['host']}:{config['port']}"
    
    # Strategy 3: Port scanning
    for port in range(8765, 8770):
        if await test_connection(f"ws://localhost:{port}"):
            return f"ws://localhost:{port}"
    
    raise ConnectionError("No getuserfiles server found")

async def test_connection(uri: str) -> bool:
    """Test if server responds correctly to getuserfiles protocol"""
    try:
        async with websockets.connect(uri, open_timeout=1) as ws:
            content = await asyncio.wait_for(ws.recv(), timeout=2)
            return content and not content.startswith("Error:")
    except:
        return False
```

## Integration Points

### Client Side (getuserfiles.py)
- Replace hardcoded URI with discovery function
- Cache successful connection for session
- Graceful fallback and error reporting

### Server Side (File Copier GUI)
- Write active port/host to config file on startup
- Optionally support environment variable configuration
- Consider service advertisement for advanced discovery

### System Integration (main.py)
- Update `_check_getuserfiles_available()` to use discovery
- Add configuration management utilities
- Support runtime reconfiguration

## Benefits

1. **Flexibility**: Run on any available port
2. **Multi-instance**: Support multiple GUI instances
3. **Environment-specific**: Different configs for dev/prod
4. **Robustness**: Automatic fallback mechanisms
5. **Caching**: Avoid repeated discovery overhead

## Migration Strategy

### Phase 1: Backward Compatible
- Add discovery logic with 8765 as fallback
- Existing setups continue working unchanged

### Phase 2: Enhanced Configuration  
- Add config file support
- Environment variable support
- Port range scanning

### Phase 3: Advanced Features
- Service discovery (optional)
- Multi-host support
- Connection pooling

## Team Discussion Points

1. **Server-side changes needed**: Can the File Copier GUI write port info to a config file?
2. **Port range preferences**: What port range is acceptable for scanning?
3. **Configuration location**: Is `~/.cli-agent/` appropriate for config files?
4. **Protocol extensions**: Any additional handshake/validation needed?
5. **Error handling**: How should connection failures be communicated to users?

## Implementation Priority

**Immediate**: Environment variable + config file support
**Short-term**: Port range scanning
**Long-term**: Service discovery (if needed)

This architecture eliminates the hardcoded dependency while maintaining simplicity and backward compatibility.