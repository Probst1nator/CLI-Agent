# Deprecated Docker Configurations

This directory contains Docker configurations that have been superseded by newer, improved implementations.

## Deprecated Files

### `docker-compose.dev.yml`
- **Deprecated**: September 25, 2025
- **Reason**: Superseded by `docker-compose.multi-dev.yml`
- **Original Purpose**: Single-instance development with live code mounting
- **Replacement**: Use `docker-compose.multi-dev.yml` for multi-instance development with live code mounting

### `docker-compose.multi-instance.yml`
- **Deprecated**: September 25, 2025  
- **Reason**: Superseded by `docker-compose.multi-simple.yml`
- **Original Purpose**: Multi-instance setup using Traefik reverse proxy
- **Replacement**: Use `docker-compose.multi-simple.yml` for simpler session manager approach

### `traefik.yml`
- **Deprecated**: September 25, 2025
- **Reason**: Only used by deprecated `docker-compose.multi-instance.yml`
- **Original Purpose**: Traefik configuration for reverse proxy routing
- **Replacement**: Session manager approach handles routing internally

## Migration Guide

### From `docker-compose.dev.yml` to `docker-compose.multi-dev.yml`:
```bash
# Old single-instance development
docker compose -f docker_files/deprecated/docker-compose.dev.yml up -d

# New multi-instance development  
docker compose -f docker_files/docker-compose.multi-dev.yml up -d
```

### From `docker-compose.multi-instance.yml` to `docker-compose.multi-simple.yml`:
```bash
# Old Traefik-based multi-instance
docker compose -f docker_files/deprecated/docker-compose.multi-instance.yml up -d

# New session manager-based multi-instance
docker compose -f docker_files/docker-compose.multi-simple.yml up -d
```

## Port Changes

| Setup | Old Ports | New Ports |
|-------|-----------|-----------|
| Single Dev | 7862 | 7871-7873 (multi-dev) |
| Multi-Instance | 7860, 8080 | 7860-7863 (simple) or 7870-7873 (dev) |

## Why These Were Deprecated

1. **Complexity Reduction**: Traefik setup was more complex than needed
2. **Development Consistency**: Multi-instance development matches production architecture  
3. **Session Management**: Custom session manager provides better control than Traefik
4. **Port Conflicts**: Better port organization with dedicated dev/prod ranges

## Safe to Remove

These files can be safely removed after confirming no active deployments use them:

```bash
# Check for active containers using old configs
docker ps --filter "label=com.docker.compose.project=deprecated"

# If no active containers, remove deprecated files
rm -rf docker_files/deprecated/
```