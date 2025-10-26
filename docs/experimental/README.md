# Experimental Documentation

**Status:** EXPERIMENTAL - Not Production Ready

This directory contains documentation for experimental features and deployment modes that are:
- Still under development
- Not fully tested in production
- Subject to significant changes
- May have incomplete implementations

## Use These Docs If:

- You're exploring advanced deployment options
- You want to understand multi-user architecture designs
- You're contributing to experimental features
- You need historical context for architectural decisions

## Production-Ready Alternatives

For production deployments, use the stable documentation in the main `docs/` directory:

- **Single-user local**: Run `python main.py` (see `CLAUDE.md`)
- **Isolated runs**: Run `python main.py --isolated` (see `docs/ISOLATED_MODE.md`)
- **Production deployment**: See `docs/INFRASTRUCTURE_STATUS.md` for current recommendations

## Files in This Directory

### Multi-User Infrastructure (EXPERIMENTAL)
- `MULTI_USER_ARCHITECTURE.md` - Multi-tenant SaaS architecture design
- `QUICK_START_MULTI_USER.md` - Quick start guide for multi-user deployment
- `MULTI_USER_IMPLEMENTATION_SUMMARY.md` - Implementation details and decisions
- `MULTI_SESSION_SETUP.md` - Session manager with multiple instances

### Docker Deployments (EXPERIMENTAL)
- `DOCKER_PERSISTENT_SETUP.md` - Always-on containerized deployment with systemd
- `DEVELOPMENT_WORKFLOW.md` - Development workflows for multi-container setup

### Historical Records
- `HISTORY_BRANCH_CLEANUP.md` - Git branch cleanup history (2025-10-16)

## Status Indicators

| Symbol | Meaning |
|--------|---------|
| EXPERIMENTAL | Feature exists but not production-ready |
| HISTORICAL | Archived for reference only |
| PROPOSED | Design document, not yet implemented |

## Important Notes

1. **Multi-user infrastructure** (Traefik + Container Manager) is experimental
   - Last tested: ~4 weeks ago (per `INFRASTRUCTURE_STATUS.md`)
   - Requires significant testing before production use
   - See `docs/INFRASTRUCTURE_STATUS.md` for current status

2. **These docs may be outdated**
   - Always check `docs/INFRASTRUCTURE_STATUS.md` for latest guidance
   - Refer to `CLAUDE.md` for current architecture
   - Verify referenced scripts/files exist before following instructions

3. **Primary deployment mode is single-user + isolated**
   - `python main.py` for local development
   - `python main.py --isolated` for containerized isolation
   - Multi-user deployments are experimental

## Contributing

If you're working on experimental features:
- Update this README when adding new experimental docs
- Mark all experimental docs with status banners
- Link to stable alternatives in main docs
- Update `docs/INFRASTRUCTURE_STATUS.md` when features stabilize

## Questions?

- Production deployments: See `docs/INFRASTRUCTURE_STATUS.md`
- Development setup: See `CLAUDE.md`
- Feature status: See `docs/INFRASTRUCTURE_STATUS.md`
