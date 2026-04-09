# Infrastructure Status

## Deployment Modes

### 1. Local Development (Stable ✅)
**Status:** Production-ready, actively used

```bash
python main.py              # CLI mode
python main.py --gui        # Web interface (localhost)
python main.py --api        # API server only
python main.py --debug      # Verbose logging
```

Use for local development, personal use, and single-session workflows.

---

### 2. Isolated Mode (Stable ✅)
**Status:** Production-ready

```bash
# First run (builds cached image, ~2-3 min)
python main.py --isolated

# Subsequent runs (~2-5 sec)
python main.py --isolated -m "Your message here"
python main.py --isolated --gui
```

Each `--isolated` run gets a fully isolated Docker container.  
**See:** [ISOLATED_MODE.md](ISOLATED_MODE.md) for details.

---

## Infrastructure Services

### Rate Limiting (`infrastructure/rate_limiting/`)
**Status:** Active, used by all LLM providers

Tracks per-provider request rates to avoid hitting API limits. Integrated into `core/llm_router.py`.

---

## GitHub Actions

### `deploy-local-runner.yml` (Active)
- **Trigger:** Push/PR to `main`, manual dispatch
- **Runner:** Self-hosted
- **Actions:** Runs `python run_tests.py --quick` + linting, builds Docker images

### `deploy-production.yml.disabled` / `deploy-staging.yml.disabled`
Both disabled — not used.

---

## Recommendations

- **Local development:** `python main.py --gui`
- **Secure/isolated execution:** `python main.py --isolated`
- **Production (single-user):** `python main.py --api` behind Nginx/Caddy with SSL and auth
