# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run server
python main.py
python main.py --port 9000
python main.py --host 127.0.0.1 --port 9000

# Install dependencies
pip install -r requirements.txt

# Tests
pytest                                                          # all tests
pytest -v                                                       # verbose
pytest tests/unit/test_auth_manager.py -v                      # single file
pytest tests/unit/test_auth_manager.py::TestClass::test_name -v # single test
pytest tests/unit/ -v                                           # unit only
pytest --cov=kiro --cov-report=html                            # with coverage

# Docker
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## Architecture

Layered pipeline — each request flows through all layers:

```
Routes (routes_*.py)
  → Converters (converters_*.py)   # OpenAI/Anthropic format → Kiro payload
  → HTTP Client (http_client.py)   # retry logic, token refresh on 403
  → Streaming (streaming_*.py)     # Kiro SSE → OpenAI/Anthropic SSE
  → Parsers (parsers.py, thinking_parser.py)
```

Two parallel stacks share core logic:
- **OpenAI**: `routes_openai` → `converters_openai` → `streaming_openai`
- **Anthropic**: `routes_anthropic` → `converters_anthropic` → `streaming_anthropic`
- **Shared**: `converters_core.py` (UnifiedMessage format), `streaming_core.py` (KiroEvent)

### Key modules

| Module | Purpose |
|--------|---------|
| `config.py` | All env vars, constants, model lists |
| `auth.py` | Token lifecycle — 4 auth methods, auto-detect, asyncio.Lock |
| `model_resolver.py` | 4-layer resolution: normalize → cache → hidden → pass-through |
| `cache.py` | TTL cache for `/ListAvailableModels` response |
| `http_client.py` | Retry: 403→refresh token, 429/5xx→exponential backoff |
| `thinking_parser.py` | FSM to extract `<thinking>` blocks split across stream chunks |
| `parsers.py` | AWS binary SSE format → JSON events; handles bracket tool calls |

### Authentication auto-detection

- Has `clientId` + `clientSecret` in creds → **AWS SSO OIDC** (`oidc.{region}.amazonaws.com/token`)
- Otherwise → **Kiro Desktop Auth** (`prod.{region}.auth.desktop.kiro.dev/refreshToken`)

Credential sources (priority order): SQLite DB → JSON file → env vars (`REFRESH_TOKEN`)

### Model name normalization

```
claude-haiku-4-5-20251001  →  claude-haiku-4.5   (dash→dot, strip date)
claude-3-7-sonnet          →  claude-3.7-sonnet  (legacy)
claude-4.5-opus-high       →  claude-opus-4.5    (inverted format)
```

Unknown models are passed through to Kiro — the gateway is not a gatekeeper.

## Critical patterns

**Streaming must use per-request HTTP clients** — reusing the shared client causes CLOSE_WAIT leaks:
```python
# Correct
async with httpx.AsyncClient(timeout=timeout) as client:
    async with client.stream("POST", url, ...) as response: ...
```

**All tests must be network-isolated** — `conftest.py` has a global `block_all_network_calls` fixture that blocks all httpx requests. Tests that attempt real network calls will fail.

**Kiro's "Improperly formed request" error is vague** — it can mean wrong role order, invalid tool schema, name length violations, or undocumented constraints. Systematic testing is required to identify the actual cause.

**Tool calls from Kiro may arrive in bracket format** — `[{"name":...}]` instead of `{"name":...}`. `parsers.py` handles this.

## Code conventions

- Type hints mandatory on all function parameters and return values
- Google-style docstrings with Args/Returns/Raises
- loguru for all logging (`from loguru import logger`); INFO for business logic, DEBUG for technical details
- Never bare `except:` — catch specific exceptions
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants

## Configuration

All config in `.env` (see `.env.example`). Key vars:

```bash
PROXY_API_KEY="..."           # required — protects the proxy
KIRO_CREDS_FILE="~/.aws/sso/cache/kiro-auth-token.json"  # auth option 1
REFRESH_TOKEN="..."           # auth option 2
KIRO_CLI_DB_FILE="~/.local/share/kiro-cli/data.sqlite3"  # auth option 3
VPN_PROXY_URL="http://127.0.0.1:7890"  # optional proxy
DEBUG_MODE="off"              # off | errors | all
```

Debug logs written to `debug_logs/` directory.
