# Maple & Ember Restaurant Chatbot

A deployable FastAPI restaurant concierge with a responsive browser interface, LangGraph orchestration, bounded conversation sessions, and a local Qwen model served by Ollama.

## Docker quick start

Requirements: Docker Desktop or Docker Engine with Compose, at least 6 GB of free memory (8 GB recommended), and roughly 4 GB of free disk space.

```powershell
docker compose up --build -d
docker compose logs -f model-init api
```

The first start downloads `qwen3:4b` (about 2.5 GB) into a persistent Docker volume. Later starts reuse it. Once the services are healthy, open:

- Web interface: `http://127.0.0.1:8000/`
- API documentation: `http://127.0.0.1:8000/docs`
- Liveness: `http://127.0.0.1:8000/health`
- Model readiness: `http://127.0.0.1:8000/ready`

Check or stop the stack with:

```powershell
docker compose ps
docker compose down
```

`docker compose down` preserves the downloaded model. `docker compose down -v` also deletes the model volume, so the next start must download Qwen again.

### NVIDIA GPU

The default Compose configuration runs on CPU. On Linux, or Windows with Docker Desktop, WSL2, current NVIDIA drivers, and GPU container support, use:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

Only the FastAPI service publishes a host port. Ollama stays on the private Compose network.

## Local Python development

Python 3.10-3.14 is supported; Python 3.12 or 3.13 is recommended for deployment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
ollama pull qwen3:4b
python -m uvicorn app.main:app --reload
```

The default `CHAT_PROVIDER=ollama` uses `qwen3:4b` at `http://127.0.0.1:11434`. Set `CHAT_PROVIDER=deterministic` for rule-based replies without a model, or `CHAT_PROVIDER=openai` plus `OPENAI_API_KEY` for the hosted alternative.

## Configuration

Compose supports these environment variables, either in the shell or in a local `.env` file:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_PORT` | `8000` | Published web/API port |
| `OLLAMA_MODEL` | `qwen3:4b` | Model pulled by the initialization service |
| `OLLAMA_REASONING` | `false` | Disable hidden reasoning for faster factual replies |
| `OLLAMA_KEEP_ALIVE` | `15m` | Keep Qwen loaded between requests |
| `MAX_TURNS_PER_SESSION` | `20` | Successful user turns per conversation |
| `SESSION_TTL_SECONDS` | `1800` | Idle expiry in seconds |
| `MAX_ACTIVE_SESSIONS` | `1000` | In-process active-session cap |
| `MAX_HISTORY_MESSAGES` | `16` | Retained user/assistant messages |
| `MAX_MESSAGE_CHARS` | `2000` | Maximum input length |

The browser stores only the current session identifier and quota metadata. Conversation history remains in the API process and is lost when that container restarts.

## Deployment notes

- Deploy to a Docker-capable VM or container host with persistent volumes. Typical serverless platforms are unsuitable for a persistent local 2.5 GB model.
- Put a TLS reverse proxy or managed load balancer in front of port 8000.
- Add authentication plus account/IP rate limiting before exposing the service publicly. Per-session turn limits are not an abuse boundary.
- Keep one API worker and one API replica while sessions use the in-memory store. Add Redis-backed sessions before horizontal scaling.
- Do not expose Ollama's port `11434` publicly.
- Back up or preserve the `restaurant-chatbot_ollama-data` volume if avoiding a model re-download matters.

## Session behavior

The turn that reaches the configured limit succeeds and reports zero remaining. Additional turns return HTTP 429 without calling the model. Expired sessions return HTTP 410; unknown or deleted sessions return HTTP 404. Successful requests refresh the idle expiry. Same-session requests are serialized, and a repeated `request_id` is idempotent.

## Tests

```powershell
python -m pytest
```

The offline suite covers graph routing, static frontend delivery, session memory and expiry, validation, idempotency, and concurrent limit enforcement.
