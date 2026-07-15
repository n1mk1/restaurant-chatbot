# Maple & Ember Restaurant Chatbot

A deployable FastAPI restaurant concierge with a responsive browser interface, LangGraph orchestration, bounded conversation sessions, and a local Qwen model served by Ollama.

The concierge is deliberately bounded to four guest needs: understanding the
current CAD-priced menu, managing dietary and allergen preferences for the
session, learning Maple & Ember's location/cuisine/atmosphere, and recording a
regular chat reservation after explicit confirmation. Recipe-level menu labels
cover vegan, vegetarian, gluten-free, pescatarian, plant-based, pork-free, and
alcohol-free requests. Certification- or nutrition-dependent needs remain
saved but are referred to staff rather than guessed.

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

## Chat slot booking

Guests can book a regular time slot entirely in chat. The flow is
deterministic and consent-gated:

1. The guest asks to book and supplies a day (Tuesday–Sunday) and an hourly
   time slot, in one message or across several turns. Slots are derived from
   the restaurant hours in `app/restaurant.py` — from opening until one hour
   before close.
2. The assistant collects the booking name and a contact phone number.
3. The assistant echoes the exact details — `confirm time slot Friday 7:00 PM
   for Jane Doe - 4165550123` — and waits.
4. Only an explicit confirmation reply appends the record to the CSV booking log
   (`BOOKINGS_CSV_PATH`, default `data/bookings.csv`) with an incrementing
   booking id and a UTC timestamp. `cancel` discards the pending request at
   any point, and nothing is ever written without the explicit confirmation.

The CSV columns are `booking_id, logged_at_utc, day, time, name, phone`.
Appends are serialized process-wide, replayed `request_id`s cannot double-log,
and the file contains guest names and phone numbers, so it is git-ignored and
persisted through the `bookings-data` volume in Compose. Recorded bookings
cannot be changed or cancelled in chat; those requests are directed to the
restaurant phone number.

## Knowledge base and retrieval

`knowledge/` is split by audience, and the split is the retrieval boundary:

- `knowledge/public/` — guest-facing documents that retrieval may quote
  verbatim: `menu.md`, `pricing.md`, `dietary_restrictions.md`, `faq.md`,
  `reservations.md`, and the curated `preset_answers.json`.
- `knowledge/agent/` — internal operating documents (`instructions.md`,
  `persona_tone.md`, `policies.md`, `selling_script.md`). These are never
  indexed or retrievable; they steer the assistant's behaviour in code, and
  `persona_tone.md` additionally briefs the optional off-topic model.

The graph keeps deterministic routing ahead of retrieval for menu facts,
allergens, preferences, reservations, and policy limits. When a message is
restaurant-related but does not match an explicit path, two retrieval layers
run in order:

1. **Curated presets** — a process-local cosine index over the example
   paraphrases in `knowledge/public/preset_answers.json`, answered from
   source-grounded templates filled with authoritative menu data.
2. **Guest documents** — every `## section` of every markdown file in
   `knowledge/public/` is indexed with IDF-weighted, stemmed token coverage.
   A confident match is quoted with its source named ("Here's what the
   Maple & Ember Menu Guide says about seasonal and local sourcing: …");
   low-confidence and off-topic queries fall back to the normal restaurant
   redirect instead of guessing.

To extend the assistant, drop a new guest-safe markdown file into
`knowledge/public/` (it becomes retrievable automatically) or add a preset
record; agent-side rules belong in `knowledge/agent/`, which retrieval can
never see. Both indexes build offline at startup — no embedding model is
downloaded.

## Configuration

Compose supports these environment variables, either in the shell or in a local `.env` file:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_PORT` | `8000` | Published web/API port |
| `API_PREFIX` | `/api/v1` | Versioned API route prefix used by both the server and browser |
| `OLLAMA_MODEL` | `qwen3:4b` | Model pulled by the initialization service |
| `OLLAMA_REASONING` | `false` | Disable hidden reasoning for faster factual replies |
| `OLLAMA_KEEP_ALIVE` | `15m` | Keep Qwen loaded between requests |
| `OLLAMA_TIMEOUT_SECONDS` | `20` | Per-request model timeout, greater than 0 and at most 120 seconds |
| `OLLAMA_OFFTOPIC_ENABLED` | `false` | Opt in to model-composed playful off-topic redirects; deterministic redirects avoid an otherwise cosmetic model call. |
| `OLLAMA_OFFTOPIC_MODEL` | *(empty)* | Optional small non-reasoning model (e.g. `llama3.2:3b`) used when model-composed off-topic redirects are enabled; empty reuses `OLLAMA_MODEL`. |
| `BOOKINGS_CSV_PATH` | `data/bookings.csv` | Append-only CSV of confirmed chat bookings (contains guest names and phone numbers) |
| `MAX_TURNS_PER_SESSION` | `20` | Successful user turns per conversation |
| `SESSION_TTL_SECONDS` | `1800` | Idle expiry in seconds |
| `MAX_ACTIVE_SESSIONS` | `1000` | In-process active-session cap |
| `MAX_HISTORY_MESSAGES` | `16` | Retained user/assistant messages |
| `MAX_MESSAGE_CHARS` | `2000` | Maximum input length |
| `LIMIT_WARNING_THRESHOLD` | `3` | Remaining-turn count at which the API and browser show a warning |

The browser receives these limits from the server-rendered page, so changing the
API prefix, turn limit, expiry, warning threshold, or message limit does not
leave the interface with stale hardcoded defaults. The readiness endpoint
checks the primary model and, when enabled, the optional off-topic Ollama model.

The browser stores only the current session identifier and quota metadata. The first chat request creates its session directly, avoiding a separate session-creation round trip. Conversation history remains in the API process and is lost when that container restarts.

## Deployment notes

- Deploy to a Docker-capable VM or container host with persistent volumes. Typical serverless platforms are unsuitable for a persistent local 2.5 GB model.
- Put a TLS reverse proxy or managed load balancer in front of port 8000.
- Add authentication plus account/IP rate limiting before exposing the service publicly. Per-session turn limits are not an abuse boundary.
- Keep one API worker and one API replica while sessions use the in-memory store. Add Redis-backed sessions before horizontal scaling.
- Do not expose Ollama's port `11434` publicly.
- Back up or preserve the `restaurant-chatbot_ollama-data` volume if avoiding a model re-download matters.
- Back up the `restaurant-chatbot_bookings-data` volume (or the file at `BOOKINGS_CSV_PATH`): it is the only record of confirmed chat bookings and contains guest contact details.

## Session behavior

The turn that reaches the configured limit succeeds and reports zero remaining. Additional turns return HTTP 429 without calling the model. Expired sessions return HTTP 410; unknown or deleted sessions return HTTP 404. Successful requests refresh the idle expiry. Same-session requests are serialized, and a repeated `request_id` is idempotent.

## Tests

```powershell
python -m pytest
python -m ruff check app tests
```

The offline suite covers graph routing, configured frontend delivery, session memory and expiry, validation, idempotency, concurrent limit enforcement, the chat booking flow (slot parsing, confirm/cancel, CSV logging, replay safety), and the retrieval boundary that keeps `knowledge/agent/` documents out of the guest-facing indexes. Ruff checks imports, dead references, modernization, and common correctness issues.
