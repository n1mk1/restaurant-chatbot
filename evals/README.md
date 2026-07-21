# Evals

Black-box evaluation of the running service over HTTP, complementary to the
in-process pytest suite. Pytest proves the cases we thought of; these measure
behaviour on phrasings we did not write while writing the router.

Requires Node (for `npx`); no install step and no Python dependency changes.

## 1. Start a server tuned for eval runs

Each eval case sends `message` with no `session_id`, so the server mints a fresh
session per request. With production defaults (`MAX_ACTIVE_SESSIONS=1000`,
`SESSION_TTL_SECONDS=1800`) sessions accumulate for 30 minutes and repeated runs
will eventually return HTTP 429 that looks like a routing failure. Raise the cap
and shorten the TTL for eval runs:

```powershell
$env:CHAT_PROVIDER="deterministic"
$env:MAX_ACTIVE_SESSIONS="50000"
$env:SESSION_TTL_SECONDS="120"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 2. Routing accuracy

```powershell
npx promptfoo@latest eval -c evals/promptfooconfig.yaml --no-cache -j 4
npx promptfoo@latest view    # web UI with the pass/fail matrix
```

`routing_cases.csv` is the labeled corpus: one guest phrasing per row plus the
intent it should route to. The config asserts three things per row — the
returned `intent` matches the label, any allergen answer carries its
cross-contact caveat, and the reply is non-empty.

**Triage every failure into one of two buckets.** Some are router gaps; some are
rows where the label is wrong and the router is right (a dietary filter
answering from `menu` with the preference saved is defensible). Fix the router
for the first, fix the CSV for the second. The corpus is only useful if its
labels are trustworthy.

Grow the corpus from sources independent of `app/intents.py` — real guest
phrasings, or a model asked to roleplay guests. Phrasings written while reading
the router only re-confirm what it already does.

## 3. Domain red-team

```powershell
npx promptfoo@latest redteam generate -c evals/redteam.yaml -o evals/redteam.generated.yaml
npx promptfoo@latest redteam run -c evals/redteam.generated.yaml
npx promptfoo@latest redteam report
```

`redteam.yaml` carries `session_id` across turns via `sessionParser`, so the
multi-turn strategies (`crescendo`, `goat`) attack a real conversation rather
than a series of cold requests. That matters here: the plausible attack is
gradually talking the router into dropping a saved allergy over several turns,
not a single-shot jailbreak.

The custom `policy` plugins carry the domain rules. The built-in packs probe
model weaknesses this bot mostly does not have — its routing is deterministic
and its one free-form call is bounded by `app/offtopic.py`.

Generation needs a model; `redteam.provider` points at local Ollama so the run
stays offline. Remove that line to use a hosted generator instead.

## Notes

- `results.json`, `redteam.generated.yaml`, and `.promptfoo/` are run artifacts,
  not source — keep them out of git.
- Treat routing accuracy as a tracked number with a floor, not a pass/fail gate.
  The useful signal is the confusion matrix: which intents bleed into which.
