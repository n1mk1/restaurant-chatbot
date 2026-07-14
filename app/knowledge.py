"""Runtime access to the curated knowledge base under ``knowledge/``.

The knowledge base is split by audience, and that split is the retrieval
security boundary:

- ``knowledge/public/`` — guest-facing documents (``menu.md``, ``pricing.md``,
  ``dietary_restrictions.md``, ``faq.md``, ``reservations.md``,
  ``preset_answers.json``). Only these are eligible for semantic retrieval in
  :mod:`app.rag`; anything placed here may be quoted to a guest verbatim.
- ``knowledge/agent/`` — internal operating documents (``instructions.md``,
  ``persona_tone.md``, ``policies.md``, ``selling_script.md``). These are never
  indexed or retrievable. ``persona_tone.md`` is read only by
  :func:`persona_offtopic` to brief the off-topic model; the rest are honoured
  deterministically in code: ``policies.md`` → ``policy_info`` /
  ``allergen_info`` + the booking confirm flow, ``selling_script.md`` →
  ``menu_info`` recommendation + beverage rule, ``instructions.md`` → routing +
  ``compose_response`` guardrails.
"""

from functools import cache
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_SOURCE_KNOWLEDGE_DIR = _APP_DIR.parent / "knowledge"
_PACKAGED_KNOWLEDGE_DIR = _APP_DIR / "data" / "knowledge"
KNOWLEDGE_DIR = (
    _SOURCE_KNOWLEDGE_DIR
    if _SOURCE_KNOWLEDGE_DIR.is_dir()
    else _PACKAGED_KNOWLEDGE_DIR
)
PUBLIC_KNOWLEDGE_DIR = KNOWLEDGE_DIR / "public"
AGENT_KNOWLEDGE_DIR = KNOWLEDGE_DIR / "agent"


def _read(directory: Path, name: str) -> str:
    """Read one file from one knowledge directory; no traversal outside it."""
    if Path(name).name != name:
        return ""
    try:
        return (directory / name).read_text(encoding="utf-8")
    except OSError:
        return ""


@cache
def load_public(name: str) -> str:
    """Raw text of a guest-facing knowledge file, or ``""`` if unavailable."""
    return _read(PUBLIC_KNOWLEDGE_DIR, name)


@cache
def load_agent(name: str) -> str:
    """Raw text of an internal agent knowledge file, or ``""`` if unavailable."""
    return _read(AGENT_KNOWLEDGE_DIR, name)


@cache
def public_documents() -> tuple[tuple[str, str], ...]:
    """(filename, text) for every guest-facing markdown document.

    This enumeration — not a per-file allowlist — defines what retrieval can
    see: dropping a new ``.md`` into ``knowledge/public/`` makes it
    retrievable, and nothing under ``knowledge/agent/`` is ever returned.
    """
    try:
        names = sorted(path.name for path in PUBLIC_KNOWLEDGE_DIR.glob("*.md"))
    except OSError:
        return ()
    documents = tuple(
        (name, text) for name in names if (text := load_public(name)).strip()
    )
    return documents


def _section(markdown: str, heading: str) -> str:
    """Extract the body of a single ``## Heading`` section from a markdown doc."""
    capturing = False
    collected: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            capturing = line[3:].strip().casefold() == heading.casefold()
            continue
        if line.startswith("# "):
            capturing = False
            continue
        if capturing:
            collected.append(line)
    return "\n".join(collected).strip()


@cache
def persona_offtopic() -> str:
    """Condensed voice + off-topic guidance from ``agent/persona_tone.md``.

    Fed to the model when steering an off-topic message back to the restaurant.
    Falls back to a compact built-in brief if the file cannot be read, so the
    off-topic path never depends on the file being present.
    """
    doc = load_agent("persona_tone.md")
    sections = [
        _section(doc, "Core Voice"),
        _section(doc, "Off-Topic Requests"),
    ]
    guidance = "\n".join(part for part in sections if part).strip()
    if guidance:
        return guidance
    return (
        "- Use a warm, calm, concise neighbourhood-bistro voice; sound conversational without "
        "pretending to be a human staff member.\n"
        "- Redirect unrelated questions politely back to Maple & Ember's menu, restaurant "
        "information, dietary help, or planning a reservation."
    )
