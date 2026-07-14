"""Runtime access to the curated knowledge base under ``knowledge/``.

The knowledge files are kept and their runtime governance is explicit, so they
are not a spec with zero effect:

- ``persona_tone.md`` is loaded at runtime and injected into the off-topic model
  prompt (see :func:`persona_offtopic`) — the documented voice governs the one
  place the model speaks freely.
- The factual files are honoured deterministically in code, so a maintainer can
  see where each is enforced rather than assume it is unused:
  ``pricing.md`` → ``menu_info`` + ``restaurant.format_item``; ``faq.md`` →
  ``restaurant_info`` / ``policy_info``; ``policies.md`` → ``policy_info`` /
  ``allergen_info``; ``selling_script.md`` → ``menu_info`` recommendation +
  beverage rule; ``instructions.md`` → routing + ``compose_response`` guardrails.
"""

from functools import lru_cache
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Return the raw text of a knowledge file, or ``""`` if it is unavailable."""
    try:
        return (KNOWLEDGE_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return ""


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


@lru_cache(maxsize=None)
def persona_offtopic() -> str:
    """Condensed voice + off-topic guidance from ``persona_tone.md``.

    Fed to the model when steering an off-topic message back to the restaurant.
    Falls back to a compact built-in brief if the file cannot be read, so the
    off-topic path never depends on the file being present.
    """
    doc = load("persona_tone.md")
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
