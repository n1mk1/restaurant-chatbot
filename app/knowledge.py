"""Runtime access to the curated knowledge base under ``knowledge/``.

R-4 decision: the knowledge files are kept, and their governance is made
explicit here so they are not a spec with zero runtime effect.

- ``persona_tone.md`` is *loaded at runtime* and injected into the off-topic
  model prompt (see :func:`persona_offtopic`), so the documented voice actually
  governs the one place the model speaks freely.
- The factual files are enforced deterministically in code; :data:`ENFORCED_BY`
  records which path owns each, so a future maintainer can see the mapping
  instead of assuming the files are dead.
"""

from functools import lru_cache
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

# Maps each knowledge file to the runtime code path that enforces it.
ENFORCED_BY: dict[str, str] = {
    "pricing.md": "app.graph.menu_info + app.restaurant.format_item (prices, dietary flags, allergens)",
    "faq.md": "app.graph.restaurant_info / policy_info (hours, location, parking, payment, services)",
    "policies.md": "app.graph.policy_info / allergen_info (allergens, substitutions, fees, specials)",
    "selling_script.md": "app.graph.menu_info recommendation path + beverage rule",
    "instructions.md": "app.graph deterministic routing + compose_response output guardrails",
    "persona_tone.md": "loaded into the off-topic model prompt via persona_offtopic()",
}


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
