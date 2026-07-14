"""Semantic retrieval over the guest-facing knowledge base.

Two retrieval layers, both grounded in ``knowledge/public/`` and nothing else:

- Curated preset answers (``preset_answers.json``): high-precision Q→A pairs
  matched by example paraphrases. Checked first.
- Document sections: every ``## section`` of every markdown file in
  ``knowledge/public/`` (menu, pricing, dietary restrictions, FAQ,
  reservations) becomes a retrievable chunk with source attribution.

The deterministic graph remains authoritative for menu facts, preferences,
and allergen safety; retrieval only supplies a semantic catch-all when a
message does not match one of those explicit paths. Agent-internal documents
under ``knowledge/agent/`` (instructions, persona, policies, selling script)
are structurally outside both indexes — the stores are built from
:func:`app.knowledge.public_documents` / :func:`app.knowledge.load_public`,
which never read that directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from functools import cache
from typing import Any

from app.knowledge import load_public, public_documents
from app.preferences import normalize
from app.restaurant import MENU, RESTAURANT, format_item

logger = logging.getLogger(__name__)

_EMBEDDING_DIMENSION = 256
_DEFAULT_MIN_SIMILARITY = 0.43
_WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")
_SUPPORTED_PLACEHOLDERS = {
    "phone",
    "vegetarian_items",
    "vegan_items",
    "meat_items",
}
_SEMANTIC_DOMAIN_WORDS = frozenset({
    "allergen", "allergens", "accommodate", "certification", "diet", "dietary", "dish", "dishes",
    "eat", "eating", "food", "gluten", "halal", "handle", "keto", "kosher", "meat", "menu", "needs",
    "need", "options", "paleo", "preference", "preferences", "restriction", "restrictions", "restaurant",
    "vegan", "vegetarian", "animal", "animals", "carnivore", "carnivorous", "meal", "meals", "order", "orders",
})


@dataclass(frozen=True, slots=True)
class PresetAnswer:
    """A curated answer and the semantic examples that should retrieve it."""

    id: str
    intent: str
    kind: str
    examples: tuple[str, ...]
    answer: str


class HashEmbeddingFunction:
    """Small deterministic local embeddings for offline semantic retrieval.

    Token and character n-gram features make paraphrases useful without
    downloading an embedding model at startup.
    """

    def __init__(self, dimension: int = _EMBEDDING_DIMENSION):
        self.dimension = dimension

    def name(self) -> str:
        return "maple-ember-hash-embeddings-v1"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(value) for value in input]

    def _embed(self, value: str) -> list[float]:
        normalized = normalize(value)
        tokens = _WORD_RE.findall(normalized)
        features: list[tuple[str, float]] = [(token, 2.0) for token in tokens]
        features.extend(
            (f"phrase:{tokens[index]}_{tokens[index + 1]}", 2.5)
            for index in range(len(tokens) - 1)
        )
        for token in tokens:
            padded = f"^{token}$"
            features.extend(
                (f"ngram:{padded[index:index + size]}", 0.35)
                for size in (3, 4)
                for index in range(len(padded) - size + 1)
            )

        vector = [0.0] * self.dimension
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


def _records() -> tuple[PresetAnswer, ...]:
    try:
        raw = json.loads(load_public("preset_answers.json"))
    except (TypeError, ValueError):
        logger.exception("Could not parse knowledge/public/preset_answers.json")
        return ()
    if not isinstance(raw, list):
        logger.error("knowledge/public/preset_answers.json must contain a list")
        return ()

    records: list[PresetAnswer] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("id")
        intent = entry.get("intent")
        kind = entry.get("kind")
        answer = entry.get("answer")
        examples = entry.get("examples")
        if (
            not isinstance(identifier, str)
            or not isinstance(intent, str)
            or not isinstance(kind, str)
            or not isinstance(answer, str)
            or not isinstance(examples, list)
        ):
            logger.warning("Skipping malformed preset answer record: %r", entry)
            continue
        cleaned_examples = tuple(example.strip() for example in examples if isinstance(example, str) and example.strip())
        if not cleaned_examples:
            logger.warning("Skipping preset answer without examples: %s", identifier)
            continue
        unknown_placeholders = {
            placeholder
            for placeholder in re.findall(r"{([a-z_]+)}", answer)
            if placeholder not in _SUPPORTED_PLACEHOLDERS
        }
        if unknown_placeholders:
            logger.warning("Skipping preset answer with unknown placeholders: %s", identifier)
            continue
        records.append(PresetAnswer(identifier, intent, kind, cleaned_examples, answer))
    return tuple(records)


def _render_menu_items(items: tuple[Any, ...] | list[Any]) -> str:
    return "\n".join(f"- {format_item(item)}" for item in items) or "- No matching listed items."


def render_preset(preset: PresetAnswer) -> str:
    """Render a preset template against the current, authoritative menu data."""
    groups = {
        "phone": RESTAURANT["phone"],
        "vegetarian_items": _render_menu_items([item for item in MENU if item.vegetarian]),
        "vegan_items": _render_menu_items([item for item in MENU if item.vegan]),
        "meat_items": _render_menu_items([item for item in MENU if not item.vegetarian]),
    }
    rendered = preset.answer
    for placeholder, value in groups.items():
        rendered = rendered.replace("{" + placeholder + "}", value)
    return rendered


class SemanticPresetStore:
    """Exact in-memory cosine index containing each preset example."""

    def __init__(self, records: tuple[PresetAnswer, ...] | None = None):
        self.records = records if records is not None else _records()
        self._by_id = {record.id: record for record in self.records}
        self._embedding = HashEmbeddingFunction()
        examples = [
            (record.id, example)
            for record in self.records
            for example in record.examples
        ]
        embeddings = self._embedding([example for _, example in examples])
        self._index = tuple(
            (preset_id, tuple(embedding))
            for (preset_id, _), embedding in zip(examples, embeddings, strict=True)
        )

    @property
    def document_count(self) -> int:
        """Number of indexed examples."""
        return len(self._index)

    def search(self, query: str, *, min_similarity: float = _DEFAULT_MIN_SIMILARITY) -> PresetAnswer | None:
        query_tokens = set(_WORD_RE.findall(normalize(query)))
        if not query_tokens & _SEMANTIC_DOMAIN_WORDS or not self._index:
            return None
        query_embedding = self._embedding([query])[0]
        preset_id, similarity = max(
            (
                (preset_id, sum(left * right for left, right in zip(query_embedding, embedding, strict=True)))
                for preset_id, embedding in self._index
            ),
            key=lambda match: match[1],
        )
        if similarity < min_similarity:
            return None
        return self._by_id.get(preset_id)


@cache
def default_preset_store() -> SemanticPresetStore:
    """Build the process-local semantic index once, on the first fallback query."""
    return SemanticPresetStore()


def retrieve_preset(query: str) -> PresetAnswer | None:
    """Return the best safe preset, or ``None`` when semantic confidence is low."""
    try:
        return default_preset_store().search(query)
    except Exception:
        logger.exception("Semantic preset retrieval failed")
        return None


# --- Guest-document retrieval ------------------------------------------------

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Generic English filler excluded from document scoring so a shared "the" or
# "please" cannot qualify a query for document retrieval on its own.
_DOC_STOPWORDS = frozenset({
    "about", "after", "all", "also", "and", "any", "are", "ask", "asks", "before", "below",
    "between", "both", "but", "call", "can", "cannot", "come", "comes", "confirm",
    "confirmed", "current", "currently", "does", "each", "ember", "every", "for", "from",
    "get", "give", "has", "have", "here", "how", "into", "its", "just", "know", "like",
    "listed", "maple", "may", "more", "most", "much", "need", "not", "off", "one", "only",
    "other", "our", "out", "per", "please", "provided", "really", "run", "runs", "say",
    "says", "see", "should", "some", "such", "tell", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "under", "until", "use",
    "used", "using", "very", "want", "wanted", "was", "what", "when", "where", "whether",
    "which", "while", "who", "why", "will", "with", "would", "yes", "you", "your",
})
_DOCUMENT_MIN_SCORE = 0.5
_HEADING_BONUS = 0.2


def _stem(token: str) -> str:
    """Tiny deterministic suffix-stemmer: booking/booked/books → book."""
    for suffix in ("ing", "ed", "es", "ly", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) >= 4 and token[-1] == token[-2]:
        token = token[:-1]
    if len(token) >= 4 and token.endswith("e"):
        token = token[:-1]
    return token


def _content_stems(text: str) -> set[str]:
    return {
        _stem(token)
        for token in _WORD_RE.findall(normalize(text))
        if len(token) >= 3 and token not in _DOC_STOPWORDS
    }


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """One retrievable ``## section`` of a guest-facing knowledge document."""

    source: str
    doc_title: str
    heading: str
    content: str

    @property
    def ref(self) -> str:
        return f"{self.source}#{self.heading}"


def _split_sections(name: str, text: str) -> list[DocumentSection]:
    cleaned = _COMMENT_RE.sub("", text)
    doc_title = name
    heading: str | None = None
    body: list[str] = []
    sections: list[DocumentSection] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if heading and content:
            sections.append(DocumentSection(name, doc_title, heading, content))
        body.clear()

    for line in cleaned.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
        elif line.startswith("# "):
            flush()
            doc_title = line[2:].strip() or name
            heading = None
        else:
            body.append(line)
    flush()
    return sections


class DocumentStore:
    """IDF-weighted token-coverage index over guest-document sections.

    Hashed cosine works for the short preset examples but is too noisy across
    multi-paragraph sections, so documents are scored by how much of the
    query's content vocabulary (stemmed, stopword-free, rarity-weighted) a
    section actually contains, plus a bonus when the section heading itself is
    hit. Unknown query words count against coverage, so "what is the meaning
    of life" cannot ride in on one incidental shared token.
    """

    def __init__(self, sections: tuple[DocumentSection, ...] | None = None):
        if sections is None:
            sections = tuple(
                section
                for name, text in public_documents()
                for section in _split_sections(name, text)
            )
        self.sections = sections
        self._section_stems = [
            _content_stems(f"{section.doc_title} {section.heading} {section.content}")
            for section in sections
        ]
        self._heading_stems = [
            _content_stems(f"{section.doc_title} {section.heading}") for section in sections
        ]
        frequency: dict[str, int] = {}
        for stems in self._section_stems:
            for stem in stems:
                frequency[stem] = frequency.get(stem, 0) + 1
        total = max(len(self.sections), 1)
        self._idf = {
            stem: math.log((1 + total) / (1 + count)) + 1.0
            for stem, count in frequency.items()
        }
        self._unknown_idf = math.log(1 + total) + 1.0

    @property
    def section_count(self) -> int:
        return len(self.sections)

    def search(self, query: str, *, min_score: float = _DOCUMENT_MIN_SCORE) -> DocumentSection | None:
        query_stems = _content_stems(query)
        if not query_stems or not self.sections:
            return None
        query_weight = sum(self._idf.get(stem, self._unknown_idf) for stem in query_stems)
        best: tuple[float, int] | None = None
        for index, stems in enumerate(self._section_stems):
            hits = query_stems & stems
            if not hits:
                continue
            coverage = sum(self._idf[stem] for stem in hits) / query_weight
            score = coverage + (_HEADING_BONUS if query_stems & self._heading_stems[index] else 0.0)
            if best is None or score > best[0]:
                best = (score, index)
        if best is None or best[0] < min_score:
            return None
        return self.sections[best[1]]


@cache
def default_document_store() -> DocumentStore:
    """Build the process-local guest-document index once, on first use."""
    return DocumentStore()


def retrieve_document(query: str) -> DocumentSection | None:
    """Best guest-document section for the query, or ``None`` below confidence."""
    try:
        return default_document_store().search(query)
    except Exception:
        logger.exception("Guest-document retrieval failed")
        return None


def render_document_section(section: DocumentSection) -> str:
    """Present a retrieved section with its source named, verbatim."""
    return f"Here’s what the {section.doc_title} says about {section.heading.lower()}:\n\n{section.content}"
