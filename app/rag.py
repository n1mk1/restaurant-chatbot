"""Semantic retrieval for curated, source-grounded preset answers.

The deterministic graph remains authoritative for menu facts, preferences, and
allergen safety. This module only supplies a semantic catch-all when a message
does not match one of those explicit paths. Preset examples and answer
templates live in ``knowledge/preset_answers.json`` so adding a paraphrase does
not require adding another prompt-specific branch to the graph.
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

from app.knowledge import load
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
        raw = json.loads(load("preset_answers.json"))
    except (TypeError, ValueError):
        logger.exception("Could not parse knowledge/preset_answers.json")
        return ()
    if not isinstance(raw, list):
        logger.error("knowledge/preset_answers.json must contain a list")
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
