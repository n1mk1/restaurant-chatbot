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
import tempfile
from dataclasses import dataclass
from functools import cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.knowledge import load
from app.preferences import normalize
from app.restaurant import MENU, RESTAURANT, format_item

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "restaurant_preset_answers_v1"
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
    """Small deterministic local embeddings for offline Chroma retrieval.

    Token and character n-gram features make paraphrases useful without
    downloading an embedding model at startup. Chroma still owns indexing and
    nearest-neighbour search; this function only defines the vectorization.
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
    """Chroma collection containing one document for each preset example."""

    def __init__(self, records: tuple[PresetAnswer, ...] | None = None):
        self.records = records if records is not None else _records()
        self._by_id = {record.id: record for record in self.records}
        self._embedding = HashEmbeddingFunction()
        # Chroma's default ephemeral client intentionally shares one in-memory
        # SQLite system across all clients in a process. A private temporary
        # persistent directory gives this index a stable lifecycle without
        # leaking state between workers or requiring a checked-in vector DB.
        self._storage = tempfile.TemporaryDirectory(
            prefix="maple-ember-chroma-",
            ignore_cleanup_errors=True,
        )
        self._client = chromadb.PersistentClient(
            path=self._storage.name,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedding,
            metadata={"hnsw:space": "cosine"},
        )
        self._document_count = 0
        if self.records:
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, str]] = []
            for record in self.records:
                for index, example in enumerate(record.examples):
                    ids.append(f"{record.id}:{index}")
                    documents.append(example)
                    metadatas.append({"preset_id": record.id, "intent": record.intent, "kind": record.kind})
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
            self._document_count = len(ids)

    @property
    def document_count(self) -> int:
        """Number of indexed examples, tracked without a database round trip."""
        return self._document_count

    def search(self, query: str, *, min_similarity: float = _DEFAULT_MIN_SIMILARITY) -> PresetAnswer | None:
        query_tokens = set(_WORD_RE.findall(normalize(query)))
        if not query_tokens & _SEMANTIC_DOMAIN_WORDS or self._document_count == 0:
            return None
        result = self._collection.query(query_texts=[query], n_results=1, include=["distances", "metadatas"])
        distances = result.get("distances") or [[]]
        metadatas = result.get("metadatas") or [[]]
        if not distances[0] or not metadatas[0] or not metadatas[0][0]:
            return None
        distance = float(distances[0][0])
        similarity = 1.0 - distance
        if similarity < min_similarity:
            return None
        preset_id = metadatas[0][0].get("preset_id")
        return self._by_id.get(preset_id)


@cache
def default_preset_store() -> SemanticPresetStore:
    """Build the process-local Chroma index once, on the first fallback query."""
    return SemanticPresetStore()


def retrieve_preset(query: str) -> PresetAnswer | None:
    """Return the best safe preset, or ``None`` when semantic confidence is low."""
    try:
        return default_preset_store().search(query)
    except Exception:
        logger.exception("Semantic preset retrieval failed")
        return None
