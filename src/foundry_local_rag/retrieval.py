"""Deterministic in-process retrieval over persisted chunk embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .embeddings import FoundryLocalEmbeddingAdapter
from .errors import RetrievalError
from .persistence import load_chunks


@dataclass(frozen=True)
class RetrievedChunk:
    """A persisted chunk paired with its query similarity score."""

    source_id: str
    chunk_index: int
    text: str
    score: float


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return cosine similarity for two compatible, non-zero finite vectors."""

    left_values = _validated_vector(left)
    right_values = _validated_vector(right)
    if len(left_values) != len(right_values):
        raise RetrievalError("Embedding dimensions must match for retrieval")

    left_scale = max(abs(value) for value in left_values)
    right_scale = max(abs(value) for value in right_values)
    if left_scale == 0 or right_scale == 0:
        raise RetrievalError("Cosine similarity is undefined for a zero-magnitude embedding")

    normalized_left = tuple(value / left_scale for value in left_values)
    normalized_right = tuple(value / right_scale for value in right_values)
    dot_product = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(normalized_left, normalized_right)
    )
    left_squared_sum = math.fsum(value * value for value in normalized_left)
    right_squared_sum = math.fsum(value * value for value in normalized_right)
    return dot_product / math.sqrt(left_squared_sum * right_squared_sum)


def retrieve_chunks(
    question: str,
    database_path: Path,
    embedding_adapter: FoundryLocalEmbeddingAdapter,
    top_k: int,
) -> tuple[RetrievedChunk, ...]:
    """Embed one question and return its most similar persisted chunks."""

    _validate_top_k(top_k)
    query_embedding = embedding_adapter.embed_text(question)
    chunks = load_chunks(database_path)

    results = tuple(
        RetrievedChunk(
            source_id=chunk.source_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            score=cosine_similarity(query_embedding, chunk.embedding),
        )
        for chunk in chunks
    )
    return tuple(
        sorted(
            results,
            key=lambda result: (-result.score, result.source_id, result.chunk_index),
        )[:top_k]
    )


def _validate_top_k(top_k: object) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalError("top_k must be a positive integer")


def _validated_vector(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise RetrievalError("Embedding vector must contain numeric values")
    try:
        vector = tuple(values)
    except TypeError as error:
        raise RetrievalError("Embedding vector must be an iterable") from error
    if not vector:
        raise RetrievalError("Embedding vector must not be empty")

    normalized: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RetrievalError("Embedding vector values must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise RetrievalError("Embedding vector values must be finite")
        normalized.append(number)
    return tuple(normalized)
