
"""Basic dense retrieval for AITA submissions (RAG-03)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Protocol

from asshole_ai_detector.schemas.case import AITACase

EMBEDDING_DIMENSIONS = 1024
DEFAULT_TOP_K = 30


class VectorIndex(Protocol):
    """Minimal interface supported by a ChromaDB collection."""

    def count(self) -> int:
        ...

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, Any]:
        ...


class MissingCaseError(LookupError):
    """An indexed post ID has no corresponding stored case."""


@dataclass(frozen=True)
class DenseHit:
    """One result from dense vector retrieval."""

    post_id: str
    cosine_distance: float
    case: AITACase


def retrieve_dense(
    query: str,
    *,
    embed_query: Callable[[str], Sequence[float]],
    index: VectorIndex,
    get_case: Callable[[str], AITACase | None],
    top_k: int = DEFAULT_TOP_K,
) -> list[DenseHit]:
    """Retrieve AITA cases using an existing cosine ChromaDB index.

    The embed_query function must use the RAG-01 query format,
    model revision, and tokenization settings used for indexing.

    Case storage is injected to avoid inventing a new storage
    contract while RAG-02 integration is being finalized.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty")

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    available = index.count()

    if available == 0:
        return []

    # RAG-01 handles the instruction prefix and token truncation.
    normalized_query = query.strip().lower()
    embedding = [float(value) for value in embed_query(normalized_query)]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError("Query embedding must have 1024 dimensions")

    if not all(isfinite(value) for value in embedding):
        raise ValueError("Query embedding contains invalid numbers")

    norm = sqrt(sum(value * value for value in embedding))

    if abs(norm - 1.0) > 0.001:
        raise ValueError("Query embedding must be L2-normalized")

    response = index.query(
        query_embeddings=[embedding],
        n_results=min(top_k, available),
        include=["distances"],
    )

    ids_batches = response.get("ids") or []
    distance_batches = response.get("distances") or []

    if not ids_batches:
        return []

    if not distance_batches:
        raise ValueError("Vector index returned no distances")

    ids = ids_batches[0]
    distances = distance_batches[0]

    if len(ids) != len(distances):
        raise ValueError("Index returned mismatched IDs and distances")

    if len(ids) != len(set(ids)):
        raise ValueError("Index returned duplicate post IDs")

    hits = []

    for post_id, raw_distance in zip(ids, distances, strict=True):
        distance = float(raw_distance)

        if not isfinite(distance) or distance < 0:
            raise ValueError("Invalid cosine distance from index")

        case = get_case(post_id)

        if case is None:
            raise MissingCaseError(
                f"Missing stored case for post_id: {post_id}"
            )

        if case.post_id != post_id:
            raise ValueError(
                f"Stored case ID does not match index ID: {post_id}"
            )

        hits.append(
            DenseHit(
                post_id=post_id,
                cosine_distance=distance,
                case=case,
            )
        )

    return sorted(
        hits,
        key=lambda hit: (hit.cosine_distance, hit.post_id),
    )
