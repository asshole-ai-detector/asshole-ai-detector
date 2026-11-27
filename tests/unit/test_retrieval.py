
"""Unit tests for basic dense retrieval (RAG-03)."""

import pytest

from asshole_ai_detector.rag.retrieval import (
    DenseHit,
    MissingCaseError,
    retrieve_dense,
)
from asshole_ai_detector.schemas.case import (
    AITACase,
    AITAComment,
    DerivedFields,
)


def make_case(post_id: str) -> AITACase:
    """Create a valid synthetic AITA case."""

    comments = [
        AITAComment(
            comment_id=f"{post_id}#{rank}",
            rank=rank,
            verdict="NTA",
            text=f"Test comment {rank}",
        )
        for rank in range(1, 6)
    ]

    return AITACase(
        post_id=post_id,
        title="A synthetic conflict",
        text=" ".join(["example"] * 55),
        url=f"https://example.com/{post_id}",
        valid_comments=comments,
        derived_fields=DerivedFields.from_comments(comments),
    )


def fake_embedding(query: str) -> list[float]:
    """Return a normalized 1024-dimensional vector."""
    return [1.0] + [0.0] * 1023


class FakeIndex:
    """Small stand-in for a ChromaDB collection."""

    def __init__(
        self,
        ids: list[str],
        distances: list[float],
        available: int | None = None,
    ):
        self.ids = ids
        self.distances = distances
        self.available = (
            len(ids) if available is None else available
        )
        self.last_query = None

    def count(self) -> int:
        return self.available

    def query(
        self,
        *,
        query_embeddings,
        n_results,
        include,
    ):
        self.last_query = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include,
        }

        return {
            "ids": [self.ids[:n_results]],
            "distances": [self.distances[:n_results]],
        }


def test_basic_retrieval():
    """Retrieve actual cases associated with vector IDs."""

    index = FakeIndex(
        ids=["post-1", "post-2"],
        distances=[0.1, 0.3],
    )

    cases = {
        "post-1": make_case("post-1"),
        "post-2": make_case("post-2"),
    }

    results = retrieve_dense(
        "A conflict with my friend",
        embed_query=fake_embedding,
        index=index,
        get_case=cases.get,
    )

    assert len(results) == 2
    assert isinstance(results[0], DenseHit)
    assert results[0].post_id == "post-1"
    assert results[0].cosine_distance == pytest.approx(0.1)
    assert results[0].case.post_id == "post-1"
    assert results[1].post_id == "post-2"


def test_results_sorted_by_distance():
    """Smaller cosine distance means a closer match."""

    index = FakeIndex(
        ids=["post-1", "post-2"],
        distances=[0.8, 0.2],
    )

    cases = {
        "post-1": make_case("post-1"),
        "post-2": make_case("post-2"),
    }

    results = retrieve_dense(
        "A family disagreement",
        embed_query=fake_embedding,
        index=index,
        get_case=cases.get,
    )

    assert [hit.post_id for hit in results] == [
        "post-2",
        "post-1",
    ]


def test_empty_index():
    """An empty vector index returns no results."""

    index = FakeIndex([], [])

    def should_not_embed(query):
        raise AssertionError("Embedding should not be called")

    results = retrieve_dense(
        "A disagreement",
        embed_query=should_not_embed,
        index=index,
        get_case=lambda post_id: None,
    )

    assert results == []


def test_fewer_results_than_top_k():
    """Requesting 30 results from two posts returns only two."""

    index = FakeIndex(
        ids=["post-1", "post-2"],
        distances=[0.1, 0.2],
    )

    cases = {
        "post-1": make_case("post-1"),
        "post-2": make_case("post-2"),
    }

    results = retrieve_dense(
        "A workplace conflict",
        embed_query=fake_embedding,
        index=index,
        get_case=cases.get,
        top_k=30,
    )

    assert len(results) == 2
    assert index.last_query["n_results"] == 2


def test_respects_top_k():
    """The search must not return more than requested."""

    index = FakeIndex(
        ids=["post-1", "post-2", "post-3"],
        distances=[0.1, 0.2, 0.3],
    )

    cases = {
        post_id: make_case(post_id)
        for post_id in index.ids
    }

    results = retrieve_dense(
        "A disagreement",
        embed_query=fake_embedding,
        index=index,
        get_case=cases.get,
        top_k=2,
    )

    assert len(results) == 2
    assert index.last_query["n_results"] == 2


def test_missing_case():
    """Missing stored cases must raise an explicit error."""

    index = FakeIndex(
        ids=["missing-post"],
        distances=[0.1],
    )

    with pytest.raises(
        MissingCaseError,
        match="missing-post",
    ):
        retrieve_dense(
            "A disagreement",
            embed_query=fake_embedding,
            index=index,
            get_case=lambda post_id: None,
        )


def test_rejects_empty_query():
    """A blank query is invalid."""

    index = FakeIndex([], [])

    with pytest.raises(ValueError, match="empty"):
        retrieve_dense(
            "   ",
            embed_query=fake_embedding,
            index=index,
            get_case=lambda post_id: None,
        )


def test_rejects_invalid_embedding_dimensions():
    """Query vectors must contain exactly 1024 elements."""

    index = FakeIndex(
        ids=["post-1"],
        distances=[0.1],
    )

    with pytest.raises(ValueError, match="1024"):
        retrieve_dense(
            "A disagreement",
            embed_query=lambda query: [1.0, 0.0],
            index=index,
            get_case=lambda post_id: make_case(post_id),
        )


def test_rejects_unnormalized_embedding():
    """Query embeddings must have an L2 norm near one."""

    index = FakeIndex(
        ids=["post-1"],
        distances=[0.1],
    )

    with pytest.raises(ValueError, match="L2-normalized"):
        retrieve_dense(
            "A disagreement",
            embed_query=lambda query: [2.0] + [0.0] * 1023,
            index=index,
            get_case=lambda post_id: make_case(post_id),
        )


def test_rejects_duplicate_ids():
    """The vector index must not return repeated post IDs."""

    index = FakeIndex(
        ids=["post-1", "post-1"],
        distances=[0.1, 0.2],
    )

    with pytest.raises(ValueError, match="duplicate"):
        retrieve_dense(
            "A disagreement",
            embed_query=fake_embedding,
            index=index,
            get_case=lambda post_id: make_case(post_id),
        )


def test_rejects_mismatched_distances():
    """Every returned ID must have a matching distance."""

    index = FakeIndex(
        ids=["post-1", "post-2"],
        distances=[0.1],
    )

    with pytest.raises(ValueError, match="mismatched"):
        retrieve_dense(
            "A disagreement",
            embed_query=fake_embedding,
            index=index,
            get_case=lambda post_id: make_case(post_id),
        )


def test_query_is_normalized():
    """Text is trimmed and lowercased before embedding."""

    received_queries = []

    def recording_embedding(query):
        received_queries.append(query)
        return fake_embedding(query)

    index = FakeIndex(
        ids=["post-1"],
        distances=[0.1],
    )

    retrieve_dense(
        "  A Family CONFLICT  ",
        embed_query=recording_embedding,
        index=index,
        get_case=lambda post_id: make_case(post_id),
    )

    assert received_queries == ["a family conflict"]
    assert index.last_query["include"] == ["distances"]
