"""Tests for the embedding pipeline, using stubs instead of the real model."""

import json
import math
from pathlib import Path

import pytest

from asshole_ai_detector.data.cleaning import CleanedAITACase, clean_cases
from asshole_ai_detector.data_processing.consensus import label_cleaned_case
from asshole_ai_detector.rag.embeddings import (
    Embedder,
    EmbeddingConfig,
    cap_tokens,
    document_text,
    format_query,
)
from asshole_ai_detector.schemas.case import NormalizedAITACase

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "normalized_cases.json"
CONFIG = EmbeddingConfig()


class WordTokenizer:
    """One token per whitespace-separated word, enough to test the token cap."""

    def __init__(self) -> None:
        self.vocab: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab.append(word)
            ids.append(self.vocab.index(word))
        return ids

    def decode(self, token_ids) -> str:
        return " ".join(self.vocab[i] for i in token_ids)


class StubBackend:
    """Deterministic, deliberately unnormalized vectors; records every call."""

    def __init__(
        self, dimensions: int = 1024, vector: list[float] | None = None
    ) -> None:
        self.dimensions = dimensions
        self.vector = vector
        self.calls: list[tuple[list[str], int]] = []

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        self.calls.append((list(texts), batch_size))
        if self.vector is not None:
            return [self.vector for _ in texts]
        return [
            [float((len(text) + i) % 7 + 1) for i in range(self.dimensions)]
            for text in texts
        ]


@pytest.fixture
def cases():
    """Valid AITA cases built with the real DATA-02 and DATA-03 steps."""
    raw = [
        NormalizedAITACase.model_validate(c) for c in json.loads(FIXTURES.read_text())
    ]
    accepted = clean_cases(raw).accepted
    assert accepted and all(isinstance(c, CleanedAITACase) for c in accepted)
    return [label_cleaned_case(case) for case in accepted]


def make_embedder(backend: StubBackend | None = None) -> Embedder:
    return Embedder(backend or StubBackend(), WordTokenizer(), CONFIG)


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


# Text preparation ------------------------------------------------------------


def test_document_text_is_title_newline_text(cases):
    case = cases[0]
    assert document_text(case) == f"{case.title}\n{case.text}"


def test_documents_have_no_prefix_and_no_comments(cases):
    backend = StubBackend()
    make_embedder(backend).embed_documents(cases)

    texts, _ = backend.calls[0]
    for case, text in zip(cases, texts, strict=True):
        assert text == document_text(case)
        assert not text.startswith("Instruct:")
        assert all(comment.text not in text for comment in case.valid_comments)


def test_short_text_is_not_changed():
    text = " ".join(f"w{i}" for i in range(CONFIG.max_tokens))
    assert cap_tokens(text, WordTokenizer(), CONFIG) == text


def test_long_text_keeps_first_700_and_last_324_tokens():
    words = [f"w{i}" for i in range(1500)]
    capped = cap_tokens(" ".join(words), WordTokenizer(), CONFIG).split()

    assert len(capped) == 1024
    assert capped[:700] == words[:700]
    assert capped[700:] == words[-324:]


def test_query_is_lowercased_capped_and_prefixed():
    embedder = make_embedder()
    long_query = "My SISTER " + " ".join(f"w{i}" for i in range(1500))
    prepared = embedder.prepare_query(long_query)

    prefix = format_query("", CONFIG)
    assert prepared.startswith(prefix)
    assert prefix == (
        "Instruct: Given a description of an interpersonal conflict, "
        "retrieve similar conflicts\nQuery: "
    )
    body = prepared.removeprefix(prefix)
    assert body.startswith("my sister")
    assert len(body.split()) == 1024


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        make_embedder().embed_queries(["   "])


# Vectors ---------------------------------------------------------------------


def test_vectors_have_1024_dimensions_and_unit_norm(cases):
    vectors = make_embedder().embed_documents(cases)

    assert len(vectors) == len(cases)
    for vector in vectors:
        assert len(vector) == 1024
        assert norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_one_batched_call_with_configured_batch_size(cases):
    backend = StubBackend()
    make_embedder(backend).embed_documents(cases)

    assert len(backend.calls) == 1
    assert backend.calls[0][1] == CONFIG.batch_size


def test_same_input_gives_same_vectors_in_input_order(cases):
    embedder = make_embedder()
    first = embedder.embed_documents(cases)
    second = embedder.embed_documents(cases)
    reversed_order = embedder.embed_documents(list(reversed(cases)))

    assert first == second
    assert reversed_order == list(reversed(first))


def test_empty_input_does_not_call_the_model():
    backend = StubBackend()
    assert make_embedder(backend).embed_documents([]) == []
    assert backend.calls == []


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (StubBackend(dimensions=768), "Expected 1024 dimensions"),
        (StubBackend(vector=[0.0] * 1024), "zero or non-finite"),
        (StubBackend(vector=[math.nan] * 1024), "zero or non-finite"),
    ],
)
def test_invalid_model_output_is_rejected(cases, backend, message):
    with pytest.raises(ValueError, match=message):
        make_embedder(backend).embed_documents(cases)


# Configuration -----------------------------------------------------------------


def test_yaml_config_matches_architecture_defaults():
    assert EmbeddingConfig.from_yaml() == CONFIG


def test_head_and_tail_must_add_up_to_the_cap():
    with pytest.raises(ValueError, match="must equal max_tokens"):
        EmbeddingConfig(head_tokens=600)


def test_manifest_records_model_revision_and_format():
    manifest = make_embedder().manifest()

    assert manifest["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert manifest["revision"] == CONFIG.revision
    assert manifest["dimensions"] == 1024
    assert manifest["document_prefix"] == ""
    assert "batch_size" not in manifest
