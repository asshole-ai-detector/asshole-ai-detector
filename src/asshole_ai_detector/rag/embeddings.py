"""Qwen3 embedding pipeline (RAG-01, Milestone 2 Sections 3.3, 3.4 and 3.6).

One vector per AITA submission: ``title + "\\n" + text``, capped at 1,024 tokens
(first 700 + last 324). Comments are never embedded. Queries get the Section 3.4
instruction prefix; documents get none. Vectors have 1,024 dimensions and are
L2-normalized, so cosine similarity equals the dot product.

The same ``Embedder`` is shared by indexing (RAG-02), retrieval and semantic
near-duplicate detection, so all vectors come from one model and revision.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"

# Extra room for the query prefix and end-of-text token, so the model never cuts
# a text that was already capped by ``cap_tokens``.
PROMPT_TOKEN_HEADROOM = 64


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding settings from ``rag`` in configs/default.yaml."""

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    dimensions: int = 1024
    max_tokens: int = 1024
    head_tokens: int = 700
    tail_tokens: int = 324
    query_instruction: str = (
        "Given a description of an interpersonal conflict, retrieve similar conflicts"
    )
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.head_tokens + self.tail_tokens != self.max_tokens:
            raise ValueError("head_tokens + tail_tokens must equal max_tokens")
        if (
            min(self.dimensions, self.head_tokens, self.tail_tokens, self.batch_size)
            < 1
        ):
            raise ValueError("Dimensions, token counts and batch size must be positive")

    @classmethod
    def from_mapping(cls, rag: Mapping[str, Any]) -> Self:
        """Build from the ``rag`` section, ignoring unrelated keys."""
        defaults = cls()
        return cls(
            model_name=str(rag.get("embedding_model", defaults.model_name)),
            revision=str(rag.get("embedding_revision", defaults.revision)),
            dimensions=int(rag.get("embedding_dimensions", defaults.dimensions)),
            max_tokens=int(rag.get("embedded_post_token_limit", defaults.max_tokens)),
            head_tokens=int(rag.get("embedded_post_head_tokens", defaults.head_tokens)),
            tail_tokens=int(rag.get("embedded_post_tail_tokens", defaults.tail_tokens)),
            query_instruction=str(
                rag.get("embedding_query_instruction", defaults.query_instruction)
            ),
            batch_size=int(rag.get("embedding_batch_size", defaults.batch_size)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Self:
        with Path(path).open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return cls.from_mapping(config.get("rag", {}))

    def manifest(self) -> dict[str, Any]:
        """Settings stored with an index; queries must use the same values.

        Batch size is left out because it does not change the vectors.
        """
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "dimensions": self.dimensions,
            "max_tokens": self.max_tokens,
            "head_tokens": self.head_tokens,
            "tail_tokens": self.tail_tokens,
            "query_instruction": self.query_instruction,
            "document_prefix": "",
            "normalized": True,
            "similarity": "cosine",
        }


class Tokenizer(Protocol):
    """The part of a Hugging Face tokenizer used for the token cap."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class EncoderBackend(Protocol):
    """Turns prepared texts into vectors (a SentenceTransformer in practice)."""

    def encode(
        self, texts: list[str], batch_size: int
    ) -> Sequence[Sequence[float]]: ...


class Submission(Protocol):
    """Anything with a title and a post text, e.g. ``AITACase``."""

    title: str
    text: str


# Text preparation ----------------------------------------------------------


def document_text(case: Submission) -> str:
    """Section 3.3: the embedded text is the title, a newline, then the post."""
    return f"{case.title}\n{case.text}"


def cap_tokens(text: str, tokenizer: Tokenizer, config: EmbeddingConfig) -> str:
    """Keep the first 700 and last 324 tokens of texts longer than 1,024 tokens.

    The cut is made on token IDs (without special tokens) and decoded back to
    text, so the model sees the same tokens. Shorter texts are unchanged.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= config.max_tokens:
        return text
    kept = [*token_ids[: config.head_tokens], *token_ids[-config.tail_tokens :]]
    return tokenizer.decode(kept)


def format_query(text: str, config: EmbeddingConfig) -> str:
    """Section 3.4 query format; documents never get this prefix."""
    return f"Instruct: {config.query_instruction}\nQuery: {text}"


def _normalized(
    vectors: Sequence[Sequence[float]], expected_count: int, dimensions: int
) -> list[list[float]]:
    """Check shape and return unit-length vectors.

    The model already normalizes; dividing again makes the guarantee explicit
    and catches broken outputs (zero, NaN or wrong size) early.
    """
    rows = [[float(value) for value in vector] for vector in vectors]
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} vectors, got {len(rows)}")
    result: list[list[float]] = []
    for row in rows:
        if len(row) != dimensions:
            raise ValueError(f"Expected {dimensions} dimensions, got {len(row)}")
        norm = math.sqrt(sum(value * value for value in row))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Embedding has a zero or non-finite norm")
        result.append([value / norm for value in row])
    return result


# Embedding interface ---------------------------------------------------------


class Embedder:
    """Shared embedding interface for documents (cases) and retrieval queries.

    Outputs keep the input order: vector ``i`` belongs to input ``i``.
    """

    def __init__(
        self,
        backend: EncoderBackend,
        tokenizer: Tokenizer,
        config: EmbeddingConfig | None = None,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config or EmbeddingConfig()

    def prepare_document(self, case: Submission) -> str:
        """Title + post, capped. Comments are not part of the embedded text."""
        return cap_tokens(document_text(case), self.tokenizer, self.config)

    def prepare_query(self, text: str) -> str:
        """Lowercased like indexed posts (Section 3.5, step 1), capped, prefixed."""
        if not text.strip():
            raise ValueError("The query must not be empty")
        capped = cap_tokens(text.lower(), self.tokenizer, self.config)
        return format_query(capped, self.config)

    def embed_documents(self, cases: Sequence[Submission]) -> list[list[float]]:
        """One vector per submission, e.g. for indexing or duplicate checks."""
        return self._embed([self.prepare_document(case) for case in cases])

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per query, e.g. a neutralized user scenario."""
        return self._embed([self.prepare_query(text) for text in texts])

    def manifest(self) -> dict[str, Any]:
        return self.config.manifest()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.backend.encode(texts, batch_size=self.config.batch_size)
        return _normalized(vectors, len(texts), self.config.dimensions)


# Real model --------------------------------------------------------------------


class SentenceTransformerBackend:
    """Adapts a SentenceTransformer to ``EncoderBackend``."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


def load_qwen3_embedder(
    config: EmbeddingConfig | None = None,
    *,
    device: str | None = None,
) -> Embedder:
    """Load Qwen3-Embedding at the pinned revision (needs ``uv sync --group rag``)."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Embedding needs the 'rag' dependency group. Install it with: "
            "uv sync --group rag"
        ) from error

    config = config or EmbeddingConfig.from_yaml()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # fp16 on GPU (about 1.2 GB, Section 3.8); fp32 on CPU, where fp16 is slow.
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = SentenceTransformer(
        config.model_name,
        revision=config.revision,
        device=device,
        model_kwargs={"dtype": dtype},
    )
    model.max_seq_length = config.max_tokens + PROMPT_TOKEN_HEADROOM

    actual = model.get_embedding_dimension()
    if actual != config.dimensions:
        raise ValueError(
            f"Model returns {actual} dimensions, expected {config.dimensions}"
        )
    return Embedder(SentenceTransformerBackend(model), model.tokenizer, config)
