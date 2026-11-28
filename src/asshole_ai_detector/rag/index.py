"""Build and open immutable preliminary RAG-02 indexes from verified DATA-05 exports."""

from __future__ import annotations

import errno
import json
import math
import re
import struct
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from itertools import batched
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from asshole_ai_detector.data.pipeline import load_prepared
from asshole_ai_detector.rag.embeddings import (
    Embedder,
    EmbeddingConfig,
    load_qwen3_embedder,
)
from asshole_ai_detector.schemas.case import VERDICTS, AITACase

if TYPE_CHECKING:
    from chromadb import Collection

COLLECTION = "aita-preliminary"
DIMENSIONS = 1024
DEFAULT_INDEX_ROOT = Path("artifacts/rag-02")


@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 42
    max_per_top_verdict: int = 5000
    max_posts: int = 20000

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("Sampling seed must be an integer")
        if (
            type(self.max_per_top_verdict) is not int
            or not 1 <= self.max_per_top_verdict <= 5000
        ):
            raise ValueError("Class cap must be in [1, 5000]")
        if (
            type(self.max_posts) is not int
            or not 4 * self.max_per_top_verdict <= self.max_posts <= 20000
        ):
            raise ValueError(
                "Total cap must allow all four class quotas and be at most 20000"
            )


def select_cases(
    cases: Sequence[AITACase], config: SamplingConfig
) -> tuple[list[AITACase], dict]:
    """Sample without replacement in stable verdict, priority and ID order."""
    groups = defaultdict(list)
    seen = set()
    for case in sorted(cases, key=lambda c: c.post_id):
        if case.post_id in seen:
            raise ValueError(f"Duplicate candidate ID: {case.post_id}")
        seen.add(case.post_id)
        fields = case.derived_fields
        groups[fields.top_verdict, fields.consensus_label].append(case)
    rng = Random(config.seed)
    selected = []
    counts = {}
    for verdict in VERDICTS:
        room = config.max_per_top_verdict
        counts[verdict] = {}
        for label in ("DISPUTED", verdict):
            pool = groups[verdict, label]
            chosen = rng.sample(pool, room) if len(pool) > room else pool
            selected.extend(chosen)
            room -= len(chosen)
            counts[verdict][label] = {
                "eligible": len(pool),
                "selected": len(chosen),
                "excluded_by_cap": len(pool) - len(chosen),
            }
    return sorted(selected, key=lambda c: c.post_id), counts


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode()).hexdigest()


def _case_line(case: AITACase) -> bytes:
    return (case.model_dump_json() + "\n").encode("utf-8")


def _metadata(case: AITACase) -> dict[str, str]:
    fields = case.derived_fields
    return {
        "post_id": case.post_id,
        "top_verdict": fields.top_verdict,
        "consensus_label": fields.consensus_label,
    }


@dataclass
class _BuildInputs:
    cases: list[AITACase]
    identity: dict
    report: dict

    @property
    def build_id(self) -> str:
        return _digest(self.identity)


def _inputs(
    prepared_dir: Path,
    config: EmbeddingConfig,
    sampling: SamplingConfig,
    expected_source: tuple[str, str] | None,
) -> _BuildInputs:
    prepared = load_prepared(prepared_dir, strict=True)
    splits, report = prepared.splits, prepared.report
    manifest = splits.manifest
    source = (report["source_dataset"], manifest.source_revision)
    if expected_source is not None and source != expected_source:
        raise ValueError(
            "Prepared dataset or revision differs from the requested source"
        )
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source[1]):
        raise ValueError(
            "Indexing requires a pinned dataset revision (commit or content digest)"
        )
    if not config.model_name.strip() or not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", config.revision
    ):
        raise ValueError(
            "Indexing requires an embedding model name and pinned revision"
        )
    if config.dimensions != DIMENSIONS:
        raise ValueError("RAG-03 requires 1024 embedding dimensions")

    cases, counts = select_cases(splits.rag_candidates, sampling)
    selected_ids = {case.post_id for case in cases}
    excluded_overlap = selected_ids & (manifest.held_out_ids | manifest.development_ids)
    if excluded_overlap or not selected_ids <= manifest.retrieval_ids:
        raise ValueError("Selection falls outside the verified retrieval inventory")
    case_digest = sha256()
    for case in cases:
        case_digest.update(_case_line(case))
    try:
        chroma_version = version("chromadb")
    except PackageNotFoundError as error:
        raise ImportError(
            "Persistent indexing requires uv sync --group index"
        ) from error
    identity = {
        "format_version": 1,
        "purpose": "preliminary_development",
        "collection": COLLECTION,
        "distance": "cosine",
        "chromadb_version": chroma_version,
        "source_dataset": source[0],
        "source_revision": source[1],
        "split_manifest_id": manifest.manifest_id,
        "split_strategy": manifest.strategy,
        "split_seed": splits.seed,
        "preprocessing_report_schema": report["schema_version"],
        "preprocessing_report_sha256": _digest(report),
        "cleaning_config": report["cleaning_config"],
        "consensus_threshold": report["consensus_threshold"],
        "embedding": config.manifest(),
        "sampling": {"policy": "disputed_first_seeded_v1", **asdict(sampling)},
        "cases_sha256": case_digest.hexdigest(),
    }
    summary = {
        "indexed_cases": len(cases),
        "per_top_verdict": {
            v: sum(c["selected"] for c in counts[v].values()) for v in VERDICTS
        },
        "selection_counts": counts,
        "excluded_partitions": {
            "golden_candidate_pool": len(manifest.held_out_ids),
            "development": len(manifest.development_ids),
            "unused_test": len(splits.unused_test),
        },
        "verified_golden_or_development_ids_in_index": len(excluded_overlap),
        "upstream_semantic_scan": report["semantic_near_duplicate_scan"],
        "semantic_scan_performed_by_indexer": False,
    }
    return _BuildInputs(cases, identity, summary)


def _client(directory: Path):
    import chromadb

    return chromadb.PersistentClient(path=str(directory / "chroma"))


def _vector_bytes(vector: Sequence[float]) -> bytes:
    if len(vector) != DIMENSIONS:
        raise ValueError("Expected 1024 dimensions per embedding")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
        raise ValueError("Embedding must be finite and L2-normalized")
    return struct.pack(f"<{DIMENSIONS}f", *vector)


def _populate(
    directory: Path,
    inputs: _BuildInputs,
    config: EmbeddingConfig,
    load_embedder: Callable[[EmbeddingConfig], Embedder],
) -> str:
    vectors_digest = sha256()
    with _client(directory) as client, (directory / "cases.jsonl").open("xb") as store:
        collection = client.create_collection(
            COLLECTION,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"build_id": inputs.build_id},
        )
        if inputs.cases:
            embedder = load_embedder(config)
            if embedder.manifest() != config.manifest():
                raise ValueError("Loaded embedder differs from requested configuration")
            batch_size = min(config.batch_size, client.get_max_batch_size())
            for batch in batched(inputs.cases, batch_size):
                vectors = embedder.embed_documents(batch)
                if len(vectors) != len(batch):
                    raise ValueError("Embedding count differs from case count")
                for vector in vectors:
                    vectors_digest.update(_vector_bytes(vector))
                collection.add(
                    ids=[c.post_id for c in batch],
                    embeddings=vectors,
                    metadatas=[_metadata(c) for c in batch],
                )
                store.writelines(_case_line(c) for c in batch)
    return vectors_digest.hexdigest()


def _check_collection(
    collection: Collection, inputs: _BuildInputs, expected_digest: str
) -> None:
    metric = collection.configuration_json["hnsw"]["space"]
    if metric != "cosine" or collection.metadata != {"build_id": inputs.build_id}:
        raise ValueError("Collection metric or build ID mismatch")
    if collection.count() != len(inputs.cases):
        raise ValueError("Collection count differs from verified selection")
    digest = sha256()
    for batch in batched(inputs.cases, 256):
        ids = [c.post_id for c in batch]
        records = collection.get(ids=ids, include=["metadatas", "embeddings"])
        rows = {
            key: (metadata, vector)
            for key, metadata, vector in zip(
                records["ids"], records["metadatas"], records["embeddings"], strict=True
            )
        }
        if rows.keys() != set(ids):
            raise ValueError("Collection IDs differ from verified selection")
        for case in batch:
            metadata, vector = rows[case.post_id]
            if metadata != _metadata(case):
                raise ValueError(f"Incorrect collection metadata for {case.post_id}")
            digest.update(_vector_bytes(vector))
    if digest.hexdigest() != expected_digest:
        raise ValueError("Stored vector checksum mismatch")


@dataclass
class OpenedIndex:
    """Use within open_index's context; Chroma operations are read-only by convention."""

    directory: Path
    collection: Collection
    manifest: dict
    _offsets: dict[str, int]

    def get_case(self, post_id: str) -> AITACase | None:
        offset = self._offsets.get(post_id)
        if offset is None:
            return None
        with (self.directory / "cases.jsonl").open("rb") as store:
            store.seek(offset)
            case = AITACase.model_validate_json(store.readline())
        if case.post_id != post_id:
            raise ValueError(f"Case-store ID mismatch for {post_id}")
        return case


@contextmanager
def _verified(directory: Path, inputs: _BuildInputs) -> Iterator[OpenedIndex]:
    manifest = json.loads((directory / "index_manifest.json").read_bytes())
    if manifest["build_id"] != inputs.build_id or manifest["inputs"] != inputs.identity:
        raise ValueError("Index manifest differs from current split or configuration")
    report = json.loads((directory / "build_report.json").read_bytes())
    if report != {"build_id": inputs.build_id, **inputs.report} or any(
        manifest[k] != inputs.report[k] for k in ("indexed_cases", "per_top_verdict")
    ):
        raise ValueError("Index counts or report disagree with verified inputs")
    offsets = {}
    with (directory / "cases.jsonl").open("rb") as store:
        for case in inputs.cases:
            offsets[case.post_id] = store.tell()
            if store.readline() != _case_line(case):
                raise ValueError(
                    f"Stored case differs from verified selection: {case.post_id}"
                )
        if store.read(1):
            raise ValueError("Unexpected extra case-store entries")
    if not (directory / "chroma" / "chroma.sqlite3").is_file():
        raise ValueError("Missing persistent Chroma database")
    with _client(directory) as client:
        collection = client.get_collection(COLLECTION, embedding_function=None)
        _check_collection(collection, inputs, manifest["vectors_sha256"])
        yield OpenedIndex(directory, collection, manifest, offsets)


def build_index(
    prepared_dir: Path,
    index_root: Path = DEFAULT_INDEX_ROOT,
    *,
    embedding_config: EmbeddingConfig,
    sampling: SamplingConfig = SamplingConfig(),
    load_embedder: Callable[[EmbeddingConfig], Embedder] = load_qwen3_embedder,
    expected_source: tuple[str, str] | None = None,
) -> Path:
    """Verify and reuse or atomically publish a complete generation. Return its path."""
    inputs = _inputs(prepared_dir, embedding_config, sampling, expected_source)
    destination = index_root / inputs.build_id
    if destination.exists():
        with _verified(destination, inputs):
            return destination
    index_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".building-", dir=index_root) as temp:
        stage = Path(temp) / "generation"
        stage.mkdir()
        vector_digest = _populate(stage, inputs, embedding_config, load_embedder)
        manifest = {
            "build_id": inputs.build_id,
            "inputs": inputs.identity,
            "vectors_sha256": vector_digest,
            "built_at": datetime.now(UTC).isoformat(),
            "embedding_batch_size": embedding_config.batch_size,
            "indexed_cases": inputs.report["indexed_cases"],
            "per_top_verdict": inputs.report["per_top_verdict"],
        }
        report = {"build_id": inputs.build_id, **inputs.report}
        for name, data in (
            ("index_manifest.json", manifest),
            ("build_report.json", report),
        ):
            (stage / name).write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        with _verified(stage, inputs):
            pass
        try:
            stage.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # Another builder published the same generation while ours was staging.
            with _verified(destination, inputs):
                pass
    return destination


@contextmanager
def open_index(
    prepared_dir: Path,
    index_root: Path = DEFAULT_INDEX_ROOT,
    *,
    embedding_config: EmbeddingConfig,
    sampling: SamplingConfig = SamplingConfig(),
    expected_source: tuple[str, str] | None = None,
) -> Iterator[OpenedIndex]:
    """Open only the generation matching the current verified export and query settings."""
    inputs = _inputs(prepared_dir, embedding_config, sampling, expected_source)
    with _verified(index_root / inputs.build_id, inputs) as opened:
        yield opened
