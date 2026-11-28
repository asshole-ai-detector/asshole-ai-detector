"""Real on-disk Chroma, synthetic DATA-05 records, no external models or data."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_prepare_data import raw_row

from asshole_ai_detector.data.pipeline import export_prepared, prepare
from asshole_ai_detector.rag import retrieval
from asshole_ai_detector.rag.embeddings import EmbeddingConfig
from asshole_ai_detector.rag.index import (
    COLLECTION,
    SamplingConfig,
    build_index,
    open_index,
)
from asshole_ai_detector.schemas.case import VERDICTS

chromadb = pytest.importorskip("chromadb")
DEMO = Path(__file__).resolve().parents[2] / "scripts/demo_index.py"
CONFIG = EmbeddingConfig(
    model_name="synthetic/index-test", revision="a" * 40, batch_size=2
)


class TestEncoder:
    """Already-normalized vectors, with distinct positions to catch case/vector swaps."""

    __test__ = False

    def __init__(self, config):
        self.config = config

    def manifest(self):
        return self.config.manifest()

    def embed_documents(self, cases):
        vectors = []
        for case in cases:
            vector = [0.0] * 1024
            vector[int(case.post_id[-2:])] = 0.6
            vector[1023] = 0.8
            vectors.append(vector)
        return vectors


def export(
    path, *, split_seed=42, strategy="combined_stratified", n_rows=20, golden=3, dev=2
):
    rows = [raw_row(f"case{i:02d}", verdict=VERDICTS[i % 4]) for i in range(n_rows)]
    prepared = prepare(
        rows[:15],
        rows[15:],
        source_revision="b" * 40,
        split_strategy=strategy,
        seed=split_seed,
        golden_minimums={},
        golden_size=golden,
        development_size=dev,
    )
    prepared.report["source_dataset"] = "synthetic/index-tests"
    export_prepared(prepared, path)
    return prepared


def build(data, root, **kwargs):
    return build_index(
        data, root, embedding_config=CONFIG, load_embedder=TestEncoder, **kwargs
    )


def never_load(config):
    raise AssertionError("Unexpected embedder load")


@pytest.mark.parametrize("strategy", ["combined_stratified", "provider_test"])
def test_persistence_metadata_case_lookup_and_idempotency(tmp_path, strategy):
    data, root = tmp_path / "prepared", tmp_path / "index"
    prepared = export(data, strategy=strategy, golden=2, dev=1)
    manifest = prepared.splits.manifest
    if strategy == "combined_stratified":
        assert any(int(i[-2:]) < 15 for i in manifest.held_out_ids)
        assert any(int(i[-2:]) < 15 for i in manifest.development_ids)
    generation = build(data, root)
    original_manifest = (generation / "index_manifest.json").read_bytes()
    assert (
        build_index(data, root, embedding_config=CONFIG, load_embedder=never_load)
        == generation
    )
    assert (generation / "index_manifest.json").read_bytes() == original_manifest
    assert list(root.iterdir()) == [generation]
    expected = {c.post_id: c for c in prepared.splits.retrieval}
    with open_index(data, root, embedding_config=CONFIG) as index:
        records = index.collection.get(include=["embeddings", "metadatas"])
        assert index.collection.count() == len(set(records["ids"])) == len(expected)
        assert set(records["ids"]) == manifest.retrieval_ids
        assert index.manifest["indexed_cases"] == len(expected)
        for key, metadata, vector in zip(
            records["ids"], records["metadatas"], records["embeddings"], strict=True
        ):
            case = expected[key]
            assert metadata == {
                "post_id": key,
                "top_verdict": case.derived_fields.top_verdict,
                "consensus_label": case.derived_fields.consensus_label,
            }
            assert vector == pytest.approx(
                TestEncoder(CONFIG).embed_documents([case])[0]
            )
            assert index.get_case(key) == case
        excluded = (
            manifest.held_out_ids
            | manifest.development_ids
            | {c.post_id for c in prepared.splits.unused_test}
        )
        assert all(index.get_case(key) is None for key in excluded)
        assert index.collection.get(ids=sorted(excluded))["ids"] == []
    report = json.loads((generation / "build_report.json").read_text())
    assert report["excluded_partitions"] == {
        "golden_candidate_pool": 2,
        "development": 1,
        "unused_test": len(prepared.splits.unused_test),
    }
    assert report["upstream_semantic_scan"] == "not_run"
    assert report["semantic_scan_performed_by_indexer"] is False


def test_separate_process_demo(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(DEMO), "--root", str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Reopened 8 records" in completed.stdout
    assert "Verified reuse without embedding" in completed.stdout


@pytest.mark.skipif(
    not hasattr(retrieval, "retrieve_dense"),
    reason="RAG-03 retrieve_dense is on draft PR #31",
)
def test_actual_rag03_query_and_evidence_in_fresh_process(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(DEMO), "--root", str(tmp_path), "--rag03"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "RAG-03 returned" in completed.stdout


def test_changed_partition_never_opens_previous_generation(tmp_path):
    a, b, root = tmp_path / "data-a", tmp_path / "data-b", tmp_path / "index"
    old = export(a)
    new = export(b, split_seed=11)
    changed = old.splits.manifest.retrieval_ids - new.splits.manifest.retrieval_ids
    assert changed & new.splits.manifest.held_out_ids
    assert changed & new.splits.manifest.development_ids
    previous = build(a, root)
    with pytest.raises(FileNotFoundError), open_index(b, root, embedding_config=CONFIG):
        pass
    assert build(b, root) != previous
    with open_index(b, root, embedding_config=CONFIG) as index:
        assert set(index.collection.get()["ids"]) == new.splits.manifest.retrieval_ids
        assert all(index.get_case(i) is None for i in changed)


def test_changed_model_sampling_and_payload_get_distinct_generations(tmp_path):
    data, other, root = tmp_path / "data", tmp_path / "updated", tmp_path / "index"
    export(data)
    original = build(data, root)
    generations = {original}
    for config, sampling in [
        (replace(CONFIG, revision="c" * 40), SamplingConfig()),
        (replace(CONFIG, model_name="synthetic/another"), SamplingConfig()),
        (CONFIG, SamplingConfig(seed=16)),
        (CONFIG, SamplingConfig(max_per_top_verdict=1)),
    ]:
        options = {"embedding_config": config, "sampling": sampling}
        with pytest.raises(FileNotFoundError), open_index(data, root, **options):
            pass
        generation = build_index(data, root, **options, load_embedder=TestEncoder)
        assert generation not in generations
        generations.add(generation)
    rows = [raw_row(f"case{i:02d}", verdict=VERDICTS[i % 4]) for i in range(20)]
    rows[0]["top_comment_1"] = "Revised full case evidence."
    revised = prepare(
        rows[:15],
        rows[15:],
        source_revision="b" * 40,
        split_strategy="combined_stratified",
        golden_minimums={},
        golden_size=3,
        development_size=2,
    )
    revised.report["source_dataset"] = "synthetic/index-tests"
    export_prepared(revised, other)
    assert build(other, root) not in generations


@pytest.mark.parametrize(
    "error",
    [
        "missing_vector",
        "wrong_dim",
        "nan",
        "infinite",
        "zero",
        "non_unit",
        "backend_revision",
    ],
)
def test_invalid_embedder_never_publishes(tmp_path, error):
    data, root = tmp_path / "data", tmp_path / "index"
    export(data)

    class Broken(TestEncoder):
        def manifest(self):
            result = super().manifest()
            if error == "backend_revision":
                result["revision"] = "wrong"
            return result

        def embed_documents(self, cases):
            rows = super().embed_documents(cases)
            if error == "missing_vector":
                return rows[:-1]
            if error == "wrong_dim":
                rows[0].pop()
            else:
                rows[0] = [
                    {
                        "nan": float("nan"),
                        "infinite": float("inf"),
                        "zero": 0.0,
                        "non_unit": 2.0,
                    }[error]
                ] + [0.0] * 1023
            return rows

    with pytest.raises(ValueError):
        build_index(data, root, embedding_config=CONFIG, load_embedder=Broken)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        "missing_manifest",
        "golden",
        "development",
        "missing_row",
        "duplicate_row",
        "unexpected_row",
        "overlap",
    ],
)
def test_invalid_export_fails_before_model_or_disk_write(tmp_path, change):
    data, root = tmp_path / "data", tmp_path / "index"
    prepared = export(data)
    file = data / "rag_candidates.jsonl"
    lines = file.read_text().splitlines()
    if change == "missing_manifest":
        (data / "split_manifest.json").unlink()
    elif change == "overlap":
        manifest = data / "split_manifest.json"
        raw = json.loads(manifest.read_text())
        raw["held_out_ids"].append(raw["retrieval_ids"][0])
        manifest.write_text(json.dumps(raw))
    else:
        if change == "golden":
            lines.append(prepared.splits.held_out[-1].model_dump_json())
        elif change == "development":
            lines.append(prepared.splits.development[0].model_dump_json())
        elif change == "missing_row":
            lines.pop()
        elif change == "duplicate_row":
            lines.append(lines[0])
        else:
            key = json.loads(lines[0])["post_id"]
            lines[0] = lines[0].replace(key, "unexpected99")
        file.write_text("\n".join(lines))
    with pytest.raises((ValueError, FileNotFoundError)):
        build_index(data, root, embedding_config=CONFIG, load_embedder=never_load)
    assert not root.exists()


@pytest.mark.parametrize(
    "config,source",
    [
        (replace(CONFIG, dimensions=16), None),
        (replace(CONFIG, revision="main"), None),
        (CONFIG, ("synthetic/other", "b" * 40)),
    ],
)
def test_incompatible_build_configuration_fails_before_writing(
    tmp_path, config, source
):
    data, root = tmp_path / "data", tmp_path / "index"
    export(data)
    with pytest.raises(ValueError):
        build_index(
            data,
            root,
            embedding_config=config,
            expected_source=source,
            load_embedder=never_load,
        )
    assert not root.exists()


def test_interrupted_build_and_retry_preserve_other_artifacts(tmp_path):
    data, root = tmp_path / "data", tmp_path / "index"
    export(data)
    root.mkdir()
    unrelated = root / "unrelated"
    unrelated.write_text("local reference")
    previous = build(data, root, sampling=SamplingConfig(seed=9))

    class FailAfterBatch(TestEncoder):
        calls = 0

        def embed_documents(self, cases):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interrupted")
            return super().embed_documents(cases)

    with pytest.raises(RuntimeError, match="interrupted"):
        build_index(data, root, embedding_config=CONFIG, load_embedder=FailAfterBatch)
    assert set(root.iterdir()) == {previous, unrelated}
    assert build(data, root) != previous
    assert unrelated.read_text() == "local reference"


@pytest.mark.parametrize(
    "damage",
    [
        "case",
        "missing_case",
        "extra_case",
        "extra_vector",
        "missing_vector",
        "wrong_metadata",
        "swapped_vector",
        "report",
        "identity",
    ],
)
def test_reopen_and_rebuild_refuse_damaged_generation(tmp_path, damage):
    data, root = tmp_path / "data", tmp_path / "index"
    export(data)
    generation = build(data, root)
    if damage in ("case", "extra_case", "missing_case"):
        path = generation / "cases.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        if damage == "case":
            lines[0] = lines[0].replace(b"opinion", b"advice")
        elif damage == "extra_case":
            lines.append(lines[0])
        else:
            lines.pop()
        path.write_bytes(b"".join(lines))
    elif damage in ("identity", "report"):
        path = generation / (
            "index_manifest.json" if damage == "identity" else "build_report.json"
        )
        obj = json.loads(path.read_text())
        obj["build_id"] = "stale"
        path.write_text(json.dumps(obj))
    else:
        with chromadb.PersistentClient(path=str(generation / "chroma")) as client:
            collection = client.get_collection(COLLECTION, embedding_function=None)
            key = collection.get(limit=1)["ids"][0]
            vector = [[1.0] + [0.0] * 1023]
            if damage == "extra_vector":
                collection.add(ids=["stale"], embeddings=vector)
            elif damage == "missing_vector":
                collection.delete(ids=[key])
            elif damage == "wrong_metadata":
                collection.update(ids=[key], metadatas=[{"post_id": "wrong"}])
            else:
                collection.update(ids=[key], embeddings=vector)
    with pytest.raises(ValueError), open_index(data, root, embedding_config=CONFIG):
        pass
    with pytest.raises(ValueError):
        build_index(data, root, embedding_config=CONFIG, load_embedder=never_load)


def test_empty_retrieval_partition_can_be_opened(tmp_path):
    data, root = tmp_path / "data", tmp_path / "index"
    export(data, n_rows=2, golden=1, dev=1)
    build_index(data, root, embedding_config=CONFIG, load_embedder=never_load)
    with open_index(data, root, embedding_config=CONFIG) as index:
        assert index.collection.count() == 0
        assert index.get_case("case00") is None


def test_concurrent_builders_publish_one_complete_generation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    data, root = tmp_path / "data", tmp_path / "index"
    export(data)
    both_staging = Barrier(2)

    def synchronized_encoder(config):
        both_staging.wait(timeout=15)
        return TestEncoder(config)

    def worker():
        return build_index(
            data, root, embedding_config=CONFIG, load_embedder=synchronized_encoder
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(worker)
        second = executor.submit(worker)
        assert first.result() == second.result()
    assert list(root.iterdir()) == [first.result()]
    with open_index(data, root, embedding_config=CONFIG) as index:
        assert index.collection.count() == 15


def test_build_cli_validates_and_reuses_generation_without_model(tmp_path):
    import yaml

    data, root = tmp_path / "data", tmp_path / "index"
    prepared = export(data)
    sampling = SamplingConfig(seed=12, max_per_top_verdict=1)
    generation = build(data, root, sampling=sampling)
    settings = {
        "project": {"seed": 42},
        "rag": {
            "embedding_model": CONFIG.model_name,
            "embedding_revision": CONFIG.revision,
            "max_per_top_verdict": 5000,
            "index_max_posts": 20000,
        },
        "data_sources": {
            "aita": {
                "hf_dataset": prepared.report["source_dataset"],
                "revision": prepared.report["source_revision"],
            }
        },
    }
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml.safe_dump(settings))
    cmd = [
        sys.executable,
        str(DEMO.with_name("build_index.py")),
        "--prepared-dir",
        str(data),
        "--index-root",
        str(root),
        "--config",
        str(config_file),
        "--seed",
        "12",
        "--max-per-top-verdict",
        "1",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=True)
    assert str(generation) in completed.stdout
    assert json.loads(completed.stdout.split("\n", 1)[1])["indexed_cases"] == 4


def _terminate_mid_build(data, root):
    import os

    class Interrupted(TestEncoder):
        calls = 0

        def embed_documents(self, cases):
            self.calls += 1
            if self.calls == 2:
                os._exit(23)
            return super().embed_documents(cases)

    build_index(data, root, embedding_config=CONFIG, load_embedder=Interrupted)


def test_process_death_leaves_only_unpublished_staging_and_retry_succeeds(tmp_path):
    import multiprocessing

    data, root = tmp_path / "data", tmp_path / "index"
    export(data)
    process = multiprocessing.get_context("spawn").Process(
        target=_terminate_mid_build, args=(data, root)
    )
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("Interrupted builder timed out")
    assert process.exitcode == 23
    [abandoned] = root.iterdir()
    assert abandoned.name.startswith(".building-")
    with (
        pytest.raises(FileNotFoundError),
        open_index(data, root, embedding_config=CONFIG),
    ):
        pass
    generation = build(data, root)
    assert set(root.iterdir()) == {abandoned, generation}
    with open_index(data, root, embedding_config=CONFIG) as index:
        assert index.collection.count() == 15


def _publish_and_record(data, root, ready, receipt):
    """Synchronize real rename calls and identify each process's staged generation."""
    from hashlib import sha256
    from unittest.mock import patch

    rename = Path.rename
    attempt = {}

    def synchronized_rename(stage, destination):
        attempt["inode"] = stage.stat().st_ino
        attempt["manifest"] = sha256(
            (stage / "index_manifest.json").read_bytes()
        ).hexdigest()
        ready.wait(timeout=30)
        try:
            return rename(stage, destination)
        except OSError as error:
            attempt["errno"] = error.errno
            raise

    with patch.object(Path, "rename", synchronized_rename):
        generation = build(data, root)
    receipt.write_text(json.dumps({**attempt, "generation": generation.name}))


def test_independent_processes_preserve_first_published_generation(tmp_path):
    import errno
    import multiprocessing
    from hashlib import sha256

    data, root = tmp_path / "data", tmp_path / "index"
    prepared = export(data)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Barrier(2)
    receipts = [tmp_path / f"receipt-{i}.json" for i in range(2)]
    writers = [
        ctx.Process(target=_publish_and_record, args=(data, root, ready, path))
        for path in receipts
    ]
    try:
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join(timeout=45)
            assert writer.exitcode == 0
    finally:
        for writer in writers:
            if writer.is_alive():
                writer.kill()
                writer.join()

    attempts = [json.loads(path.read_text()) for path in receipts]
    [winner] = [a for a in attempts if "errno" not in a]
    [loser] = [a for a in attempts if "errno" in a]
    assert loser["errno"] in (errno.EEXIST, errno.ENOTEMPTY)
    assert loser["inode"] != winner["inode"]
    [published] = root.iterdir()
    assert published.name == winner["generation"] == loser["generation"]
    assert published.stat().st_ino == winner["inode"]
    assert (
        sha256((published / "index_manifest.json").read_bytes()).hexdigest()
        == winner["manifest"]
    )
    with open_index(data, root, embedding_config=CONFIG) as index:
        assert (
            set(index.collection.get()["ids"]) == prepared.splits.manifest.retrieval_ids
        )
