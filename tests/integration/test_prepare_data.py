"""DATA-05 integration tests, with synthetic Hugging Face-style rows only."""

import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from asshole_ai_detector.data.pipeline import (
    export_prepared,
    load_prepared,
    prepare,
)


def story(topic: str) -> str:
    return (
        f"I had an argument involving {topic} and I am trying to understand what happened. "
        + "My roommate and I discussed the situation several times and considered the consequences. "
        * 7
    )


def raw_row(
    post_id: str,
    *,
    narrative: str | None = None,
    verdict: str = "NTA",
    include_info: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "submission_url": f"https://www.reddit.com/r/AmItheAsshole/comments/{post_id}/story/",
        "submission_title": f"AITA for a conflict involving {post_id}",
        "submission_text": narrative if narrative is not None else story(post_id),
        "submission_date": "2022-01-10 12:00:00",
        "ambiguity_score": 0.1,
    }
    for rank in range(1, 6):
        row[f"top_comment_{rank}"] = f"This is my genuine opinion number {rank}."
        row[f"top_comment_{rank}_classification"] = verdict
    if include_info:
        row["top_comment_6"] = "We need more information about what happened."
        row["top_comment_6_classification"] = "INFO"
    return row


class FakeDataset:
    """Represents the relevant HF Dataset behavior: iterating yields mappings."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        yield from self.rows


def sample_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # The first train record duplicates a test narrative with another post ID.
    duplicate_narrative = story("shared-narrative")
    train = [
        raw_row("train001", narrative=duplicate_narrative),
        raw_row("train002"),
    ]
    test = [
        raw_row("test001"),
        raw_row("test002", narrative=duplicate_narrative, include_info=True),
        raw_row("test003", verdict="ESH"),
        raw_row("test004", verdict="YTA"),
    ]
    return train, test


def run_test_pipeline(*, seed: int = 42):
    train, test = sample_rows()
    return prepare(
        FakeDataset(train),
        FakeDataset(test),
        source_revision="synthetic-test-revision",
        golden_size=2,
        development_size=1,
        golden_minimums={},  # Real quota rules are tested by DATA-04.
        seed=seed,
    )


def test_accepts_dataset_iterable_and_preserves_provider_split() -> None:
    prepared = run_test_pipeline()
    split = prepared.splits
    assert len(split.held_out) == 2
    assert len(split.development) == 1
    assert len(split.retrieval) == 1
    assert len(split.unused_test) == 1
    assert split.retrieval[0].post_id == "train002"
    assert all(case.post_id.startswith("test") for case in split.held_out)
    assert all(case.post_id.startswith("test") for case in split.development)
    assert prepared.report["deduplication"]["removed_count"] == 1
    assert split.manifest.exact_groups == ()
    assert split.manifest.semantic_groups is None


def test_reproducibility_and_input_order() -> None:
    train, test = sample_rows()
    first = run_test_pipeline()
    second = prepare(
        reversed(train),
        reversed(test),
        source_revision="synthetic-test-revision",
        golden_size=2,
        development_size=1,
        golden_minimums={},
        seed=42,
    )
    assert first.splits.manifest == second.splits.manifest


def test_original_source_keeps_info_and_matches_selected_record() -> None:
    train, test = sample_rows()
    prepared = prepare(
        train,
        test,
        source_revision="synthetic-test-revision",
        golden_size=4,
        development_size=0,
        golden_minimums={},
    )
    assert set(prepared.golden_sources) == prepared.splits.manifest.held_out_ids
    info_source = prepared.golden_sources["test002"]
    assert any(comment.verdict == "INFO" for comment in info_source.comments)
    for case in prepared.splits.held_out:
        assert prepared.golden_sources[case.post_id].post_id == case.post_id


def test_export_and_reload_for_eval01(tmp_path) -> None:
    prepared = run_test_pipeline()
    output = tmp_path / "private_data"
    export_prepared(prepared, output)
    reloaded = load_prepared(output)
    assert reloaded.splits.manifest == prepared.splits.manifest
    assert reloaded.splits.held_out == prepared.splits.held_out
    assert reloaded.splits.retrieval == prepared.splits.retrieval
    assert reloaded.golden_sources == prepared.golden_sources
    assert json.loads((output / "split_manifest.json").read_text())["semantic_groups"] is None
    assert (output / "preprocessing_report.json").is_file()
    with pytest.raises(FileExistsError):
        export_prepared(prepared, output)


def test_fails_with_actionable_quota_error() -> None:
    train, test = sample_rows()
    with pytest.raises(ValueError, match="Eligible provider-test counts"):
        prepare(
            train,
            test,
            source_revision="synthetic-test-revision",
            golden_size=2,
            development_size=1,
            golden_minimums={"ESH": 3},
        )


def test_load_rejects_tampered_partition_membership(tmp_path) -> None:
    prepared = run_test_pipeline()
    output = tmp_path / "private_data"
    export_prepared(prepared, output)
    manifest_path = output / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["held_out_ids"].append("forged_id")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree with the split manifest"):
        load_prepared(output)


def test_combined_golden_keeps_original_train_source(tmp_path) -> None:
    train, test = sample_rows()
    # Three distinct NTA records exist across both original partitions:
    # train002, test001, and test002. The shared train001 narrative is removed.
    prepared = prepare(
        train,
        test,
        source_revision="synthetic-test-revision",
        golden_size=3,
        development_size=1,
        golden_minimums={"NTA": 3},
        development_minimums={},
        split_strategy="combined_stratified",
    )
    split = prepared.splits
    assert split.manifest.strategy == "combined_stratified"
    assert any(case.post_id == "train002" for case in split.held_out)
    assert prepared.golden_sources["train002"].post_id == "train002"
    assert split.unused_test == []
    assert len(split.retrieval) == 1
    assert prepared.report["splits"]["source_partition_counts"]["held_out"]["original_train"] >= 1
    output = tmp_path / "combined"
    export_prepared(prepared, output)
    reloaded = load_prepared(output)
    assert reloaded.splits.manifest == split.manifest
    assert reloaded.golden_sources == prepared.golden_sources


def test_combined_fails_on_impossible_rare_class_quota() -> None:
    train, test = sample_rows()
    with pytest.raises(ValueError, match="Eligible combined counts"):
        prepare(
            train, test,
            source_revision="synthetic-test-revision",
            golden_size=2,
            development_size=1,
            golden_minimums={"ESH": 2},
            split_strategy="combined_stratified",
        )


def test_duplicate_provider_id_prefers_original_test_source_row() -> None:
    train = [raw_row("shared"), raw_row("trainonly")]
    test = [raw_row("shared", include_info=True), raw_row("testonly")]
    prepared = prepare(
        train,
        test,
        source_revision="synthetic-test-revision",
        golden_size=3,
        development_size=0,
        golden_minimums={},
        split_strategy="combined_stratified",
    )
    assert prepared.report["deduplication"]["removed_count"] == 1
    assert any(
        comment.verdict == "INFO"
        for comment in prepared.golden_sources["shared"].comments
    )
    assert "trainonly" in prepared.golden_sources


def test_strict_verification_is_opt_in_for_historical_exports(tmp_path):
    prepared = run_test_pipeline()
    export_prepared(prepared, tmp_path)
    assert load_prepared(tmp_path, strict=True).splits == prepared.splits
    path = tmp_path / "split_manifest.json"
    raw = json.loads(path.read_text())
    del raw["strategy"]
    path.write_text(json.dumps(raw))
    assert load_prepared(tmp_path).splits == prepared.splits
    with pytest.raises(ValueError, match="re-export"):
        load_prepared(tmp_path, strict=True)


@pytest.mark.parametrize(
    "inventory", ["held_out_ids", "development_ids", "retrieval_ids"]
)
def test_strict_verification_detects_repeated_raw_inventory_ids(tmp_path, inventory):
    export_prepared(run_test_pipeline(), tmp_path)
    path = tmp_path / "split_manifest.json"
    raw = json.loads(path.read_text())
    raw[inventory].append(raw[inventory][0])
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Duplicate IDs in manifest"):
        load_prepared(tmp_path, strict=True)


@pytest.mark.parametrize(
    "partition", ["rag_candidates", "development", "golden_pool", "unused_test"]
)
def test_strict_verification_binds_full_payloads(tmp_path, partition):
    export_prepared(run_test_pipeline(), tmp_path)
    path = tmp_path / f"{partition}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["valid_comments"][0]["text"] = "Different evidence under an unchanged ID."
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="fingerprint"):
        load_prepared(tmp_path, strict=True)


@pytest.mark.parametrize(
    "change", ["size", "source", "strategy", "seed", "cleaning", "semantic"]
)
def test_strict_verification_rejects_inconsistent_report(tmp_path, change):
    export_prepared(run_test_pipeline(), tmp_path)
    path = tmp_path / "preprocessing_report.json"
    report = json.loads(path.read_text())
    if change == "size":
        report["splits"]["retrieval"] += 1
    elif change == "source":
        report["source_revision"] = "another-revision"
    elif change == "strategy":
        report["partition_strategy"] = "combined_stratified"
    elif change == "seed":
        report["seed"] += 1
    elif change == "cleaning":
        report["cleaning_config"]["min_post_words"] = 1
    else:
        report["semantic_near_duplicate_scan"] = "verified"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_prepared(tmp_path, strict=True)


@pytest.mark.parametrize(
    ("strategy", "field", "partition", "bad_counts"),
    [
        # Correct row total, wrong consensus labels.
        ("combined_stratified", "label_counts", "held_out", {"NAH": 2}),
        ("provider_test", "label_counts", "retrieval", {"NTA": True}),
        ("combined_stratified", "eligible_combined_counts", None, {"NTA": 5}),
        # Same test total, but too many ESH cases for the combined corpus.
        ("combined_stratified", "eligible_test_counts", None, {"ESH": 4}),
        # Feasible for the whole corpus, incorrect for the known provider-test cases.
        ("provider_test", "eligible_test_counts", None, {"NTA": 3, "ESH": 1}),
        (
            "combined_stratified",
            "source_partition_counts",
            "retrieval",
            {"original_train": 1, "original_test": 0},
        ),
        (
            "combined_stratified",
            "source_partition_counts",
            "retrieval",
            {"original_train": 3, "original_test": -1},
        ),
        (
            "combined_stratified",
            "source_partition_counts",
            "retrieval",
            {"original_train": True, "original_test": 1},
        ),
        # Correct row size, impossible when reconciled with the eligible-test total.
        (
            "combined_stratified",
            "source_partition_counts",
            "development",
            {"original_train": 1, "original_test": 0},
        ),
    ],
)
def test_report_distributions_checked_only_in_strict_mode(
    tmp_path, strategy, field, partition, bad_counts
):
    train, test = sample_rows()
    prepared = prepare(
        train,
        test,
        split_strategy=strategy,
        golden_size=2,
        development_size=1,
        golden_minimums={},
    )
    export_prepared(prepared, tmp_path)
    assert load_prepared(tmp_path, strict=True).splits == prepared.splits
    path = tmp_path / "preprocessing_report.json"
    report = json.loads(path.read_text())
    if partition is None:
        report["splits"][field] = bad_counts
    else:
        report["splits"][field][partition] = bad_counts
    path.write_text(json.dumps(report))
    assert load_prepared(tmp_path).splits == prepared.splits
    with pytest.raises(ValueError, match="counts"):
        load_prepared(tmp_path, strict=True)


def test_provider_source_swap_with_plausible_margins_is_rejected(tmp_path):
    prepared = run_test_pipeline()
    sources = prepared.report["splits"]["source_partition_counts"]
    sources["development"], sources["retrieval"] = (
        sources["retrieval"],
        sources["development"],
    )
    export_prepared(prepared, tmp_path)
    with pytest.raises(ValueError, match="source_partition_counts"):
        load_prepared(tmp_path, strict=True)


def test_combined_export_cannot_identify_provider_origin_per_case(tmp_path):
    rows = [raw_row(f"case{i}") for i in range(4)]
    options = {
        "split_strategy": "combined_stratified",
        "golden_size": 1,
        "development_size": 1,
        "golden_minimums": {},
    }
    first = prepare(rows[:2], rows[2:], **options)
    second = prepare(rows[2:], rows[:2], **options)
    assert first.splits == second.splits
    assert (
        first.report["splits"]["source_partition_counts"]
        != second.report["splits"]["source_partition_counts"]
    )
    for name, prepared in (("first", first), ("second", second)):
        export_prepared(prepared, tmp_path / name)
        assert load_prepared(tmp_path / name, strict=True).splits == prepared.splits


def test_source_counts_cannot_exceed_available_provider_rows(tmp_path):
    prepared = run_test_pipeline()
    prepared.report["original_counts"]["test"] = len(prepared.splits.held_out)
    export_prepared(prepared, tmp_path)
    with pytest.raises(ValueError, match="source counts"):
        load_prepared(tmp_path, strict=True)
