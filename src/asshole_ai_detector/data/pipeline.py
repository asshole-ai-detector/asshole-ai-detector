"""DATA-05: integrate DATA-01..04, preserving exact source provenance.

This exports preliminary pools, not final Golden decisions or semantic scans.
"""

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from asshole_ai_detector.data.aita_adapter import (
    HF_DATASET,
    HF_REVISION,
    iter_normalized_cases,
)
from asshole_ai_detector.data.cleaning import (
    DEFAULT_CLEANING_CONFIG,
    CleanedAITACase,
    CleaningReport,
    clean_case,
)
from asshole_ai_detector.data.consensus import label_cleaned_case
from asshole_ai_detector.data.deduplication import (
    DeduplicationResult,
    deduplicate_cases,
)
from asshole_ai_detector.data.splits import (
    DEFAULT_GOLDEN_MINIMUMS,
    SplitManifest,
    SplitResult,
    SplitStrategy,
    split_cases,
)
from asshole_ai_detector.schemas.case import AITACase, NormalizedAITACase


@dataclass
class PreparedData:
    splits: SplitResult
    golden_sources: dict[str, NormalizedAITACase]
    report: dict[str, Any]


def _label_rows(
    rows: Iterable[Mapping[str, Any]],
    report: CleaningReport,
    *,
    source_index: dict[tuple[str, str], NormalizedAITACase] | None = None,
) -> tuple[list[AITACase], int]:
    """Normalize iterable raw rows; optionally preserve original rows for either source.

    A Hugging Face Dataset is an iterable whose individual rows are dicts.
    `iter_normalized_cases(rows)` therefore accepts a Dataset directly.
    """
    labeled: list[AITACase] = []
    total = 0
    for normalized in iter_normalized_cases(rows):
        total += 1
        cleaned = clean_case(normalized, report=report)
        if not isinstance(cleaned, CleanedAITACase):
            continue
        case = label_cleaned_case(cleaned)
        labeled.append(case)
        if source_index is not None:
            # Same-ID source rows can differ. Match the *canonical labeled
            # record* selected by DATA-04, not just its post ID.
            key = (case.post_id, case.model_dump_json())
            existing = source_index.get(key)
            if existing is None or normalized.model_dump_json() < existing.model_dump_json():
                source_index[key] = normalized
    return labeled, total


def prepare(
    train_rows: Iterable[Mapping[str, Any]],
    test_rows: Iterable[Mapping[str, Any]],
    *,
    source_revision: str = HF_REVISION,
    golden_size: int = 800,
    development_size: int = 500,
    seed: int = 42,
    golden_minimums: Mapping[str, int] | None = None,
    development_minimums: Mapping[str, int] | None = None,
    split_strategy: SplitStrategy = "provider_test",
) -> PreparedData:
    """Clean both provider partitions, deduplicate together, then split.

    ``provider_test`` preserves the old default for existing callers/tests.
    The CLI explicitly opts into ``combined_stratified`` via configuration.
    Retain ORIGINAL source rows from both provider partitions, because a
    combined Golden pool can contain former train records.
    """
    cleaning = CleaningReport()
    train_sources: dict[tuple[str, str], NormalizedAITACase] = {}
    test_sources: dict[tuple[str, str], NormalizedAITACase] = {}
    train_cases, n_train = _label_rows(
        train_rows, cleaning, source_index=train_sources
    )
    test_cases, n_test = _label_rows(
        test_rows, cleaning, source_index=test_sources
    )
    # Match DATA-04's preferred_ids policy if identical labeled records came
    # from both original sources: prefer the original test source row.
    indexed_sources = {**train_sources, **test_sources}

    test_ids = {case.post_id for case in test_cases}
    dedup: DeduplicationResult = deduplicate_cases(
        [*train_cases, *test_cases], preferred_ids=test_ids
    )
    eligible_test_counts = Counter(
        case.derived_fields.consensus_label
        for case in dedup.cases
        if case.post_id in test_ids
    )
    eligible_combined_counts = Counter(
        case.derived_fields.consensus_label for case in dedup.cases
    )
    effective_golden_minimums = (
        dict(DEFAULT_GOLDEN_MINIMUMS)
        if golden_minimums is None
        else dict(golden_minimums)
    )
    try:
        splits = split_cases(
            dedup.cases,
            test_ids=test_ids,
            source_revision=source_revision,
            golden_size=golden_size,
            development_size=development_size,
            seed=seed,
            golden_minimums=effective_golden_minimums,
            development_minimums=development_minimums,
            strategy=split_strategy,
        )
    except ValueError as error:
        raise ValueError(
            f"DATA-04 splitting failed: {error}. "
            f"Eligible combined counts: {dict(sorted(eligible_combined_counts.items()))}; "
            f"Eligible provider-test counts: {dict(sorted(eligible_test_counts.items()))}. "
            "Coordinate allocations with EVAL-01 rather than silently relaxing quotas."
        ) from error

    golden_sources: dict[str, NormalizedAITACase] = {}
    for case in splits.held_out:
        source = indexed_sources.get((case.post_id, case.model_dump_json()))
        if source is None:
            raise ValueError(
                f"Held-out case {case.post_id} has no matching original source row"
            )
        golden_sources[case.post_id] = source

    def label_counts(cases: list[AITACase]) -> dict[str, int]:
        return dict(sorted(Counter(c.derived_fields.consensus_label for c in cases).items()))

    def source_counts(cases: list[AITACase]) -> dict[str, int]:
        count_test = sum(case.post_id in test_ids for case in cases)
        return {"original_train": len(cases) - count_test, "original_test": count_test}

    report: dict[str, Any] = {
        "schema_version": 1,
        "source_dataset": HF_DATASET,
        "source_revision": source_revision,
        "seed": seed,
        "cleaning_config": asdict(DEFAULT_CLEANING_CONFIG),
        "consensus_threshold": 0.70,
        "partition_strategy": split_strategy,
        "golden_minimums": dict(sorted(effective_golden_minimums.items())),
        "development_minimums": dict(sorted((development_minimums or {}).items())),
        "original_counts": {"train": n_train, "test": n_test},
        "cleaning": cleaning.to_dict(),
        "deduplication": {
            "input_count": dedup.input_count,
            "unique_count": len(dedup.cases),
            "removed_count": dedup.removed_count,
            "removal_reasons": dict(sorted(Counter(d.reason for d in dedup.duplicates).items())),
        },
        "splits": {
            "golden_candidate_pool": len(splits.held_out),
            "development": len(splits.development),
            "retrieval": len(splits.retrieval),
            "unused_test": len(splits.unused_test),
            "label_counts": {
                "held_out": label_counts(splits.held_out),
                "development": label_counts(splits.development),
                "retrieval": label_counts(splits.retrieval),
            },
            "source_partition_counts": {
                "held_out": source_counts(splits.held_out),
                "development": source_counts(splits.development),
                "retrieval": source_counts(splits.retrieval),
            },
            "eligible_test_counts": dict(sorted(eligible_test_counts.items())),
            "eligible_combined_counts": dict(sorted(eligible_combined_counts.items())),
        },
        "manifest_id": splits.manifest.manifest_id,
        "semantic_near_duplicate_scan": "not_run",
        "golden_selection": "not_run_by_DATA-05",
    }
    return PreparedData(splits, golden_sources, report)


def _manifest_to_json(manifest: SplitManifest) -> dict[str, Any]:
    def groups(value: tuple[frozenset[str], ...] | None) -> list[list[str]] | None:
        return None if value is None else [sorted(group) for group in value]

    return {
        "manifest_id": manifest.manifest_id,
        "source_revision": manifest.source_revision,
        "strategy": manifest.strategy,
        "held_out_ids": sorted(manifest.held_out_ids),
        "development_ids": sorted(manifest.development_ids),
        "retrieval_ids": sorted(manifest.retrieval_ids),
        "partition_sizes": list(manifest.partition_sizes),
        "exact_groups": groups(manifest.exact_groups),
        "semantic_groups": groups(manifest.semantic_groups),
    }


def _manifest_from_json(data: Mapping[str, Any]) -> SplitManifest:
    def groups(value: list[list[str]] | None) -> tuple[frozenset[str], ...] | None:
        return None if value is None else tuple(frozenset(group) for group in value)

    return SplitManifest(
        manifest_id=data["manifest_id"],
        source_revision=data["source_revision"],
        strategy=data.get("strategy", "provider_test"),
        held_out_ids=frozenset(data["held_out_ids"]),
        development_ids=frozenset(data["development_ids"]),
        retrieval_ids=frozenset(data["retrieval_ids"]),
        partition_sizes=tuple(data["partition_sizes"]),
        exact_groups=groups(data["exact_groups"]),
        semantic_groups=groups(data["semantic_groups"]),
    )


def _write_jsonl(path: Path, records: Iterable[AITACase | NormalizedAITACase]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json() + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_prepared(prepared: PreparedData, output_dir: Path) -> None:
    """Write private local artifacts. Refuse nonempty folders to avoid stale mixing."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split = prepared.splits
    _write_jsonl(output_dir / "golden_pool.jsonl", split.held_out)
    _write_jsonl(output_dir / "development.jsonl", split.development)
    _write_jsonl(output_dir / "rag_candidates.jsonl", split.retrieval)
    _write_jsonl(output_dir / "unused_test.jsonl", split.unused_test)
    _write_jsonl(
        output_dir / "golden_source_rows.jsonl",
        (prepared.golden_sources[case.post_id] for case in split.held_out),
    )
    _write_json(output_dir / "split_manifest.json", _manifest_to_json(split.manifest))
    _write_json(output_dir / "preprocessing_report.json", prepared.report)


def _read_jsonl(path: Path, model: type[AITACase] | type[NormalizedAITACase]) -> list:
    with path.open(encoding="utf-8") as file:
        return [model.model_validate_json(line) for line in file if line.strip()]


def load_prepared(output_dir: Path) -> PreparedData:
    """Restore DATA-04 SplitResult and source rows for EVAL-01."""
    manifest_data = json.loads((output_dir / "split_manifest.json").read_text("utf-8"))
    manifest = _manifest_from_json(manifest_data)
    report = json.loads((output_dir / "preprocessing_report.json").read_text("utf-8"))
    held_out = _read_jsonl(output_dir / "golden_pool.jsonl", AITACase)
    development = _read_jsonl(output_dir / "development.jsonl", AITACase)
    retrieval = _read_jsonl(output_dir / "rag_candidates.jsonl", AITACase)
    unused = _read_jsonl(output_dir / "unused_test.jsonl", AITACase)
    sources = _read_jsonl(output_dir / "golden_source_rows.jsonl", NormalizedAITACase)
    partition_ids = (
        frozenset(c.post_id for c in held_out),
        frozenset(c.post_id for c in development),
        frozenset(c.post_id for c in retrieval),
    )
    if partition_ids != (
        manifest.held_out_ids,
        manifest.development_ids,
        manifest.retrieval_ids,
    ) or manifest.partition_sizes != tuple(map(len, partition_ids)):
        raise ValueError("Saved partitions disagree with the split manifest")
    if sum(map(len, (held_out, development, retrieval))) != sum(map(len, partition_ids)):
        raise ValueError("Duplicate IDs found in saved partitions")
    if any(partition_ids[i] & partition_ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Saved partitions overlap")
    source_map = {s.post_id: s for s in sources}
    if len(source_map) != len(sources) or source_map.keys() != manifest.held_out_ids:
        raise ValueError("Golden source rows don't match the held-out partition")
    if report["manifest_id"] != manifest.manifest_id or report["source_revision"] != manifest.source_revision:
        raise ValueError("Preparation report disagrees with the manifest")
    splits = SplitResult(held_out, development, retrieval, unused, manifest, report["seed"])
    return PreparedData(splits, source_map, report)


