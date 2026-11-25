"""DATA-04: split isolation, reproducibility, and EVAL-01 compatibility."""

from typing import Literal

import pytest
from pydantic import HttpUrl

from asshole_ai_detector.data.splits import attach_semantic_scan, split_cases
from asshole_ai_detector.schemas.annotation import Verdict
from asshole_ai_detector.schemas.case import AITACase, AITAComment, DerivedFields

MINIMUMS = {"NTA": 1, "YTA": 1, "ESH": 1, "NAH": 1, "DISPUTED": 1}


def make_case(
    post_id: str, label: Verdict | Literal["DISPUTED"] = "NTA"
) -> AITACase:
    story = (
        f"I need advice about situation {post_id} in my household. "
        + "Several people were involved in the disagreement and we tried to resolve it calmly. " * 6
    )
    verdicts: list[Verdict]
    if label == "DISPUTED":
        verdicts = ["NTA", "NTA", "NTA", "YTA", "YTA"]
    else:
        verdicts = [label, label, label, label, "YTA" if label != "YTA" else "NTA"]
    comments = [
        AITAComment(
            comment_id=f"{post_id}#{i}",
            rank=i,
            verdict=verdict,
            text=f"Synthetic moral opinion {i}",
        )
        for i, verdict in enumerate(verdicts, start=1)
    ]
    return AITACase(
        post_id=post_id,
        title=f"Question about {post_id}",
        text=story,
        url=HttpUrl(f"https://example.com/{post_id}"),
        valid_comments=comments,
        derived_fields=DerivedFields.from_comments(comments),
    )


def make_records() -> tuple[list[AITACase], set[str]]:
    labels: tuple[Verdict | Literal["DISPUTED"], ...] = (
        "NTA", "YTA", "ESH", "NAH", "DISPUTED"
    )
    train = [make_case(f"train-{i:02d}", labels[i % 5]) for i in range(15)]
    test = [make_case(f"test-{i:02d}", labels[i % 5]) for i in range(25)]
    return train + test, {case.post_id for case in test}


def run_split(
    cases: list[AITACase] | None = None,
    test_ids: set[str] | None = None,
    *,
    seed: int = 42,
    golden_size: int = 10,
    development_size: int = 5,
    minimums: dict[str, int] | None = None,
):
    initial, provider_test_ids = make_records()
    return split_cases(
        cases if cases is not None else initial,
        test_ids=test_ids if test_ids is not None else provider_test_ids,
        source_revision="pinned-hf-revision-123",
        golden_size=golden_size,
        development_size=development_size,
        golden_minimums=MINIMUMS if minimums is None else minimums,
        seed=seed,
    )


def test_partitions_are_disjoint_and_keep_provider_separation() -> None:
    result = run_split()
    assert len(result.held_out) == 10
    assert len(result.development) == 5
    assert len(result.retrieval) == 15
    assert len(result.unused_test) == 10
    assert all(case.post_id.startswith("train") for case in result.retrieval)
    assert all(case.post_id.startswith("test") for case in result.held_out)
    assert all(case.post_id.startswith("test") for case in result.development)
    held, dev, retrieval = (
        result.manifest.held_out_ids,
        result.manifest.development_ids,
        result.manifest.retrieval_ids,
    )
    assert not held & dev
    assert not held & retrieval
    assert not dev & retrieval
    assert result.manifest.partition_sizes == (10, 5, 15)
    assert result.manifest.exact_groups == ()
    assert result.manifest.semantic_groups is None


def test_reordered_input_is_identical() -> None:
    cases, ids = make_records()
    original = run_split(cases, ids)
    reversed_order = run_split(list(reversed(cases)), ids)
    assert original.manifest == reversed_order.manifest
    assert original.held_out == reversed_order.held_out
    assert original.development == reversed_order.development


def test_same_seed_produces_same_manifest_id() -> None:
    assert run_split().manifest.manifest_id == run_split().manifest.manifest_id


def test_different_seed_changes_assignment() -> None:
    first = run_split(seed=42)
    second = run_split(seed=43)
    assert first.manifest.held_out_ids != second.manifest.held_out_ids


def test_respects_gold_minimums() -> None:
    result = run_split()
    from collections import Counter
    counts = Counter(case.derived_fields.consensus_label for case in result.held_out)
    assert all(counts[label] >= count for label, count in MINIMUMS.items())


def test_rejects_input_duplicate_id() -> None:
    cases, ids = make_records()
    with pytest.raises(ValueError, match="Repeated post IDs"):
        run_split(cases + [cases[0]], ids)


def test_rejects_identical_content_across_different_ids() -> None:
    cases, ids = make_records()
    duplicated = cases[0].model_copy(deep=True)
    duplicated.post_id = "new-id"
    duplicated.url = HttpUrl("https://example.com/new-id")
    duplicated.valid_comments = [
        comment.model_copy(update={"comment_id": f"new-id#{comment.rank}"})
        for comment in duplicated.valid_comments
    ]
    # model_copy won't run model validators, but this remains schema-consistent.
    with pytest.raises(ValueError, match="Repeated normalized narratives"):
        run_split(cases + [duplicated], ids)


def test_rejects_insufficient_provider_test_cases() -> None:
    with pytest.raises(ValueError, match="only 25"):
        run_split(golden_size=20, development_size=10)


def test_rejects_insufficient_rare_label_quota() -> None:
    with pytest.raises(ValueError, match="Not enough NAH"):
        run_split(minimums={"NAH": 6})


def test_manifest_bound_to_source_and_content() -> None:
    cases, ids = make_records()
    before = run_split(cases, ids)
    updated = cases[0].model_copy(deep=True)
    updated.text += " This is an additional distinct detail."
    after = run_split([updated, *cases[1:]], ids)
    assert before.manifest.manifest_id != after.manifest.manifest_id


def test_semantic_scan_status_requires_actual_evidence() -> None:
    original = run_split().manifest
    assert original.semantic_groups is None
    scanned = attach_semantic_scan(original, [])
    assert scanned.semantic_groups == ()
    with pytest.raises(ValueError, match="2\\+ retained"):
        attach_semantic_scan(original, [{"unknown-id", "test-01"}])
