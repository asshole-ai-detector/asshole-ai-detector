"""Tests for preliminary Golden Dataset candidate selection."""

import json
from dataclasses import replace

import pytest

from asshole_ai_detector.data.deduplication import deduplicate_cases
from asshole_ai_detector.data.splits import (
    SplitResult,
    attach_semantic_scan,
    split_cases,
)
from asshole_ai_detector.evaluation.golden_selection import (
    GoldenSelectionResult,
    export_golden_candidates,
    export_selection_report,
    select_golden_candidates,
)
from asshole_ai_detector.evaluation.golden_validation import validate_golden_selection
from asshole_ai_detector.schemas.case import (
    AITACase,
    AITAComment,
    DerivedFields,
)

VERDICTS = ("NTA", "YTA", "ESH", "NAH")


def make_case(
    post_id: str,
    votes: list[str],
) -> AITACase:
    comments = [
        AITAComment(
            comment_id=f"{post_id}#{rank}",
            rank=rank,
            verdict=verdict,
            text=f"Valid moral comment {rank}",
        )
        for rank, verdict in enumerate(votes, start=1)
    ]

    return AITACase(
        post_id=post_id,
        title="A sample interpersonal conflict",
        text=f"Synthetic scenario {post_id}. "
        + "This is a detailed example of a conflict. " * 12,
        url=f"https://www.reddit.com/r/AmItheAsshole/comments/{post_id}/",
        valid_comments=comments,
        derived_fields=DerivedFields.from_comments(comments),
    )


@pytest.fixture
def split() -> SplitResult:
    test_cases = [
        make_case("consensus", ["NTA"] * 7 + ["YTA"] * 3),
        make_case("nominee", ["NTA"] * 5),
        make_case("split-vote", ["YTA"] * 3 + ["NTA"] * 2),
        make_case("too-many", ["ESH"] * 11),
        make_case("other-test-1", ["YTA"] * 5),
        make_case("other-test-2", ["YTA"] * 5),
    ]
    test_ids = {case.post_id for case in test_cases}
    corpus = deduplicate_cases(
        [make_case("train", ["NAH"] * 5), *test_cases], preferred_ids=test_ids
    )
    return split_cases(
        corpus.cases,
        test_ids=test_ids,
        source_revision="synthetic-revision",
        golden_size=4,
        development_size=1,
        golden_minimums={"NTA": 2, "DISPUTED": 1, "ESH": 1},
    )


def test_split_selection_uses_held_out_eligibility_and_provenance(split):
    result = select_golden_candidates(
        split, missing_context_review_ids={"nominee"}, seed=37
    )

    assert result.seed == 37
    assert (result.manifest_id, result.source_revision) == (
        split.manifest.manifest_id,
        split.manifest.source_revision,
    )
    assert {case.case_id: case.category for case in result.selected} == {
        "consensus": "NTA",
        "nominee": "missing_context",
        "split-vote": "disputed",
    }
    assert len(result.selected) == 3
    assert all(case.requires_manual_review for case in result.selected)
    assert result.available_counts == result.selected_counts
    assert result.shortages == {
        "NTA": 39,
        "YTA": 40,
        "ESH": 40,
        "NAH": 40,
        "disputed": 19,
        "missing_context": 19,
    }
    assert not result.to_report()["complete"]
    report = validate_golden_selection(result, manifest=split.manifest)
    assert report.valid
    assert "Missing-context candidates require human confirmation" in report.warnings
    assert "Semantic duplicate scan not verified" in report.warnings
    assert "Exact duplicate scan not verified" not in report.warnings


def test_split_rejects_ineligible_nominee_even_alongside_an_eligible_nominee(split):
    with pytest.raises(
        ValueError, match=r"too-many.*5–10 valid moral comments, found 11"
    ):
        select_golden_candidates(
            split, missing_context_review_ids={"nominee", "too-many"}
        )


def test_split_accepts_nominees_at_both_comment_count_limits(split):
    result = select_golden_candidates(
        split, missing_context_review_ids={"nominee", "consensus"}
    )
    assert {
        case.case_id: case.n_valid_comments
        for case in result.selected
        if case.category == "missing_context"
    } == {"nominee": 5, "consensus": 10}


def test_split_sampling_reproduces_list_selection_with_the_same_seed(split):
    # Both NTA cases are in the same sampling bucket without a nomination.
    expected = select_golden_candidates(
        [case for case in split.golden_candidates if case.post_id != "too-many"],
        manifest_id=split.manifest.manifest_id,
        source_revision=split.manifest.source_revision,
        seed=37,
    )
    reordered = replace(split, held_out=list(reversed(split.golden_candidates)))
    assert select_golden_candidates(split, seed=37) == expected
    assert select_golden_candidates(reordered, seed=37) == expected


@pytest.mark.parametrize("partition", ["development", "retrieval", "unused_test"])
def test_split_rejects_foreign_candidates_even_before_exclusions(split, partition):
    foreign = getattr(split, partition)[0]
    with pytest.raises(ValueError, match="absent"):
        select_golden_candidates(split, missing_context_review_ids={foreign.post_id})
    contaminated = replace(split, held_out=[*split.held_out, foreign])
    with pytest.raises(ValueError, match="held-out partition"):
        select_golden_candidates(contaminated, excluded_ids={foreign.post_id})


def test_split_rejects_truncated_pool(split):
    with pytest.raises(ValueError, match="held-out partition"):
        select_golden_candidates(replace(split, held_out=split.held_out[1:]))


@pytest.mark.parametrize("partition", ["development_ids", "retrieval_ids"])
def test_split_rejects_held_out_ids_listed_in_another_partition(split, partition):
    overlap = getattr(split.manifest, partition) | {"nominee"}
    manifest = replace(split.manifest, **{partition: overlap})
    with pytest.raises(ValueError, match="overlaps"):
        select_golden_candidates(replace(split, manifest=manifest))


@pytest.mark.parametrize("field", ["manifest_id", "source_revision"])
def test_split_requires_its_own_provenance(split, field):
    with pytest.raises(ValueError, match="supplies its own"):
        select_golden_candidates(split, **{field: "other"})
    incomplete = replace(split.manifest, **{field: " "})
    with pytest.raises(ValueError, match="Missing manifest ID or source revision"):
        select_golden_candidates(replace(split, manifest=incomplete))
    selection = select_golden_candidates(split)
    stale = replace(split.manifest, **{field: "other"})
    assert "Selection provenance disagrees with source manifest" in (
        validate_golden_selection(selection, manifest=stale).errors
    )


def test_split_rejects_reported_cross_partition_semantic_match(split):
    selection = select_golden_candidates(split)
    # Synthetic scan finding: one held-out case and one retrieval case are linked.
    scanned = attach_semantic_scan(split.manifest, [{"consensus", "train"}])
    assert "Duplicate group crosses partitions" in (
        validate_golden_selection(selection, manifest=scanned).errors
    )


def test_select_consensus_categories():
    cases = [
        make_case(f"case-{v}", [v] * 8 + ["YTA" if v != "YTA" else "NTA"] * 2)
        for v in VERDICTS
    ]

    result = select_golden_candidates(cases)

    assert result.selected_counts["NTA"] == 1
    assert result.selected_counts["YTA"] == 1
    assert result.selected_counts["ESH"] == 1
    assert result.selected_counts["NAH"] == 1


def test_disputed_requires_less_than_70_percent():
    disputed = make_case("disputed", ["NTA"] * 6 + ["YTA"] * 4)
    consensus = make_case("consensus", ["NTA"] * 7 + ["YTA"] * 3)

    result = select_golden_candidates([disputed, consensus])

    assert result.selected_counts["disputed"] == 1
    assert result.selected_counts["NTA"] == 1


def test_missing_context_review_is_exclusive():
    case = make_case("review", ["NTA"] * 10)

    result = select_golden_candidates(
        [case],
        missing_context_review_ids={"review"},
    )

    assert result.selected_counts["missing_context"] == 1
    assert result.selected_counts["NTA"] == 0


def test_excluded_cases_are_not_selected():
    case = make_case("excluded", ["YTA"] * 10)

    result = select_golden_candidates(
        [case],
        excluded_ids={"excluded"},
    )

    assert result.selected == []


def test_selection_is_deterministic_and_order_independent():
    cases = [make_case(f"case-{i}", ["NTA"] * 10) for i in range(60)]

    first = select_golden_candidates(cases, seed=123)
    second = select_golden_candidates(list(reversed(cases)), seed=123)

    assert first.selected == second.selected
    assert first.selected_counts["NTA"] == 40


def test_shortages_are_reported():
    result = select_golden_candidates([make_case("one", ["ESH"] * 10)])

    assert result.shortages["ESH"] == 39
    assert result.shortages["NTA"] == 40
    assert result.to_report()["complete"] is False


def test_duplicate_ids_are_rejected():
    import pytest

    case = make_case("duplicate", ["NTA"] * 10)

    with pytest.raises(ValueError, match="Duplicate"):
        select_golden_candidates([case, case])


def test_unknown_review_ids_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="absent"):
        select_golden_candidates(
            [],
            missing_context_review_ids={"unknown"},
        )


def test_export_candidates_creates_valid_json(tmp_path):
    case = make_case("candidate-1", ["NTA"] * 10)
    result = select_golden_candidates([case], seed=123)
    output = tmp_path / "candidates.json"

    export_golden_candidates(result, output)

    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["schema_version"] == "1.0"
    assert data["status"] == "preliminary_candidates"
    assert data["seed"] == 123
    assert data["total_candidates"] == 1
    assert data["candidates"][0]["case_id"] == "candidate-1"
    assert data["candidates"][0]["category"] == "NTA"
    assert data["candidates"][0]["category_status"] == "provisional"


def test_export_report_contains_shortages(tmp_path):
    result = select_golden_candidates([make_case("one", ["ESH"] * 10)])
    output = tmp_path / "report.json"

    export_selection_report(result, output)

    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["selected_counts"]["ESH"] == 1
    assert data["shortages"]["ESH"] == 39
    assert data["total_selected"] == 1
    assert data["complete"] is False


def test_export_does_not_include_narrative_or_comments(tmp_path):
    case = make_case("private-case", ["YTA"] * 10)
    result = select_golden_candidates([case])
    output = tmp_path / "candidates.json"

    export_golden_candidates(result, output)

    content = output.read_text(encoding="utf-8")
    assert case.text not in content
    assert case.title not in content
    assert "Valid moral comment" not in content


def test_export_prevents_accidental_overwrite(tmp_path):
    result = select_golden_candidates([])
    output = tmp_path / "report.json"

    export_selection_report(result, output)
    original = output.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_selection_report(result, output)

    assert output.read_text(encoding="utf-8") == original


def test_export_allows_explicit_overwrite(tmp_path):
    output = tmp_path / "report.json"

    first = select_golden_candidates([])
    second = select_golden_candidates([make_case("one", ["NTA"] * 10)])

    export_selection_report(first, output)
    export_selection_report(second, output, overwrite=True)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["total_selected"] == 1


def test_export_creates_parent_directory(tmp_path):
    result = select_golden_candidates([])
    output = tmp_path / "nested" / "evaluation" / "report.json"

    export_selection_report(result, output)

    assert output.exists()


@pytest.mark.parametrize("targets", [{}, {"NTA": 0}, {"unknown": 200}])
def test_malformed_targets_cannot_claim_completion(targets):
    result = GoldenSelectionResult(target_counts=targets)
    with pytest.raises(ValueError, match="canonical target"):
        result.to_report()
    assert sum(result.shortages.values()) == 200


def test_review_nomination_cannot_override_exclusion():
    case = make_case("nominated", ["NTA"] * 5)
    with pytest.raises(ValueError, match="overlap"):
        select_golden_candidates(
            [case], excluded_ids={"nominated"}, missing_context_review_ids={"nominated"}
        )


def test_selector_rejects_rank_outside_original_slots():
    case = make_case("oversize", ["YTA"] * 11)
    with pytest.raises(ValueError, match="top-ten"):
        select_golden_candidates([case])


@pytest.mark.parametrize("share", [float("nan"), float("inf"), True, "0.8", 0.81, 0.2])
def test_invalid_candidate_statistics_cannot_be_reported(share):
    result = select_golden_candidates([make_case("one", ["NTA"] * 10)])
    result.selected[0] = replace(result.selected[0], community_agreement=share)
    with pytest.raises(ValueError, match="[Aa]greement"):
        result.to_report()


def test_inventory_cannot_undercount_selection():
    result = select_golden_candidates([make_case("one", ["ESH"] * 5)])
    result.available_counts["ESH"] = 0
    with pytest.raises(ValueError, match="available counts"):
        result.to_report()
