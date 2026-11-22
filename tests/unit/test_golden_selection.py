"""Tests for preliminary Golden Dataset candidate selection."""

import json

import pytest

from asshole_ai_detector.evaluation.golden_selection import (
    export_golden_candidates,
    export_selection_report,
    select_golden_candidates,
)
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
        text=" ".join(["This is a detailed example of a conflict."] * 12),
        url=f"https://www.reddit.com/r/AmItheAsshole/comments/{post_id}/",
        valid_comments=comments,
        derived_fields=DerivedFields.from_comments(comments),
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
