
"""Tests for consensus labeling and AITA case integration."""

import pytest
from pydantic import ValidationError

from asshole_ai_detector.data_processing.consensus import label_cleaned_case
from asshole_ai_detector.schemas.annotation import Verdict
from asshole_ai_detector.schemas.case import (
    AITACase,
    AITAComment,
    NormalizedAITACase,
)


def make_case() -> NormalizedAITACase:
    """Create a synthetic normalized AITA case."""
    return NormalizedAITACase(
        post_id="test-001",
        title="A synthetic AITA scenario",
        text=" ".join(["example"] * 55),
        url="https://example.com/test-001",
        provider_ambiguity=0.3,
    )


def make_comments(
    verdicts: tuple[Verdict, ...],
    ranks: tuple[int, ...] | None = None,
) -> list[AITAComment]:
    """Create valid comments with their original ranks."""
    if ranks is None:
        ranks = tuple(range(1, len(verdicts) + 1))

    assert len(verdicts) == len(ranks)

    return [
        AITAComment(
            comment_id=f"test-001#{rank}",
            rank=rank,
            verdict=verdict,
            text=f"Synthetic comment {rank}",
        )
        for rank, verdict in zip(ranks, verdicts, strict=True)
    ]


def test_consensus_labeling() -> None:
    """80% NTA should produce an NTA consensus."""
    comments = make_comments(
        ("NTA", "NTA", "NTA", "NTA", "YTA")
    )

    result = label_cleaned_case(make_case(), comments)
    fields = result.derived_fields

    assert fields.n_valid == 5
    assert fields.top_verdict == "NTA"
    assert fields.consensus_label == "NTA"
    assert fields.verdict_distribution["NTA"] == pytest.approx(0.8)
    assert fields.verdict_distribution["YTA"] == pytest.approx(0.2)


def test_exact_70_percent_threshold() -> None:
    """Exactly 70% must still count as consensus."""
    comments = make_comments(
        ("NTA",) * 7 + ("YTA",) * 3
    )

    result = label_cleaned_case(make_case(), comments)

    assert result.derived_fields.n_valid == 10
    assert result.derived_fields.consensus_label == "NTA"
    assert result.derived_fields.verdict_distribution["NTA"] == pytest.approx(0.7)


def test_below_70_percent_is_disputed() -> None:
    """60% agreement must produce DISPUTED."""
    comments = make_comments(
        ("NTA",) * 6 + ("YTA",) * 4
    )

    result = label_cleaned_case(make_case(), comments)

    assert result.derived_fields.top_verdict == "NTA"
    assert result.derived_fields.consensus_label == "DISPUTED"


def test_tie_breaking_uses_lowest_rank() -> None:
    """A tied plurality is broken by the highest-ranked comment."""
    comments = make_comments(
        ("YTA", "NTA", "YTA", "NTA", "ESH"),
        ranks=(7, 2, 9, 4, 11),
    )

    result = label_cleaned_case(make_case(), comments)

    # Both YTA and NTA have two votes.
    # NTA has the highest-ranked tied comment (rank 2).
    assert result.derived_fields.top_verdict == "NTA"
    assert result.derived_fields.consensus_label == "DISPUTED"

    # Original ranks and comment order must be preserved.
    assert [c.rank for c in result.valid_comments] == [
        7, 2, 9, 4, 11
    ]


def test_verdict_distribution_sums_to_one() -> None:
    """All four moral verdict probabilities must sum to one."""
    comments = make_comments(
        ("NTA", "YTA", "ESH", "NAH", "NTA")
    )

    result = label_cleaned_case(make_case(), comments)
    distribution = result.derived_fields.verdict_distribution

    assert set(distribution) == {"NTA", "YTA", "ESH", "NAH"}
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution["NTA"] == pytest.approx(0.4)
    assert distribution["YTA"] == pytest.approx(0.2)
    assert distribution["ESH"] == pytest.approx(0.2)
    assert distribution["NAH"] == pytest.approx(0.2)


def test_info_is_not_a_valid_moral_verdict() -> None:
    """INFO must not enter the moral-verdict calculation."""
    with pytest.raises(ValidationError):
        AITAComment.model_validate(
            {
                "comment_id": "test-001#1",
                "rank": 1,
                "verdict": "INFO",
                "text": "More information is needed.",
            }
        )


def test_rejects_fewer_than_five_valid_comments() -> None:
    """A case needs at least five valid moral verdicts."""
    comments = make_comments(
        ("NTA", "NTA", "YTA", "ESH")
    )

    with pytest.raises(ValueError, match="At least 5"):
        label_cleaned_case(make_case(), comments)


def test_rejects_inconsistent_derived_fields() -> None:
    """Stored verdict statistics must match actual comments."""
    comments = make_comments(
        ("NTA", "NTA", "NTA", "NTA", "YTA")
    )
    result = label_cleaned_case(make_case(), comments)

    payload = result.model_dump(mode="json")

    # Internally valid distribution, but inconsistent with comments.
    payload["derived_fields"]["verdict_distribution"] = {
        "NTA": 1.0,
        "YTA": 0.0,
        "ESH": 0.0,
        "NAH": 0.0,
    }

    with pytest.raises(
        ValidationError,
        match="Incorrect verdict distribution",
    ):
        AITACase.model_validate(payload)


def test_rejects_duplicate_comment_ranks() -> None:
    """A case cannot contain repeated original comment ranks."""
    comments = make_comments(
        ("NTA", "NTA", "NTA", "NTA", "YTA"),
        ranks=(1, 2, 3, 4, 4),
    )

    with pytest.raises(
        ValidationError,
        match="ranks must be unique",
    ):
        label_cleaned_case(make_case(), comments)


def test_preserves_case_metadata() -> None:
    """Labeling must preserve identifiers and source metadata."""
    case = make_case()
    comments = make_comments(("NTA",) * 5)

    result = label_cleaned_case(case, comments)

    assert result.post_id == case.post_id
    assert result.title == case.title
    assert result.text == case.text
    assert str(result.url) == case.url
    assert result.provider_ambiguity == case.provider_ambiguity
    assert [c.comment_id for c in result.valid_comments] == [
        c.comment_id for c in comments
    ]
