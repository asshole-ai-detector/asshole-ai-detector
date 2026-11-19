"""DATA-03 tests: consensus rules and the DATA-02 -> DATA-03 handoff."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from asshole_ai_detector.data.cleaning import CleanedAITACase
from asshole_ai_detector.data_processing.consensus import label_cleaned_case
from asshole_ai_detector.schemas.annotation import CommunityVotes, Verdict
from asshole_ai_detector.schemas.case import (
    AITACase,
    NormalizedAITAComment,
)


POST_ID = "test-001"
URL = "https://example.com/test-001"
LONG_TEXT = " ".join(["example"] * 55)


def make_cleaned_case(
    verdicts: tuple[Verdict, ...],
    ranks: tuple[int, ...] | None = None,
) -> CleanedAITACase:
    """Build the DATA-02 output contract without using the DATA-02 implementation.

    This isolates DATA-03 consensus behavior from preprocessing behavior.
    """
    if ranks is None:
        ranks = tuple(range(1, len(verdicts) + 1))
    assert len(verdicts) == len(ranks)

    comments = [
        NormalizedAITAComment(
            rank=rank,
            verdict=verdict,
            text=f"Synthetic comment {rank}",
        )
        for rank, verdict in zip(ranks, verdicts, strict=True)
    ]

    return CleanedAITACase(
        post_id=POST_ID,
        title="A synthetic AITA scenario",
        text=LONG_TEXT,
        url=URL,
        date=datetime(2026, 1, 10, tzinfo=UTC),
        provider_ambiguity=0.3,
        # DATA-02 sorts surviving comments but does not renumber them.
        comments=sorted(comments, key=lambda comment: comment.rank),
        community_votes=CommunityVotes(
            NTA=verdicts.count("NTA"),
            YTA=verdicts.count("YTA"),
            ESH=verdicts.count("ESH"),
            NAH=verdicts.count("NAH"),
            INFO=0,
        ),
    )


def test_consensus_at_80_percent() -> None:
    result = label_cleaned_case(make_cleaned_case(("NTA",) * 4 + ("YTA",)))

    assert isinstance(result, AITACase)
    assert result.derived_fields.n_valid == 5
    assert result.derived_fields.top_verdict == "NTA"
    assert result.derived_fields.consensus_label == "NTA"
    assert result.derived_fields.verdict_distribution["NTA"] == pytest.approx(0.8)
    assert result.derived_fields.verdict_distribution["YTA"] == pytest.approx(0.2)


def test_exactly_70_percent_is_consensus() -> None:
    result = label_cleaned_case(make_cleaned_case(("NTA",) * 7 + ("YTA",) * 3))

    assert result.derived_fields.n_valid == 10
    assert result.derived_fields.consensus_label == "NTA"
    assert result.derived_fields.verdict_distribution["NTA"] == pytest.approx(0.7)


def test_below_70_percent_is_disputed() -> None:
    result = label_cleaned_case(make_cleaned_case(("NTA",) * 6 + ("YTA",) * 4))

    assert result.derived_fields.top_verdict == "NTA"
    assert result.derived_fields.consensus_label == "DISPUTED"


def test_tie_break_uses_lowest_original_rank() -> None:
    # NTA and YTA each have 2 votes. Rank 2 favors NTA, not rank 7's YTA.
    case = make_cleaned_case(
        ("YTA", "NTA", "YTA", "NTA", "ESH"),
        ranks=(7, 2, 9, 4, 11),
    )
    result = label_cleaned_case(case)

    assert result.derived_fields.top_verdict == "NTA"
    assert result.derived_fields.consensus_label == "DISPUTED"
    assert [comment.rank for comment in result.valid_comments] == [2, 4, 7, 9, 11]
    assert [comment.comment_id for comment in result.valid_comments] == [
        "test-001#2",
        "test-001#4",
        "test-001#7",
        "test-001#9",
        "test-001#11",
    ]


def test_distribution_contains_all_four_verdicts_and_sums_to_one() -> None:
    result = label_cleaned_case(
        make_cleaned_case(("NTA", "YTA", "ESH", "NAH", "NTA"))
    )
    distribution = result.derived_fields.verdict_distribution

    assert set(distribution) == {"NTA", "YTA", "ESH", "NAH"}
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution == pytest.approx(
        {"NTA": 0.4, "YTA": 0.2, "ESH": 0.2, "NAH": 0.2}
    )


def test_missing_verdict_classes_have_zero_probability() -> None:
    result = label_cleaned_case(make_cleaned_case(("NAH",) * 5))

    assert result.derived_fields.top_verdict == "NAH"
    assert result.derived_fields.verdict_distribution == {
        "NTA": 0.0,
        "YTA": 0.0,
        "ESH": 0.0,
        "NAH": 1.0,
    }


def test_rejects_fewer_than_five_comments() -> None:
    # This deliberately constructs an invalid "cleaned" input to test
    # DATA-03's defensive check, regardless of DATA-02 filtering.
    case = make_cleaned_case(("NTA", "NTA", "ESH", "YTA"))

    with pytest.raises(ValueError, match="At least 5"):
        label_cleaned_case(case)


@pytest.mark.parametrize(
    ("verdict", "text"),
    [
        ("INFO", "I need more information"),
        (None, "Unlabeled comment"),
        ("NTA", None),
    ],
)
def test_rejects_comments_that_are_not_final_moral_comments(
    verdict: str | None, text: str | None
) -> None:
    # CleanedAITACase uses the permissive normalized-comment schema.
    # DATA-03 must not silently accept invalid comments if they slip through.
    case = make_cleaned_case(("NTA",) * 5)
    case.comments[0] = NormalizedAITAComment(rank=1, verdict=verdict, text=text)

    with pytest.raises(ValidationError):
        label_cleaned_case(case)


def test_rejects_duplicate_comment_ranks() -> None:
    case = make_cleaned_case(("NTA",) * 5, ranks=(1, 2, 3, 4, 4))

    with pytest.raises(ValidationError, match="ranks must be unique"):
        label_cleaned_case(case)


def test_final_case_rejects_derived_fields_inconsistent_with_comments() -> None:
    result = label_cleaned_case(make_cleaned_case(("NTA",) * 4 + ("YTA",)))
    payload = result.model_dump(mode="json")
    # This is a mathematically valid distribution, but it does not describe
    # the underlying 4 NTA + 1 YTA comments.
    payload["derived_fields"]["verdict_distribution"] = {
        "NTA": 1.0,
        "YTA": 0.0,
        "ESH": 0.0,
        "NAH": 0.0,
    }

    with pytest.raises(ValidationError, match="Incorrect verdict distribution"):
        AITACase.model_validate(payload)


def test_case_metadata_is_preserved() -> None:
    case = make_cleaned_case(("NTA",) * 5)
    result = label_cleaned_case(case)

    assert result.post_id == case.post_id
    assert result.title == case.title
    assert result.text == case.text
    assert str(result.url) == case.url
    assert result.date == case.date
    assert result.provider_ambiguity == case.provider_ambiguity
    assert [comment.comment_id for comment in result.valid_comments] == [
        f"{case.post_id}#{rank}" for rank in range(1, 6)
    ]


