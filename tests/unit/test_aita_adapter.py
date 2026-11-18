"""Unit tests for the raw AITA dataset adapter (no network or downloads)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from asshole_ai_detector.data.aita_adapter import (
    extract_post_id,
    iter_normalized_cases,
    parse_date,
    row_to_normalized,
)

URL = "https://www.reddit.com/r/AmItheAsshole/comments/gr8bp3/aita_for_suing_my_girlfriend/"


def make_row(**overrides):
    """Build a raw row with the same columns as the Hugging Face dataset."""
    row = {
        "submission_title": "AITA for suing my girlfriend?",
        "submission_text": "i had a car project in my garage and she had it scrapped.",
        "submission_score": 70805,
        "submission_url": URL,
        "submission_date": "2020-05-26 23:35:45",
        "ambiguity_score": 0.0,
        "flanT5_instruction": "Classify the interpersonal conflict ...",
        "llama2_instruction": "<s>[INST] ...",
    }
    for rank in range(1, 11):
        row[f"top_comment_{rank}"] = None
        row[f"top_comment_{rank}_classification"] = None
    row.update(overrides)
    return row


def test_extract_post_id_from_reddit_url():
    assert extract_post_id(URL) == "gr8bp3"
    # A URL without a trailing slash still has an ID.
    assert extract_post_id(URL.split("/aita_")[0]) == "gr8bp3"


@pytest.mark.parametrize("url", [None, "", "https://example.com/post/123"])
def test_extract_post_id_returns_none_when_missing(url):
    assert extract_post_id(url) is None


def test_parse_date_marks_source_dates_as_utc():
    assert parse_date("2020-05-26 23:35:45") == datetime(2020, 5, 26, 23, 35, 45, tzinfo=UTC)


@pytest.mark.parametrize("value", [None, "", "26/05/2020", 1590536145])
def test_parse_date_returns_none_for_missing_or_malformed_values(value):
    assert parse_date(value) is None


def test_row_maps_submission_fields():
    case = row_to_normalized(make_row(ambiguity_score=0.25))

    assert case.post_id == "gr8bp3"
    assert case.title == "AITA for suing my girlfriend?"
    assert case.text.startswith("i had a car project")
    assert case.url == URL
    assert case.date == datetime(2020, 5, 26, 23, 35, 45, tzinfo=UTC)
    assert case.provider_ambiguity == 0.25
    assert case.comments == []


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_submission_text_becomes_none(text):
    assert row_to_normalized(make_row(submission_text=text)).text is None


def test_comments_keep_rank_raw_label_and_deleted_text():
    case = row_to_normalized(
        make_row(
            top_comment_1="nta, she went too far.",
            top_comment_1_classification="NTA",
            top_comment_2="[deleted]",  # Deleted comments have no label in the source.
            top_comment_3="did she ask you first?",
            top_comment_3_classification="INFO",
            top_comment_5="just waiting for the plot twist",  # No verdict found.
        )
    )

    assert [(c.rank, c.verdict, c.text) for c in case.comments] == [
        (1, "NTA", "nta, she went too far."),
        (2, None, "[deleted]"),
        (3, "INFO", "did she ask you first?"),
        (5, None, "just waiting for the plot twist"),
    ]


def test_empty_comment_slots_are_skipped():
    # Slot 4 has only an empty string and no label, so it is treated as absent.
    case = row_to_normalized(
        make_row(top_comment_1="yta.", top_comment_1_classification="YTA", top_comment_4="")
    )
    assert [c.rank for c in case.comments] == [1]


def test_nan_ambiguity_becomes_none():
    assert row_to_normalized(make_row(ambiguity_score=float("nan"))).provider_ambiguity is None


def test_out_of_range_ambiguity_is_rejected_by_schema():
    with pytest.raises(ValidationError):
        row_to_normalized(make_row(ambiguity_score=1.5))


def test_iter_normalized_cases_is_lazy_and_preserves_order():
    rows = [make_row(), make_row(submission_url=URL.replace("gr8bp3", "e5k3z2"))]
    cases = iter_normalized_cases(rows)

    assert not isinstance(cases, list)
    assert [case.post_id for case in cases] == ["gr8bp3", "e5k3z2"]
