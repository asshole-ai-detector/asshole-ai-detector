"""Unit tests for AITA cleaning and filtering (DATA-02, Milestone 2 Section 3.2)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from asshole_ai_detector.data.cleaning import (
    DEFAULT_CONFIG_PATH,
    CleanedAITACase,
    CleaningConfig,
    CleaningReport,
    CommentStatus,
    RejectedAITACase,
    RejectionReason,
    classify_comment,
    clean_case,
    clean_cases,
    normalize_text,
)
from asshole_ai_detector.schemas.case import (
    NormalizedAITACase,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "normalized_cases.json"
URL = "https://www.reddit.com/r/AmItheAsshole/comments/abc123/aita/"
LONG_TEXT = " ".join(["word"] * 60)


def comment(rank, verdict="NTA", text="some reasoning"):
    return {"rank": rank, "verdict": verdict, "text": text}


def make_case(comments=None, **overrides):
    data = {
        "post_id": "abc123",
        "title": "AITA for testing?",
        "text": LONG_TEXT,
        "url": URL,
        "date": datetime(2020, 1, 1, tzinfo=UTC),
        "provider_ambiguity": 0.3,
        "comments": comments
        if comments is not None
        else [comment(r) for r in range(1, 6)],
    }
    data.update(overrides)
    return NormalizedAITACase.model_validate(data)


def status(verdict, text="text"):
    return classify_comment(
        NormalizedAITACase.model_validate(
            {"comments": [comment(1, verdict, text)]}
        ).comments[0]
    )


# Text normalization -----------------------------------------------------------


def test_normalize_text_removes_links_and_mentions_and_lowercases():
    raw = "Ask U/Some_User in r/AmItheAsshole, see https://x.com/a?b=1 or www.y.org &amp; [this](https://z.io)"
    assert normalize_text(raw) == "ask in , see or & this"


def test_normalize_text_keeps_paragraphs_but_collapses_whitespace():
    assert (
        normalize_text("  First\t\tline \r\n\r\n\r\n\nSecond  ")
        == "first line\n\nsecond"
    )


def test_normalize_text_does_not_treat_word_internal_slashes_as_mentions():
    assert normalize_text("w/r/t my and/or plan") == "w/r/t my and/or plan"


def test_normalize_text_can_keep_case():
    assert normalize_text("AITA?", lowercase=False) == "AITA?"


@pytest.mark.parametrize("value", [None, "", "   ", "https://only.a/link"])
def test_normalize_text_empty_results(value):
    assert normalize_text(value) == ""


# Comment classification -------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "text", "expected"),
    [
        ("NTA", "nta", CommentStatus.VALID),
        ("YTA", "yta", CommentStatus.VALID),
        ("ESH", "esh", CommentStatus.VALID),
        ("NAH", "nah", CommentStatus.VALID),
        (" yta ", "lowercase label is normalized", CommentStatus.VALID),
        ("INFO", "did she ask?", CommentStatus.INFO),
        (None, "just a joke", CommentStatus.NULL_LABEL),
        ("   ", "blank label", CommentStatus.NULL_LABEL),
        ("UNKNOWN", "odd label", CommentStatus.UNSUPPORTED_LABEL),
        ("NTA", "[deleted]", CommentStatus.DELETED),
        ("NTA", " [Removed] ", CommentStatus.DELETED),
        (None, "[removed by reddit]", CommentStatus.DELETED),
        ("NTA", None, CommentStatus.EMPTY_TEXT),
        ("NTA", "https://just.a/link", CommentStatus.EMPTY_TEXT),
    ],
)
def test_classify_comment(verdict, text, expected):
    assert status(verdict, text) is expected


# Submission filtering ---------------------------------------------------------


def test_valid_case_is_accepted_with_metadata_preserved():
    outcome = clean_case(make_case())

    assert isinstance(outcome, CleanedAITACase)
    assert outcome.post_id == "abc123"
    assert outcome.title == "aita for testing?"
    assert outcome.url == URL
    assert outcome.date == datetime(2020, 1, 1, tzinfo=UTC)
    assert outcome.provider_ambiguity == 0.3
    assert outcome.n_valid == 5


def test_invalid_comments_are_dropped_and_ranks_preserved():
    comments = [
        comment(7, "YTA"),  # Out of order on purpose.
        comment(1, "NTA"),
        comment(2, "INFO", "info?"),
        comment(3, "NTA", "[deleted]"),
        comment(4, None, "no label"),
        comment(5, "ESH"),
        comment(6, "NAH"),
        comment(9, "NTA", "Ask u/someone, NTA"),
    ]
    outcome = clean_case(make_case(comments))

    assert isinstance(outcome, CleanedAITACase)
    assert [(c.rank, c.verdict) for c in outcome.comments] == [
        (1, "NTA"),
        (5, "ESH"),
        (6, "NAH"),
        (7, "YTA"),
        (9, "NTA"),
    ]
    assert outcome.comments[-1].text == "ask , nta"


def test_info_is_excluded_from_verdicts_but_counted():
    comments = [comment(r, "NTA") for r in range(1, 6)] + [
        comment(6, "INFO", "q1?"),
        comment(7, "INFO", "q2?"),
    ]
    outcome = clean_case(make_case(comments))

    assert isinstance(outcome, CleanedAITACase)
    assert outcome.n_valid == 5
    assert all(c.verdict != "INFO" for c in outcome.comments)
    assert outcome.community_votes.model_dump() == {
        "NTA": 5,
        "YTA": 0,
        "ESH": 0,
        "NAH": 0,
        "INFO": 2,
    }


def test_info_comments_do_not_count_towards_minimum():
    comments = [comment(r, "NTA") for r in range(1, 5)] + [comment(5, "INFO", "q?")]
    outcome = clean_case(make_case(comments))

    assert isinstance(outcome, RejectedAITACase)
    assert outcome.reasons == (RejectionReason.TOO_FEW_VALID_VERDICTS,)
    assert (outcome.n_valid, outcome.info_count) == (4, 1)


def test_word_threshold_boundary():
    words = lambda n: " ".join(["word"] * n)
    assert isinstance(clean_case(make_case(text=words(50))), CleanedAITACase)
    rejected = clean_case(make_case(text=words(49)))
    assert rejected.reasons == (RejectionReason.TOO_FEW_WORDS,)


def test_links_do_not_count_as_words():
    text = " ".join(["word"] * 49) + " https://example.com/padding"
    assert clean_case(make_case(text=text)).reasons == (RejectionReason.TOO_FEW_WORDS,)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"post_id": None}, RejectionReason.MISSING_POST_ID),
        ({"title": None}, RejectionReason.MISSING_TITLE),
        ({"text": None}, RejectionReason.MISSING_TEXT),
        ({"text": "[deleted]"}, RejectionReason.DELETED_TEXT),
        ({"text": "[removed]"}, RejectionReason.DELETED_TEXT),
        ({"url": None}, RejectionReason.INVALID_URL),
        ({"url": "reddit.com/r/aita/comments/abc123"}, RejectionReason.INVALID_URL),
    ],
)
def test_invalid_submissions_are_rejected(overrides, reason):
    outcome = clean_case(make_case(**overrides))
    assert isinstance(outcome, RejectedAITACase)
    assert outcome.reasons == (reason,)


def test_all_rejection_reasons_are_recorded():
    outcome = clean_case(make_case(comments=[], text="too short", url=None))
    assert outcome.reasons == (
        RejectionReason.TOO_FEW_WORDS,
        RejectionReason.INVALID_URL,
        RejectionReason.TOO_FEW_VALID_VERDICTS,
    )



def test_config_thresholds_are_respected():
    config = CleaningConfig(
        min_post_words=61,
        min_valid_verdict_comments=6,
    )

    outcome = clean_case(make_case(), config)

    assert isinstance(outcome, RejectedAITACase)
    assert RejectionReason.TOO_FEW_WORDS in outcome.reasons
    assert RejectionReason.TOO_FEW_VALID_VERDICTS in outcome.reasons


def test_config_rejects_thresholds_below_schema_minimums():
    with pytest.raises(ValueError):
        CleaningConfig(min_post_words=49)

    with pytest.raises(ValueError):
        CleaningConfig(min_valid_verdict_comments=4)



def test_cleaning_is_deterministic():
    case = make_case()
    assert clean_case(case) == clean_case(case)


# Report and batch -------------------------------------------------------------


def test_report_counts_records_reasons_and_comments():
    cases = [
        make_case(),
        make_case(text="short"),
        make_case(comments=[comment(1, "INFO", "q?"), comment(2, "NTA", "[deleted]")]),
    ]
    result = clean_cases(cases)

    assert [c.post_id for c in result.accepted] == ["abc123"]
    assert len(result.rejected) == 2
    assert result.report.to_dict() == {
        "total": 3,
        "accepted": 1,
        "rejected": 2,
        "rejection_reasons": {"too_few_valid_verdicts": 1, "too_few_words": 1},
        "comment_statuses": {"deleted": 1, "info": 1, "valid": 10},
    }


def test_clean_case_without_report_has_no_side_effects():
    report = CleaningReport()
    clean_case(make_case())
    assert report.total == 0


def test_config_loads_from_default_yaml():
    assert CleaningConfig.from_yaml(DEFAULT_CONFIG_PATH) == CleaningConfig()


def test_config_rejects_unknown_verdicts():
    with pytest.raises(ValueError):
        CleaningConfig(valid_verdicts=("NTA", "INFO"))


# Contract with the fixtures and DATA-03 ------------------------------------------


def test_fixture_cases():
    cases = [
        NormalizedAITACase.model_validate(c) for c in json.loads(FIXTURES.read_text())
    ]
    result = clean_cases(cases)

    assert [c.post_id for c in result.accepted] == [
        "mock-consensus-001",
        "mock-disputed-002",
        "mock-invalid-comments-004",
    ]
    assert [(r.post_id, r.reasons) for r in result.rejected] == [
        (
            "mock-missing-003",
            (RejectionReason.MISSING_TITLE, RejectionReason.INVALID_URL),
        ),
        (
            "mock-short-005",
            (RejectionReason.TOO_FEW_WORDS, RejectionReason.TOO_FEW_VALID_VERDICTS),
        ),
    ]
    invalid = result.accepted[2]
    assert [c.rank for c in invalid.comments] == [1, 3, 6, 7, 8]
    assert invalid.community_votes.INFO == 1
    assert result.report.comment_statuses["deleted"] == 2
    assert result.report.comment_statuses["unsupported_label"] == 1


def test_cleaned_case_is_still_a_normalized_case():
    outcome = clean_case(make_case())
    assert isinstance(outcome, NormalizedAITACase)
    assert all(c.verdict in {"NTA", "YTA", "ESH", "NAH"} for c in outcome.comments)

