"""Integration test for the real DATA-02 -> DATA-03 boundary."""

import pytest

from asshole_ai_detector.data.cleaning import CleanedAITACase, clean_case
from asshole_ai_detector.data.consensus import label_cleaned_case
from asshole_ai_detector.schemas.case import NormalizedAITACase

LONG_TEXT = " ".join(["example"] * 55)

def test_real_cleaning_to_consensus_handoff() -> None:
    """DATA-02 -> DATA-03: discard INFO/deleted; preserve gaps and IDs."""
    raw_case = NormalizedAITACase.model_validate(
        {
            "post_id": "integration-001",
            "title": "A synthetic disagreement",
            "text": LONG_TEXT,
            "url": "https://example.com/integration-001",
            "comments": [
                {"rank": 1, "verdict": "NTA", "text": "Reason 1"},
                {"rank": 2, "verdict": "INFO", "text": "Need more context"},
                {"rank": 3, "verdict": "NTA", "text": "Reason 3"},
                {"rank": 4, "verdict": "NTA", "text": "[deleted]"},
                {"rank": 5, "verdict": "YTA", "text": "Reason 5"},
                {"rank": 6, "verdict": "NTA", "text": "Reason 6"},
                {"rank": 7, "verdict": "NTA", "text": "Reason 7"},
                {"rank": 8, "verdict": "NTA", "text": "Reason 8"},
            ],
        }
    )

    cleaned = clean_case(raw_case)
    assert isinstance(cleaned, CleanedAITACase)
    assert cleaned.community_votes.INFO == 1

    labeled = label_cleaned_case(cleaned)
    assert labeled.derived_fields.n_valid == 6
    assert labeled.derived_fields.consensus_label == "NTA"
    assert labeled.derived_fields.verdict_distribution["NTA"] == pytest.approx(5 / 6)
    assert [comment.comment_id for comment in labeled.valid_comments] == [
        "integration-001#1",
        "integration-001#3",
        "integration-001#5",
        "integration-001#6",
        "integration-001#7",
        "integration-001#8",
    ]
    assert all(comment.verdict != "INFO" for comment in labeled.valid_comments)
