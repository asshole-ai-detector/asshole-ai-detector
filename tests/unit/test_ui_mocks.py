"""Tests for UI mock assessments and display helpers (no Streamlit needed)."""

import pytest

from asshole_ai_detector.schemas.result import AssessmentResult
from asshole_ai_detector.ui.mock_assessments import (
    MOCK_AFTER_CLARIFICATION,
    MOCK_ASSESSMENTS,
    MockAssessmentBackend,
)
from asshole_ai_detector.ui.presentation import (
    TIER_STYLES,
    VERDICT_STYLES,
    WITHHELD_STYLE,
    displayed_verdicts,
    needs_clarification,
)

SCENARIO = "I borrowed my friend's car without asking and returned it late."


def test_mocks_cover_all_tiers_and_verdicts():
    results = MOCK_ASSESSMENTS.values()
    assert {r.uncertainty_tier for r in results} == set(TIER_STYLES)

    shown = {r.verdict for r in results} | {r.secondary_verdict for r in results}
    assert set(VERDICT_STYLES) <= shown


def test_mocks_are_valid_assessment_results():
    # Revalidate the serialized form so the examples match the shared contract.
    for result in MOCK_ASSESSMENTS.values():
        AssessmentResult.model_validate(result.model_dump())


def test_clarification_mock_has_at_most_three_questions():
    questions = MOCK_ASSESSMENTS["clarification"].clarification_questions
    assert 1 <= len(questions) <= 3


def test_displayed_verdicts_handles_withheld_single_and_split():
    assert displayed_verdicts(MOCK_ASSESSMENTS["withhold"]) == [WITHHELD_STYLE]
    assert displayed_verdicts(MOCK_ASSESSMENTS["confident"]) == [VERDICT_STYLES["YTA"]]
    assert displayed_verdicts(MOCK_ASSESSMENTS["split"]) == [
        VERDICT_STYLES["ESH"],
        VERDICT_STYLES["NAH"],
    ]


def test_clarification_is_requested_only_once():
    result = MOCK_ASSESSMENTS["clarification"]
    assert needs_clarification(result, round_used=False)
    assert not needs_clarification(result, round_used=True)
    assert not needs_clarification(MOCK_ASSESSMENTS["leaning"], round_used=False)


@pytest.mark.parametrize("key", list(MOCK_ASSESSMENTS))
def test_backend_returns_selected_mock(key):
    assert MockAssessmentBackend(key).assess(SCENARIO) == MOCK_ASSESSMENTS[key]


def test_backend_uses_answers_or_withholds_when_skipped():
    backend = MockAssessmentBackend("clarification")
    answered = backend.assess(SCENARIO, {"Is the conflict still ongoing?": "yes"})
    skipped = backend.assess(SCENARIO, {"Is the conflict still ongoing?": "  "})

    assert answered == MOCK_AFTER_CLARIFICATION
    assert skipped.uncertainty_tier == "withhold"
    assert skipped.verdict is None


def test_backend_rejects_empty_scenario():
    with pytest.raises(ValueError, match="must not be empty"):
        MockAssessmentBackend("confident").assess("   ")
