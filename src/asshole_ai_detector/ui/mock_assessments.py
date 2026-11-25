"""Synthetic assessments for UI development (UI-01, Milestone 2 Sections 1.1 and 4.2).

These examples are hand-written and validated against ``AssessmentResult``.
They let the Streamlit interface be reviewed without the classifier, retrieval
pipeline, LLM or MCP server. Nothing here is a real model output.
"""

from collections.abc import Mapping
from typing import Literal, Protocol

from asshole_ai_detector.schemas.result import AssessmentResult

type MockKey = Literal["confident", "leaning", "split", "withhold", "clarification"]

# One example per uncertainty tier, plus a clarification request. Together
# they also cover all four moral verdicts (YTA, NTA, ESH, NAH).
MOCK_ASSESSMENTS: dict[MockKey, AssessmentResult] = {
    "confident": AssessmentResult(
        verdict="YTA",
        explanation=(
            "You read your roommate's private messages without permission and then "
            "shared them with friends. The scenario does not describe any urgent "
            "reason that would justify this, so most of the responsibility for the "
            "conflict lies with your actions."
        ),
        alternative_perspectives=[
            (
                "Your roommate may feel that their trust and privacy were broken, "
                "independently of what the messages contained."
            ),
            (
                "From your side, you may have felt excluded and wanted to confirm a "
                "suspicion, which explains the action but does not justify sharing it."
            ),
        ],
        uncertainty_tier="confident",
        limitations=[
            (
                "Community judgments on similar cases are reference opinions, not "
                "objective moral truth."
            ),
        ],
    ),
    "leaning": AssessmentResult(
        verdict="NTA",
        explanation=(
            "You declined to cover your sister's rent for a third month after "
            "agreeing to two months in advance. Keeping to a stated limit is "
            "generally reasonable, although the way the refusal was communicated "
            "may have made the conflict worse."
        ),
        alternative_perspectives=[
            (
                "Your sister may have expected support because of her current job "
                "search and may see the refusal as sudden."
            ),
            "Other relatives may value family support over prior agreements.",
        ],
        uncertainty_tier="leaning",
        missing_information=[
            "Whether the two-month limit was clearly agreed by both of you.",
        ],
        limitations=[
            "The story is told only from your perspective.",
            "Similar cases show some disagreement about family financial support.",
        ],
    ),
    "split": AssessmentResult(
        verdict="ESH",
        secondary_verdict="NAH",
        explanation=(
            "Both of you changed the holiday plans without telling the other, but "
            "it is unclear whether either change was meant to hurt. Similar cases "
            "are divided between seeing this as a mutual failure and as an honest "
            "misunderstanding."
        ),
        alternative_perspectives=[
            "Your partner may have assumed the original plan was still flexible.",
            "You may have felt that your earlier message was enough notice.",
        ],
        uncertainty_tier="split",
        limitations=[
            "The available evidence does not clearly favor one verdict.",
        ],
    ),
    "withhold": AssessmentResult(
        verdict=None,
        explanation=(
            "A verdict was withheld because essential details are still missing "
            "after the clarification round. A defensible assessment cannot be "
            "made from the current description."
        ),
        uncertainty_tier="withhold",
        missing_information=[
            "What the other party said or did in response.",
            "Whether the borrowed item was returned or replaced.",
        ],
        limitations=[
            "No verdict is given instead of guessing from incomplete information.",
        ],
    ),
    "clarification": AssessmentResult(
        verdict=None,
        explanation=(
            "Before giving an assessment, a few details are needed. Your answers "
            "will be added to the scenario for one more analysis."
        ),
        uncertainty_tier="withhold",
        missing_information=[
            "The other party's reaction.",
            "The current state of the conflict.",
        ],
        clarification_questions=[
            "How did your friend react when you told them?",
            "Had you discussed this situation with them before?",
            "Is the conflict still ongoing?",
        ],
    ),
}

# Shown after the user answers the clarification questions.
MOCK_AFTER_CLARIFICATION: AssessmentResult = MOCK_ASSESSMENTS["leaning"]


class AssessmentBackend(Protocol):
    """What the UI needs from the assessment workflow.

    The Week 9 controller should implement this protocol, so the UI does not
    change when the mock backend is replaced.
    """

    def assess(
        self,
        scenario: str,
        clarification_answers: Mapping[str, str] | None = None,
    ) -> AssessmentResult: ...


class MockAssessmentBackend:
    """Returns a fixed synthetic assessment, without calling any model.

    The first call returns the selected example. If clarification answers are
    provided, the follow-up result is returned (or a withheld verdict when the
    user skipped every question).
    """

    def __init__(self, key: MockKey) -> None:
        self.key = key

    def assess(
        self,
        scenario: str,
        clarification_answers: Mapping[str, str] | None = None,
    ) -> AssessmentResult:
        if not scenario.strip():
            raise ValueError("The scenario must not be empty")
        if clarification_answers is None:
            return MOCK_ASSESSMENTS[self.key]
        if any(answer.strip() for answer in clarification_answers.values()):
            return MOCK_AFTER_CLARIFICATION
        # Section 4.2: without new information, withhold instead of guessing.
        return MOCK_ASSESSMENTS["withhold"]
