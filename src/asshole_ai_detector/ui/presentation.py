"""Display text for assessments, kept separate from Streamlit so it can be tested.

Labels follow Milestone 2, Section 1.1 (verdicts) and Section 4.2 (uncertainty
tiers). Tiers describe the strength of the available evidence, not objective
certainty about moral responsibility.
"""

from dataclasses import dataclass

from asshole_ai_detector.schemas.annotation import UncertaintyStatus, Verdict
from asshole_ai_detector.schemas.result import AssessmentResult


@dataclass(frozen=True)
class VerdictStyle:
    label: str
    description: str
    # Streamlit markdown color name; the label text always carries the meaning.
    color: str


VERDICT_STYLES: dict[Verdict, VerdictStyle] = {
    "NTA": VerdictStyle(
        "Not the asshole", "The other party is mainly responsible.", "green"
    ),
    "YTA": VerdictStyle("You're the asshole", "You are mainly responsible.", "red"),
    "ESH": VerdictStyle(
        "Everyone sucks here", "Both sides share responsibility.", "orange"
    ),
    "NAH": VerdictStyle("No assholes here", "Nobody acted wrongly.", "blue"),
}

WITHHELD_STYLE = VerdictStyle(
    "Verdict withheld",
    "There is not enough information for a defensible assessment.",
    "gray",
)


@dataclass(frozen=True)
class TierStyle:
    title: str
    description: str


TIER_STYLES: dict[UncertaintyStatus, TierStyle] = {
    "confident": TierStyle(
        "Confident",
        "The available evidence consistently supports this verdict.",
    ),
    "leaning": TierStyle(
        "Leaning",
        "A tentative verdict: some uncertainty or potential bias remains.",
    ),
    "split": TierStyle(
        "Split",
        "The evidence is divided between two verdicts.",
    ),
    "withhold": TierStyle(
        "Withheld",
        "A verdict is not given because essential information is missing.",
    ),
}

# Shown under every result, because community judgments are reference labels.
EVIDENCE_NOTE = (
    "Tiers describe the strength and limitations of the available evidence, "
    "not objective certainty about who is right."
)


def displayed_verdicts(result: AssessmentResult) -> list[VerdictStyle]:
    """Return the verdicts to show: none (withheld), one, or two for split.

    The UI never assumes that a single verdict always exists.
    """
    if result.verdict is None:
        return [WITHHELD_STYLE]
    verdicts = [VERDICT_STYLES[result.verdict]]
    if result.secondary_verdict is not None:
        verdicts.append(VERDICT_STYLES[result.secondary_verdict])
    return verdicts


def needs_clarification(result: AssessmentResult, *, round_used: bool) -> bool:
    """True when the user should answer questions before the next analysis.

    Only one clarification round is allowed (Section 4.2); afterwards the
    questions are shown for context but no new answers are requested.
    """
    return bool(result.clarification_questions) and not round_used
