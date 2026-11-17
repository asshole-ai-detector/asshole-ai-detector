"""Golden Dataset annotation contracts (Milestone 2, Section 2.3).

These are *human reference labels*, not the classifier's predictions.
"""

from math import isclose
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# INFO signals a need for clarification; it is never a final moral verdict.
# INFO signals a need for clarification; it is never a final moral verdict.
type Verdict = Literal["NTA", "YTA", "ESH", "NAH"]
type ClassifierLabel = Verdict | Literal["INFO"]
type UncertaintyStatus = Literal["withhold", "split", "leaning", "confident"]
type AnnotatorVerdict = Verdict | Literal["UNDETERMINED"]


class ContractModel(BaseModel):
    """Reject unexpected fields and strip surrounding string whitespace.

    These contracts validate normalized records; source-specific adapters should
    handle additional columns from the raw Reddit dataset.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CommunityVotes(ContractModel):
    """Raw counts; INFO is recorded but excluded from moral-vote agreement."""

    NTA: int = Field(ge=0)
    YTA: int = Field(ge=0)
    ESH: int = Field(ge=0)
    NAH: int = Field(ge=0)
    INFO: int = Field(ge=0)

    @property
    def moral_count(self) -> int:
        return self.NTA + self.YTA + self.ESH + self.NAH

    @property
    def winning_share(self) -> float | None:
        # A moral-vote agreement is undefined if there are only INFO votes.
        if self.moral_count == 0:
            return None
        return max(self.NTA, self.YTA, self.ESH, self.NAH) / self.moral_count


class GoldenAnnotation(ContractModel):
    """One independently annotated Golden Dataset story.

    Missing-context cases can have strong community agreement while still
    withholding a *reference* verdict because essential facts are absent.
    """

    case_id: str = Field(min_length=1)
    narrative: str = Field(min_length=1)
    community_votes: CommunityVotes
    community_agreement: float = Field(ge=0, le=1)
    reference_verdict: Verdict | None
    case_type: Literal["consensus", "disputed", "missing_context"]
    key_facts: list[str]
    missing_information: list[str] = Field(default_factory=list)
    alternative_perspectives: list[str] = Field(default_factory=list)
    framing_bias: Literal["none", "possible", "clear"]
    # Preserve the two *independent* judgments for Cohen's kappa calculation.
    annotator_verdicts: list[AnnotatorVerdict] = Field(min_length=2, max_length=2)
    annotation_notes: str = Field(min_length=1)
    expected_uncertainty_tier: UncertaintyStatus
    expected_initial_action: Literal["proceed", "request_clarification"]

    @model_validator(mode="after")
    def validate_reference_label(self) -> Self:
        # Golden labels use up to ten top comments, including INFO comments.
        total_votes = self.community_votes.moral_count + self.community_votes.INFO
        if total_votes > 10:
            raise ValueError("Golden annotations count at most 10 source comments")
        winning_share = self.community_votes.winning_share
        if winning_share is None:
            raise ValueError(
                "community_agreement cannot be computed with zero moral votes; "
                "exclude this record or explicitly revise the contract to allow null"
            )
        if not isclose(self.community_agreement, winning_share, rel_tol=0, abs_tol=1e-6):
            raise ValueError("community_agreement must exclude INFO votes")

        if self.case_type == "missing_context":
            # This category takes precedence even when community agreement >= 70%.
            if self.reference_verdict is not None:
                raise ValueError("missing_context cases must have a null reference verdict")
            return self

        if self.community_votes.moral_count < 5:
            raise ValueError("consensus/disputed cases need at least 5 moral verdicts")

        if self.case_type == "disputed":
            if winning_share >= 0.70 or self.reference_verdict is not None:
                raise ValueError("disputed cases require <70% agreement and a null verdict")
        else:
            if winning_share < 0.70 or self.reference_verdict is None:
                raise ValueError("consensus cases require >=70% agreement and a verdict")
            winning_verdicts = (
                verdict
                for verdict in ("NTA", "YTA", "ESH", "NAH")
                if getattr(self.community_votes, verdict) / self.community_votes.moral_count
                == winning_share
            )
            if self.reference_verdict not in winning_verdicts:
                raise ValueError("reference_verdict must match the community majority")

        return self