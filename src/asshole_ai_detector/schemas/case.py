"""Normalized AITA index payloads (Milestone 2, Section 3.2)."""

from collections import Counter
from datetime import datetime
from math import isclose
from typing import Annotated, Literal, Self

from pydantic import Field, HttpUrl, model_validator

from .annotation import ContractModel, Verdict

VERDICTS: tuple[Verdict, ...] = ("NTA", "YTA", "ESH", "NAH")

type Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
type VerdictDistribution = dict[Verdict, Probability]
type ConsensusLabel = Verdict | Literal["DISPUTED"]


class AITAComment(ContractModel):
    """A valid, ranked comment with a four-class moral verdict (never INFO)."""

    comment_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    verdict: Verdict
    text: str = Field(min_length=1)


class DerivedFields(ContractModel):
    """Consensus statistics computed from *all* valid comments of a post."""

    verdict_distribution: VerdictDistribution = Field(min_length=4, max_length=4)
    top_verdict: Verdict
    consensus_label: ConsensusLabel
    n_valid: int = Field(ge=5)

    @classmethod
    def from_comments(cls, comments: list[AITAComment]) -> Self:
        """Apply Section 3.2's >=5 rule, plurality tie-break, and 70% threshold."""
        if len(comments) < 5:
            raise ValueError("At least 5 valid moral-verdict comments are required")

        counts = Counter(comment.verdict for comment in comments)
        max_votes = max(counts.values())
        tied = {verdict for verdict in VERDICTS if counts[verdict] == max_votes}

        # The highest-ranked (smallest rank number) comment breaks plurality ties.
        top_verdict: Verdict = next(
            comment.verdict
            for comment in sorted(comments, key=lambda c: c.rank)
            if comment.verdict in tied
        )
        n_valid = len(comments)
        share = max_votes / n_valid
        consensus_label: ConsensusLabel = (
            top_verdict if share >= 0.70 else "DISPUTED"
        )
        return cls(
            verdict_distribution={
                verdict: counts[verdict] / n_valid for verdict in VERDICTS
            },
            top_verdict=top_verdict,
            consensus_label=consensus_label,
            n_valid=n_valid,
        )

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        # Standalone DerivedFields must also be internally coherent. The full
        # AITACase validator checks these statistics against the comments.
        scores = self.verdict_distribution
        if set(scores) != set(VERDICTS):
            raise ValueError("Distribution must contain exactly four moral verdicts")
        if not isclose(sum(scores.values()), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Verdict distribution must sum to 1")
        if scores[self.top_verdict] < max(scores.values()) - 1e-6:
            raise ValueError("top_verdict must be a plurality winner")
        expected = self.top_verdict if scores[self.top_verdict] >= 0.70 else "DISPUTED"
        if self.consensus_label != expected:
            raise ValueError("consensus_label does not match the 70% rule")
        return self


class AITACase(ContractModel):
    """A cleaned, indexed post; not the shorter evidence shown by retrieval."""

    post_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    url: HttpUrl
    date: datetime | None = None
    derived_fields: DerivedFields
    valid_comments: list[AITAComment] = Field(min_length=5)
    # Diagnostic provider score: never an input to consensus or uncertainty.
    provider_ambiguity: Probability | None = None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if len(self.text.split()) < 50:
            raise ValueError("Indexed posts must contain at least 50 words")

        ranks = [comment.rank for comment in self.valid_comments]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Valid comment ranks must be unique")
        # Ranks need not be consecutive: removed/INFO comments can leave gaps.
        for comment in self.valid_comments:
            expected_id = f"{self.post_id}#{comment.rank}"
            if comment.comment_id != expected_id:
                raise ValueError(f"Invalid comment_id: expected {expected_id}")

        expected = DerivedFields.from_comments(self.valid_comments)
        actual = self.derived_fields
        if (
            actual.n_valid != expected.n_valid
            or actual.top_verdict != expected.top_verdict
            or actual.consensus_label != expected.consensus_label
        ):
            raise ValueError("Derived fields do not match the post's valid comments")
        for verdict in VERDICTS:
            if not isclose(
                actual.verdict_distribution[verdict],
                expected.verdict_distribution[verdict],
                rel_tol=0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"Incorrect verdict distribution for {verdict}")
        return self
    


class NormalizedAITAComment(ContractModel):
    """Intermediate comment record before filtering invalid verdicts.

    Comments may have missing text or unsupported verdict labels.
    The preprocessing pipeline determines whether they are valid.
    """

    # One-based position in the original comment ordering.
    # Preserved for deterministic tie-breaking.
    rank: int = Field(ge=1)

    # Raw verdict label; may be INFO, missing, or unsupported.
    verdict: str | None = None

    # Missing or deleted comment text is handled during filtering.
    text: str | None = None


class NormalizedAITACase(ContractModel):
    """Intermediate AITA submission before filtering and deriving labels.

    Fields remain optional because source records may be incomplete.
    Only eligible records are converted into validated AITACase objects.
    """

    # Source identifier used for deduplication and comment IDs.
    post_id: str | None = None

    # Original submission content, validated during preprocessing.
    title: str | None = None
    text: str | None = None

    # Source metadata; URL validation happens in the final case schema.
    url: str | None = None
    date: datetime | None = None

    # Original comments, including those later excluded by filtering.
    comments: list[NormalizedAITAComment] = Field(default_factory=list)

    # Dataset-provided ambiguity score; diagnostic only.
    # Not used for filtering or consensus calculations.
    provider_ambiguity: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
