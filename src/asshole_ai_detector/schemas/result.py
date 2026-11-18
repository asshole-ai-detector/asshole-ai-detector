"""Final user-facing assessment contract (Milestone 2, Sections 1.1 and 4.2)."""

from typing import Self

from pydantic import Field, model_validator

from .annotation import ContractModel, UncertaintyStatus, Verdict


class AssessmentResult(ContractModel):
    verdict: Verdict | None
    secondary_verdict: Verdict | None = None
    explanation: str = Field(min_length=1)
    alternative_perspectives: list[str] = Field(default_factory=list)
    uncertainty_tier: UncertaintyStatus
    missing_information: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_tier(self) -> Self:
        if self.uncertainty_tier == "withhold":
            if self.verdict is not None or self.secondary_verdict is not None:
                raise ValueError("Withhold tier must not return a verdict")
        elif self.verdict is None:
            raise ValueError("A non-withhold tier must include a verdict")

        if self.uncertainty_tier == "split":
            if self.secondary_verdict is None or self.secondary_verdict == self.verdict:
                raise ValueError("Split tier must provide two distinct verdicts")
        elif self.secondary_verdict is not None:
            raise ValueError("secondary_verdict is only used for split")
        return self
