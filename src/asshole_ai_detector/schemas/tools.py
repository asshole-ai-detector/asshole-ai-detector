"""Input/output contracts for the six MCP tools (Milestone 2, Section 4.3).

These describe validated messages, NOT MCP decorators or tool implementations.
The agent/workflow controller is responsible for calling them in the right order.
"""

from datetime import datetime
from math import isclose, log
from typing import Literal, Self, TypeAlias

from pydantic import Field, HttpUrl, model_validator

from .annotation import ClassifierLabel, ContractModel, UncertaintyStatus, Verdict
from .case import (
    AITAComment,
    DerivedFields,
    Probability,
    VerdictDistribution,
    VERDICTS,
)

CLASSIFIER_LABELS: tuple[ClassifierLabel, ...] = (
    "NTA", "YTA", "ESH", "NAH", "INFO"
)
ClassifierProbabilities: TypeAlias = dict[ClassifierLabel, Probability]


# 1. classify_verdict ----------------------------------------------------------

class ClassifyVerdictInput(ContractModel):
    # In the proposed workflow, RoBERTa classifies the *neutralized* narrative.
    neutralized_scenario: str = Field(min_length=1)


class ClassifyVerdictOutput(ContractModel):
    """Calibrated five-class RoBERTa output; INFO is not a final verdict."""

    probabilities: ClassifierProbabilities = Field(min_length=5, max_length=5)
    predicted_class: ClassifierLabel
    top_two_margin: Probability
    entropy: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_predictions(self) -> Self:
        if set(self.probabilities) != set(CLASSIFIER_LABELS):
            raise ValueError("All five classifier labels must be present")
        scores = sorted(self.probabilities.values(), reverse=True)
        if not isclose(sum(scores), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Classifier probabilities must sum to 1")
        if not isclose(
            self.probabilities[self.predicted_class], scores[0], rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError("predicted_class must have the highest probability")
        if not isclose(self.top_two_margin, scores[0] - scores[1], rel_tol=0, abs_tol=1e-6):
            raise ValueError("top_two_margin is inconsistent with the probabilities")
        expected_entropy = -sum(p * log(p) for p in scores if p > 0)
        if not isclose(self.entropy, expected_entropy, rel_tol=0, abs_tol=1e-4):
            raise ValueError("entropy must be the unnormalized Shannon entropy")
        return self


# 2. neutralize_narrative ------------------------------------------------------

class NeutralizeNarrativeInput(ContractModel):
    original_scenario: str = Field(min_length=1)


class NeutralizeNarrativeOutput(ContractModel):
    """Bias-reduced wording plus facts retained from the original story."""

    neutralized_narrative: str = Field(min_length=1)
    preserved_facts: list[str] = Field(default_factory=list)
    # These fields enable Section 3.5's single deterministic retrieval retry.
    # They must describe supplied facts, not invented motivations or events.
    relationship: str | None = None
    trigger: str | None = None
    actions: list[str] = Field(default_factory=list)


# 3. retrieve_similar_cases ----------------------------------------------------

class RetrievalParameters(ContractModel):
    """Initial values from Section 3.5; final values are frozen after dev tuning."""

    dense_top_k: int = Field(default=30, ge=1)
    rerank_top_k: int = Field(default=15, ge=1)
    evidence_top_k: int = Field(default=5, ge=1)
    reranker_threshold: Probability = 0.5

    @model_validator(mode="after")
    def validate_k_order(self) -> Self:
        if not self.dense_top_k >= self.rerank_top_k >= self.evidence_top_k:
            raise ValueError("Expected dense_top_k >= rerank_top_k >= evidence_top_k")
        return self


class RetrievedCaseEvidence(ContractModel):
    """One displayed precedent, NOT a full indexed AITACase.

    DerivedFields refers to *all* original valid comments. selected_comments
    contains at most three (one per distinct verdict), capped at 60 words each.
    """

    post_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    url: HttpUrl
    date: datetime | None = None
    derived_fields: DerivedFields
    reranker_score: Probability
    selected_comments: list[AITAComment] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_selected_comments(self) -> Self:
        seen_verdicts: set[Verdict] = set()
        for comment in self.selected_comments:
            if comment.comment_id != f"{self.post_id}#{comment.rank}":
                raise ValueError("Retrieved comment ID must match post_id#rank")
            if comment.verdict in seen_verdicts:
                raise ValueError("Selected evidence may include only one comment per verdict")
            if self.derived_fields.verdict_distribution[comment.verdict] == 0:
                raise ValueError("Evidence comment contradicts the source distribution")
            if len(comment.text.split()) > 60:
                raise ValueError("Retrieved comment excerpts must be <=60 words")
            seen_verdicts.add(comment.verdict)
        return self


class RetrieveSimilarCasesInput(ContractModel):
    """Neutralized text and deterministic retry fields from neutralization."""

    neutralized_scenario: str = Field(min_length=1)
    relationship: str | None = None
    trigger: str | None = None
    actions: list[str] = Field(default_factory=list)
    parameters: RetrievalParameters = Field(default_factory=RetrievalParameters)


class RetrieveSimilarCasesOutput(ContractModel):
    """Separate top-15 *statistics* from up-to-five *displayed* precedents.

    Neighbor statistics use the reranked pool; displayed cases are selected
    verdict-first for balanced evidence. The two sets are not interchangeable.
    """

    retrieved_cases: list[RetrievedCaseEvidence] = Field(default_factory=list)
    neighbor_verdict_distribution: VerdictDistribution | None = None
    # Normalized Shannon entropy of the four moral verdict probabilities.
    retrieval_disagreement: Probability | None = None
    disputed_share: Probability | None = None
    # Average score of top five *reranked* neighbors, not balanced evidence.
    retrieval_confidence: Probability
    best_reranker_score: Probability
    neighbors_considered: int = Field(ge=0)
    retry_performed: bool = False

    @model_validator(mode="after")
    def validate_retrieval_statistics(self) -> Self:
        if len({case.post_id for case in self.retrieved_cases}) != len(self.retrieved_cases):
            raise ValueError("Displayed retrieved cases must have distinct post IDs")
        if self.retrieval_confidence > self.best_reranker_score + 1e-6:
            raise ValueError("Mean retrieval confidence cannot exceed the best score")
        if any(case.reranker_score > self.best_reranker_score + 1e-6 for case in self.retrieved_cases):
            raise ValueError("A displayed case cannot score above the best reranker score")
        if len(self.retrieved_cases) > self.neighbors_considered:
            raise ValueError("Cannot display more retrieved cases than scored neighbors")
        if self.neighbors_considered == 0:
            if (self.retrieval_confidence != 0 or self.best_reranker_score != 0):
                raise ValueError("Empty retrieval must have zero confidence and best score")
            if (
                self.neighbor_verdict_distribution is not None
                or self.retrieval_disagreement is not None
                or self.disputed_share is not None
            ):
                raise ValueError("Empty retrieval has no disagreement statistics")
            return self

        if self.disputed_share is None:
            raise ValueError("disputed_share is required when neighbors exist")
        if self.neighbor_verdict_distribution is None:
            # Zero total relevance weight has no well-defined weighted average.
            if self.retrieval_disagreement is not None:
                raise ValueError("Cannot give entropy without a distribution")
            return self

        distribution = self.neighbor_verdict_distribution
        if set(distribution) != set(VERDICTS):
            raise ValueError("Neighbor distribution must contain four verdicts")
        if not isclose(sum(distribution.values()), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Neighbor verdict distribution must sum to 1")
        if self.retrieval_disagreement is None:
            raise ValueError("retrieval_disagreement is required with a distribution")
        expected_entropy = (
            -sum(p * log(p) for p in distribution.values() if p > 0) / log(4)
        )
        if not isclose(self.retrieval_disagreement, expected_entropy, rel_tol=0, abs_tol=1e-4):
            raise ValueError("retrieval_disagreement must be normalized entropy")
        return self


# 4. generate_counter_perspective ---------------------------------------------

class CounterPerspectiveReference(ContractModel):
    """Claim-level evidence pointers; reference validity is checked later."""

    scenario_facts: list[str] = Field(default_factory=list)
    comment_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def has_reference(self) -> Self:
        if not self.scenario_facts and not self.comment_ids:
            raise ValueError("A claim needs at least one scenario fact or comment ID")
        return self


class GroundedPerspective(ContractModel):
    text: str = Field(min_length=1)
    references: CounterPerspectiveReference


class GenerateCounterPerspectiveInput(ContractModel):
    # The workflow controller selects the candidate moral verdict, not the LLM.
    # When INFO requires withholding, the controller handles it separately.
    neutralized_scenario: str = Field(min_length=1)
    candidate_verdict: Verdict
    retrieved_cases: list[RetrievedCaseEvidence] = Field(default_factory=list)


class GenerateCounterPerspectiveOutput(ContractModel):
    """One LLM call returns perspectives AND a preliminary explanation."""

    alternative_perspectives: list[GroundedPerspective] = Field(min_length=1)
    preliminary_explanation: str = Field(min_length=1)
    explanation_references: CounterPerspectiveReference


# 5. check_sycophancy_and_bias -----------------------------------------------

class CheckSycophancyAndBiasInput(ContractModel):
    original_scenario: str = Field(min_length=1)
    draft: GenerateCounterPerspectiveOutput
    classifier_output: ClassifyVerdictOutput
    retrieval_output: RetrieveSimilarCasesOutput


class CheckSycophancyAndBiasOutput(ContractModel):
    """Results of role swap, grounding, and classifier/retrieval checks."""

    role_swapped_narrative: str | None = None
    role_swapped_prediction: ClassifierLabel | None = None
    # None means unassessed, e.g. a role swap could not be constructed.
    role_swap_consistent: bool | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_comment_ids: list[str] = Field(default_factory=list)
    # Disagreement alone is uncertainty, not automatically narrator favoritism.
    classifier_retrieval_disagreement: bool = False
    unjustified_narrator_agreement: bool = False
    limitations: list[str] = Field(default_factory=list)
    passed: bool

    @model_validator(mode="after")
    def validate_pass_flag(self) -> Self:
        if self.role_swap_consistent is not None and (
            self.role_swapped_narrative is None or self.role_swapped_prediction is None
        ):
            raise ValueError("An assessed role swap needs a narrative and prediction")
        has_bias_issue = (
            self.role_swap_consistent is False
            or bool(self.unsupported_claims)
            or bool(self.invalid_comment_ids)
            or self.unjustified_narrator_agreement
        )
        if self.passed == has_bias_issue:
            raise ValueError("passed must be false if a bias check flags an issue")
        return self


# 6. assess_verdict_uncertainty -----------------------------------------------

class CompletenessStatus(ContractModel):
    """Deterministic controller state after structural/clarification checks."""

    initial_action: Literal["proceed", "request_clarification"]
    clarification_already_requested: bool = False
    essential_information_missing: bool = False
    assessment_possible: bool = True
    missing_information: list[str] = Field(default_factory=list)


class AssessVerdictUncertaintyInput(ContractModel):
    classifier_output: ClassifyVerdictOutput
    retrieval_output: RetrieveSimilarCasesOutput
    completeness: CompletenessStatus
    bias_check: CheckSycophancyAndBiasOutput
    # Thresholds such as m_low, m_high, t_info and retrieval-disagreement
    # are deployment configuration, tuned on development data (not per-call).


class AssessVerdictUncertaintyOutput(ContractModel):
    """Deterministic four-tier decision, before formatting AssessmentResult."""

    uncertainty_tier: UncertaintyStatus
    verdict: Verdict | None
    secondary_verdict: Verdict | None = None
    reasons: list[str] = Field(min_length=1)
    missing_information: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list, max_length=3)
    request_clarification: bool = False

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.uncertainty_tier == "withhold":
            if self.verdict is not None or self.secondary_verdict is not None:
                raise ValueError("Withhold must not return a verdict")
        elif self.verdict is None:
            raise ValueError("Only the withhold tier can have a null verdict")

        if self.uncertainty_tier == "split":
            if self.secondary_verdict is None or self.secondary_verdict == self.verdict:
                raise ValueError("Split requires two distinct moral verdicts")
        elif self.secondary_verdict is not None:
            raise ValueError("Secondary verdict is permitted only for split")

        if self.request_clarification and not self.clarification_questions:
            raise ValueError("Requesting clarification requires questions")
        return self
