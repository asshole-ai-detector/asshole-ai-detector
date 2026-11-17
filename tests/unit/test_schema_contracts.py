"""Validation tests for the Week 7 Pydantic v2 contracts.

Run from the folder containing ``schemas/`` and ``tests/``:
    python -m pytest -q

Important typing distinction:
* Valid examples use the exact declared types (Literal aliases, HttpUrl, etc.).
* Intentionally invalid examples go through ``model_validate``. This lets
  Pydantic reject bad runtime data without making the test source itself
  invalid under Pylance/Pyright static type checking.
"""

import math

import pytest
from pydantic import HttpUrl, ValidationError

from asshole_ai_detector.schemas.annotation import (
    CommunityVotes,
    GoldenAnnotation,
    Verdict,
)
from asshole_ai_detector.schemas.case import (
    AITACase,
    AITAComment,
    DerivedFields,
    VerdictDistribution,
)
from asshole_ai_detector.schemas.result import AssessmentResult
from asshole_ai_detector.schemas.tools import (
    AssessVerdictUncertaintyInput,
    AssessVerdictUncertaintyOutput,
    CheckSycophancyAndBiasInput,
    CheckSycophancyAndBiasOutput,
    ClassifierProbabilities,
    ClassifyVerdictInput,
    ClassifyVerdictOutput,
    CompletenessStatus,
    CounterPerspectiveReference,
    GenerateCounterPerspectiveInput,
    GenerateCounterPerspectiveOutput,
    GroundedPerspective,
    NeutralizeNarrativeInput,
    NeutralizeNarrativeOutput,
    RetrievalParameters,
    RetrievedCaseEvidence,
    RetrieveSimilarCasesInput,
    RetrieveSimilarCasesOutput,
)

# ---------------------------------------------------------------------------
# Small, reusable VALID fixtures. Each helper returns its real Pydantic type.
# ---------------------------------------------------------------------------


def sample_comments() -> list[AITAComment]:
    # YTA/NTA tie at 2 votes each; rank 1 YTA should break the tie.
    # Explicit annotation prevents Pylance inferring the labels as plain str.
    labels: tuple[Verdict, ...] = ("YTA", "NTA", "NTA", "YTA", "ESH")
    return [
        AITAComment(comment_id=f"abc#{rank}", rank=rank, verdict=verdict, text=f"Reason {rank}")
        for rank, verdict in enumerate(labels, start=1)
    ]


def sample_golden(**overrides: object) -> GoldenAnnotation:
    # ``model_validate`` accepts raw input, just like JSON loading does.
    # Overrides are useful for testing both valid and invalid payloads.
    payload: dict[str, object] = {
        "case_id": "example-1",
        "narrative": "A made a decision that upset B.",
        "community_votes": {"NTA": 1, "YTA": 4, "ESH": 0, "NAH": 0, "INFO": 2},
        "community_agreement": 0.8,  # 4 / 5 MORAL verdicts, not 4 / 7 total
        "reference_verdict": "YTA",
        "case_type": "consensus",
        "key_facts": ["A made a decision", "B was upset"],
        "framing_bias": "possible",
        "annotator_verdicts": ["YTA", "YTA"],
        "annotation_notes": "Both annotators independently chose YTA.",
        "expected_uncertainty_tier": "confident",
        "expected_initial_action": "proceed",
    }
    payload.update(overrides)
    return GoldenAnnotation.model_validate(payload)


def sample_classifier() -> ClassifyVerdictOutput:
    # Match ClassifierProbabilities = dict[ClassifierLabel, Probability].
    # A plain dict[str, float] is too broad for Pylance's Literal key types.
    probabilities: ClassifierProbabilities = {
        "NTA": 0.1,
        "YTA": 0.6,
        "ESH": 0.1,
        "NAH": 0.1,
        "INFO": 0.1,
    }
    return ClassifyVerdictOutput(
        probabilities=probabilities,
        predicted_class="YTA",
        top_two_margin=0.5,
        entropy=-sum(p * math.log(p) for p in probabilities.values()),
    )


def sample_evidence() -> RetrievedCaseEvidence:
    comments = sample_comments()
    return RetrievedCaseEvidence(
        post_id="abc",
        title="A situation with a friend",
        text="A conflict occurred involving two friends.",
        # HttpUrl's runtime constructor also matches Pylance's expected type.
        url=HttpUrl("https://reddit.com/r/AmItheAsshole/comments/abc"),
        derived_fields=DerivedFields.from_comments(comments),
        reranker_score=0.95,
        selected_comments=[comments[0], comments[1], comments[4]],
    )


# ---------------------------------------------------------------------------
# Golden Dataset annotations — Section 2.3
# ---------------------------------------------------------------------------


def test_info_votes_are_excluded_from_community_agreement() -> None:
    annotation = sample_golden()
    assert annotation.community_votes.moral_count == 5
    assert annotation.community_votes.winning_share == pytest.approx(0.8)
    with pytest.raises(ValidationError, match="exclude INFO"):
        sample_golden(community_agreement=4 / 7)


def test_consensus_requires_correct_majority_and_threshold() -> None:
    assert sample_golden().reference_verdict == "YTA"
    with pytest.raises(ValidationError, match="community majority"):
        sample_golden(reference_verdict="NTA")
    with pytest.raises(ValidationError, match="70%"):
        sample_golden(community_agreement=0.6, community_votes={
            "NTA": 2, "YTA": 3, "ESH": 0, "NAH": 0, "INFO": 0,
        })


def test_missing_context_can_have_high_agreement_but_no_reference() -> None:
    annotation = sample_golden(case_type="missing_context", reference_verdict=None)
    assert annotation.community_agreement == 0.8
    assert annotation.reference_verdict is None
    with pytest.raises(ValidationError, match="null reference"):
        sample_golden(case_type="missing_context")


def test_annotation_rejects_more_than_ten_source_comments() -> None:
    with pytest.raises(ValidationError, match="at most 10"):
        sample_golden(community_votes={
            "NTA": 2, "YTA": 8, "ESH": 0, "NAH": 0, "INFO": 2,
        })


def test_community_votes_cannot_be_negative() -> None:
    # Negative values deliberately enter as untrusted JSON-like data.
    with pytest.raises(ValidationError):
        CommunityVotes.model_validate({"NTA": -1, "YTA": 5, "ESH": 0, "NAH": 0, "INFO": 0})


# ---------------------------------------------------------------------------
# Indexed cases and derived labels — Section 3.2
# ---------------------------------------------------------------------------


def test_derived_fields_tie_break_uses_comment_rank() -> None:
    fields = DerivedFields.from_comments(sample_comments())
    assert fields.n_valid == 5
    assert fields.top_verdict == "YTA"
    assert fields.consensus_label == "DISPUTED"
    assert fields.verdict_distribution["ESH"] == pytest.approx(0.2)


def test_derived_fields_reject_inconsistent_consensus() -> None:
    fields = DerivedFields.from_comments(sample_comments())
    with pytest.raises(ValidationError, match="70% rule"):
        DerivedFields(
            verdict_distribution=fields.verdict_distribution,
            top_verdict="YTA",
            consensus_label="YTA",  # Invalid: only 40% support YTA.
            n_valid=5,
        )


def test_case_rejects_derived_fields_not_matching_comments() -> None:
    comments = sample_comments()
    data: dict[str, object] = {
        "post_id": "abc",
        "title": "An argument",
        "text": " ".join(["word"] * 50),
        "url": "https://reddit.com/r/AmItheAsshole/comments/abc",
        "derived_fields": DerivedFields.from_comments(comments),
        "valid_comments": comments,
    }
    assert AITACase.model_validate(data).derived_fields.n_valid == 5

    # Turning the last ESH into YTA changes the true distribution, but not
    # the supplied derived_fields. The full case validator should catch it.
    changed_comments = [
        *comments[:-1], comments[4].model_copy(update={"verdict": "YTA"})
    ]
    with pytest.raises(ValidationError, match="Incorrect verdict distribution"):
        AITACase.model_validate({**data, "valid_comments": changed_comments})


def test_case_rejects_short_post_and_wrong_comment_id() -> None:
    comments = sample_comments()
    data: dict[str, object] = {
        "post_id": "abc", "title": "Title", "text": "too short",
        "url": "https://reddit.com/r/AmItheAsshole/comments/abc",
        "derived_fields": DerivedFields.from_comments(comments),
        "valid_comments": comments,
    }
    with pytest.raises(ValidationError, match="at least 50 words"):
        AITACase.model_validate(data)

    wrong_id = comments[0].model_copy(update={"comment_id": "other#1"})
    with pytest.raises(ValidationError, match="Invalid comment_id"):
        AITACase.model_validate({
            **data,
            "text": " ".join(["word"] * 50),
            "valid_comments": [wrong_id, *comments[1:]],
        })


# ---------------------------------------------------------------------------
# Classifier and retrieval tool outputs — Section 4.3
# ---------------------------------------------------------------------------


def test_classifier_probabilities_margin_and_entropy() -> None:
    output = sample_classifier()
    assert output.predicted_class == "YTA"
    assert sum(output.probabilities.values()) == pytest.approx(1.0)
    assert output.top_two_margin == pytest.approx(0.5)

    with pytest.raises(ValidationError, match="top_two_margin"):
        ClassifyVerdictOutput.model_validate({
            **output.model_dump(), "top_two_margin": 0.0,
        })
    with pytest.raises(ValidationError, match="entropy"):
        ClassifyVerdictOutput.model_validate({
            **output.model_dump(), "entropy": 0.0,
        })


def test_classifier_must_contain_all_five_labels() -> None:
    data = sample_classifier().model_dump()
    del data["probabilities"]["INFO"]
    with pytest.raises(ValidationError, match="at least 5 items"):
        ClassifyVerdictOutput.model_validate(data)


def test_retrieved_evidence_has_distinct_verdict_comments() -> None:
    evidence = sample_evidence()
    assert len(evidence.selected_comments) == 3
    assert evidence.url.host == "reddit.com"

    # This bad input is intentional: test a Pydantic rejection at runtime.
    comments = sample_comments()
    with pytest.raises(ValidationError, match="one comment per verdict"):
        RetrievedCaseEvidence.model_validate({
            **evidence.model_dump(),
            "selected_comments": [comments[0].model_dump(), comments[3].model_dump()],
        })


def test_retrieval_stats_are_not_derived_from_displayed_cases() -> None:
    # The top-15 pool produces these stats; the verdict-balanced cases shown
    # in the UI can be a different subset. Keep these notions independent.
    distribution: VerdictDistribution = {
        "NTA": 0.5, "YTA": 0.5, "ESH": 0.0, "NAH": 0.0,
    }
    retrieval = RetrieveSimilarCasesOutput(
        retrieved_cases=[sample_evidence()],
        neighbor_verdict_distribution=distribution,
        retrieval_disagreement=0.5,  # Entropy normalized by ln(4).
        disputed_share=0.4,
        retrieval_confidence=0.82,
        best_reranker_score=0.95,
        neighbors_considered=5,
    )
    assert retrieval.neighbors_considered == 5
    assert len(retrieval.retrieved_cases) == 1


def test_retrieval_rejects_incorrect_entropy_and_invalid_empty_stats() -> None:
    distribution: VerdictDistribution = {
        "NTA": 0.5, "YTA": 0.5, "ESH": 0.0, "NAH": 0.0,
    }
    with pytest.raises(ValidationError, match="normalized entropy"):
        RetrieveSimilarCasesOutput(
            neighbor_verdict_distribution=distribution,
            retrieval_disagreement=0.9,
            disputed_share=0.0,
            retrieval_confidence=0.7,
            best_reranker_score=0.8,
            neighbors_considered=5,
        )
    with pytest.raises(ValidationError, match="Empty retrieval"):
        RetrieveSimilarCasesOutput(
            retrieval_confidence=0.7,
            best_reranker_score=0.8,
            neighbors_considered=0,
        )


def test_retrieval_parameters_enforce_rank_order() -> None:
    with pytest.raises(ValidationError, match="dense_top_k"):
        RetrievalParameters(dense_top_k=5, rerank_top_k=15, evidence_top_k=5)


# ---------------------------------------------------------------------------
# Perspective, bias checks, and final uncertainty — Sections 4.2 and 4.3
# ---------------------------------------------------------------------------


def test_perspective_reference_must_not_be_empty() -> None:
    with pytest.raises(ValidationError, match="one scenario fact or comment ID"):
        CounterPerspectiveReference()


def test_bias_disagreement_alone_is_not_a_failed_grounding_check() -> None:
    result = CheckSycophancyAndBiasOutput(
        classifier_retrieval_disagreement=True,
        passed=True,
    )
    assert result.passed
    with pytest.raises(ValidationError, match="passed"):
        CheckSycophancyAndBiasOutput(
            unsupported_claims=["Unfounded motive"],
            passed=True,
        )


def test_uncertainty_split_requires_two_distinct_moral_verdicts() -> None:
    decision = AssessVerdictUncertaintyOutput(
        uncertainty_tier="split",
        verdict="YTA",
        secondary_verdict="ESH",
        reasons=["Low classifier margin"],
    )
    assert decision.secondary_verdict == "ESH"
    with pytest.raises(ValidationError, match="distinct"):
        AssessmentResult(
            uncertainty_tier="split",
            verdict="YTA",
            secondary_verdict="YTA",
            explanation="Mixed signals.",
        )


def test_final_verdict_never_allows_info() -> None:
    # Correct way to test bad external JSON: pass it to model_validate.
    # Do NOT write AssessmentResult(verdict="INFO", ...): the static checker
    # should correctly reject that call before the test even runs.
    with pytest.raises(ValidationError):
        AssessmentResult.model_validate({
            "uncertainty_tier": "confident",
            "verdict": "INFO",
            "explanation": "Need more facts.",
        })


def test_withhold_must_have_null_verdict_and_clarification_needs_questions() -> None:
    assert AssessmentResult(
        verdict=None, uncertainty_tier="withhold", explanation="Not enough context."
    ).verdict is None
    with pytest.raises(ValidationError, match="Withhold"):
        AssessmentResult(
            verdict="YTA", uncertainty_tier="withhold", explanation="Contradiction."
        )
    with pytest.raises(ValidationError, match="requires questions"):
        AssessVerdictUncertaintyOutput(
            uncertainty_tier="withhold",
            verdict=None,
            reasons=["A critical fact is missing"],
            request_clarification=True,
        )


def test_all_six_tool_contracts_accept_a_connected_mock_workflow() -> None:
    # Schema-only smoke test: DOES NOT call an LLM, RoBERTa, MCP or Chroma.
    original = "I refused to help my friend at short notice."
    neutralization_input = NeutralizeNarrativeInput(original_scenario=original)
    neutralization = NeutralizeNarrativeOutput(
        neutralized_narrative="A person refused a last-minute request from a friend.",
        preserved_facts=["A refused the request"],
        relationship="friends", trigger="a last-minute request", actions=["refused to help"],
    )
    classifier_input = ClassifyVerdictInput(
        neutralized_scenario=neutralization.neutralized_narrative,
    )
    classifier_output = sample_classifier()
    retrieval_input = RetrieveSimilarCasesInput(
        neutralized_scenario=classifier_input.neutralized_scenario,
        relationship=neutralization.relationship,
        trigger=neutralization.trigger,
        actions=neutralization.actions,
    )
    # No matches is valid; no invented neighbor statistics or precedents.
    retrieval_output = RetrieveSimilarCasesOutput(
        retrieved_cases=[], retrieval_confidence=0.0,
        best_reranker_score=0.0, neighbors_considered=0,
    )
    perspective_input = GenerateCounterPerspectiveInput(
        neutralized_scenario=neutralization.neutralized_narrative,
        candidate_verdict="YTA",
        retrieved_cases=retrieval_output.retrieved_cases,
    )
    draft = GenerateCounterPerspectiveOutput(
        alternative_perspectives=[GroundedPerspective(
            text="The friend may have been under pressure.",
            references=CounterPerspectiveReference(scenario_facts=["A refused the request"]),
        )],
        preliminary_explanation="The request and refusal both matter.",
        explanation_references=CounterPerspectiveReference(scenario_facts=["A refused the request"]),
    )
    bias_input = CheckSycophancyAndBiasInput(
        original_scenario=original,
        draft=draft, classifier_output=classifier_output,
        retrieval_output=retrieval_output,
    )
    bias_output = CheckSycophancyAndBiasOutput(
        role_swapped_narrative="I asked for help but was refused.",
        role_swapped_prediction="NTA",
        role_swap_consistent=True,
        passed=True,
    )
    uncertainty_input = AssessVerdictUncertaintyInput(
        classifier_output=classifier_output,
        retrieval_output=retrieval_output,
        completeness=CompletenessStatus(initial_action="proceed"),
        bias_check=bias_output,
    )
    decision = AssessVerdictUncertaintyOutput(
        uncertainty_tier="leaning", verdict="YTA",
        reasons=["Limited retrieval support"],
    )
    final = AssessmentResult(
        verdict=decision.verdict,
        explanation=draft.preliminary_explanation,
        alternative_perspectives=[p.text for p in draft.alternative_perspectives],
        uncertainty_tier=decision.uncertainty_tier,
        limitations=["No relevant retrieved cases"],
    )

    assert neutralization_input.original_scenario == original
    assert retrieval_input.parameters.evidence_top_k == 5
    assert perspective_input.candidate_verdict == "YTA"
    assert bias_input.draft is draft
    assert uncertainty_input.bias_check is bias_output
    assert final.model_dump(mode="json")["uncertainty_tier"] == "leaning"


def test_all_six_tool_json_schemas_are_exportable() -> None:
    # The MCP server will later use these classes as its contracts.
    tool_contracts = (
        ClassifyVerdictInput, ClassifyVerdictOutput,
        NeutralizeNarrativeInput, NeutralizeNarrativeOutput,
        RetrieveSimilarCasesInput, RetrieveSimilarCasesOutput,
        GenerateCounterPerspectiveInput, GenerateCounterPerspectiveOutput,
        CheckSycophancyAndBiasInput, CheckSycophancyAndBiasOutput,
        AssessVerdictUncertaintyInput, AssessVerdictUncertaintyOutput,
    )
    for contract in tool_contracts:
        assert "properties" in contract.model_json_schema()
