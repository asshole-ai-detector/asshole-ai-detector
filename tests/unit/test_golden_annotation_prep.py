import json
from dataclasses import replace
from uuid import UUID

import pytest
from pydantic import ValidationError

from asshole_ai_detector.data.aita_adapter import row_to_normalized
from asshole_ai_detector.data.consensus import label_cleaned_case
from asshole_ai_detector.evaluation.golden_annotation_prep import (
    Adjudication,
    AnnotationSource,
    export_annotation_review_manifest,
    export_blinded_assignments,
    prepare_annotation_records,
    validate_completed_annotation,
)
from asshole_ai_detector.evaluation.golden_selection import (
    GoldenCandidate,
    GoldenSelectionResult,
    select_golden_candidates,
)
from asshole_ai_detector.schemas.case import (
    AITACase,
    AITAComment,
    DerivedFields,
)


def make_case(post_id="example", votes=None):
    votes = votes if votes is not None else ["NTA"] * 8 + ["YTA"] * 2

    comments = [
        AITAComment(
            comment_id=f"{post_id}#{rank}",
            rank=rank,
            verdict=verdict,
            text=f"Comment {rank}",
        )
        for rank, verdict in enumerate(votes, start=1)
    ]

    return AITACase(
        post_id=post_id,
        title="Example interpersonal conflict",
        text=" ".join(["A detailed account of an interpersonal conflict."] * 12),
        url=f"https://www.reddit.com/r/AmItheAsshole/comments/{post_id}/",
        valid_comments=comments,
        derived_fields=DerivedFields.from_comments(comments),
    )


def make_selection(case, category="NTA"):
    fields = case.derived_fields

    return GoldenSelectionResult(
        selected=[
            GoldenCandidate(
                case_id=case.post_id,
                category=category,
                community_agreement=fields.verdict_distribution[fields.top_verdict],
                n_valid_comments=fields.n_valid,
            )
        ]
    )


def test_prepares_review_record():
    case = make_case()
    result = make_selection(case)

    records = prepare_annotation_records(result, [case])

    assert len(records) == 1
    assert records[0].case_id == "example"
    assert records[0].proposed_category == "NTA"
    assert records[0].review_status == "pending_human_review"
    assert records[0].community_agreement_all_valid == 0.8
    assert records[0].requires_source_comment_review


def test_missing_context_requires_review():
    case = make_case()
    result = make_selection(case, category="missing_context")

    records = prepare_annotation_records(result, [case])

    assert records[0].requires_missing_context_review


def test_missing_source_case_is_rejected():
    case = make_case()
    result = make_selection(case)

    with pytest.raises(ValueError, match="not found"):
        prepare_annotation_records(result, [])


def test_duplicate_source_cases_are_rejected():
    case = make_case()
    result = make_selection(case)

    with pytest.raises(ValueError, match="Duplicate source"):
        prepare_annotation_records(result, [case, case])


def test_inconsistent_candidate_statistics_are_rejected():
    case = make_case()
    result = make_selection(case)
    result.selected[0] = GoldenCandidate(
        case_id=case.post_id,
        category="NTA",
        community_agreement=0.9,
        n_valid_comments=10,
    )

    with pytest.raises(ValueError, match="statistics mismatch"):
        prepare_annotation_records(result, [case])


def test_review_manifest_excludes_story_content(tmp_path):
    case = make_case()
    result = make_selection(case)
    records = prepare_annotation_records(result, [case])

    path = tmp_path / "review.json"
    export_annotation_review_manifest(records, path)

    content = path.read_text(encoding="utf-8")
    data = json.loads(content)

    assert data["status"] == "pending_human_review"
    assert data["records"][0]["case_id"] == "example"
    assert case.text not in content
    assert case.title not in content
    assert "Comment 1" not in content


def test_review_manifest_prevents_overwrite(tmp_path):
    path = tmp_path / "review.json"

    export_annotation_review_manifest([], path)

    with pytest.raises(FileExistsError):
        export_annotation_review_manifest([], path)


def test_review_manifest_rejects_duplicate_ids(tmp_path):
    case = make_case()
    record = prepare_annotation_records(make_selection(case), [case])[0]

    with pytest.raises(ValueError, match="Duplicate IDs"):
        export_annotation_review_manifest(
            [record, record],
            tmp_path / "review.json",
        )


def completed_annotation():
    return {
        "case_id": "example",
        "narrative": "A sufficiently detailed example narrative.",
        "community_votes": {
            "NTA": 8,
            "YTA": 2,
            "ESH": 0,
            "NAH": 0,
            "INFO": 0,
        },
        "community_agreement": 0.8,
        "reference_verdict": "NTA",
        "case_type": "consensus",
        "key_facts": ["The narrator described the conflict."],
        "missing_information": [],
        "alternative_perspectives": [],
        "framing_bias": "possible",
        "annotator_verdicts": ["NTA", "NTA"],
        "annotation_notes": "Independent review completed.",
        "expected_uncertainty_tier": "confident",
        "expected_initial_action": "proceed",
    }


def make_source(votes=None):
    case = make_case()
    row = {
        "submission_title": case.title,
        "submission_text": case.text,
        "submission_url": str(case.url),
    }
    for rank, verdict in enumerate(votes or ["NTA"] * 8 + ["YTA"] * 2, 1):
        row[f"top_comment_{rank}"] = f"Private source comment {rank}"
        row[f"top_comment_{rank}_classification"] = verdict
    return AnnotationSource(row_to_normalized(row), "fixture-revision", "source_text")


def accepted_inputs(source=None):
    return {
        "expected_case_id": "example",
        "expected_category": "NTA",
        "source": source or make_source(),
        "expected_revision": "fixture-revision",
    }


def test_valid_completed_annotation():
    data = completed_annotation()
    data["narrative"] = make_case().text
    annotation = validate_completed_annotation(data, **accepted_inputs())

    assert annotation.case_type == "consensus"
    assert annotation.reference_verdict == "NTA"


def test_completed_annotation_requires_two_verdicts():
    data = completed_annotation()
    data["annotator_verdicts"] = ["NTA"]

    with pytest.raises(ValidationError):
        validate_completed_annotation(data)


def test_completed_annotation_rejects_wrong_case():
    with pytest.raises(ValueError, match="case ID"):
        validate_completed_annotation(
            completed_annotation(),
            expected_case_id="different",
        )


def test_completed_annotation_rejects_wrong_category():
    with pytest.raises(ValueError, match="verdict"):
        validate_completed_annotation(
            completed_annotation(),
            expected_category="YTA",
        )


def test_completed_annotation_rejects_more_than_ten_votes():
    data = completed_annotation()
    data["community_votes"]["NTA"] = 9
    data["community_agreement"] = 9 / 11

    with pytest.raises(ValidationError):
        validate_completed_annotation(data)


@pytest.mark.parametrize("proposal", ["YTA", "disputed"])
def test_preparation_rejects_wrong_category_even_with_matching_share(proposal):
    case = make_case(votes=["NTA"] * 8 + ["YTA"] * 2)
    with pytest.raises(ValueError, match="category"):
        prepare_annotation_records(make_selection(case, category=proposal), [case])


def test_completion_requires_evidence_even_with_correct_id_and_proposal():
    with pytest.raises(ValueError, match="pinned source"):
        validate_completed_annotation(
            completed_annotation(), expected_case_id="example", expected_category="NTA"
        )


def test_completion_rejects_wrong_revision_identity_and_narrative():
    data = completed_annotation()
    data["narrative"] = make_case().text
    source = make_source()
    for changes, message in [
        ({"source": replace(source, revision="other")}, "revision"),
        (
            {
                "source": replace(
                    source, row=source.row.model_copy(update={"post_id": "other"})
                )
            },
            "case ID",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_completed_annotation(data, **(accepted_inputs() | changes))
    with pytest.raises(ValueError, match="narrative"):
        validate_completed_annotation(
            data | {"narrative": "Different story"}, **accepted_inputs()
        )


def test_complete_source_reconciliation_keeps_info_and_original_slots():
    source = make_source(
        ["INFO", "NTA", None, "YTA", "NTA", "NTA", "NTA", "NTA", "YTA", "INFO"]
    )
    source.row.comments[-1].text = "[deleted]"
    narrative, clean = source.resolve(source.revision)
    case = label_cleaned_case(clean)
    result = select_golden_candidates([case])
    review = prepare_annotation_records(result, [case])[0]
    assert review.top_moral_comment_ranks == (2, 4, 5, 6, 7, 8, 9)
    data = completed_annotation() | {
        "narrative": narrative,
        "community_votes": clean.community_votes.model_dump(),
        "community_agreement": 5 / 7,
    }
    annotation = validate_completed_annotation(data, **accepted_inputs(source))
    assert annotation.community_votes.INFO == 1
    for stale_votes in ({"INFO": 0}, {"NTA": 6}, {"YTA": 1}):
        votes = data["community_votes"] | stale_votes
        stale = data | {
            "community_votes": votes,
            "community_agreement": votes["NTA"] / (votes["NTA"] + votes["YTA"]),
        }
        with pytest.raises(ValueError, match="votes disagree"):
            validate_completed_annotation(stale, **accepted_inputs(source))


def test_filtered_source_cannot_silently_replace_info_with_zero():
    source = make_source(["NTA"] * 9 + ["INFO"])
    _, cleaned = source.resolve(source.revision)
    assert cleaned.community_votes.INFO == 1
    with pytest.raises(TypeError, match="unfiltered DATA-01"):
        replace(source, row=cleaned).resolve(source.revision)


def test_resolution_rejects_out_of_range_info_slot():
    source = make_source(["NTA"] * 9 + ["INFO"])
    source.row.comments[-1].rank = 11
    with pytest.raises(ValueError, match="top-ten"):
        source.resolve(source.revision)


def test_adjudicated_missing_context_requires_decision_and_evidence():
    data = completed_annotation() | {
        "narrative": make_case().text,
        "case_type": "missing_context",
        "reference_verdict": None,
        "expected_uncertainty_tier": "withhold",
        "missing_information": ["Was consent obtained before disclosure?"],
        "annotation_notes": "Both reviewers found the missing consent materially changes the assessment.",
    }
    decision = Adjudication(
        "example", "NTA", "missing_context", data["annotation_notes"]
    )
    inputs = accepted_inputs()
    with pytest.raises(ValueError, match="requires adjudication"):
        validate_completed_annotation(data, **inputs)
    for bad in (
        replace(decision, rationale=" "),
        replace(decision, case_id="other"),
        replace(decision, final_category="YTA"),
    ):
        with pytest.raises(ValueError, match="decision"):
            validate_completed_annotation(data, adjudication=bad, **inputs)
    for empty in ([], [" "], ["??"], ["N/A"]):
        with pytest.raises(ValueError, match="essential missing information"):
            validate_completed_annotation(
                data | {"missing_information": empty}, adjudication=decision, **inputs
            )
    final = validate_completed_annotation(data, adjudication=decision, **inputs)
    assert final.case_type == "missing_context" and final.annotator_verdicts == [
        "NTA",
        "NTA",
    ]


def test_blinded_export_only_contains_opaque_assignment_and_approved_narrative(
    tmp_path,
):
    source = make_source()
    narrative, clean = source.resolve(source.revision)
    case = label_cleaned_case(clean)
    records = prepare_annotation_records(select_golden_candidates([case]), [case])
    path = tmp_path / "blind.json"
    mapping = export_blinded_assignments(
        records, {case.post_id: source}, path, expected_revision=source.revision
    )
    data = json.loads(path.read_text())
    (assignment_id,) = mapping
    assert UUID(assignment_id).version == 4 and mapping[assignment_id] == case.post_id
    assert data == {
        "assignments": [{"assignment_id": assignment_id, "narrative": narrative}]
    }
    for secret in (case.post_id, "NTA", "0.8", "Private source comment", case.title):
        assert secret not in path.read_text()


def test_blinded_export_refuses_unblinded_narrative_before_writing(tmp_path):
    source = make_source()
    source.row.text += " Final verdict was NTA."
    _, clean = source.resolve(source.revision)
    case = label_cleaned_case(clean)
    records = prepare_annotation_records(select_golden_candidates([case]), [case])
    path = tmp_path / "blind.json"
    with pytest.raises(ValueError, match="identifying or verdict"):
        export_blinded_assignments(
            records, {case.post_id: source}, path, expected_revision=source.revision
        )
    assert not path.exists()


def test_blind_join_rejects_changed_source_text_with_same_id_and_votes(tmp_path):
    source = make_source()
    _, clean = source.resolve(source.revision)
    case = label_cleaned_case(clean)
    records = prepare_annotation_records(select_golden_candidates([case]), [case])
    source.row.text += " A different ending."
    with pytest.raises(ValueError, match="narrative or statistics"):
        export_blinded_assignments(
            records,
            {case.post_id: source},
            tmp_path / "blind.json",
            expected_revision=source.revision,
        )
