"""Tests for EVAL-02 review, agreement and progress tracking (synthetic data only)."""

import json

import pytest
from pydantic import ValidationError

from asshole_ai_detector.data.aita_adapter import row_to_normalized
from asshole_ai_detector.evaluation.golden_annotation_prep import (
    AnnotationSource,
    validate_completed_annotation,
)
from asshole_ai_detector.evaluation.golden_annotation_review import (
    BatchPlan,
    FinalDecision,
    ReviewerFile,
    ReviewerSubmission,
    build_golden_record,
    build_progress_report,
    cohens_kappa,
    compute_agreement,
    export_review_template,
    final_category,
    load_locked_submissions,
    lock_submissions,
    pair_submissions,
)
from asshole_ai_detector.schemas.annotation import GoldenAnnotation

REVISION = "fixture-revision"
PLAN = BatchPlan(
    "week8-batch1",
    {"a1": ("ana", "sara"), "a2": ("ana", "sara"), "a3": ("sara", "dan")},
)


def submission(assignment_id="a1", reviewer="ana", **overrides):
    data = {
        "assignment_id": assignment_id,
        "reviewer": reviewer,
        "verdict": "NTA",
        "key_facts": ["The narrator declined to clean after a party."],
        "missing_information": [],
        "alternative_perspectives": ["The roommate expected shared chores."],
        "framing_bias": "possible",
        "expected_uncertainty_tier": "confident",
        "expected_initial_action": "proceed",
        "notes": "Clear refusal with a stated reason.",
    }
    data.update(overrides)
    return ReviewerSubmission.model_validate(data)


def make_source(post_id="abc123", votes=None):
    row = {
        "submission_title": "AITA for refusing to clean?",
        "submission_text": " ".join(["story"] * 60),
        "submission_url": f"https://www.reddit.com/r/AmItheAsshole/comments/{post_id}/x/",
    }
    for rank, verdict in enumerate(votes or ["NTA"] * 8 + ["INFO", "YTA"], 1):
        row[f"top_comment_{rank}"] = f"comment {rank}"
        row[f"top_comment_{rank}_classification"] = verdict
    return AnnotationSource(row_to_normalized(row), REVISION, "source_text")


def decision(**overrides):
    data = {
        "reference_verdict": "NTA",
        "case_type": "consensus",
        "key_facts": ["The narrator declined to clean after a party."],
        "missing_information": [],
        "alternative_perspectives": ["The roommate expected shared chores."],
        "framing_bias": "possible",
        "expected_uncertainty_tier": "confident",
        "expected_initial_action": "proceed",
        "annotation_notes": "Both reviewers agreed.",
    }
    data.update(overrides)
    return FinalDecision(**data)


def write_reviewer_file(path, reviewer, submissions, batch_id=PLAN.batch_id):
    data = {
        "batch_id": batch_id,
        "reviewer": reviewer,
        "submissions": [s.model_dump() for s in submissions],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# Submissions -------------------------------------------------------------------


def test_submission_accepts_undetermined():
    assert submission(verdict="UNDETERMINED").verdict == "UNDETERMINED"


@pytest.mark.parametrize(
    "overrides",
    [
        {"verdict": "INFO"},
        {"verdict": ""},
        {"key_facts": []},
        {"key_facts": ["  "]},
        {"framing_bias": "maybe"},
        {"expected_uncertainty_tier": ""},
        {"notes": ""},
        {"extra_field": "x"},
    ],
)
def test_incomplete_or_invalid_submissions_are_rejected(overrides):
    with pytest.raises(ValidationError):
        submission(**overrides)


def test_reviewer_file_rejects_mixed_reviewers_and_repeats():
    with pytest.raises(ValidationError):
        ReviewerFile(
            batch_id="b", reviewer="ana", submissions=[submission(reviewer="dan")]
        )
    with pytest.raises(ValidationError):
        ReviewerFile(batch_id="b", reviewer="ana", submissions=[submission()] * 2)


# Plan and template ---------------------------------------------------------------


@pytest.mark.parametrize(
    "reviewers", [{}, {"a1": ("ana", "ana")}, {"a1": ("ana",)}, {"a1": ("ana", " ")}]
)
def test_plan_requires_two_different_reviewers(reviewers):
    with pytest.raises(ValueError):
        BatchPlan("b", reviewers)


def test_plan_round_trips_and_lists_assignments():
    assert BatchPlan.from_dict(PLAN.to_dict()) == PLAN
    assert PLAN.assignments_for("sara") == ["a1", "a2", "a3"]
    assert PLAN.assignments_for("dan") == ["a3"]


def test_template_lists_only_the_reviewers_assignments(tmp_path):
    path = tmp_path / "dan.json"
    export_review_template(PLAN, "dan", path)
    data = json.loads(path.read_text())

    assert data["reviewer"] == "dan"
    assert [s["assignment_id"] for s in data["submissions"]] == ["a3"]
    # An unfilled template must not pass validation.
    with pytest.raises(ValidationError):
        ReviewerFile.model_validate(data)
    with pytest.raises(FileExistsError):
        export_review_template(PLAN, "dan", path)
    with pytest.raises(ValueError):
        export_review_template(PLAN, "nobody", tmp_path / "x.json")


# Locking -------------------------------------------------------------------------


def test_locked_files_load_and_changes_are_refused(tmp_path):
    ana = write_reviewer_file(
        tmp_path / "ana.json", "ana", [submission("a1"), submission("a2")]
    )
    lock = tmp_path / "lock.json"
    lock_submissions([ana], lock)

    assert len(load_locked_submissions([ana], lock, PLAN)) == 2

    write_reviewer_file(ana, "ana", [submission("a1", verdict="YTA"), submission("a2")])
    with pytest.raises(ValueError, match="changed"):
        load_locked_submissions([ana], lock, PLAN)


def test_unlocked_or_foreign_files_are_refused(tmp_path):
    lock = tmp_path / "lock.json"
    ana = write_reviewer_file(tmp_path / "ana.json", "ana", [submission("a1")])
    lock_submissions([ana], lock)
    dan = write_reviewer_file(tmp_path / "dan.json", "dan", [submission("a3", "dan")])
    with pytest.raises(ValueError, match="not locked"):
        load_locked_submissions([ana, dan], lock, PLAN)

    other = write_reviewer_file(tmp_path / "o.json", "ana", [submission()], "other")
    lock_submissions([other], tmp_path / "lock2.json")
    with pytest.raises(ValueError, match="another batch"):
        load_locked_submissions([other], tmp_path / "lock2.json", PLAN)


# Pairing -------------------------------------------------------------------------


def test_pairs_follow_plan_order_and_skip_incomplete():
    subs = [submission("a1", "sara"), submission("a1", "ana"), submission("a3", "sara")]
    pairs = pair_submissions(subs, PLAN)

    assert list(pairs) == ["a1"]
    assert [s.reviewer for s in pairs["a1"]] == ["ana", "sara"]


@pytest.mark.parametrize(
    "subs",
    [
        [submission("a9")],
        [submission("a3", "ana")],
        [submission("a1"), submission("a1")],
    ],
)
def test_pairing_rejects_unplanned_or_duplicate_answers(subs):
    with pytest.raises(ValueError):
        pair_submissions(subs, PLAN)


# Agreement ------------------------------------------------------------------------


def test_cohens_kappa_known_values():
    assert cohens_kappa(["NTA", "YTA"], ["NTA", "YTA"]) == (1.0, None)
    # Observed 0.5, expected 0.5 -> 0.
    assert (
        cohens_kappa(["NTA", "NTA", "YTA", "YTA"], ["NTA", "YTA", "NTA", "YTA"])[0] == 0
    )
    # Textbook example: po = 0.7, pe = 0.5 -> 0.4.
    first = ["Y"] * 20 + ["Y"] * 5 + ["N"] * 10 + ["N"] * 15
    second = ["Y"] * 20 + ["N"] * 5 + ["Y"] * 10 + ["N"] * 15
    assert cohens_kappa(first, second)[0] == pytest.approx(0.4)


def test_cohens_kappa_undefined_cases():
    assert cohens_kappa([], []) == (None, "no complete pairs")
    kappa, note = cohens_kappa(["NTA"] * 3, ["NTA"] * 3)
    assert kappa is None and "undefined" in note
    with pytest.raises(ValueError):
        cohens_kappa(["NTA"], [])


def test_agreement_counts_undetermined_and_lists_disagreements():
    pairs = {
        "a1": (submission("a1", "ana"), submission("a1", "sara")),
        "a2": (
            submission("a2", "ana", verdict="UNDETERMINED"),
            submission("a2", "sara", verdict="YTA", framing_bias="clear"),
        ),
    }
    report = compute_agreement(pairs)

    assert report.n_pairs == 2
    assert report.percent_agreement == 0.5
    assert report.field_agreement["framing_bias"] == 0.5
    assert report.field_agreement["expected_initial_action"] == 1.0
    assert [(d.assignment_id, d.fields) for d in report.disagreements] == [
        ("a2", ("verdict", "framing_bias"))
    ]


# Final records ----------------------------------------------------------------------


def test_final_record_keeps_original_verdicts_and_source_votes():
    pair = (submission("a1", "ana"), submission("a1", "sara", verdict="ESH"))
    source = make_source()
    record = build_golden_record(
        "abc123", pair, decision(), source, expected_revision=REVISION
    )

    assert record["annotator_verdicts"] == ["NTA", "ESH"]
    assert record["community_votes"] == {
        "NTA": 8,
        "YTA": 1,
        "ESH": 0,
        "NAH": 0,
        "INFO": 1,
    }
    assert record["community_agreement"] == pytest.approx(8 / 9)
    annotation = validate_completed_annotation(
        record,
        expected_case_id="abc123",
        expected_category="NTA",
        source=source,
        expected_revision=REVISION,
    )
    assert final_category(annotation) == "NTA"


def test_final_record_rejects_wrong_source_or_pair():
    pair = (submission("a1", "ana"), submission("a1", "sara"))
    with pytest.raises(ValueError, match="Source"):
        build_golden_record(
            "other", pair, decision(), make_source(), expected_revision=REVISION
        )
    with pytest.raises(ValueError, match="pair"):
        build_golden_record(
            "abc123",
            (submission("a1", "ana"), submission("a2", "sara")),
            decision(),
            make_source(),
            expected_revision=REVISION,
        )


def test_missing_context_record_counts_once():
    record = build_golden_record(
        "abc123",
        (submission("a1", "ana"), submission("a1", "sara")),
        decision(
            reference_verdict=None,
            case_type="missing_context",
            missing_information=["Whether chores were agreed beforehand."],
            expected_uncertainty_tier="withhold",
            expected_initial_action="request_clarification",
        ),
        make_source(),
        expected_revision=REVISION,
    )
    assert final_category(GoldenAnnotation.model_validate(record)) == "missing_context"


# Progress ----------------------------------------------------------------------------


def completed(case_id, **overrides):
    record = build_golden_record(
        case_id,
        (submission("a1", "ana"), submission("a1", "sara")),
        decision(**overrides),
        make_source(case_id),
        expected_revision=REVISION,
    )
    return GoldenAnnotation.model_validate(record)


def test_progress_report():
    subs = [
        submission("a1", "ana"),
        submission("a1", "sara"),
        submission("a2", "ana"),
        submission("a2", "sara", verdict="YTA"),
        submission("a3", "sara"),
    ]
    mapping = {"a1": "case1", "a2": "case2", "a3": "case3"}
    report = build_progress_report(PLAN, subs, [completed("case1")], mapping)

    batch = report["batch"]
    assert batch["reviews_expected"] == 6
    assert batch["reviews_received"] == 5
    assert batch["pairs_complete"] == 2
    assert batch["finalized"] == 1
    assert batch["awaiting_adjudication"] == ["a2"]
    assert batch["missing_reviews"] == [{"assignment_id": "a3", "reviewer": "dan"}]
    assert batch["agreement"]["verdict_percent_agreement"] == 0.5

    golden = report["golden_dataset"]
    assert golden["completed"] == 1
    assert golden["categories"]["NTA"] == {
        "completed": 1,
        "target": 40,
        "remaining": 39,
    }
    assert golden["categories"]["missing_context"]["remaining"] == 20
    # Coordinator report names assignments, never source case IDs.
    assert "case2" not in json.dumps(report)


def test_progress_report_rejects_double_counting_and_bad_maps():
    mapping = {"a1": "case1", "a2": "case2", "a3": "case3"}
    with pytest.raises(ValueError, match="more than once"):
        build_progress_report(PLAN, [], [completed("case1")] * 2, mapping)
    with pytest.raises(ValueError, match="different assignments"):
        build_progress_report(PLAN, [], [], {"a1": "case1"})
