"""Independent review, agreement and progress tracking for Golden annotation (EVAL-02).

Workflow for one batch (see docs/evaluation/golden_dataset.md, "Annotation batches"):

1. The coordinator issues blinded assignments (``export_blinded_assignments``) and
   writes a ``BatchPlan`` naming the two reviewers of each assignment, in a fixed
   order.
2. ``export_review_template`` gives each reviewer an empty submission file.
   Reviewers fill it in privately.
3. ``lock_submissions`` records a hash of every returned file *before* anyone
   compares answers. ``load_locked_submissions`` refuses files that changed later,
   so the original independent answers stay available for agreement analysis.
4. ``compute_agreement`` gives percentage agreement and Cohen's kappa on the
   original verdicts, before any disagreement is resolved.
5. After adjudication, ``build_golden_record`` assembles the final record. The two
   annotator verdicts always come from the locked submissions, never from the
   final decision. Accept it with ``validate_completed_annotation``.
6. ``build_progress_report`` summarizes completed cases, category counts and
   outstanding work.

Nothing here judges a story: every verdict and field comes from a human reviewer.
"""

import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from asshole_ai_detector.evaluation.golden_annotation_prep import AnnotationSource
from asshole_ai_detector.evaluation.golden_selection import (
    TARGET_COUNTS,
    GoldenCategory,
    write_golden_json,
)
from asshole_ai_detector.schemas.annotation import (
    AnnotatorVerdict,
    ContractModel,
    GoldenAnnotation,
    UncertaintyStatus,
    Verdict,
)

type FramingBias = Literal["none", "possible", "clear"]
type InitialAction = Literal["proceed", "request_clarification"]
type CaseType = Literal["consensus", "disputed", "missing_context"]

# Categorical fields compared between the two reviewers of a case.
COMPARED_FIELDS: tuple[str, ...] = (
    "verdict",
    "framing_bias",
    "expected_uncertainty_tier",
    "expected_initial_action",
)


def _clean_items(items: list[str]) -> list[str]:
    if any(not item.strip() for item in items):
        raise ValueError("List entries must not be blank")
    return [item.strip() for item in items]


# Reviewer submissions --------------------------------------------------------


class ReviewerSubmission(ContractModel):
    """One reviewer's independent answer for one blinded assignment."""

    assignment_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    verdict: AnnotatorVerdict
    key_facts: list[str] = Field(min_length=1)
    missing_information: list[str] = Field(default_factory=list)
    alternative_perspectives: list[str] = Field(default_factory=list)
    framing_bias: FramingBias
    expected_uncertainty_tier: UncertaintyStatus
    expected_initial_action: InitialAction
    notes: str = Field(min_length=1)

    @field_validator("key_facts", "missing_information", "alternative_perspectives")
    @classmethod
    def check_items(cls, items: list[str]) -> list[str]:
        return _clean_items(items)


class ReviewerFile(ContractModel):
    """The file a reviewer returns: their answers for one batch."""

    batch_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    submissions: list[ReviewerSubmission]

    @model_validator(mode="after")
    def check_reviewer_and_ids(self) -> Self:
        ids = [submission.assignment_id for submission in self.submissions]
        if len(ids) != len(set(ids)):
            raise ValueError("A reviewer file lists the same assignment twice")
        if any(submission.reviewer != self.reviewer for submission in self.submissions):
            raise ValueError("Every submission must name the file's reviewer")
        return self


# Batch plan ------------------------------------------------------------------


@dataclass(frozen=True)
class BatchPlan:
    """Which two reviewers answer each blinded assignment, in a fixed order."""

    batch_id: str
    reviewers: dict[str, tuple[str, str]]

    def __post_init__(self) -> None:
        if not self.batch_id.strip():
            raise ValueError("Batch ID must not be empty")
        if not self.reviewers:
            raise ValueError("A batch needs at least one assignment")
        for assignment_id, pair in self.reviewers.items():
            if len(pair) != 2 or not all(name.strip() for name in pair):
                raise ValueError(f"{assignment_id}: exactly two reviewers required")
            if pair[0] == pair[1]:
                raise ValueError(f"{assignment_id}: reviewers must be different people")

    def assignments_for(self, reviewer: str) -> list[str]:
        return sorted(a for a, pair in self.reviewers.items() if reviewer in pair)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "reviewers": {a: list(pair) for a, pair in sorted(self.reviewers.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            batch_id=data["batch_id"],
            reviewers={a: tuple(pair) for a, pair in data["reviewers"].items()},
        )


def export_review_template(
    plan: BatchPlan, reviewer: str, output_path: Path, *, overwrite: bool = False
) -> None:
    """Write an empty submission file for one reviewer's assignments.

    The file has no narratives, IDs of source posts or votes; reviewers read the
    narratives from the blinded assignment export.
    """
    assignments = plan.assignments_for(reviewer)
    if not assignments:
        raise ValueError(f"{reviewer} has no assignments in this batch")
    data = {
        "batch_id": plan.batch_id,
        "reviewer": reviewer,
        "submissions": [
            {
                "assignment_id": assignment_id,
                "reviewer": reviewer,
                "verdict": "",
                "key_facts": [],
                "missing_information": [],
                "alternative_perspectives": [],
                "framing_bias": "",
                "expected_uncertainty_tier": "",
                "expected_initial_action": "",
                "notes": "",
            }
            for assignment_id in assignments
        ],
    }
    write_golden_json(data, output_path, overwrite=overwrite)


# Locking and loading ---------------------------------------------------------


def _digest(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def lock_submissions(
    paths: list[Path], lock_path: Path, *, overwrite: bool = False
) -> dict[str, str]:
    """Record a hash of every returned reviewer file before comparing answers."""
    digests = {Path(path).name: _digest(path) for path in paths}
    if len(digests) != len(paths):
        raise ValueError("Reviewer files must have distinct file names")
    write_golden_json({"files": digests}, lock_path, overwrite=overwrite)
    return digests


def load_locked_submissions(
    paths: list[Path], lock_path: Path, plan: BatchPlan
) -> list[ReviewerSubmission]:
    """Load reviewer files, refusing any file that changed after locking."""
    locked = json.loads(Path(lock_path).read_text(encoding="utf-8"))["files"]
    submissions: list[ReviewerSubmission] = []
    for path in paths:
        name = Path(path).name
        if name not in locked:
            raise ValueError(f"{name} was not locked")
        if locked[name] != _digest(path):
            raise ValueError(f"{name} changed after it was locked")
        reviewer_file = ReviewerFile.model_validate_json(Path(path).read_text("utf-8"))
        if reviewer_file.batch_id != plan.batch_id:
            raise ValueError(f"{name} belongs to another batch")
        submissions.extend(reviewer_file.submissions)
    _check_against_plan(submissions, plan)
    return submissions


def _check_against_plan(submissions: list[ReviewerSubmission], plan: BatchPlan) -> None:
    seen: set[tuple[str, str]] = set()
    for submission in submissions:
        key = (submission.assignment_id, submission.reviewer)
        if key in seen:
            raise ValueError(f"Duplicate submission: {key}")
        seen.add(key)
        pair = plan.reviewers.get(submission.assignment_id)
        if pair is None:
            raise ValueError(f"Unknown assignment: {submission.assignment_id}")
        if submission.reviewer not in pair:
            raise ValueError(
                f"{submission.reviewer} is not assigned to {submission.assignment_id}"
            )


def pair_submissions(
    submissions: list[ReviewerSubmission], plan: BatchPlan
) -> dict[str, tuple[ReviewerSubmission, ReviewerSubmission]]:
    """Pair the two answers of each assignment in the plan's reviewer order.

    Assignments with fewer than two answers are left out (they are outstanding).
    """
    _check_against_plan(submissions, plan)
    by_key = {(s.assignment_id, s.reviewer): s for s in submissions}
    pairs = {}
    for assignment_id, (first, second) in sorted(plan.reviewers.items()):
        a, b = by_key.get((assignment_id, first)), by_key.get((assignment_id, second))
        if a is not None and b is not None:
            pairs[assignment_id] = (a, b)
    return pairs


# Agreement -------------------------------------------------------------------


@dataclass(frozen=True)
class Disagreement:
    assignment_id: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class AgreementReport:
    """Agreement on the original, unresolved answers."""

    n_pairs: int
    percent_agreement: float | None
    cohens_kappa: float | None
    kappa_note: str | None
    field_agreement: dict[str, float]
    disagreements: tuple[Disagreement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "verdict_percent_agreement": self.percent_agreement,
            "verdict_cohens_kappa": self.cohens_kappa,
            "kappa_note": self.kappa_note,
            "field_agreement": self.field_agreement,
            "disagreements": [
                {"assignment_id": d.assignment_id, "fields": list(d.fields)}
                for d in self.disagreements
            ],
        }


def cohens_kappa(
    first: list[str], second: list[str]
) -> tuple[float | None, str | None]:
    """Cohen's kappa for two raters. Returns (kappa, note); kappa is None if undefined."""
    if len(first) != len(second):
        raise ValueError("Both raters need the same number of labels")
    n = len(first)
    if n == 0:
        return None, "no complete pairs"
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    first_counts, second_counts = Counter(first), Counter(second)
    expected = sum(
        first_counts[label] * second_counts[label] for label in first_counts
    ) / (n * n)
    if expected == 1:
        return None, "undefined: both reviewers used a single, identical label"
    return (observed - expected) / (1 - expected), None


def compute_agreement(
    pairs: dict[str, tuple[ReviewerSubmission, ReviewerSubmission]],
) -> AgreementReport:
    """Percentage agreement and Cohen's kappa on verdicts, plus field-level agreement.

    Verdicts include UNDETERMINED as its own label.
    """
    ordered = sorted(pairs.items())
    first = [a.verdict for _, (a, _) in ordered]
    second = [b.verdict for _, (_, b) in ordered]
    kappa, note = cohens_kappa(first, second)
    n = len(ordered)

    disagreements = []
    field_matches: Counter[str] = Counter()
    for assignment_id, (a, b) in ordered:
        differing = []
        for name in COMPARED_FIELDS:
            if getattr(a, name) == getattr(b, name):
                field_matches[name] += 1
            else:
                differing.append(name)
        if differing:
            disagreements.append(Disagreement(assignment_id, tuple(differing)))

    return AgreementReport(
        n_pairs=n,
        percent_agreement=field_matches["verdict"] / n if n else None,
        cohens_kappa=kappa,
        kappa_note=note,
        field_agreement={name: field_matches[name] / n for name in COMPARED_FIELDS}
        if n
        else {},
        disagreements=tuple(disagreements),
    )


# Final records ---------------------------------------------------------------


@dataclass(frozen=True)
class FinalDecision:
    """The agreed (or adjudicated) content of a final Golden record."""

    reference_verdict: Verdict | None
    case_type: CaseType
    key_facts: list[str]
    missing_information: list[str]
    alternative_perspectives: list[str]
    framing_bias: FramingBias
    expected_uncertainty_tier: UncertaintyStatus
    expected_initial_action: InitialAction
    annotation_notes: str


def build_golden_record(
    case_id: str,
    pair: tuple[ReviewerSubmission, ReviewerSubmission],
    decision: FinalDecision,
    source: AnnotationSource,
    *,
    expected_revision: str,
) -> dict[str, Any]:
    """Assemble a final record for ``validate_completed_annotation``.

    Narrative and community votes (including INFO) come from the pinned source,
    and ``annotator_verdicts`` from the two locked submissions, so neither can be
    edited by hand.
    """
    first, second = pair
    if first.assignment_id != second.assignment_id or first.reviewer == second.reviewer:
        raise ValueError("A pair needs two reviewers of the same assignment")
    narrative, cleaned = source.resolve(expected_revision)
    if cleaned.post_id != case_id:
        raise ValueError("Source does not belong to this case")
    votes = cleaned.community_votes
    record = {
        "case_id": case_id,
        "narrative": narrative,
        "community_votes": votes.model_dump(),
        "community_agreement": votes.winning_share,
        "reference_verdict": decision.reference_verdict,
        "case_type": decision.case_type,
        "key_facts": decision.key_facts,
        "missing_information": decision.missing_information,
        "alternative_perspectives": decision.alternative_perspectives,
        "framing_bias": decision.framing_bias,
        "annotator_verdicts": [first.verdict, second.verdict],
        "annotation_notes": decision.annotation_notes,
        "expected_uncertainty_tier": decision.expected_uncertainty_tier,
        "expected_initial_action": decision.expected_initial_action,
    }
    # Early structural check; acceptance still uses validate_completed_annotation.
    GoldenAnnotation.model_validate(record)
    return record


def final_category(annotation: GoldenAnnotation) -> GoldenCategory:
    """The single category a final record counts towards."""
    if annotation.case_type == "consensus":
        return annotation.reference_verdict
    return annotation.case_type


# Progress --------------------------------------------------------------------


def build_progress_report(
    plan: BatchPlan,
    submissions: list[ReviewerSubmission],
    completed: list[GoldenAnnotation],
    assignment_to_case: dict[str, str],
) -> dict[str, Any]:
    """Coordinator-only summary of one batch and of the whole Golden Dataset.

    ``completed`` holds every accepted record so far (all batches). Each case is
    counted once, under one final category, so missing-context cases are never
    double-counted. The report lists assignment IDs and counts, not case IDs.
    """
    if set(assignment_to_case) != set(plan.reviewers):
        raise ValueError("Assignment map and batch plan list different assignments")
    case_ids = [annotation.case_id for annotation in completed]
    duplicates = sorted(c for c, n in Counter(case_ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"Cases finalized more than once: {len(duplicates)}")

    pairs = pair_submissions(submissions, plan)
    received = {(s.assignment_id, s.reviewer) for s in submissions}
    finalized = set(case_ids)
    missing_reviews = [
        {"assignment_id": a, "reviewer": reviewer}
        for a, pair in sorted(plan.reviewers.items())
        for reviewer in pair
        if (a, reviewer) not in received
    ]
    awaiting_adjudication = sorted(
        a for a in pairs if assignment_to_case[a] not in finalized
    )
    batch_finalized = sorted(
        a for a, case_id in assignment_to_case.items() if case_id in finalized
    )

    category_counts = Counter(final_category(annotation) for annotation in completed)
    categories = {
        category: {
            "completed": category_counts[category],
            "target": target,
            "remaining": max(target - category_counts[category], 0),
        }
        for category, target in TARGET_COUNTS.items()
    }
    return {
        "audience": "coordinator_only",
        "batch_id": plan.batch_id,
        "batch": {
            "assignments": len(plan.reviewers),
            "reviews_expected": 2 * len(plan.reviewers),
            "reviews_received": len(received),
            "pairs_complete": len(pairs),
            "finalized": len(batch_finalized),
            "awaiting_adjudication": awaiting_adjudication,
            "missing_reviews": missing_reviews,
            "agreement": compute_agreement(pairs).to_dict(),
        },
        "golden_dataset": {
            "completed": len(completed),
            "target": sum(TARGET_COUNTS.values()),
            "categories": categories,
        },
    }
