"""Prepare Golden Dataset candidates for independent human review.

Review records are not final GoldenAnnotation objects.
Only the blinded reviewer export contains approved narrative text.
"""

import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isclose
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from asshole_ai_detector.data.aita_adapter import MAX_COMMENTS
from asshole_ai_detector.data.cleaning import (
    CleanedAITACase,
    RejectedAITACase,
    clean_case,
)
from asshole_ai_detector.evaluation.golden_selection import (
    GoldenCategory,
    GoldenSelectionResult,
    validate_source_slots,
    write_golden_json,
)
from asshole_ai_detector.schemas.annotation import GoldenAnnotation
from asshole_ai_detector.schemas.case import VERDICTS, AITACase, NormalizedAITACase

type ReviewStatus = Literal["pending_human_review"]


@dataclass(frozen=True)
class GoldenReviewRecord:
    """Metadata-only review record for a selected candidate."""

    case_id: str
    proposed_category: GoldenCategory
    review_status: ReviewStatus
    community_agreement_all_valid: float
    n_valid_moral_comments: int
    top_moral_comment_ranks: tuple[int, ...]
    requires_source_comment_review: bool
    requires_missing_context_review: bool
    cleaned_text_digest: str


def prepare_annotation_records(
    result: GoldenSelectionResult,
    cases: list[AITACase],
) -> list[GoldenReviewRecord]:
    """Match selected candidates to their original validated cases.

    This does not compute final Golden community votes.
    Those require pinned source evidence including INFO.
    """
    cases_by_id: dict[str, AITACase] = {}

    for case in cases:
        if case.post_id in cases_by_id:
            raise ValueError(f"Duplicate source case ID: {case.post_id}")
        cases_by_id[case.post_id] = case

    records: list[GoldenReviewRecord] = []
    selected_ids: set[str] = set()

    for candidate in result.selected:
        if candidate.case_id in selected_ids:
            raise ValueError(f"Duplicate selected candidate: {candidate.case_id}")
        selected_ids.add(candidate.case_id)

        case = cases_by_id.get(candidate.case_id)
        if case is None:
            raise ValueError(f"Selected case not found: {candidate.case_id}")

        candidate.validate()
        validate_source_slots(case)
        fields = case.derived_fields
        source_category = (
            "disputed"
            if fields.consensus_label == "DISPUTED"
            else fields.consensus_label
        )
        if candidate.category not in (source_category, "missing_context"):
            raise ValueError(
                f"Candidate category disagrees with source: {candidate.case_id}"
            )
        agreement = fields.verdict_distribution[fields.top_verdict]

        if (
            not isclose(
                candidate.community_agreement, agreement, rel_tol=0, abs_tol=1e-6
            )
            or candidate.n_valid_comments != fields.n_valid
        ):
            raise ValueError(f"Candidate statistics mismatch: {candidate.case_id}")

        # Only moral comments are present in AITACase.
        # Missing INFO comments must be recovered from the
        # original source before computing final Golden votes.
        ranked_comments = sorted(
            case.valid_comments,
            key=lambda comment: comment.rank,
        )

        records.append(
            GoldenReviewRecord(
                case_id=candidate.case_id,
                proposed_category=candidate.category,
                review_status="pending_human_review",
                cleaned_text_digest=sha256(case.text.encode()).hexdigest(),
                community_agreement_all_valid=agreement,
                n_valid_moral_comments=fields.n_valid,
                top_moral_comment_ranks=tuple(
                    comment.rank for comment in ranked_comments
                ),
                requires_source_comment_review=True,
                requires_missing_context_review=(
                    candidate.category == "missing_context"
                ),
            )
        )

    return records


def export_annotation_review_manifest(
    records: list[GoldenReviewRecord],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Export metadata-only review records as JSON."""
    output_path = Path(output_path)

    ids = [record.case_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in review records")

    data = {
        "schema_version": "1.0",
        "status": "pending_human_review",
        "audience": "coordinator_only",
        "source_comment_convention": "original_rank_slots_1_to_10",
        "records": [asdict(record) for record in records],
    }

    write_golden_json(data, output_path, overwrite=overwrite)


@dataclass(frozen=True)
class AnnotationSource:
    """Coordinator-supplied DATA-01 row from a pinned revision."""

    row: NormalizedAITACase
    revision: str
    representation: Literal["source_text", "cleaned_text"]

    def resolve(self, expected_revision: str) -> tuple[str, CleanedAITACase]:
        if (
            not isinstance(expected_revision, str)
            or not expected_revision.strip()
            or self.revision != expected_revision
        ):
            raise ValueError("Missing or mismatched source revision")
        if type(self.row) is not NormalizedAITACase:
            raise TypeError(
                "Source evidence must be the unfiltered DATA-01 normalized row"
            )
        ranks = [comment.rank for comment in self.row.comments]
        if len(ranks) != len(set(ranks)) or any(
            not 1 <= rank <= MAX_COMMENTS for rank in ranks
        ):
            raise ValueError("Source must contain unique original top-ten ranks")
        cleaned = clean_case(self.row)
        match cleaned:
            case RejectedAITACase(reasons=reasons):
                raise ValueError(f"Source case rejected by DATA-02: {reasons}")
        if self.representation == "cleaned_text":
            return cleaned.text, cleaned
        if self.representation == "source_text" and self.row.text:
            return self.row.text.strip(), cleaned
        raise ValueError("Unrecognized narrative representation")


def _source_category(cleaned: CleanedAITACase) -> GoldenCategory:
    votes = cleaned.community_votes
    winner = max(VERDICTS, key=lambda verdict: getattr(votes, verdict))
    return winner if getattr(votes, winner) / votes.moral_count >= 0.70 else "disputed"


def export_blinded_assignments(
    records: list[GoldenReviewRecord],
    sources: dict[str, AnnotationSource],
    output_path: Path,
    *,
    expected_revision: str,
    overwrite: bool = False,
) -> dict[str, str]:
    """Export approved narratives. Return the coordinator's private assignment-to-case map."""
    rows = []
    assignments = {}
    seen = set()
    for record in records:
        key = record.case_id
        if key in seen or key not in sources:
            raise ValueError("Missing source or duplicate assignment for case")
        seen.add(key)
        narrative, cleaned = sources[key].resolve(expected_revision)
        if cleaned.post_id != key or record.proposed_category not in (
            _source_category(cleaned),
            "missing_context",
        ):
            raise ValueError("Source ID or category mismatch")
        if (
            record.cleaned_text_digest != sha256(cleaned.text.encode()).hexdigest()
            or record.n_valid_moral_comments != cleaned.n_valid
            or tuple(comment.rank for comment in cleaned.comments)
            != record.top_moral_comment_ranks
            or not isclose(
                record.community_agreement_all_valid,
                cleaned.community_votes.winning_share,
                rel_tol=0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                "Review narrative or statistics disagree with pinned source"
            )
        if re.search(
            rf"(?<!\w){re.escape(key)}(?!\w)|\b(?:NTA|YTA|ESH|NAH)\b|reddit\.com/",
            narrative,
            re.IGNORECASE,
        ):
            raise ValueError(
                "Narrative needs coordinator review for identifying or verdict text"
            )
        assignment_id = uuid4().hex
        assignments[assignment_id] = key
        rows.append({"assignment_id": assignment_id, "narrative": narrative})
    rows.sort(key=lambda row: row["assignment_id"])
    write_golden_json({"assignments": rows}, output_path, overwrite=overwrite)
    return assignments


@dataclass(frozen=True)
class Adjudication:
    case_id: str
    proposed_category: GoldenCategory
    final_category: GoldenCategory
    rationale: str


def _has_evidence(text: str) -> bool:
    words = re.findall(r"\w+", text.casefold())
    return bool(words) and " ".join(words) not in {
        "n a",
        "na",
        "none",
        "unknown",
        "tbd",
        "unspecified",
        "not applicable",
    }


def validate_completed_annotation(
    record: dict[str, Any],
    *,
    expected_case_id: str | None = None,
    expected_category: GoldenCategory | None = None,
    source: AnnotationSource | None = None,
    expected_revision: str | None = None,
    adjudication: Adjudication | None = None,
) -> GoldenAnnotation:
    """Accept a completed record only after reconciling it to its proposal and pinned source."""
    annotation = GoldenAnnotation.model_validate(record)
    if expected_case_id is not None and annotation.case_id != expected_case_id:
        raise ValueError("Completed annotation case ID does not match candidate")
    final_category = (
        annotation.reference_verdict
        if annotation.case_type == "consensus"
        else annotation.case_type
    )
    if annotation.case_type == "missing_context" and not any(
        _has_evidence(fact) for fact in annotation.missing_information
    ):
        raise ValueError(
            "Missing-context acceptance requires essential missing information"
        )
    if adjudication is not None:
        if (
            adjudication.case_id != annotation.case_id
            or adjudication.proposed_category != expected_category
            or adjudication.final_category != final_category
            or not _has_evidence(adjudication.rationale)
        ):
            raise ValueError("Invalid adjudication decision or missing rationale")
    elif expected_category is not None and expected_category != final_category:
        raise ValueError(
            "Completed annotation verdict/category change requires adjudication"
        )
    if (
        source is None
        or expected_revision is None
        or expected_case_id is None
        or expected_category is None
    ):
        raise ValueError(
            "Acceptance requires pinned source evidence, revision, candidate ID and proposal"
        )
    narrative, cleaned = source.resolve(expected_revision)
    if cleaned.post_id != annotation.case_id or annotation.narrative != narrative:
        raise ValueError("Source case ID or narrative mismatch")
    if annotation.community_votes != cleaned.community_votes:
        raise ValueError("Community votes disagree with pinned source")
    if expected_category not in (_source_category(cleaned), "missing_context"):
        raise ValueError("Proposed category disagrees with pinned source")
    return annotation
