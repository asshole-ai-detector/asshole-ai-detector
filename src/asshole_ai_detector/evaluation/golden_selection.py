"""Deterministic candidate selection for the held-out AITA Golden Dataset.

This module does not create final human annotations or perform dataset splitting.
The caller must supply a deduplicated, held-out pool from DATA-04.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from math import ceil, isclose
from pathlib import Path
from random import Random
from types import MappingProxyType
from typing import Literal

from asshole_ai_detector.data.aita_adapter import MAX_COMMENTS
from asshole_ai_detector.data.splits import SplitResult
from asshole_ai_detector.schemas.case import VERDICTS, AITACase

type GoldenCategory = Literal["NTA", "YTA", "ESH", "NAH", "disputed", "missing_context"]

TARGET_COUNTS = MappingProxyType(
    {
        "NTA": 40,
        "YTA": 40,
        "ESH": 40,
        "NAH": 40,
        "disputed": 20,
        "missing_context": 20,
    }
)


@dataclass(frozen=True)
class GoldenCandidate:
    """A candidate awaiting independent human annotation."""

    case_id: str
    category: GoldenCategory
    community_agreement: float
    n_valid_comments: int
    requires_manual_review: bool = True

    def validate(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("Candidate requires a nonempty case ID")
        if not isinstance(self.category, str) or self.category not in TARGET_COUNTS:
            raise ValueError(f"Unknown candidate category: {self.category!r}")
        if self.requires_manual_review is not True:
            raise ValueError("Candidates must await human review")
        count, share = self.n_valid_comments, self.community_agreement
        if type(count) is not int or not 5 <= count <= MAX_COMMENTS:
            raise ValueError("Expected 5–10 valid moral comments")
        if type(share) not in (int, float) or not 0 <= share <= 1:
            raise ValueError("Expected a finite agreement in [0, 1]")
        votes = round(count * share)
        if votes < ceil(count / len(VERDICTS)) or not isclose(
            share, votes / count, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError("Agreement is inconsistent with moral comment count")
        if self.category != "missing_context" and (
            (share >= 0.70) != (self.category in VERDICTS)
            or (votes / count >= 0.70) != (self.category in VERDICTS)
        ):
            raise ValueError("Agreement is inconsistent with candidate category")


@dataclass
class GoldenSelectionResult:
    selected: list[GoldenCandidate] = field(default_factory=list)
    available_counts: dict[str, int] = field(default_factory=dict)
    target_counts: dict[str, int] = field(default_factory=lambda: dict(TARGET_COUNTS))
    seed: int = 42
    manifest_id: str | None = None
    source_revision: str | None = None

    @property
    def selected_counts(self) -> dict[str, int]:
        counts = Counter(
            candidate.category
            for candidate in self.selected
            if isinstance(candidate.category, str)
        )
        return {category: counts[category] for category in TARGET_COUNTS}

    @property
    def shortages(self) -> dict[str, int]:
        return {
            category: max(0, target - self.selected_counts[category])
            for category, target in TARGET_COUNTS.items()
        }

    def integrity_errors(self, *, require_inventory: bool = True) -> list[str]:
        errors = []
        provenance = (self.manifest_id, self.source_revision)
        if provenance != (None, None) and not all(
            isinstance(value, str) and value.strip() for value in provenance
        ):
            errors.append(
                "Selection provenance requires both manifest ID and source revision"
            )
        if type(self.seed) is not int:
            errors.append("Selection seed must be an integer")
        if (
            not isinstance(self.target_counts, dict)
            or self.target_counts != TARGET_COUNTS
            or any(type(value) is not int for value in self.target_counts.values())
        ):
            errors.append("Invalid canonical target counts")
        seen = set()
        for candidate in self.selected:
            try:
                candidate.validate()
            except ValueError as exc:
                errors.append(f"{candidate.case_id!r}: {exc}")
                continue
            if candidate.case_id in seen:
                errors.append("Selected candidate IDs are not unique")
            seen.add(candidate.case_id)
        counts = self.selected_counts
        if any(counts[key] > target for key, target in TARGET_COUNTS.items()):
            errors.append("Selected counts exceed category quotas")
        inventory = self.available_counts
        if (require_inventory or inventory != {}) and (
            not isinstance(inventory, dict)
            or inventory.keys() != TARGET_COUNTS.keys()
            or any(
                type(value) is not int or value < counts[key]
                for key, value in inventory.items()
            )
        ):
            errors.append("Invalid available counts for selected candidates")
        return errors

    def to_report(self) -> dict:
        errors = self.integrity_errors()
        if errors:
            raise ValueError("; ".join(errors))
        return {
            "seed": self.seed,
            "manifest_id": self.manifest_id,
            "source_revision": self.source_revision,
            "target_counts": self.target_counts,
            "available_counts": self.available_counts,
            "selected_counts": self.selected_counts,
            "shortages": self.shortages,
            "total_selected": len(self.selected),
            "complete": self.selected_counts == TARGET_COUNTS,
        }


def _candidate(case: AITACase, category: GoldenCategory) -> GoldenCandidate:
    fields = case.derived_fields
    agreement = fields.verdict_distribution[fields.top_verdict]

    return GoldenCandidate(
        case_id=case.post_id,
        category=category,
        community_agreement=agreement,
        n_valid_comments=fields.n_valid,
    )


def validate_source_slots(case: AITACase) -> None:
    if any(comment.rank > MAX_COMMENTS for comment in case.valid_comments):
        raise ValueError(f"Comment outside original top-ten slots: {case.post_id}")


def select_golden_candidates(
    cases: list[AITACase] | SplitResult,
    *,
    missing_context_review_ids: set[str] | None = None,
    excluded_ids: set[str] | None = None,
    seed: int = 42,
    manifest_id: str | None = None,
    source_revision: str | None = None,
) -> GoldenSelectionResult:
    """Sample preliminary candidates from a trusted DATA-04 SplitResult.

    The split supplies the entire held-out pool and both provenance fields.
    Legacy list callers must establish isolation and provenance themselves.
    Human nominations reserve eligible IDs exclusively for missing-context review.
    """
    excluded = excluded_ids or set()
    review_ids = missing_context_review_ids or set()

    manifest = None
    if isinstance(cases, SplitResult):
        if manifest_id is not None or source_revision is not None:
            raise ValueError(
                "SplitResult supplies its own manifest ID and source revision"
            )
        manifest = cases.manifest
        manifest_id, source_revision = manifest.manifest_id, manifest.source_revision
        if not all(
            isinstance(value, str) and value.strip()
            for value in (manifest_id, source_revision)
        ):
            raise ValueError("Missing manifest ID or source revision")
        cases = cases.golden_candidates

    case_ids = {case.post_id for case in cases}
    if len(case_ids) != len(cases):
        raise ValueError("Duplicate post IDs in held-out candidate pool")
    if manifest is not None:
        if case_ids != manifest.held_out_ids:
            raise ValueError("Candidate pool IDs disagree with the held-out partition")
        if not case_ids.isdisjoint(manifest.development_ids) or not case_ids.isdisjoint(
            manifest.retrieval_ids
        ):
            raise ValueError("Held-out pool overlaps development or retrieval")

    unknown_review_ids = review_ids - case_ids
    if unknown_review_ids:
        raise ValueError(
            "Missing-context review IDs are absent from the candidate pool"
        )

    if review_ids & excluded:
        raise ValueError("Missing-context review IDs overlap excluded IDs")

    buckets: dict[GoldenCategory, list[GoldenCandidate]] = {
        category: [] for category in TARGET_COUNTS
    }

    for case in cases:
        if case.post_id in excluded:
            continue

        fields = case.derived_fields
        # DATA-04 reserves by label before EVAL-01 applies its comment limit.
        if manifest is not None and not 5 <= fields.n_valid <= MAX_COMMENTS:
            if case.post_id in review_ids:
                raise ValueError(
                    f"Missing-context nominee {case.post_id!r} requires "
                    f"5–{MAX_COMMENTS} valid moral comments, found {fields.n_valid}"
                )
            continue

        validate_source_slots(case)
        agreement = fields.verdict_distribution[fields.top_verdict]

        # Human-nominated review cases are held out of all other
        # categories to ensure mutually exclusive assignments.
        if case.post_id in review_ids:
            category: GoldenCategory = "missing_context"
        elif fields.consensus_label == "DISPUTED" and agreement < 0.70:
            category = "disputed"
        elif fields.consensus_label in VERDICTS and agreement >= 0.70:
            category = fields.consensus_label
        else:
            continue

        buckets[category].append(_candidate(case, category))

    rng = Random(seed)
    result = GoldenSelectionResult(
        seed=seed, manifest_id=manifest_id, source_revision=source_revision
    )

    for category, target in TARGET_COUNTS.items():
        # Sorting before shuffling makes selection independent
        # of the incoming record order.
        candidates = sorted(buckets[category], key=lambda candidate: candidate.case_id)
        rng.shuffle(candidates)

        result.available_counts[category] = len(candidates)
        result.selected.extend(candidates[:target])

    return result


def write_golden_json(
    data: dict,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write JSON without silently overwriting an existing artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents overwriting an existing file.
    mode = "w" if overwrite else "x"

    with output_path.open(mode, encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def export_golden_candidates(
    result: GoldenSelectionResult,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Export preliminary candidate metadata, not final annotations.

    Story narratives and comment texts are deliberately excluded.
    Missing-context categories remain provisional.
    """
    errors = result.integrity_errors()
    if errors:
        raise ValueError("; ".join(errors))
    data = {
        "schema_version": "1.0",
        "status": "preliminary_candidates",
        "seed": result.seed,
        "manifest_id": result.manifest_id,
        "source_revision": result.source_revision,
        "total_candidates": len(result.selected),
        "candidates": [
            {
                **asdict(candidate),
                "category_status": "provisional",
            }
            for candidate in result.selected
        ],
    }

    write_golden_json(data, output_path, overwrite=overwrite)


def export_selection_report(
    result: GoldenSelectionResult,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Export selection statistics and category shortages."""
    data = {
        "schema_version": "1.0",
        "status": "preliminary_selection",
        **result.to_report(),
    }

    write_golden_json(data, output_path, overwrite=overwrite)
