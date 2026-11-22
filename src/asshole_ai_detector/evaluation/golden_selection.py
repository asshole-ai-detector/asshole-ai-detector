"""Deterministic candidate selection for the held-out AITA Golden Dataset.

This module does not create final human annotations or perform dataset splitting.
The caller must supply a deduplicated, held-out pool from DATA-04.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import Literal

from asshole_ai_detector.schemas.case import VERDICTS, AITACase

type GoldenCategory = Literal["NTA", "YTA", "ESH", "NAH", "disputed", "missing_context"]

TARGET_COUNTS: dict[GoldenCategory, int] = {
    "NTA": 40,
    "YTA": 40,
    "ESH": 40,
    "NAH": 40,
    "disputed": 20,
    "missing_context": 20,
}


@dataclass(frozen=True)
class GoldenCandidate:
    """A candidate awaiting independent human annotation."""

    case_id: str
    category: GoldenCategory
    community_agreement: float
    n_valid_comments: int
    requires_manual_review: bool = True


@dataclass
class GoldenSelectionResult:
    selected: list[GoldenCandidate] = field(default_factory=list)
    available_counts: dict[str, int] = field(default_factory=dict)
    target_counts: dict[str, int] = field(default_factory=lambda: dict(TARGET_COUNTS))
    seed: int = 42

    @property
    def selected_counts(self) -> dict[str, int]:
        counts = Counter(candidate.category for candidate in self.selected)
        return {category: counts[category] for category in TARGET_COUNTS}

    @property
    def shortages(self) -> dict[str, int]:
        return {
            category: max(0, target - self.selected_counts[category])
            for category, target in self.target_counts.items()
        }

    def to_report(self) -> dict:
        return {
            "seed": self.seed,
            "target_counts": self.target_counts,
            "available_counts": self.available_counts,
            "selected_counts": self.selected_counts,
            "shortages": self.shortages,
            "total_selected": len(self.selected),
            "complete": not any(self.shortages.values()),
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


def select_golden_candidates(
    cases: list[AITACase],
    *,
    missing_context_review_ids: set[str] | None = None,
    excluded_ids: set[str] | None = None,
    seed: int = 42,
) -> GoldenSelectionResult:
    """Select preliminary candidates from an isolated held-out pool.

    Preconditions:
    - DATA-04 has already deduplicated and isolated the pool.
    - excluded_ids contains any IDs forbidden by the split manifest.
    - missing_context_review_ids contains human-nominated review candidates,
      not confirmed missing-context annotations.

    A selected category is provisional until independent human review.
    """
    excluded = excluded_ids or set()
    review_ids = missing_context_review_ids or set()

    case_ids = [case.post_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate post IDs in held-out candidate pool")

    unknown_review_ids = review_ids - set(case_ids)
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
        agreement = fields.verdict_distribution[fields.top_verdict]

        if fields.n_valid < 5:
            continue

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
    result = GoldenSelectionResult(seed=seed)

    for category, target in TARGET_COUNTS.items():
        # Sorting before shuffling makes selection independent
        # of the incoming record order.
        candidates = sorted(buckets[category], key=lambda candidate: candidate.case_id)
        rng.shuffle(candidates)

        result.available_counts[category] = len(candidates)
        result.selected.extend(candidates[:target])

    return result


def _write_json(
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
    data = {
        "schema_version": "1.0",
        "status": "preliminary_candidates",
        "seed": result.seed,
        "total_candidates": len(result.selected),
        "candidates": [
            {
                **asdict(candidate),
                "category_status": "provisional",
            }
            for candidate in result.selected
        ],
    }

    _write_json(data, output_path, overwrite=overwrite)


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

    _write_json(data, output_path, overwrite=overwrite)
