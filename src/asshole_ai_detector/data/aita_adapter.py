"""Adapter for the Reddit AITA multiclass dataset (Milestone 2, Sections 1.2 and 3.6).

Maps raw Hugging Face rows to NormalizedAITACase records. Nothing is filtered or
cleaned here: deleted comments, INFO/null labels and short posts are kept, and
later preprocessing steps decide what is valid.
"""

import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from asshole_ai_detector.schemas.case import NormalizedAITACase, NormalizedAITAComment

if TYPE_CHECKING:
    from datasets import Dataset

# Defaults mirror `data_sources.aita` in configs/default.yaml.
HF_DATASET = "MattBoraske/reddit-AITA-submissions-and-comments-multiclass"
HF_REVISION = "84c229ffd926e8efaa1828a5d3803f29231f2d0e"
HF_SPLITS: tuple[str, ...] = ("train", "test")

# The source stores up to ten ranked comments as flat columns.
MAX_COMMENTS = 10
SOURCE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# The dataset has no ID column; the Reddit post ID is part of the URL.
_POST_ID_PATTERN = re.compile(r"/comments/([a-z0-9]+)(?:/|$)", re.IGNORECASE)


def extract_post_id(url: str | None) -> str | None:
    """Return the Reddit post ID from a submission URL, or None if absent."""
    if not url:
        return None
    match = _POST_ID_PATTERN.search(url)
    return match.group(1).lower() if match else None


def parse_date(value: Any) -> datetime | None:
    """Parse the source timestamp and mark it as UTC.

    The source dates have no timezone; we treat them as UTC so they can be
    compared with timezone-aware dates from other sources.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip(), SOURCE_DATE_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        # Malformed dates stay missing instead of failing the whole load.
        return None


def _optional_text(value: Any) -> str | None:
    """Treat missing or empty strings as None so filtering has one check."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    score = float(value)
    return None if math.isnan(score) else score


def row_to_normalized(row: Mapping[str, Any]) -> NormalizedAITACase:
    """Convert one raw dataset row into a NormalizedAITACase.

    Comment slots where both text and label are missing are skipped. All
    other comments keep their original rank, text and raw label.
    """
    comments: list[NormalizedAITAComment] = []
    for rank in range(1, MAX_COMMENTS + 1):
        text = _optional_text(row.get(f"top_comment_{rank}"))
        verdict = _optional_text(row.get(f"top_comment_{rank}_classification"))
        if text is None and verdict is None:
            continue
        comments.append(NormalizedAITAComment(rank=rank, verdict=verdict, text=text))

    url = _optional_text(row.get("submission_url"))
    return NormalizedAITACase(
        post_id=extract_post_id(url),
        title=_optional_text(row.get("submission_title")),
        text=_optional_text(row.get("submission_text")),
        url=url,
        date=parse_date(row.get("submission_date")),
        comments=comments,
        # Diagnostic only; never used for filtering or consensus.
        provider_ambiguity=_optional_score(row.get("ambiguity_score")),
    )


def iter_normalized_cases(rows: Iterable[Mapping[str, Any]]) -> Iterator[NormalizedAITACase]:
    """Lazily normalize raw rows (a Hugging Face Dataset or any row iterable)."""
    for row in rows:
        yield row_to_normalized(row)


def load_aita_dataset(
    revision: str = HF_REVISION,
    splits: Sequence[str] = HF_SPLITS,
    cache_dir: str | None = None,
) -> "Dataset":
    """Load the raw dataset from Hugging Face and concatenate the given splits.

    The provider's train/test split has no meaning for us: Golden, development
    and index splits are created later from the combined pool.
    """
    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The 'datasets' package is required. Install it with: uv sync --group data"
        ) from error

    dataset_dict = load_dataset(HF_DATASET, revision=revision, cache_dir=cache_dir)
    return concatenate_datasets([dataset_dict[split] for split in splits])
