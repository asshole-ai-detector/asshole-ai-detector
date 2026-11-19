"""Cleaning and filtering of normalized AITA records (Milestone 2, Sections 3.2 and 3.6).

Input: ``NormalizedAITACase`` records from ``aita_adapter`` (nothing filtered yet).
Output: ``CleanedAITACase`` records (still ``NormalizedAITACase`` objects, but with
cleaned text and only valid comments) ready for consensus labeling (DATA-03), plus
``RejectedAITACase`` records that say why a submission was dropped. DATA-03 builds
the final ``AITAComment`` objects and their ``post_id#rank`` IDs.

Rules:

* Comments: deleted/removed comments, comments with a null or unsupported label and
  comments with no text left after cleaning are dropped. ``INFO`` comments are not
  moral verdicts: they are dropped from the valid comments but counted in
  ``community_votes.INFO``. Valid comments keep their original rank and their
  labels are normalized to upper case.
* Submissions: rejected if the post ID, title, text or URL is missing or invalid, if
  the text is deleted/removed, if the cleaned text has fewer than 50 words, or if
  fewer than 5 valid moral verdicts remain. All matching reasons are recorded.
* Text: HTML entities decoded, links and Reddit mentions removed, lowercased and
  whitespace normalized (Section 3.6, step 2). Word counts use the cleaned text,
  which is also what ``AITACase`` validates.

Consensus labels (``DerivedFields``) and comment IDs are built in DATA-03.
"""

import html
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import Field

from asshole_ai_detector.schemas.annotation import CommunityVotes
from asshole_ai_detector.schemas.case import (
    VERDICTS,
    NormalizedAITACase,
    NormalizedAITAComment,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


class CommentStatus(StrEnum):
    """Outcome for one comment. Only VALID comments count as moral verdicts."""

    VALID = "valid"
    INFO = "info"
    DELETED = "deleted"
    NULL_LABEL = "null_label"
    UNSUPPORTED_LABEL = "unsupported_label"
    EMPTY_TEXT = "empty_text"


class RejectionReason(StrEnum):
    """Why a submission was rejected. A submission can have several reasons."""

    MISSING_POST_ID = "missing_post_id"
    MISSING_TITLE = "missing_title"
    MISSING_TEXT = "missing_text"
    DELETED_TEXT = "deleted_text"
    INVALID_URL = "invalid_url"
    TOO_FEW_WORDS = "too_few_words"
    TOO_FEW_VALID_VERDICTS = "too_few_valid_verdicts"


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds from ``preprocessing`` in configs/default.yaml."""

    min_post_words: int = 50
    min_valid_verdict_comments: int = 5
    valid_verdicts: tuple[str, ...] = VERDICTS
    clarification_label: str = "INFO"
    lowercase: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.valid_verdicts) - set(VERDICTS)
        if unknown:
            raise ValueError(f"Unsupported valid verdicts: {sorted(unknown)}")

    @classmethod
    def from_mapping(cls, preprocessing: Mapping[str, Any]) -> Self:
        """Build from the ``preprocessing`` section, ignoring unrelated keys."""
        defaults = cls()
        return cls(
            min_post_words=int(
                preprocessing.get("min_post_words", defaults.min_post_words)
            ),
            min_valid_verdict_comments=int(
                preprocessing.get(
                    "min_valid_verdict_comments", defaults.min_valid_verdict_comments
                )
            ),
            valid_verdicts=tuple(
                preprocessing.get("valid_verdicts", defaults.valid_verdicts)
            ),
            clarification_label=str(
                preprocessing.get("clarification_label", defaults.clarification_label)
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Self:
        with Path(path).open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return cls.from_mapping(config.get("preprocessing", {}))


DEFAULT_CLEANING_CONFIG = CleaningConfig()


class CleanedAITACase(NormalizedAITACase):
    """An eligible submission: cleaned text and only valid moral-verdict comments.

    It is still a ``NormalizedAITACase``, so DATA-03 can call
    ``label_cleaned_case(case)`` and build ``AITAComment`` objects (and their
    ``post_id#rank`` IDs) from ``case.comments`` in one place.
    """

    post_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    url: str = Field(min_length=1)
    # Only valid NTA/YTA/ESH/NAH comments, sorted by original rank. Ranks may
    # have gaps where comments were dropped.
    comments: list[NormalizedAITAComment] = Field(default_factory=list)
    # Counts over the original comments; INFO is kept here, not in comments.
    community_votes: CommunityVotes

    @property
    def n_valid(self) -> int:
        return len(self.comments)


@dataclass(frozen=True)
class RejectedAITACase:
    """A rejected submission, kept for the cleaning report."""

    post_id: str | None
    url: str | None
    reasons: tuple[RejectionReason, ...]
    n_valid: int
    info_count: int


@dataclass
class CleaningReport:
    """Counts of accepted and rejected records and why they were rejected."""

    total: int = 0
    accepted: int = 0
    rejected: int = 0
    # A record with several reasons is counted under each of them.
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    # Status of every comment seen, across accepted and rejected records.
    comment_statuses: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "comment_statuses": dict(sorted(self.comment_statuses.items())),
        }


@dataclass
class CleaningResult:
    accepted: list[CleanedAITACase]
    rejected: list[RejectedAITACase]
    report: CleaningReport


# Text normalization ---------------------------------------------------------

_DELETED_PATTERN = re.compile(r"^\[\s*(deleted|removed)[^\]]*\]$", re.IGNORECASE)
_MARKDOWN_LINK = re.compile(
    r"\[([^\]]*)\]\((?:https?://|www\.)[^)\s]*\)", re.IGNORECASE
)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# u/name, /u/name, r/sub, /r/sub, not inside a word (e.g. "w/r/t").
_MENTION = re.compile(r"(?<![\w/])/?[ur]/[A-Za-z0-9_-]+", re.IGNORECASE)
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_EXTRA_NEWLINES = re.compile(r"\n{3,}")


def is_deleted(text: str | None) -> bool:
    """True for Reddit placeholders such as ``[deleted]`` or ``[removed]``."""
    return text is not None and bool(_DELETED_PATTERN.match(text.strip()))


def normalize_text(text: str | None, *, lowercase: bool = True) -> str:
    """Decode entities, remove links and mentions, lowercase and tidy whitespace.

    Markdown links keep their visible text. Paragraph breaks are kept (at most
    one blank line) because they are part of how the story reads.
    """
    if not text:
        return ""
    cleaned = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = _URL.sub(" ", cleaned)
    cleaned = _MENTION.sub(" ", cleaned)
    if lowercase:
        cleaned = cleaned.lower()
    lines = (_HORIZONTAL_SPACE.sub(" ", line).strip() for line in cleaned.split("\n"))
    cleaned = "\n".join(lines)
    return _EXTRA_NEWLINES.sub("\n\n", cleaned).strip()


def count_words(text: str) -> int:
    """Whitespace word count, the same rule ``AITACase`` validates."""
    return len(text.split())


def normalize_label(label: str | None) -> str | None:
    if label is None or not label.strip():
        return None
    return label.strip().upper()


# Filtering --------------------------------------------------------------------


def classify_comment(
    comment: NormalizedAITAComment, config: CleaningConfig = DEFAULT_CLEANING_CONFIG
) -> CommentStatus:
    """Decide whether a comment is a valid verdict, INFO or invalid.

    Deleted text wins over any label, because a label on a deleted comment
    cannot be checked against its text.
    """
    if is_deleted(comment.text):
        return CommentStatus.DELETED
    label = normalize_label(comment.verdict)
    if label is None:
        return CommentStatus.NULL_LABEL
    if label != config.clarification_label and label not in config.valid_verdicts:
        return CommentStatus.UNSUPPORTED_LABEL
    if not normalize_text(comment.text, lowercase=config.lowercase):
        return CommentStatus.EMPTY_TEXT
    if label == config.clarification_label:
        return CommentStatus.INFO
    return CommentStatus.VALID


def _is_http_url(url: str | None) -> bool:
    return bool(url) and url.strip().lower().startswith(("http://", "https://"))


def clean_case(
    case: NormalizedAITACase,
    config: CleaningConfig = DEFAULT_CLEANING_CONFIG,
    report: CleaningReport | None = None,
) -> CleanedAITACase | RejectedAITACase:
    """Clean one record and either accept it or say why it was rejected.

    If ``report`` is given, it is updated with this record's outcome.
    """
    statuses = [
        (comment, classify_comment(comment, config)) for comment in case.comments
    ]
    valid = sorted(
        (comment for comment, status in statuses if status is CommentStatus.VALID),
        key=lambda comment: comment.rank,
    )
    votes = Counter(
        normalize_label(c.verdict) for c, s in statuses if s is CommentStatus.VALID
    )
    info_count = sum(1 for _, status in statuses if status is CommentStatus.INFO)

    title = normalize_text(case.title, lowercase=config.lowercase)
    text = normalize_text(case.text, lowercase=config.lowercase)

    reasons: list[RejectionReason] = []
    if not case.post_id:
        reasons.append(RejectionReason.MISSING_POST_ID)
    if not title:
        reasons.append(RejectionReason.MISSING_TITLE)
    if is_deleted(case.text):
        reasons.append(RejectionReason.DELETED_TEXT)
    elif not text:
        reasons.append(RejectionReason.MISSING_TEXT)
    elif count_words(text) < config.min_post_words:
        reasons.append(RejectionReason.TOO_FEW_WORDS)
    if not _is_http_url(case.url):
        reasons.append(RejectionReason.INVALID_URL)
    if len(valid) < config.min_valid_verdict_comments:
        reasons.append(RejectionReason.TOO_FEW_VALID_VERDICTS)

    outcome: CleanedAITACase | RejectedAITACase
    if reasons:
        outcome = RejectedAITACase(
            post_id=case.post_id,
            url=case.url,
            reasons=tuple(reasons),
            n_valid=len(valid),
            info_count=info_count,
        )
    else:
        outcome = CleanedAITACase(
            post_id=case.post_id,
            title=title,
            text=text,
            url=case.url.strip(),
            date=case.date,
            comments=[
                NormalizedAITAComment(
                    rank=comment.rank,
                    verdict=normalize_label(comment.verdict),
                    text=normalize_text(comment.text, lowercase=config.lowercase),
                )
                for comment in valid
            ],
            community_votes=CommunityVotes(
                **{verdict: votes[verdict] for verdict in VERDICTS}, INFO=info_count
            ),
            provider_ambiguity=case.provider_ambiguity,
        )

    if report is not None:
        report.total += 1
        report.comment_statuses.update(status.value for _, status in statuses)
        if reasons:
            report.rejected += 1
            report.rejection_reasons.update(reason.value for reason in reasons)
        else:
            report.accepted += 1
    return outcome


def clean_cases(
    cases: Iterable[NormalizedAITACase],
    config: CleaningConfig = DEFAULT_CLEANING_CONFIG,
) -> CleaningResult:
    """Clean every record, keeping input order, and build the report."""
    report = CleaningReport()
    accepted: list[CleanedAITACase] = []
    rejected: list[RejectedAITACase] = []
    for case in cases:
        outcome = clean_case(case, config, report)
        if isinstance(outcome, CleanedAITACase):
            accepted.append(outcome)
        else:
            rejected.append(outcome)
    return CleaningResult(accepted=accepted, rejected=rejected, report=report)
