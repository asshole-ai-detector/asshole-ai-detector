"""DATA-04: deterministic exact deduplication of validated AITA cases.

This does not detect semantic paraphrases; that is a separate embedding scan.
"""

from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

from asshole_ai_detector.schemas.case import AITACase

from .cleaning import normalize_text


def normalize_for_dedup(text: str) -> str:
    """Reuse DATA-02 cleaning, ignoring whitespace/paragraph differences."""
    return " ".join(normalize_text(text).split())


def content_hash(case: AITACase) -> str:
    """Stable SHA-256 of the normalized narrative, independent of its title."""
    normalized = normalize_for_dedup(case.text)
    if not normalized:
        raise ValueError(f"Empty normalized narrative: {case.post_id}")
    return sha256(normalized.encode("utf-8")).hexdigest()


DuplicateReason = Literal[
    "duplicate_id", "duplicate_content", "transitive_duplicate"
]


@dataclass(frozen=True)
class DuplicateRecord:
    post_id: str
    duplicate_of: str
    reason: DuplicateReason


@dataclass
class DeduplicationResult:
    cases: list[AITACase] = field(default_factory=list)
    duplicates: list[DuplicateRecord] = field(default_factory=list)
    input_count: int = 0

    @property
    def removed_count(self) -> int:
        return len(self.duplicates)


def deduplicate_cases(
    cases: Iterable[AITACase],
    *,
    preferred_ids: Collection[str] = (),
) -> DeduplicationResult:
    """Collapse connected duplicate groups using IDs OR normalized narratives.

    A group may be connected transitively (ID match + content match).
    The canonical record is deterministic. ``preferred_ids`` can give
    original test IDs priority when collapsing train/test duplicates.

    IMPORTANT: Pass the combined train + test records, not each split alone.
    """
    records = list(cases)
    hashes = [content_hash(case) for case in records]
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}
    for index, (case, digest) in enumerate(zip(records, hashes, strict=True)):
        if case.post_id in seen_ids:
            union(index, seen_ids[case.post_id])
        else:
            seen_ids[case.post_id] = index
        if digest in seen_hashes:
            union(index, seen_hashes[digest])
        else:
            seen_hashes[digest] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)

    preferred = set(preferred_ids)
    result = DeduplicationResult(input_count=len(records))
    for members in groups.values():
        # model_dump_json is a stable tiebreaker for same-ID conflicting rows.
        canonical_idx = min(
            members,
            key=lambda i: (
                records[i].post_id not in preferred,
                records[i].post_id,
                records[i].model_dump_json(),
            ),
        )
        canonical = records[canonical_idx]
        result.cases.append(canonical)
        for index in members:
            if index == canonical_idx:
                continue
            rejected = records[index]
            if rejected.post_id == canonical.post_id:
                reason: DuplicateReason = "duplicate_id"
            elif hashes[index] == hashes[canonical_idx]:
                reason = "duplicate_content"
            else:
                reason = "transitive_duplicate"
            result.duplicates.append(
                DuplicateRecord(rejected.post_id, canonical.post_id, reason)
            )

    result.cases.sort(key=lambda case: case.post_id)
    result.duplicates.sort(
        key=lambda row: (row.post_id, row.duplicate_of, row.reason)
    )
    return result
