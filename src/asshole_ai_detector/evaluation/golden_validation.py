"""Check Golden selection against DATA-04 evidence supplied by a trusted loader."""

from dataclasses import dataclass

from asshole_ai_detector.data.splits import SplitManifest
from asshole_ai_detector.evaluation.golden_selection import (
    TARGET_COUNTS,
    GoldenSelectionResult,
)


@dataclass(frozen=True)
class GoldenValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    selected_count: int
    manifest_id: str | None = None
    source_revision: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors


def _check_groups(
    groups: list[set[str] | frozenset[str]],
    partitions: tuple[frozenset[str], ...],
    selected: set[str],
) -> list[str]:
    """Check connected components of both scans, including transitive links."""
    neighbors: dict[str, set[str]] = {}
    errors = []
    universe = frozenset.union(*partitions)
    for index, group in enumerate(groups):
        if not isinstance(group, (set, frozenset)) or len(group) < 2:
            errors.append(f"Duplicate group {index} needs at least two IDs")
            continue
        if not group <= universe:
            errors.append(f"Duplicate group {index} contains unmapped IDs")
            continue
        anchor = next(iter(group))
        neighbors.setdefault(anchor, set()).update(group)
        for member in group:
            neighbors.setdefault(member, set()).add(anchor)
    while neighbors:
        pending = [next(iter(neighbors))]
        cluster = set()
        while pending:
            member = pending.pop()
            if member in neighbors:
                cluster.add(member)
                pending.extend(neighbors.pop(member))
        if sum(bool(cluster & partition) for partition in partitions) > 1:
            errors.append("Duplicate group crosses partitions")
        if len(cluster & selected) > 1:
            errors.append("Duplicate group contains multiple selected candidates")
    return errors


def validate_golden_selection(
    result: GoldenSelectionResult,
    *,
    held_out_ids: set[str] | None = None,
    development_ids: set[str] | None = None,
    retrieval_ids: set[str] | None = None,
    duplicate_groups: list[set[str]] | None = None,
    require_complete: bool = False,
    manifest: SplitManifest | None = None,
) -> GoldenValidationReport:
    """Legacy ID sets permit preliminary checks. Strict mode needs manifest evidence."""
    errors = result.integrity_errors(require_inventory=require_complete)
    warnings = []
    pending = errors if require_complete else warnings
    groups: list[set[str] | frozenset[str]] = []
    if manifest is None:
        pending.append("DATA-04 manifest provenance not verified")
        pending.append("Exact duplicate scan status not verified")
        pending.append("Near-duplicate isolation not verified")
        inventories = (held_out_ids, development_ids, retrieval_ids)
        groups.extend(duplicate_groups or ())
    else:
        if any(
            arg is not None
            for arg in (held_out_ids, development_ids, retrieval_ids, duplicate_groups)
        ):
            errors.append(
                "Use one manifest without separate partition or cluster inputs"
            )
        if not all(
            isinstance(value, str) and value.strip()
            for value in (manifest.manifest_id, manifest.source_revision)
        ):
            errors.append("Missing manifest ID or source revision")
        if (result.manifest_id, result.source_revision) == (None, None):
            pending.append("Selection is not bound to its source manifest")
        elif (result.manifest_id, result.source_revision) != (
            manifest.manifest_id,
            manifest.source_revision,
        ):
            errors.append("Selection provenance disagrees with source manifest")
        inventories = (
            manifest.held_out_ids,
            manifest.development_ids,
            manifest.retrieval_ids,
        )
        for name, scan in (
            ("Exact", manifest.exact_groups),
            ("Semantic", manifest.semantic_groups),
        ):
            if scan is None:
                pending.append(f"{name} duplicate scan not verified")
            elif not isinstance(scan, tuple):
                errors.append(f"Invalid {name} duplicate scan payload")
            else:
                groups.extend(scan)

    if any(not isinstance(ids, (set, frozenset)) for ids in inventories):
        errors.append("All three partition inventories must be sets of IDs")
    partitions = tuple(
        frozenset(ids) if isinstance(ids, (set, frozenset)) else frozenset()
        for ids in inventories
    )
    if manifest is not None and (
        manifest.partition_sizes != tuple(map(len, partitions))
        or any(type(size) is not int for size in manifest.partition_sizes)
    ):
        errors.append("Partition inventories disagree with manifest sizes")
    held, dev, retrieval = partitions
    if any(
        not isinstance(key, str) or not key.strip()
        for part in partitions
        for key in part
    ):
        errors.append("Invalid partition ID")
    selected = {
        item.case_id for item in result.selected if isinstance(item.case_id, str)
    }
    if not selected <= held:
        errors.append("Some selected IDs are outside the held-out pool")
    if held & dev:
        errors.append("Held-out and development partitions overlap")
    if held & retrieval:
        errors.append("Held-out and retrieval partitions overlap")
    if dev & retrieval:
        errors.append("Development and retrieval partitions overlap")
    errors.extend(_check_groups(groups, partitions, selected))
    if not result.available_counts:
        warnings.append("Available inventory counts not supplied")
    elif not errors and sum(result.available_counts.values()) > len(held):
        errors.append("Available counts exceed held-out population")
    if require_complete:
        if len(selected) != sum(TARGET_COUNTS.values()) or len(result.selected) != 200:
            errors.append("Complete selection expected exactly 200 unique records")
        for category, count in result.selected_counts.items():
            if count != TARGET_COUNTS[category]:
                errors.append(
                    f"{category}: expected {TARGET_COUNTS[category]}, got {count}"
                )
    if any(result.shortages.values()):
        warnings.append("Selection has category shortages")
    if any(candidate.category == "missing_context" for candidate in result.selected):
        warnings.append("Missing-context candidates require human confirmation")
    return GoldenValidationReport(
        tuple(errors),
        tuple(warnings),
        len(result.selected),
        manifest.manifest_id if manifest else None,
        manifest.source_revision if manifest else None,
    )
