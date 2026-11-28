"""DATA-04: reproducible isolated AITA partitions with optional combined stratification.

Two policies are supported:
- provider_test: original HF test cases are eligible for Golden/dev; train for retrieval.
- combined_stratified: any clean/deduplicated case may be sampled into Golden/dev;
  all remaining cases are retrieval-eligible. Never use Golden/dev in training or RAG.

Only call on cases deduplicated across ALL original provider partitions.
"""

import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from random import Random
from typing import Literal

from asshole_ai_detector.schemas.case import AITACase

from .deduplication import content_hash

SplitStrategy = Literal["provider_test", "combined_stratified"]

DEFAULT_GOLDEN_MINIMUMS: dict[str, int] = {
    "NTA": 40,
    "YTA": 40,
    "ESH": 40,
    "NAH": 40,
    "DISPUTED": 20,
}


@dataclass(frozen=True)
class SplitManifest:
    """Partition inventories and leakage-scan evidence for EVAL-01.

    None = scan not yet verified; () = verified with zero duplicate groups.
    """

    manifest_id: str
    source_revision: str
    held_out_ids: frozenset[str]
    development_ids: frozenset[str]
    retrieval_ids: frozenset[str]
    partition_sizes: tuple[int, ...]
    exact_groups: tuple[frozenset[str], ...] | None
    semantic_groups: tuple[frozenset[str], ...] | None
    strategy: SplitStrategy = "provider_test"


@dataclass(frozen=True)
class SplitResult:
    held_out: list[AITACase]
    development: list[AITACase]
    retrieval: list[AITACase]
    unused_test: list[AITACase]
    manifest: SplitManifest
    seed: int

    @property
    def golden_candidates(self) -> list[AITACase]:
        return self.held_out

    @property
    def rag_candidates(self) -> list[AITACase]:
        return self.retrieval


def _validate_minimums(minimums: Mapping[str, int], name: str, size: int) -> None:
    if any(
        label not in DEFAULT_GOLDEN_MINIMUMS
        or type(count) is not int
        or count < 0
        for label, count in minimums.items()
    ):
        raise ValueError(f"Invalid {name} minimum label/count")
    if sum(minimums.values()) > size:
        raise ValueError(f"{name} minimum quotas exceed partition size")


def _pick_golden(
    test_cases: list[AITACase],
    count: int,
    minimums: Mapping[str, int],
    rng: Random,
) -> list[AITACase]:
    """Keep the original provider-test sampler for backwards compatibility."""
    buckets: dict[str, list[AITACase]] = {}
    for case in test_cases:
        buckets.setdefault(case.derived_fields.consensus_label, []).append(case)

    selected: list[AITACase] = []
    for category in sorted(minimums):
        requested = minimums[category]
        available = sorted(buckets.get(category, []), key=lambda c: c.post_id)
        if len(available) < requested:
            raise ValueError(
                f"Not enough {category} cases for Golden candidates: "
                f"need {requested}, have {len(available)}"
            )
        rng.shuffle(available)
        selected.extend(available[:requested])

    already = {case.post_id for case in selected}
    remainder = sorted(
        (case for case in test_cases if case.post_id not in already),
        key=lambda case: case.post_id,
    )
    rng.shuffle(remainder)
    selected.extend(remainder[: count - len(selected)])
    return sorted(selected, key=lambda case: case.post_id)


def _allocate_with_minimums(
    eligible: list[AITACase],
    *,
    golden_size: int,
    development_size: int,
    golden_minimums: Mapping[str, int],
    development_minimums: Mapping[str, int],
    rng: Random,
) -> tuple[list[AITACase], list[AITACase], list[AITACase]]:
    """Reserve BOTH partitions' rare classes before filling remaining slots.

    This is deterministic for the same source records, revision, quotas and seed;
    it does not artificially duplicate or resample any post.
    """
    buckets: dict[str, list[AITACase]] = {}
    for case in eligible:
        buckets.setdefault(case.derived_fields.consensus_label, []).append(case)

    held_out: list[AITACase] = []
    development: list[AITACase] = []
    for label in sorted(set(golden_minimums) | set(development_minimums)):
        n_gold = golden_minimums.get(label, 0)
        n_dev = development_minimums.get(label, 0)
        available = sorted(buckets.get(label, []), key=lambda c: c.post_id)
        if n_gold + n_dev > len(available):
            raise ValueError(
                f"Not enough {label} cases: need {n_gold} Golden + "
                f"{n_dev} development, have {len(available)}"
            )
        rng.shuffle(available)
        held_out.extend(available[:n_gold])
        development.extend(available[n_gold : n_gold + n_dev])

    reserved_ids = {case.post_id for case in (*held_out, *development)}
    remaining = sorted(
        (case for case in eligible if case.post_id not in reserved_ids),
        key=lambda c: c.post_id,
    )
    rng.shuffle(remaining)
    gold_fill = golden_size - len(held_out)
    dev_fill = development_size - len(development)
    held_out.extend(remaining[:gold_fill])
    development.extend(remaining[gold_fill : gold_fill + dev_fill])
    unused = remaining[gold_fill + dev_fill :]
    return (
        sorted(held_out, key=lambda c: c.post_id),
        sorted(development, key=lambda c: c.post_id),
        sorted(unused, key=lambda c: c.post_id),
    )


def _ensure_exactly_unique(cases: list[AITACase]) -> None:
    ids = [case.post_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated post IDs: run deduplicate_cases before splitting")
    hashes = [content_hash(case) for case in cases]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Repeated normalized narratives: deduplicate before splitting")


def _manifest_id(
    partitions: dict[str, list[AITACase]],
    *,
    source_revision: str,
    seed: int,
    strategy: SplitStrategy,
    golden_minimums: Mapping[str, int],
    development_minimums: Mapping[str, int],
) -> str:
    """Bind ID to provenance, partition contents, policy and both quota sets."""
    snapshot = {
        "schema_version": 2,
        "source_revision": source_revision,
        "seed": seed,
        "strategy": strategy,
        "golden_minimums": dict(sorted(golden_minimums.items())),
        "development_minimums": dict(sorted(development_minimums.items())),
        "partitions": {
            name: [
                (case.post_id, sha256(case.model_dump_json().encode("utf-8")).hexdigest())
                for case in cases
            ]
            for name, cases in sorted(partitions.items())
        },
    }
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def split_cases(
    cases: Iterable[AITACase],
    *,
    test_ids: Collection[str],
    source_revision: str,
    golden_size: int = 800,
    development_size: int = 500,
    seed: int = 42,
    golden_minimums: Mapping[str, int] | None = None,
    development_minimums: Mapping[str, int] | None = None,
    strategy: SplitStrategy = "provider_test",
) -> SplitResult:
    """Split cases; EVAL-01 separately selects the final 200 Golden candidates.

    ``test_ids`` identifies original eligible HF test cases. The combined policy
    can sample from both HF partitions; original test IDs remain useful for
    provenance and compatibility. ``unused_test`` is empty for combined policy:
    every other eligible case becomes retrieval-eligible.
    """
    if not source_revision or not source_revision.strip():
        raise ValueError("Provide a pinned, nonempty dataset source_revision")
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    if type(golden_size) is not int or golden_size < 0:
        raise ValueError("golden_size must be a nonnegative integer")
    if type(development_size) is not int or development_size < 0:
        raise ValueError("development_size must be a nonnegative integer")
    if strategy not in ("provider_test", "combined_stratified"):
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    gold_mins = dict(DEFAULT_GOLDEN_MINIMUMS if golden_minimums is None else golden_minimums)
    dev_mins = dict(development_minimums or {})
    _validate_minimums(gold_mins, "Golden", golden_size)
    _validate_minimums(dev_mins, "Development", development_size)

    unique_cases = sorted(cases, key=lambda case: case.post_id)
    _ensure_exactly_unique(unique_cases)
    test_id_set = set(test_ids)
    test_cases = [case for case in unique_cases if case.post_id in test_id_set]
    train_cases = [case for case in unique_cases if case.post_id not in test_id_set]
    eligible = test_cases if strategy == "provider_test" else unique_cases
    if golden_size + development_size > len(eligible):
        raise ValueError(
            f"Need {golden_size + development_size} held-out/development cases, "
            f"but only {len(eligible)} eligible cases exist"
        )

    rng = Random(seed)
    if strategy == "provider_test" and not dev_mins:
        # Original behavior for existing callers and historical tests.
        held_out = _pick_golden(test_cases, golden_size, gold_mins, rng)
        held_ids = {case.post_id for case in held_out}
        remaining = [case for case in test_cases if case.post_id not in held_ids]
        rng.shuffle(remaining)
        dev_ids = {case.post_id for case in remaining[:development_size]}
        development = sorted(
            (case for case in remaining if case.post_id in dev_ids),
            key=lambda c: c.post_id,
        )
        unused_test = sorted(
            (case for case in remaining if case.post_id not in dev_ids),
            key=lambda c: c.post_id,
        )
        retrieval = train_cases
    else:
        held_out, development, leftover = _allocate_with_minimums(
            eligible,
            golden_size=golden_size,
            development_size=development_size,
            golden_minimums=gold_mins,
            development_minimums=dev_mins,
            rng=rng,
        )
        if strategy == "combined_stratified":
            retrieval = leftover
            unused_test = []
        else:
            retrieval = train_cases
            unused_test = leftover

    held_ids = frozenset(c.post_id for c in held_out)
    dev_ids = frozenset(c.post_id for c in development)
    retrieval_ids = frozenset(c.post_id for c in retrieval)
    inventories = (held_ids, dev_ids, retrieval_ids)
    if any(inventories[i] & inventories[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Cross-partition post-ID overlap")
    if len(held_out) != golden_size or len(development) != development_size:
        raise ValueError("Partition allocation did not meet requested sizes")

    manifest = SplitManifest(
        manifest_id=_manifest_id(
            {
                "held_out": held_out,
                "development": development,
                "retrieval": retrieval,
                "unused_test": unused_test,
            },
            source_revision=source_revision,
            seed=seed,
            strategy=strategy,
            golden_minimums=gold_mins,
            development_minimums=dev_mins,
        ),
        source_revision=source_revision,
        held_out_ids=held_ids,
        development_ids=dev_ids,
        retrieval_ids=retrieval_ids,
        partition_sizes=tuple(map(len, inventories)),
        exact_groups=(),
        semantic_groups=None,  # NOT scanned yet; RAG-04 owns the semantic scan.
        strategy=strategy,
    )
    return SplitResult(held_out, development, retrieval, unused_test, manifest, seed)


def attach_semantic_scan(
    manifest: SplitManifest,
    groups: Iterable[Collection[str]],
) -> SplitManifest:
    """Attach VERIFIED scan results only after scanning retained partitions."""
    universe = (
        manifest.held_out_ids | manifest.development_ids | manifest.retrieval_ids
    )
    cleaned_groups: set[frozenset[str]] = set()
    for group in groups:
        group_ids = frozenset(group)
        if len(group_ids) < 2 or not group_ids <= universe:
            raise ValueError("Semantic group must contain 2+ retained partition IDs")
        cleaned_groups.add(group_ids)
    ordered = tuple(sorted(cleaned_groups, key=lambda group: tuple(sorted(group))))
    return replace(manifest, semantic_groups=ordered)
