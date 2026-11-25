"""DATA-04: reproducible RAG/development/Golden-candidate partitions.

Uses the upstream Hugging Face test IDs as the only evaluation source.
Call only on the output of deduplicate_cases() over combined train+test.
"""

import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from random import Random

from asshole_ai_detector.schemas.case import AITACase

from .deduplication import content_hash

# Minimum available cases for EVAL-01's 40/40/40/40/20 consensus quotas.
# Increase these margins if the team needs extra candidates for manual review.
DEFAULT_GOLDEN_MINIMUMS: dict[str, int] = {
    "NTA": 40,
    "YTA": 40,
    "ESH": 40,
    "NAH": 40,
    "DISPUTED": 20,
}


@dataclass(frozen=True)
class SplitManifest:
    """Contract consumed by EVAL-01/golden_validation.py.

    None = scan not yet verified; () = verified, with zero duplicate groups.
    Group IDs must refer to *retained* partition members, never removed IDs.
    """

    manifest_id: str
    source_revision: str
    held_out_ids: frozenset[str]
    development_ids: frozenset[str]
    retrieval_ids: frozenset[str]
    partition_sizes: tuple[int, ...]
    exact_groups: tuple[frozenset[str], ...] | None
    semantic_groups: tuple[frozenset[str], ...] | None


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


def _pick_golden(
    test_cases: list[AITACase],
    count: int,
    minimums: Mapping[str, int],
    rng: Random,
) -> list[AITACase]:
    if sum(minimums.values()) > count:
        raise ValueError("Golden minimum quotas exceed the held-out pool size")

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
    minimums: Mapping[str, int],
) -> str:
    """Bind manifest ID to the source, exact contents and split assignments."""
    snapshot = {
        "schema_version": 1,
        "source_revision": source_revision,
        "seed": seed,
        "minimums": dict(sorted(minimums.items())),
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
) -> SplitResult:
    """Split deduplicated cases while preserving provider train/test roles.

    `test_ids` are original provider *test* post IDs. Other IDs become RAG
    candidates. Golden and development are sampled solely from provider test.
    Golden *final* category selection is delegated to EVAL-01.
    """
    if not source_revision or not source_revision.strip():
        raise ValueError("Provide a pinned, nonempty dataset source_revision")
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    if type(golden_size) is not int or golden_size < 0:
        raise ValueError("golden_size must be a nonnegative integer")
    if type(development_size) is not int or development_size < 0:
        raise ValueError("development_size must be a nonnegative integer")
    minimums = (
        dict(DEFAULT_GOLDEN_MINIMUMS)
        if golden_minimums is None
        else dict(golden_minimums)
    )
    if any(
        label not in DEFAULT_GOLDEN_MINIMUMS
        or type(count) is not int
        or count < 0
        for label, count in minimums.items()
    ):
        raise ValueError("Invalid Golden minimum label/count")

    unique_cases = sorted(cases, key=lambda case: case.post_id)
    _ensure_exactly_unique(unique_cases)
    test_id_set = set(test_ids)
    test_cases = [case for case in unique_cases if case.post_id in test_id_set]
    train_cases = [case for case in unique_cases if case.post_id not in test_id_set]
    if golden_size + development_size > len(test_cases):
        raise ValueError(
            f"Need {golden_size + development_size} held-out/development cases, "
            f"but only {len(test_cases)} eligible provider-test cases exist"
        )

    rng = Random(seed)
    held_out = _pick_golden(test_cases, golden_size, minimums, rng)
    held_ids = {case.post_id for case in held_out}
    remaining = [case for case in test_cases if case.post_id not in held_ids]
    rng.shuffle(remaining)
    dev_ids = {case.post_id for case in remaining[:development_size]}
    development = sorted(
        (case for case in remaining if case.post_id in dev_ids),
        key=lambda case: case.post_id,
    )
    unused_test = sorted(
        (case for case in remaining if case.post_id not in dev_ids),
        key=lambda case: case.post_id,
    )

    inventories = (
        frozenset(held_ids),
        frozenset(dev_ids),
        frozenset(case.post_id for case in train_cases),
    )
    assert not (inventories[0] & inventories[1])
    assert not (inventories[0] & inventories[2])
    assert not (inventories[1] & inventories[2])
    manifest = SplitManifest(
        manifest_id=_manifest_id(
            {
                "held_out": held_out,
                "development": development,
                "retrieval": train_cases,
                "unused_test": unused_test,
            },
            source_revision=source_revision,
            seed=seed,
            minimums=minimums,
        ),
        source_revision=source_revision,
        held_out_ids=inventories[0],
        development_ids=inventories[1],
        retrieval_ids=inventories[2],
        partition_sizes=tuple(map(len, inventories)),
        exact_groups=(),  # All retained cases were explicitly checked above.
        semantic_groups=None,  # NOT scanned yet; never pretend this equals ().
    )
    return SplitResult(held_out, development, train_cases, unused_test, manifest, seed)


def attach_semantic_scan(
    manifest: SplitManifest,
    groups: Iterable[Collection[str]],
) -> SplitManifest:
    """Attach VERIFIED near-duplicate scan results, after embeddings exist.

    Only call after actually scanning the *entire* retained dataset for
    near duplicates. Detected cross-partition groups remain visible for
    EVAL-01 validation to reject; they must be resolved before final use.
    """
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
