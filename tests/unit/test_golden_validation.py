from dataclasses import replace

import pytest

from asshole_ai_detector.evaluation.golden_selection import (
    TARGET_COUNTS,
    GoldenCandidate,
    GoldenSelectionResult,
)
from asshole_ai_detector.evaluation.golden_validation import (
    SplitManifest,
    validate_golden_selection,
)


def make_result() -> GoldenSelectionResult:
    return GoldenSelectionResult(
        selected=[
            GoldenCandidate(
                case_id="gold-1",
                category="NTA",
                community_agreement=0.8,
                n_valid_comments=10,
            )
        ]
    )


def validate(result=None, **overrides):
    arguments = {
        "held_out_ids": {"gold-1"},
        "development_ids": {"dev-1"},
        "retrieval_ids": {"rag-1"},
        "duplicate_groups": [],
    }
    arguments.update(overrides)

    return validate_golden_selection(
        result or make_result(),
        **arguments,
    )


def test_valid_isolated_selection():
    report = validate()
    assert report.valid
    assert report.errors == ()


def test_rejects_candidate_outside_held_out_pool():
    report = validate(held_out_ids={"other"})
    assert not report.valid
    assert any("outside" in error for error in report.errors)


def test_rejects_development_leakage():
    report = validate(development_ids={"gold-1"})
    assert not report.valid
    assert any("development" in error for error in report.errors)


def test_rejects_retrieval_leakage():
    report = validate(retrieval_ids={"gold-1"})
    assert not report.valid
    assert any("retrieval" in error for error in report.errors)


def test_rejects_cross_partition_duplicates():
    report = validate(duplicate_groups=[{"gold-1", "rag-1"}])
    assert not report.valid
    assert any("Duplicate group" in error for error in report.errors)


def test_warns_when_duplicate_groups_are_missing():
    report = validate(duplicate_groups=None)
    assert report.valid
    assert any("not verified" in warning for warning in report.warnings)


def test_rejects_duplicate_selected_ids():
    result = make_result()
    result.selected.append(result.selected[0])

    report = validate(result)
    assert not report.valid
    assert any("not unique" in error for error in report.errors)


def test_incomplete_selection_fails_when_required():
    report = validate(require_complete=True)
    assert not report.valid
    assert any("expected" in error for error in report.errors)


def test_incomplete_selection_is_allowed_in_draft():
    report = validate(require_complete=False)
    assert report.valid
    assert any("shortages" in warning for warning in report.warnings)


def test_complete_selection_requires_duplicate_groups():
    report = validate(
        duplicate_groups=None,
        require_complete=True,
    )

    assert not report.valid
    assert any(
        "Near-duplicate isolation not verified" in error for error in report.errors
    )


def complete_fixture():
    result = GoldenSelectionResult(
        available_counts=dict(TARGET_COUNTS),
        manifest_id="fixture-manifest",
        source_revision="fixture-revision",
    )
    for category, count in TARGET_COUNTS.items():
        for index in range(count):
            result.selected.append(
                GoldenCandidate(
                    f"{category}-{index}",
                    category,
                    0.6 if category == "disputed" else 0.8,
                    10,
                )
            )
    held = frozenset(candidate.case_id for candidate in result.selected)
    manifest = SplitManifest(
        manifest_id="fixture-manifest",
        source_revision="fixture-revision",
        held_out_ids=held,
        development_ids=frozenset({"dev"}),
        retrieval_ids=frozenset({"rag"}),
        partition_sizes=(200, 1, 1),
        exact_groups=(),
        semantic_groups=(),
    )
    return result, manifest


def test_complete_zero_match_checks_require_a_manifest():
    result, manifest = complete_fixture()
    strict = validate_golden_selection(result, manifest=manifest, require_complete=True)
    assert strict.valid and strict.source_revision == manifest.source_revision
    legacy = validate_golden_selection(
        result,
        held_out_ids=set(manifest.held_out_ids),
        development_ids=set(),
        retrieval_ids=set(),
        duplicate_groups=[],
        require_complete=True,
    )
    assert not legacy.valid
    assert any("provenance" in error for error in legacy.errors)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"semantic_groups": None}, "Semantic"),
        ({"exact_groups": None}, "Exact"),
        ({"source_revision": " "}, "source revision"),
        ({"manifest_id": ""}, "manifest ID"),
        ({"partition_sizes": (200, 1, 40)}, "inventories"),
    ],
)
def test_incomplete_evidence_cannot_certify_complete_selection(changes, message):
    result, manifest = complete_fixture()
    report = validate_golden_selection(
        result, manifest=replace(manifest, **changes), require_complete=True
    )
    assert not report.valid
    assert any(message in error for error in report.errors)


def test_201st_info_candidate_fails_strict_with_correct_known_quotas():
    result, manifest = complete_fixture()
    result.selected.append(GoldenCandidate("extra", "INFO", 1, 10))
    manifest = replace(
        manifest,
        held_out_ids=manifest.held_out_ids | {"extra"},
        partition_sizes=(201, 1, 1),
    )
    errors = validate_golden_selection(
        result, manifest=manifest, require_complete=True
    ).errors
    assert any("Unknown candidate category" in error for error in errors)
    assert any("200 unique" in error for error in errors)


@pytest.mark.parametrize(
    "changes",
    [
        {"n_valid_comments": 2},
        {"community_agreement": 0.5},
        {"n_valid_comments": True},
        {"n_valid_comments": 11},
        {"community_agreement": float("nan")},
    ],
)
def test_ineligible_quota_filling_nta_rejected(changes):
    result, manifest = complete_fixture()
    result.selected[0] = replace(result.selected[0], **changes)
    assert not validate_golden_selection(
        result, manifest=manifest, require_complete=True
    ).valid


def test_duplicate_selected_cluster_and_transitive_cross_scan_leakage():
    result, manifest = complete_fixture()
    manifest = replace(
        manifest,
        exact_groups=(frozenset({"NTA-0", "NTA-1"}),),
        semantic_groups=(frozenset({"NTA-1", "rag"}),),
    )
    errors = validate_golden_selection(
        result, manifest=manifest, require_complete=True
    ).errors
    assert any("multiple selected" in error for error in errors)
    assert any("crosses partitions" in error for error in errors)


def test_transitive_cluster_links_two_selected_via_unselected_held_out():
    result, manifest = complete_fixture()
    manifest = replace(
        manifest,
        held_out_ids=manifest.held_out_ids | {"bridge"},
        partition_sizes=(201, 1, 1),
        exact_groups=(frozenset({"NTA-0", "bridge"}),),
        semantic_groups=(frozenset({"bridge", "ESH-0"}),),
    )
    report = validate_golden_selection(result, manifest=manifest, require_complete=True)
    assert any("multiple selected" in error for error in report.errors)


def test_draft_with_missing_scan_warns_without_certifying_it():
    result, manifest = complete_fixture()
    report = validate_golden_selection(
        result, manifest=replace(manifest, semantic_groups=None)
    )
    assert report.valid
    assert "Semantic duplicate scan not verified" in report.warnings


@pytest.mark.parametrize("revision", [None, "stale-revision"])
def test_strict_requires_selection_bound_to_same_snapshot(revision):
    result, manifest = complete_fixture()
    result.source_revision = revision
    if revision is None:
        result.manifest_id = None
    report = validate_golden_selection(result, manifest=manifest, require_complete=True)
    assert not report.valid
    assert any("manifest" in error for error in report.errors)
