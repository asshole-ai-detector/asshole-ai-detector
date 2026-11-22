
# EVAL-01 — Golden Dataset Candidate Selection

## Overview

This module prepares preliminary candidates for the 200-example
AITA Golden Dataset used in final system evaluation.

It implements deterministic candidate selection and JSON exports.
It does not perform human annotation or dataset splitting.

## Target Distribution

| Category | Target |
|---|---:|
| NTA | 40 |
| YTA | 40 |
| ESH | 40 |
| NAH | 40 |
| Community disagreement | 20 |
| Insufficient information | 20 |
| **Total** | **200** |

## Input

The selection function accepts validated `AITACase` objects
produced by the existing DATA-02/DATA-03 pipeline.

DATA-04 must provide a deduplicated, isolated held-out
candidate pool before production selection can be performed.

## Selection Rules

- Consensus cases require at least five valid moral comments
  and at least 70% agreement.
- Disputed cases require at least five valid moral comments
  and less than 70% agreement.
- INFO comments are excluded from moral consensus.
- For EVAL-01, a coordinator supplies human nominations by held-out post ID.
  Only DATA-02-accepted cases enter this pool (at least five moral comments).
  Reserve nominees for missing-context review before sampling other categories,
  even if a nominee has strong consensus. INFO or disagreement alone never
  nominates a case. Record the nomination reason privately. A separate path
  for low-vote cases would require team approval and DATA-04 isolation.
- Categories are mutually exclusive.
- Duplicate post IDs are rejected.
- Explicitly excluded post IDs are not selected.
- Selection uses a configurable random seed.

If insufficient candidates exist, the module reports shortages
rather than filling categories with invalid examples.

## JSON Artifacts

Selection produces two coordinator artifacts:

- `export_golden_candidates`: preliminary candidate metadata.
- `export_selection_report`: counts, targets, and shortages.

Example:

```python
from pathlib import Path

from asshole_ai_detector.evaluation.golden_selection import (
    export_golden_candidates,
    export_selection_report,
    select_golden_candidates,
)

from asshole_ai_detector.evaluation.golden_validation import validate_golden_selection

# The loader and nomination register are supplied by the coordinator.
result = select_golden_candidates(
    cases, missing_context_review_ids=nominated_ids, seed=42,
    manifest_id=manifest.manifest_id, source_revision=manifest.source_revision,
)
report = validate_golden_selection(result, manifest=manifest)
if report.errors:
    raise ValueError(report.errors)
# Record report.warnings with the preliminary outputs.

export_golden_candidates(
    result,
    Path("artifacts/annotations/golden_candidates.json"),
)

export_selection_report(
    result,
    Path("artifacts/annotations/golden_selection_report.json"),
)
```

The `cases` variable is illustrative; a DATA-04 loader is
not yet implemented.

Candidate exports contain IDs and selection statistics,
not narrative or comment text.

Exports do not overwrite existing files unless explicitly
requested.

## Human Annotation

Selected categories are provisional.

Final `GoldenAnnotation` records require:

1. Two independent annotator verdicts.
2. Key facts and missing-information assessment.
3. Alternative perspectives and framing-bias assessment.
4. Annotation notes and uncertainty expectations.
5. Adjudication and schema validation.

The existing `GoldenAnnotation` schema is the authoritative
contract for final annotations.

The exact review protocol and inter-annotator agreement
workflow require team confirmation.

## Leakage Prevention

The Golden Dataset must remain separate from development
and retrieval partitions.

Golden cases, their near-duplicates, labels, and comments
must not be used for model selection, threshold tuning,
or retrieval evidence.

### Preliminary and strict checks

`validate_golden_selection` checks recognized categories, integer moral counts,
feasible winning shares, canonical quotas, unique IDs, partition membership,
partition disjointness, and any supplied duplicate groups. Known inventories
must cover all six categories, with nonnegative integer counts at least as large
as their selected counts. The report export rejects missing or inconsistent
inventories. Its `complete` field describes the quota counts only.

Preliminary mode permits shortages. Legacy callers can pass all three ID sets
and optional duplicate groups. Missing inventory or scan evidence is reported
in warnings. Read both errors and warnings: preliminary `valid` means the
checks on supplied evidence passed. Do not use it to release evaluation data.

Use `validate_golden_selection(result, manifest=manifest, require_complete=True)`
for the strict selection gate. It requires 200 unique records in the exact
six quotas, available counts and one trusted DATA-04 snapshot:

- Populate `SplitManifest.manifest_id` and `source_revision` from the actual
  manifest. Supply full held-out, development and retrieval ID sets and
  `partition_sizes` **in that order**, taken independently from the split report.
- Supply `exact_groups` and `semantic_groups` separately as tuples of ID sets.
  Use `None` for a pending or unavailable scan. An empty tuple means a completed
  scan of this inventory found zero matches. Never translate missing output into
  an empty tuple. A legacy `duplicate_groups=[]` establishes neither status.
- Group members must map to these partitions. Supply pairs or groups of at
  least two IDs. Validation connects overlapping groups across both scans,
  rejects components crossing any partition, and allows at most one selected
  Golden ID per component. Upstream must map discarded IDs to representatives
  or resolve them before supplying this inventory.
- Bind the selection to its manifest ID and revision when loading its cases. Strict validation rejects a stale or unbound selection. Keep the manifest with the report. The validator records its ID and revision
  and checks internal consistency. Authenticating its origin, verifying scan
  coverage, obtaining training-data overlap checks where applicable, and
  enforcing the index/prompt exclusions are upstream or downstream responsibilities.

DATA-04 can deliver exact deduplication before semantic verification exists.
A preliminary report can be useful then. Strict validation stays blocked
until actual results for both checks exist. Neither mode certifies human labels.

## Reproducibility

Both selection JSON artifacts record the seed, manifest ID and source revision. For an unbound preliminary selection, both provenance fields are null.

The final integrated workflow must additionally record:

- Dataset revision
- Split manifest and selected IDs
- Deduplication/grouping decisions
- Preprocessing version
- Selection code version

The current JSON exports are preliminary and do not yet
provide a complete reproducibility manifest.


## Annotation Review Preparation

`golden_annotation_prep.py` prepares metadata-only review records
from selected candidates and validated `AITACase` objects.

The preparation step:
- Matches selected candidates to their source cases.
- Rejects missing or duplicate source IDs.
- Checks count, share, original ranks, and proposed consensus/disputed label against the source case.
- Exports a **coordinator-only** review manifest with proposed labels, IDs, and a cleaned-narrative digest. Restrict it even though it omits story and comment text.
- Flags all candidates for source-comment review.
- Separately flags proposed missing-context cases.
- Validates completed human annotations using `GoldenAnnotation`.

Review manifests are not completed Golden annotations.

### Source comment limitation

`AITACase.valid_comments` contains moral-verdict comments only,
while the shared `GoldenAnnotation` schema counts up to ten
top source comments, including INFO.

Therefore, the preliminary review manifest cannot reconstruct
final Golden community votes.

DATA-01 supplies original slots 1–10. DATA-02 preserves surviving moral ranks
and counts valid INFO separately. Deleted, empty, null or unsupported comments
do not contribute votes. Never fill gaps using lower-ranked comments. With the
same source and cleaning policy, restoring INFO leaves the winning moral share
unchanged.

Wrap the pinned normalized DATA-01 row in `AnnotationSource` with the recorded
revision and the team's approved `source_text` or `cleaned_text` representation.
The helper checks original slot bounds/uniqueness and reruns DATA-02 cleaning.
Use the same representation throughout the batch; confirm against §2.3's
original-text requirement before approving a transformation. If the row or
revision is unavailable, stop at metadata preparation.

`export_blinded_assignments(records, sources, path, expected_revision=...)`
joins by ID and checks the cleaned narrative digest and selection statistics.
After a coordinator has screened the narrative itself, issue the resulting
assignment list to reviewers. Only random opaque IDs and narrative text are
included. Save its returned assignment-to-case mapping privately. Preserve the
issued batch; generating a new one creates new IDs. See
[the annotation guidelines](annotation_guidelines.md) for independent review.

### Human review requirements

The following cannot be generated automatically:
- Two independent annotator judgments
- Key facts and missing-information assessment
- Framing-bias assessment
- Alternative perspectives
- Adjudication notes

For structural checks during drafting use `GoldenAnnotation.model_validate`.
For acceptance, call `validate_completed_annotation` with the private
`expected_case_id`, original `expected_category`, `source`, and
`expected_revision` from the same manifest. It checks the chosen narrative and
all five source counts, including INFO. A changed category requires an explicit
`Adjudication(case_id, proposed_category, final_category, rationale)`. Store
that decision alongside the untouched reviewer submissions. Recount final
categories and select replacements from the same held-out pool as needed.
Passing these checks does not establish the independence or quality of human judgments.


## Current Status

Implemented:
- Deterministic candidate selection
- Category quotas and shortage reporting
- Duplicate-ID validation
- Explicit ID exclusions
- Provisional missing-context review category
- JSON candidate and report exports
- Coordinator and blinded review exports, source reconciliation, and regression tests

Pending:
- DATA-04 manifest integration
- Verified near-duplicate and partition exclusion
- Real candidate selection
- Human annotation and adjudication
- Final GoldenAnnotation validation
- Final dataset freeze

## References

- `docs/asshole-ai-detector-milestone-2.md`
- `docs/architecture/technical-specifications.md`
- `src/asshole_ai_detector/schemas/annotation.py`
- `src/asshole_ai_detector/schemas/case.py`
