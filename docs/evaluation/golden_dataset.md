
# EVAL-01 — Golden Dataset Candidate Selection

## Overview

EVAL-01 selects up to 200 preliminary candidates for final system evaluation
from DATA-04's deduplicated held-out pool. Each selected case awaits independent
human annotation and adjudication.

## Target Distribution

| Category | Target |
|---|---:|
| NTA | 40 |
| YTA | 40 |
| ESH | 40 |
| NAH | 40 |
| `disputed` (community disagreement) | 20 |
| `missing_context` (human-nominated for review) | 20 |
| **Total** | **200** |

## Input

DATA-04 deduplicates validated DATA-02/DATA-03 cases across the combined
provider train and test inputs. It draws the held-out pool (default 800) and
development set (default 500) from provider test cases. Retained provider
train cases supply retrieval. Surplus test cases remain unused.

Pass the resulting `SplitResult` to `select_golden_candidates`. It reads
`split.golden_candidates` (an alias for `split.held_out`) and takes the manifest ID
and pinned source revision directly from `split.manifest`. The selector checks
the full pool against the manifest before applying nominations, exclusions,
or quotas. A missing, extra, repeated, or cross-partition pool ID raises an error.
Keep the cases together with the manifest returned by DATA-04.

## Selection Rules

- All categories require 5–10 valid moral comments from the original top ten
  ranked slots. DATA-04's minimum consensus-label counts reserve candidates.
  EVAL-01 counts eligible cases after applying exclusions and nominations.
- A winning share of at least 70% qualifies for the corresponding moral verdict.
  Below 70%, DATA-04's `DISPUTED` label maps to EVAL-01's `disputed` category.
- INFO comments are excluded from moral consensus.
- For EVAL-01, a coordinator supplies human nominations by held-out post ID.
  A nomination outside the 5–10 valid moral comment range stops selection
  from a DATA-04 split with an error identifying the case and its count.
  Only DATA-02-accepted cases enter this pool (at least five moral comments).
  Reserve nominees for missing-context review before sampling other categories,
  even if a nominee has strong consensus. INFO or disagreement alone never
  nominates a case. Record the nomination reason privately. A separate path
  for low-vote cases would require team approval and DATA-04 isolation.
- Categories are mutually exclusive.
- Duplicate post IDs are rejected.
- Explicitly excluded post IDs are not selected.
- Selection uses its own configurable random seed (default 42).

The report gives the shortfall for each category when the eligible pool cannot
fill its quota.

## Run selection and preliminary validation

Start with validated `train_cases` and `test_cases` from the same pinned
`source_revision`, the original `provider_test_ids`, and a coordinator's set of
human-nominated held-out IDs (`nominated_ids`, possibly empty). If DATA-04 has
already supplied `split`, start at the selection call.

```python
from pathlib import Path

from asshole_ai_detector.data.deduplication import deduplicate_cases
from asshole_ai_detector.data.splits import split_cases
from asshole_ai_detector.evaluation.golden_selection import (
    export_golden_candidates,
    export_selection_report,
    select_golden_candidates,
)
from asshole_ai_detector.evaluation.golden_validation import validate_golden_selection

deduplicated = deduplicate_cases(
    [*train_cases, *test_cases], preferred_ids=provider_test_ids
)
split = split_cases(
    deduplicated.cases, test_ids=provider_test_ids, source_revision=source_revision
)
result = select_golden_candidates(
    split, missing_context_review_ids=nominated_ids, seed=42
)
report = validate_golden_selection(result, manifest=split.manifest)
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

Both exports are coordinator-only. They contain candidate metadata or aggregate
counts and omit narratives and comments. Each export requires a new output path
unless the coordinator explicitly requests overwrite.

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

Preliminary mode permits shortages and warns about pending scan evidence.
Existing callers can select from a list of validated cases and pass three ID
sets and optional duplicate groups to validation. These callers establish pool
isolation themselves. Use the integrated path for a DATA-04 selection.

The shared `asshole_ai_detector.data.splits.SplitManifest` is authoritative.
Use the same manifest for selection and validation. Validation checks the
manifest ID, source revision, the three disjoint ID inventories, and their sizes
in held-out, development, retrieval order.

DATA-04 returns `exact_groups=()` after verifying exact uniqueness of retained
cases. Its initial `semantic_groups=None` means semantic isolation is unverified.
Strict validation fails in this state even when all 200 places are filled.
Only a completed scan of the full held-out, development, and retrieval
inventories can establish a verified empty semantic result (`()`).

Once the coordinator has verified scan results for this snapshot,
`attach_semantic_scan` attaches those findings to a copy of the same manifest.
Use that manifest throughout the strict run.

```python
from dataclasses import replace

from asshole_ai_detector.data.splits import attach_semantic_scan

scanned_manifest = attach_semantic_scan(split.manifest, verified_groups)
scanned_split = replace(split, manifest=scanned_manifest)
result = select_golden_candidates(
    scanned_split, missing_context_review_ids=nominated_ids, seed=42
)
strict = validate_golden_selection(
    result, manifest=scanned_split.manifest, require_complete=True
)
if strict.errors:
    raise ValueError(strict.errors)
```

Strict validation requires exactly 200 unique eligible candidates in the six
quotas and complete inventory counts. It connects overlapping duplicate groups
across both scans, rejects groups with unknown IDs or cross-partition members,
and permits at most one selected candidate per connected group.

A valid strict report verifies candidate selection only. Final annotations still
require two independent human judgments per case, human confirmation of
missing-context labels, and adjudication. The coordinator remains
responsible for verifying the source and scan coverage, documenting any training
overlap checks, and enforcing index and prompt exclusions.

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
from selected candidates and validated `AITACase` objects. Use
`prepare_annotation_records(result, split.golden_candidates)` for this batch.

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


## Annotation batches (EVAL-02)

`golden_annotation_review.py` tracks independent review, agreement and progress.
Every verdict and field comes from a human reviewer; the module only checks,
stores and counts them. Keep all files below in `artifacts/annotations/`
(ignored by Git) or other restricted storage.

| Step | Who | Tool | Output (private) |
|---|---|---|---|
| 1. Issue narratives | Coordinator | `export_blinded_assignments` | assignment file + private assignment-to-case map |
| 2. Assign two reviewers per case, fixed order | Coordinator | `BatchPlan(batch_id, {assignment_id: (first, second)})`, saved with `to_dict()` | batch plan |
| 3. Hand out answer files | Coordinator | `export_review_template(plan, reviewer, path)` | one empty file per reviewer |
| 4. Annotate privately | Each reviewer | edit their own file only | filled reviewer file |
| 5. Lock originals before comparing | Coordinator | `lock_submissions(files, lock_path)` | lock file with SHA-256 hashes |
| 6. Agreement before discussion | Coordinator | `load_locked_submissions`, `pair_submissions`, `compute_agreement` | percentage agreement, Cohen's kappa, disagreement list |
| 7. Resolve and finalize | Adjudicator | `build_golden_record` + `validate_completed_annotation` (with `Adjudication` if the category changed) | final `GoldenAnnotation` records |
| 8. Report | Coordinator | `build_progress_report` | batch and category progress |

Rules enforced in code:

- A reviewer file is valid only when every field is filled: verdict
  (NTA/YTA/ESH/NAH/UNDETERMINED), at least one key fact, framing bias,
  uncertainty tier, initial action and notes. An unfilled template fails.
- Each assignment has exactly two different reviewers; answers from unassigned
  reviewers, unknown assignments or other batches are refused.
- A locked file that changes afterwards is refused, so the original answers stay
  available for agreement analysis.
- Kappa uses the original verdicts, with UNDETERMINED as its own label, and is
  reported as undefined (with a note) when both reviewers used one identical label.
- Final records take `annotator_verdicts` from the locked submissions and the
  narrative and community votes (including INFO) from the pinned source.
- Each case counts once, under one final category, so missing-context cases are
  never counted under a verdict as well. The progress report lists assignment
  IDs, not source case IDs.

### Week 8 batch

Planning target: 40–60 double-annotated cases (final 200 in Week 10). Record the
batch here when it is agreed:

| Field | Value |
|---|---|
| Batch ID | _to agree_ |
| Batch size and category mix | _to agree_ |
| Reviewers and pairing | _to agree_ |
| Adjudicator | _to agree_ |
| Deadline | _to agree_ |

Choose the batch mix with the remaining category counts in mind (for example,
ESH and NAH candidates are scarce). Golden candidate IDs stay excluded from RAG
indexing, threshold tuning and model selection throughout.

## Current Status

The integrated selection, validation, and review-preparation workflow is tested
with synthetic cases. A final dataset still requires a real pinned split, a
verified semantic scan, sufficient eligible candidates and human nominations,
independent annotation, adjudication, final quota and source checks, and a
dataset freeze.

## References

- `docs/asshole-ai-detector-milestone-2.md`
- `docs/architecture/technical-specifications.md`
- `src/asshole_ai_detector/schemas/annotation.py`
- `src/asshole_ai_detector/schemas/case.py`
