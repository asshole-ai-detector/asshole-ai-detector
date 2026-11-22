
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
- Missing-context candidates are nominated for manual review,
  not automatically confirmed.
- Categories are mutually exclusive.
- Duplicate post IDs are rejected.
- Explicitly excluded post IDs are not selected.
- Selection uses a configurable random seed.

If insufficient candidates exist, the module reports shortages
rather than filling categories with invalid examples.

## JSON Artifacts

Two export functions are provided:

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

# cases must come from the isolated DATA-04 held-out pool.
result = select_golden_candidates(cases, seed=42)

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

The current selection module does not independently verify
partition provenance or near-duplicate exclusion.

These guarantees depend on DATA-04 integration.

## Reproducibility

The selection seed is recorded in both JSON artifacts.

The final integrated workflow must additionally record:

- Dataset revision
- Split manifest and selected IDs
- Deduplication/grouping decisions
- Preprocessing version
- Selection code version

The current JSON exports are preliminary and do not yet
provide a complete reproducibility manifest.

## Current Status

Implemented:
- Deterministic candidate selection
- Category quotas and shortage reporting
- Duplicate-ID validation
- Explicit ID exclusions
- Provisional missing-context review category
- JSON candidate and report exports
- Unit tests

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
