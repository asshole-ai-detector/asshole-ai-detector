# DATA-05 — Real AITA preprocessing integration

Copy `scripts/prepare_data.py` and `tests/integration/test_prepare_data.py` into the corresponding locations in your repository.

## Why HF Dataset works with the adapter

The adapter signature is `iter_normalized_cases(rows: Iterable[Mapping[str, Any]])`. **It does not expect a single mapping**; it expects an iterable of mappings. Hugging Face `Dataset` yields `dict` rows when iterated, so a dataset can be passed directly. For a single row, use `row_to_normalized(row)`.

## Execute

```bash
uv sync --group data
uv run pytest tests/integration/test_prepare_data.py -v
uv run ruff check scripts/prepare_data.py tests/integration/test_prepare_data.py
uv run python scripts/prepare_data.py
```

Run from the repository root. `scripts/prepare_data.py` calls DATA-01..04 but **does not select the final 200 cases** (EVAL-01 does that). The `--output-dir` default is `artifacts/data/`, which is ignored by Git. It refuses to overwrite a nonempty output directory. Set `--output-dir artifacts/data-run2` for subsequent runs.

The script loads the `train` and `test` HF provider splits individually to preserve their membership, normalizes/cleans/labels the data, deduplicates all eligible cases together, then calls DATA-04 to produce: held-out Golden candidate pool (default 800), development (500), RAG candidates, and unused test records. It saves the source test rows needed for EVAL-01 to verify INFO votes.

Outputs:

- `golden_pool.jsonl`: deduplicated held-out `AITACase` objects from provider test
- `golden_source_rows.jsonl`: corresponding original `NormalizedAITACase` rows (INFO comments retained)
- `development.jsonl`: held-out cases for tuning
- `rag_candidates.jsonl`: candidate retrieval source records, **not** capped at 20,000
- `unused_test.jsonl`: remaining eligible provider test records
- `split_manifest.json`: provenance, partition inventories, exact/semantic scan status
- `preprocessing_report.json`: counts, rejection reasons, duplicate totals, per-label splits

`prepare()` accepts synthetic iterables and a custom `golden_minimums` dict for testing. The normal CLI uses DATA-04's default Golden minimums. If the real pinned dataset has fewer eligible rare-label examples than expected, splitting raises with label counts; coordinate quota/sampling changes with EVAL-01, rather than silently relaxing constraints.

## EVAL-01 handoff

After the run, reload the split and original rows (from repository root):

```python
from pathlib import Path
from scripts.prepare_data import load_prepared
from asshole_ai_detector.evaluation.golden_selection import select_golden_candidates
from asshole_ai_detector.evaluation.golden_annotation_prep import prepare_annotation_records
from asshole_ai_detector.evaluation.golden_validation import validate_golden_selection

prepared = load_prepared(Path("artifacts/data"))
split = prepared.splits
selection = select_golden_candidates(split, missing_context_review_ids=set())
report = validate_golden_selection(selection, manifest=split.manifest)
review_records = prepare_annotation_records(selection, split.held_out)
# Coordinator supplies real missing_context_review_ids after screening;
# source rows are in prepared.golden_sources (keyed by ID).
```

`missing_context_review_ids=set()` produces a preliminary shortage for that category. **Do not treat this as the complete Golden Dataset**. Human nominations and semantic near-duplicate checks must be completed before strict validation/freeze. The record's original source can be wrapped in `AnnotationSource` using the pinned `source_revision` for blinded assignments.

**Privacy:** Output contains original Reddit narratives and provisional evaluation metadata. Keep it in `artifacts/` or other private ignored storage. Never commit these outputs to Git.

## Known limitations

- This is integration, not semantic near-duplicate detection (RAG-04 owns that).
- EVAL-01 chooses 200 provisional candidates, including the 20 *human-nominated* missing-context examples.
- EVAL-02 performs independent human annotation/adjudication.
- Current DATA-04 can reject the real split if rare-label quotas cannot be satisfied. Record and address any shortage with the team.
- This bundle has been syntax checked, but not executed against the full remote repository or Hugging Face dataset in this environment. Validate with your repo's `uv run pytest` and Ruff before merging.
