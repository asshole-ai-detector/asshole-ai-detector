# DATA-04 / DATA-05 — Combined stratified AITA preprocessing

This version uses the **same cleaning, 70% consensus rule and exact deduplication** as before. The change is **which accepted records are eligible for Golden and development**.

## Why this change?

The original Hugging Face test split contained only 51 eligible ESH and 80 eligible NAH consensus posts after cleaning. Across both original splits there are approximately 276 ESH and 409 NAH cases. Restricting Golden and development to the original test split left almost no rare-class development examples or reserve Golden candidates.

The combined strategy cleans original `train` and `test` separately, **deduplicates across their union**, and then allocates cases from that union using reproducible category minimums. It **replaces the provider's original test split as an untouched benchmark**, so the change must be agreed upon and documented before freezing final evaluation.

## Configuration

`configs/default.yaml` contains:

- `splits.strategy: combined_stratified` (new default for the CLI)
- `splits.golden_candidate_pool_size: 800`
- `splits.development_size: 500`
- `splits.golden_candidate_minimums` (NTA 40, YTA 40, ESH 60, NAH 60, DISPUTED 20)
- `splits.development_minimums` (NTA 50, YTA 50, ESH 40, NAH 50, DISPUTED 50)
- `splits.golden_targets`: unchanged final EVAL-01 200-case targets

The category minimums are preliminary reserves, **not fixed distributions** or guaranteed human-approved cases. The remainder is uniformly shuffled with a fixed seed, after reserving both partitions' minimums. If any quota cannot be satisfied, the pipeline fails with the eligible category counts instead of silently compromising the Golden Dataset.

To preserve legacy behavior, callers of `split_cases()` or `prepare()` that omit the new strategy argument still default to `provider_test`. The CLI explicitly reads `combined_stratified` from YAML.

## Run (repository root)

```bash
uv sync --group data
uv run pytest
uv run ruff check .
uv run python scripts/prepare_data.py --output-dir artifacts/data-combined-v1
```

On Windows PowerShell, run the final command on one line. The output folder must be empty or absent. **Never overwrite or mix older partitions with a new run**. Use `--config`, `--seed`, `--golden-size` or `--development-size` only when intentionally changing the experiment configuration.

`prepare()` accepts Hugging Face datasets directly because `iter_normalized_cases()` accepts an iterable of mappings and HF datasets yield dictionaries on iteration. For a single raw row, use `row_to_normalized(row)`.

## Artifacts and behavior

- `golden_pool.jsonl`: deduplicated provisional held-out cases (800 by default)
- `golden_source_rows.jsonl`: original **train or test** normalized rows for each selected Golden case; includes INFO comments for manual review
- `development.jsonl`: 500 disjoint development cases
- `rag_candidates.jsonl`: all other accepted cases, **RAG-eligible**, not automatically indexed or capped at 20,000
- `unused_test.jsonl`: **empty under `combined_stratified`**, retained to maintain the current `SplitResult` and loader interface. Under legacy `provider_test`, remaining provider-test records go here
- `split_manifest.json`: IDs, source revision, policy and current exact/semantic scan status
- `preprocessing_report.json`: counts, category distributions, quotas, strategy, source partition counts, rejection and deduplication stats

The `manifest_id` binds contents, seed, quota settings, strategy and source revision. Only a **matching new manifest** can be used for downstream outputs. The original source split provenance remains visible in the report, though original provider train/test is no longer the evaluation boundary.

## EVAL-01 handoff

```python
from pathlib import Path
from asshole_ai_detector.data.pipeline import load_prepared
from asshole_ai_detector.evaluation.golden_selection import select_golden_candidates
from asshole_ai_detector.evaluation.golden_annotation_prep import prepare_annotation_records
from asshole_ai_detector.evaluation.golden_validation import validate_golden_selection

prepared = load_prepared(Path("artifacts/data-combined-v1"))
selection = select_golden_candidates(
    prepared.splits,
    missing_context_review_ids=set(),  # coordinator adds real nominations
)
report = validate_golden_selection(selection, manifest=prepared.splits.manifest)
review_records = prepare_annotation_records(selection, prepared.splits.held_out)
# Original source rows for any Golden case: prepared.golden_sources[case_id]
```

A blank missing-context nomination set gives a **preliminary shortage**. EVAL-01 selects final provisional cases and EVAL-02 performs independent human review and adjudication. The held-out 800 cases are **not** themselves the final Golden Dataset.

## Downstream contracts and leakage prevention

- RAG-02 must rebuild/version the ChromaDB index from **new** `rag_candidates.jsonl` and use the matching split manifest; do not keep vectors from an old partition where they may now belong to Golden/dev.
- RAG-03's retrieval interface is unchanged; point it at the rebuilt RAG-02 index and case store.
- CLF-01/CLF-02 currently train on Scruples, so their initial training code does not change. If any model is trained or tuned on AITA, **exclude both Golden and development from fitting** and use development for tuning only.
- RAG-04 must run semantic near-duplicate leakage checking on the updated partitions. `semantic_groups=None` means **not yet scanned**, never 'clean'. Fix cross-partition near-duplicates before freezing the evaluation assets.
- Golden cases must never enter training, threshold tuning, or RAG indexing; development cases must never enter training or RAG indexing. Selection from the same source corpus does not by itself guarantee semantic isolation.

**Privacy:** Original Reddit narratives and provisional evaluation metadata are private working artifacts. Keep everything in Git-ignored `artifacts/`; never commit datasets, indexes, model checkpoints, secrets or review assignments.

## Known limitations

- No new stories are created by stratification; ESH and NAH remain minority classes.
- The Golden pool is intentionally class-balanced toward uncommon labels; report class-conditional metrics separately from natural-frequency performance.
- Human annotation and semantic de-duplication are not performed by DATA-05.
- The proposed quotas are starting values requiring approval from EVAL-01 and the team; freeze them before final experiments.
