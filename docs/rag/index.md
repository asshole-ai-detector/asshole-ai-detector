# RAG-02 preliminary persistent index

A build consumes a complete local DATA-05 export and publishes one immutable generation under `artifacts/rag-02/`. Each selected submission has one cosine vector in Chroma and one complete validated `AITACase` in a JSONL case store. The index is for development experiments. Golden evaluation requires a subsequent RAG-04 semantic near-duplicate scan, resolution of exclusions, and a newly verified evaluation artifact.

## Build with the real model

Use Python 3.12 and the repository lockfile. The `index` dependency group provides persistent ChromaDB. The existing `rag` group supplies the SentenceTransformer backend. Run from the repository root with the complete DATA-05 export available locally.

```sh
uv sync --frozen --group index --group rag
uv run --group index --group rag python scripts/build_index.py \
  --prepared-dir artifacts/data-combined-v1
```

If preparing real data for the first time, the existing DATA-05 command is `uv run --group data python scripts/prepare_data.py --output-dir artifacts/data-combined-v1`. It loads the configured dataset and requires a new or empty output directory. A real build needs the pinned Qwen weights locally or access to download them. The automated suite and synthetic demo run offline after dependency installation.

`--config` selects the YAML configuration. The CLI checks the export against `data_sources.aita.hf_dataset` and `revision`. Defaults are `MattBoraske/reddit-AITA-submissions-and-comments-multiclass` at `84c229ffd926e8efaa1828a5d3803f29231f2d0e` and `Qwen/Qwen3-Embedding-0.6B` at `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. The CLI prints the generation path and JSON report. `--index-root`, `--seed`, and `--max-per-top-verdict` override the output root, sampling seed, and class cap. Keep local artifacts in Git-ignored storage.

## Input contract and selection

1. `load_prepared(prepared_dir, strict=True)` reads `split_manifest.json`, `preprocessing_report.json`, and the five DATA-05 JSONL files. It verifies typed case records, raw ID inventories, counts, disjointness, report provenance, and the DATA-04 content fingerprint including the split seed, quotas, and all four partitions. Strict verification is opt-in for DATA-05 callers. Both current `combined_stratified` and `provider_test` exports are supported. Re-export older data if the strategy or fingerprint inputs are missing.
2. Candidate IDs must exactly match `retrieval_ids` before sampling. The complete preliminary Golden pool corresponds to `held_out_ids`. It and `development_ids` are excluded regardless of the original HF split. Legacy `unused_test` records are also excluded. An export reporting unresolved semantic groups requires upstream resolution. The DATA-05 value `semantic_groups: null` records a pending scan.
3. Group by the validated plurality `derived_fields.top_verdict`. Within each verdict, keep `consensus_label == "DISPUTED"` first. If this group exceeds the cap, draw the cap uniformly without replacement from that group. Otherwise draw consensus cases to fill remaining slots. Sampling uses sorted post IDs, the schema's verdict order (NTA, YTA, ESH, NAH), and one local seeded Python RNG. Write the final selection in post-ID order. The defaults are seed 42, up to 5,000 per verdict and 20,000 total. For a smaller experiment, lower the per-class cap. The configured total must accommodate all four quotas.
4. RAG-01 embeds `title + "\n" + text` with no document prefix. It retains the first 700 and last 324 tokens at the 1,024-token cap. Comments go into the case store only. The writer verifies batch cardinality and finite, 1,024-dimensional, L2-normalized vectors before insertion. Chroma receives explicit embeddings, unique post IDs, and scalar string metadata for `post_id`, `top_verdict`, and `consensus_label`.

Strict loading recomputes consensus-label counts from all four exported partitions. With `provider_test`, partition membership also determines the original provider split. With `combined_stratified`, the export contains only aggregate provider counts. Strict loading checks their types, row totals, eligible-test totals and bounds against the available input rows. Verifying individual provider origins would require upstream per-ID provenance.

## Files, publication, and recovery

| File under `artifacts/rag-02/<build-id>/` | Contents |
|---|---|
| `chroma/` | Local persistent Chroma database with the cosine collection `aita-preliminary` and its build ID |
| `cases.jsonl` | One UTF-8 `AITACase.model_dump_json()` line per selected ID, including derived fields, URL, date, all valid ranked comments, and provider diagnostics |
| `index_manifest.json` | Build ID, format version, collection/metric, Chroma version, source and split provenance, cleaning settings, preprocessing report schema and checksum, embedding manifest, sampling settings, checksums of selected cases and stored vectors, counts, UTC timestamp, and embedding batch size |
| `build_report.json` | Eligible, selected, and cap-excluded counts per verdict and consensus label, held-out/dev/unused counts, verified ID overlap, and upstream semantic-scan status |

The build ID is SHA-256 over canonical JSON in the manifest's `inputs` field. The timestamp and execution batch size are recorded outside that identity. The case checksum covers the exact JSONL bytes. The vector checksum covers little-endian float32 rows in selected ID order. Reproducibility applies to membership and input identity. Different precision or hardware can change floating-point outputs. DATA-05 exports include a report schema and split fingerprint but omit the producer's Git commit.

A writer builds in a private sibling directory, closes Chroma, reopens it, checks the collection, every vector and metadata row, and the exact case-store contents against verified source cases. It then publishes by a same-filesystem directory rename. On local macOS and Linux filesystems the first writer publishes a nonempty directory. A competing rename fails with `ENOTEMPTY` or `EEXIST`, and that writer verifies the published generation before returning it. A repeated build validates the existing generation before returning it, without loading the model. Changed membership, payloads, model settings, sampling, or preprocessing provenance select a fresh directory. `open_index` resolves only the generation for the supplied current export and settings.

A failed validation leaves the published generation untouched. A hard interruption can leave a `.building-*` directory, which readers ignore. Rerun the same command to retry. If a published generation is damaged, preserve it for diagnosis and build in a fresh ignored root. Old generations stay available to explicitly matched historical exports. No automatic pruning takes place.

## Public API and RAG-03

`build_index(prepared_dir, index_root, *, embedding_config, sampling, expected_source)` returns the published generation path. The default loader uses RAG-01. `load_embedder` is an injection point for controlled tests. `open_index(...)` is a context manager that verifies the same export, model settings, and sampling configuration. Keep its handle open across queries to amortize the full startup audit. The byte-offset map provides direct case lookup. `get_case(id)` returns the validated complete case or `None` for an unknown ID.

With the RAG-03 implementation available, a query uses the existing callable and collection contracts directly.

```python
from pathlib import Path

from asshole_ai_detector.rag.embeddings import EmbeddingConfig, load_qwen3_embedder
from asshole_ai_detector.rag.index import open_index
from asshole_ai_detector.rag.retrieval import retrieve_dense

config = EmbeddingConfig.from_yaml()
embedder = load_qwen3_embedder(config)
with open_index(Path("artifacts/data-combined-v1"), embedding_config=config) as index:
    hits = retrieve_dense(
        "a disagreement with my roommate about rent",
        embed_query=lambda query: embedder.embed_queries([query])[0],
        index=index.collection,
        get_case=index.get_case,
    )
```

Use the same `SamplingConfig` on both calls if you override sampling. The RAG-01 query interface supplies the instruction prefix, token cap and normalization. RAG-03 requests cosine distances and resolves result IDs through the case store. Treat the opened collection as read-only. Keep the matching DATA-05 export with the generation when transferring it to another machine. Opening loads the preparatory partitions and scans the selected vectors once, so allow memory and startup time for the full export.

## Offline demonstration and tests

```sh
uv sync --frozen --group index
uv run --group index python scripts/demo_index.py
uv run --group index pytest
```

The demo creates 24 invented posts, uses DATA-05 combined splitting, and uses a deterministic 1,024-dimensional topic encoder through RAG-01 with a synthetic model identity. It publishes under `artifacts/rag-02-demo/`, launches a second Python process, reads metadata and full comments, checks every excluded ID and queries Chroma. The parent repeats the build with a loader that raises if called. Use `--rag03` to require the real dense-retrieval function.

PR #31's `retrieve_dense` is absent from the inspected `main`. To reproduce the integration before that PR lands, use the following Bash or Zsh commands with the PR commit available locally. The isolated checkout uses the current RAG-02 files, the unchanged retrieval module and tests from the pinned PR, and the already-installed project interpreter.

```sh
rag02_python="$PWD/.venv/bin/python"
rag02_checkout="$(mktemp -d)/checkout"
rag03_revision=6c2c7c5dc03b99f95b24103001a3bbf8bd6c4d43
git clone --quiet --local --no-hardlinks . "$rag02_checkout"
git -C "$rag02_checkout" restore --source="$rag03_revision" -- \
  src/asshole_ai_detector/rag/retrieval.py tests/unit/test_retrieval.py
rsync -aR src/asshole_ai_detector/data/{pipeline,splits}.py \
  src/asshole_ai_detector/rag/index.py scripts/{build_index,demo_index}.py \
  tests/integration/{test_prepare_data,test_persistent_index}.py "$rag02_checkout/"
(cd "$rag02_checkout" && PYTHONPATH=src "$rag02_python" scripts/demo_index.py --rag03)
(cd "$rag02_checkout" && PYTHONPATH=src "$rag02_python" -m pytest -q \
  tests/unit/test_retrieval.py tests/integration/test_persistent_index.py)
```

The integration fixture checks process persistence, exact set and payload equality, deterministic reuse, incompatible builds, damaged stores, interruption, and concurrent publication. The synthetic retrieval result exercises wiring and evidence lookup. Real AITA retrieval quality and semantic overlap require separate experiments.
