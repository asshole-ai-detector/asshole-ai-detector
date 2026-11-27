# RAG-01: Qwen3 embedding pipeline

`asshole_ai_detector.rag.embeddings` turns AITA submissions and retrieval queries into vectors
(Milestone 2, Sections 3.3, 3.4 and 3.6). The same `Embedder` is meant for indexing (RAG-02), retrieval
and semantic near-duplicate detection, so every vector comes from one model and revision.

Install: `uv sync --group rag` (adds `sentence-transformers`, `transformers` and `torch`). CI does not install
this group; unit tests use stubs.

## Usage

```python
from asshole_ai_detector.rag.embeddings import load_qwen3_embedder

embedder = load_qwen3_embedder()             # reads `rag` in configs/default.yaml
doc_vectors = embedder.embed_documents(cases)  # e.g. AITACase objects, one vector each
query_vectors = embedder.embed_queries([neutralized_scenario])
embedder.manifest()                          # store this with the index
```

Vectors are returned as `list[list[float]]` in input order (vector `i` belongs to input `i`).

## Rules

| Rule | Implementation |
| --- | --- |
| Model | `Qwen/Qwen3-Embedding-0.6B`, pinned at revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Document text | `title + "\n" + text`, no prefix. Comments are never embedded. |
| Token cap | Texts over 1,024 tokens keep the first 700 and last 324 tokens. The cut is made on token IDs and decoded back to text. |
| Query text | Lowercased (Section 3.5, step 1), capped the same way, then prefixed with `Instruct: Given a description of an interpersonal conflict, retrieve similar conflicts\nQuery: ` |
| Output | 1,024 dimensions, L2-normalized (cosine similarity = dot product). Wrong sizes, zero or NaN vectors raise an error. |
| Batching | One call per list of texts; the model batches internally with `embedding_batch_size` (32). |
| Precision | fp16 on GPU (about 1.2 GB, Section 3.8), fp32 on CPU. |

Configuration lives in `rag` in `configs/default.yaml` (`embedding_model`, `embedding_revision`,
`embedding_query_instruction`, `embedding_batch_size`, `embedding_dimensions` and the
`embedded_post_*` token limits). `Embedder.manifest()` returns the settings that must match between an
index and its queries; batch size is left out because it does not change the vectors.

## Real-model experiment

`notebooks/rag_experiments.ipynb` loads the pinned model on Colab, embeds 200 cleaned and labeled cases from
the provider `test` split and checks:

- token lengths of `title + "\n" + text` and how many posts hit the 1,024-token cap;
- vector dimensions and norms;
- time per post and per query;
- a query smoke test: the first two sentences of each post should retrieve that post first.

The query check is only a smoke test; retrieval relevance is evaluated separately (Section 3.9).

| Measurement | Result |
| --- | --- |
| Hardware | Google Colab, NVIDIA T4 GPU, Python 3.13.16 |
| Model load time | 40.5 s (first run, includes downloading the model) |
| Sample | 200 cleaned and labeled cases from the provider `test` split |
| Token length (median / max) | 501.5 / 724 tokens |
| Posts over 1,024 tokens | 0 of 200 |
| Dimensions / norm range | 1,024 / 1.000000 to 1.000000 |
| Time per post / per query | 59 ms (11.8 s for 200 posts) / 8 ms |
| Top-1 source match | 129 of 200 (64%) |

**Observations:**

- All vectors have 1,024 dimensions and unit norm, as required for cosine similarity.
- No post reached the 1,024-token cap (maximum 724 tokens), which matches the dataset's length filtering seen in DATA-01. The cap remains a safety limit for other sources.
- Embedding the full retrieval index (up to 20,000 posts) should take roughly 20 minutes on a T4 at this speed, within the "under an hour" estimate of Section 3.8.
- The top-1 check is far above chance (1 in 200), but 36% of queries do not retrieve their own post first. A query built from only the first two sentences is often too generic (for example a background introduction), so this is a smoke test, not a retrieval quality measure. Relevance is evaluated on the development set (Section 3.9).
- The `pip` warning about `cudf` / `pyarrow` comes from a library preinstalled on Colab that this project does not use.

## Notes

- The architecture writes the query format as `Query: ...`, with a space after the colon; Qwen's model card
  example has no space. We follow the architecture. Only queries are prefixed, so this has to stay the same
  between indexing-time tests and retrieval.
- Posts in this dataset are at most about 650 words, so the cap is expected to trigger rarely; it protects
  future sources with longer posts (Section 3.3).
- The vector database, reranking and the near-duplicate scan belong to RAG-02, RAG-03 and RAG-04.
