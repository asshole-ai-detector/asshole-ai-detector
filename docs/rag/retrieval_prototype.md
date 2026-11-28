
# RAG-03 — Basic Dense Retrieval Prototype

## Objective

Implement and validate basic dense retrieval for AITA submissions
using a ChromaDB vector index.

This prototype covers dense search only, not reranking or the
complete `retrieve_similar_cases` MCP contract.

## Architecture

The retrieval pipeline:

1. Receives a conflict query.
2. Normalizes the query text.
3. Calls an injected query embedding function.
4. Validates the 1024-dimensional L2-normalized vector.
5. Searches a cosine-distance ChromaDB collection.
6. Returns up to 30 matching post IDs and distances.
7. Resolves each post ID through an injected case lookup.
8. Raises an explicit error when a stored case is missing.

The embedding function and case lookup are injected so that
RAG-01 and RAG-02 can provide their agreed implementations.

## Reproducible demonstration

Run from the repository root:

    uv run --with chromadb python scripts/demo_retrieval.py

The demonstration uses:

- An actual in-memory ChromaDB collection.
- Three synthetic AITA submissions.
- Deterministic toy embeddings with 1024 dimensions.
- Cosine-distance search.
- A synthetic in-memory case lookup.

No real AITA or Golden Dataset records are used.

## Observed results

| Query | Top result | Cosine distance |
| --- | --- | --- |
| Roommate conflict about rent | post-roommate | 0.0000 |
| Problem with a coworker | post-work | 0.0000 |
| Argument with family | post-family | 0.0000 |

All three retrieval demonstrations passed.

The matching toy vectors have cosine distance 0.0000.
The other orthogonal vectors have cosine distance 1.0000.

These results demonstrate correct vector-index integration,
not real-world semantic retrieval quality.

## Verification

Run:

    uv run pytest
    uv run ruff check .

Observed local test result:

- 243 tests passed.
- 1 test skipped.
- 12 RAG-03 unit tests passed.
- Ruff passed for retrieval.py and test_retrieval.py.

## Limitations and remaining integration

- Qwen3-Embedding-0.6B is not used in this demonstration.
- RAG-01 must supply the documented instruction-aware
  query embedding interface.
- RAG-02 must supply the persistent cosine ChromaDB index
  and agreed case-store lookup.
- Model revision, embedding dimensions, index configuration,
  and manifest compatibility must be verified during integration.
- Retrieval relevance on real AITA queries has not been evaluated.
- Reranking, verdict-balanced evidence selection, disagreement,
  confidence calculations and retry logic are outside RAG-03.

The prototype must not be presented as the completed RAG pipeline.
