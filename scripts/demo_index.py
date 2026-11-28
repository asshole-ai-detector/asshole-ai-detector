"""Offline RAG-02 demonstration using invented records and a deterministic test encoder."""

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from asshole_ai_detector.data.pipeline import export_prepared, load_prepared, prepare
from asshole_ai_detector.rag.embeddings import Embedder, EmbeddingConfig
from asshole_ai_detector.rag.index import SamplingConfig, build_index, open_index
from asshole_ai_detector.schemas.case import VERDICTS

TOPICS = ("rent", "chores", "garden", "concert")
CONFIG = EmbeddingConfig(
    model_name="synthetic/rag02-topic-encoder",
    revision=sha256(b"rag02-topic-encoder-v1").hexdigest(),
)
SAMPLING = SamplingConfig(seed=42, max_per_top_verdict=2)


class DemoTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


class DemoEncoder:
    def encode(self, texts, batch_size):
        vectors = []
        for text in texts:
            vector = [0.0] * CONFIG.dimensions
            topic = next(word for word in TOPICS if word in text.split())
            vector[TOPICS.index(topic)] = 1.0
            vectors.append(vector)
        return vectors


def demo_embedder(config):
    return Embedder(DemoEncoder(), DemoTokenizer(), config)


def demo_data(seed=42, strategy="combined_stratified"):
    rows = []
    for i in range(24):
        post_id = f"demo{i:02d}"
        topic = TOPICS[i % 4]
        verdict = VERDICTS[i % 4]
        row = {
            "submission_url": f"https://example.org/comments/{post_id}/discussion",
            "submission_title": f"A disagreement about {topic}",
            "submission_text": (
                f"In invented situation {post_id} we disagreed over {topic} last week. "
                + "We talked about what each of us expected and how to make a fair decision. "
                * 4
            ),
        }
        for rank in range(1, 6):
            row[f"top_comment_{rank}"] = (
                f"My opinion about example {post_id} at rank {rank}."
            )
            row[f"top_comment_{rank}_classification"] = (
                VERDICTS[(i + 1) % 4] if i < 8 and rank >= 4 else verdict
            )
        rows.append(row)
    prepared = prepare(
        rows[:18],
        rows[18:],
        source_revision=sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        golden_size=3,
        development_size=2,
        seed=seed,
        golden_minimums={},
        split_strategy=strategy,
    )
    prepared.report["source_dataset"] = "synthetic/rag02-demo"
    return prepared


def verify(root, *, use_rag03):
    data, indexes = root / "prepared", root / "indexes"
    prepared = load_prepared(data, strict=True)
    split = prepared.splits.manifest
    excluded = split.held_out_ids | split.development_ids
    with open_index(data, indexes, embedding_config=CONFIG, sampling=SAMPLING) as index:
        records = index.collection.get(include=["metadatas"])
        ids = set(records["ids"])
        assert ids <= split.retrieval_ids and ids.isdisjoint(excluded)
        assert all(index.get_case(i) is None for i in excluded)
        first_id = min(ids)
        case = index.get_case(first_id)
        assert case == next(
            c for c in prepared.splits.retrieval if c.post_id == first_id
        )
        metadata = records["metadatas"][records["ids"].index(first_id)]
        print(
            f"Reopened {len(ids)} records. Excluded all {len(excluded)} Golden/dev IDs."
        )
        print(
            f"Metadata {metadata}. Full case includes {len(case.valid_comments)} comments."
        )
        query = f"A disagreement about {TOPICS[VERDICTS.index(case.derived_fields.top_verdict)]}"
        embedder = demo_embedder(CONFIG)
        if use_rag03:
            from asshole_ai_detector.rag.retrieval import retrieve_dense

            hits = retrieve_dense(
                query,
                embed_query=lambda q: embedder.embed_queries([q])[0],
                index=index.collection,
                get_case=index.get_case,
            )
            assert hits and hits[0].cosine_distance == 0
            assert (
                hits[0].case.derived_fields.top_verdict
                == case.derived_fields.top_verdict
            )
            print(
                f"RAG-03 returned {hits[0].post_id} at cosine distance {hits[0].cosine_distance}."
            )
        else:
            result = index.collection.query(
                query_embeddings=embedder.embed_queries([query]), n_results=1
            )
            assert result["distances"][0][0] == 0
            print(
                "Direct Chroma query passed. Use --rag03 with PR #31 to exercise retrieve_dense."
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/rag-02-demo"))
    parser.add_argument(
        "--rag03",
        action="store_true",
        help="Require the RAG-03 implementation on the Python path",
    )
    parser.add_argument("--reopen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reopen:
        verify(args.root, use_rag03=args.rag03)
        return
    data, indexes = args.root / "prepared", args.root / "indexes"
    prepared = demo_data()
    if not data.exists():
        export_prepared(prepared, data)
    source = (
        prepared.report["source_dataset"],
        prepared.splits.manifest.source_revision,
    )
    options = {
        "embedding_config": CONFIG,
        "sampling": SAMPLING,
        "expected_source": source,
    }
    generation = build_index(data, indexes, **options, load_embedder=demo_embedder)
    cmd = [sys.executable, __file__, "--root", str(args.root), "--reopen"]
    if args.rag03:
        cmd.append("--rag03")
    subprocess.run(cmd, check=True)

    def unexpected_load(config):
        raise AssertionError("Unchanged build requested an embedder")

    assert (
        build_index(data, indexes, **options, load_embedder=unexpected_load)
        == generation
    )
    print(f"Verified reuse without embedding. Synthetic generation {generation.name}")


if __name__ == "__main__":
    main()
