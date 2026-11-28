"""Build or verify the preliminary persistent RAG-02 index using real RAG-01 embeddings."""

import argparse
from pathlib import Path

import yaml

from asshole_ai_detector.rag.embeddings import DEFAULT_CONFIG_PATH, EmbeddingConfig
from asshole_ai_detector.rag.index import (
    DEFAULT_INDEX_ROOT,
    SamplingConfig,
    build_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-per-top-verdict", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rag = config["rag"]
    source = config["data_sources"]["aita"]
    sampling = SamplingConfig(
        seed=config["project"]["seed"] if args.seed is None else args.seed,
        max_per_top_verdict=rag["max_per_top_verdict"]
        if args.max_per_top_verdict is None
        else args.max_per_top_verdict,
        max_posts=rag["index_max_posts"],
    )
    generation = build_index(
        args.prepared_dir,
        args.index_root,
        embedding_config=EmbeddingConfig.from_mapping(rag),
        sampling=sampling,
        expected_source=(source["hf_dataset"], source["revision"]),
    )
    print(generation)
    print((generation / "build_report.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
