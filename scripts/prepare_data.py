"""CLI entry point: DATA-05 preprocessing with configured DATA-04 split policy."""

import argparse
import json
from pathlib import Path

import yaml

from asshole_ai_detector.data.aita_adapter import HF_REVISION, load_aita_dataset
from asshole_ai_detector.data.pipeline import export_prepared, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="DATA-05 AITA preprocessing")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--golden-size", type=int, default=None)
    parser.add_argument("--development-size", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    splits = config["splits"]
    revision = args.revision or config["data_sources"]["aita"].get("revision", HF_REVISION)
    seed = args.seed if args.seed is not None else config["project"]["seed"]
    golden_size = (
        args.golden_size
        if args.golden_size is not None
        else splits["golden_candidate_pool_size"]
    )
    development_size = (
        args.development_size
        if args.development_size is not None
        else splits["development_size"]
    )

    train = load_aita_dataset(revision=revision, splits=("train",))
    test = load_aita_dataset(revision=revision, splits=("test",))
    prepared = prepare(
        train,
        test,
        source_revision=revision,
        seed=seed,
        golden_size=golden_size,
        development_size=development_size,
        golden_minimums=splits["golden_candidate_minimums"],
        development_minimums=splits["development_minimums"],
        split_strategy=splits["strategy"],
    )
    export_prepared(prepared, args.output_dir)
    print(json.dumps(prepared.report, indent=2))
    print(f"\nPrivate outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
