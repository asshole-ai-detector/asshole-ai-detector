
"""CLI entry point for DATA-05 preprocessing."""

import argparse
import json
from pathlib import Path

from asshole_ai_detector.data.aita_adapter import (
    HF_REVISION,
    load_aita_dataset,
)
from asshole_ai_detector.data.pipeline import (
    export_prepared,
    prepare,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DATA-05 AITA preprocessing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data"),
    )
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--golden-size", type=int, default=800)
    parser.add_argument("--development-size", type=int, default=500)

    args = parser.parse_args()

    train = load_aita_dataset(
        revision=args.revision,
        splits=("train",),
    )
    test = load_aita_dataset(
        revision=args.revision,
        splits=("test",),
    )

    prepared = prepare(
        train,
        test,
        source_revision=args.revision,
        seed=args.seed,
        golden_size=args.golden_size,
        development_size=args.development_size,
    )

    export_prepared(prepared, args.output_dir)

    print(json.dumps(prepared.report, indent=2))
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
