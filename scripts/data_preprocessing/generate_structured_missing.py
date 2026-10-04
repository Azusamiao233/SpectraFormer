"""Generate the six fixed-rate structured-missingness dataset variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mssa_transformer.config import DATA_DIR, MASKED_DATA_DIR
from mssa_transformer.structured_missingness import (
    DATASET_SPECS,
    PATTERN_NAMES,
    apply_structured_missing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SWaT/WADI structured missing data at a fixed group rate."
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASET_SPECS),
        default=sorted(DATASET_SPECS),
    )
    parser.add_argument(
        "--patterns", nargs="+", choices=PATTERN_NAMES, default=list(PATTERN_NAMES)
    )
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=MASKED_DATA_DIR / "structured"
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Validate and write metadata/masks without duplicating the large CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for dataset in args.datasets:
        dataset_spec = DATASET_SPECS[dataset]
        source_path = DATA_DIR / dataset_spec.filename
        print(f"Loading {dataset.upper()}: {source_path}")
        frame = pd.read_csv(source_path)

        for pattern_index, pattern in enumerate(args.patterns):
            masked, mask, metadata = apply_structured_missing(
                frame,
                dataset=dataset,
                pattern=pattern,
                missing_rate=args.missing_rate,
                n_blocks=args.blocks,
                seed=args.seed + pattern_index,
            )
            stem = f"{dataset}_{pattern}_missing_{int(args.missing_rate * 100):02d}"
            mask_path = args.output_dir / f"{stem}_mask.npz"
            metadata_path = args.output_dir / f"{stem}.json"
            data_path = args.output_dir / f"{stem}.csv"

            # Save a compact boolean matrix while retaining column names in metadata.
            import numpy as np

            np.savez_compressed(mask_path, mask=mask.to_numpy(dtype=bool))
            metadata.update(
                {
                    "source_path": str(source_path),
                    "mask_path": str(mask_path),
                    "data_path": None if args.metadata_only else str(data_path),
                }
            )
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not args.metadata_only:
                masked.to_csv(data_path, index=False)
            manifest.append(metadata)
            print(
                f"  {pattern}: group={metadata['actual_group_missing_rate']:.2%}, "
                f"overall={metadata['actual_overall_missing_rate']:.2%}"
            )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
