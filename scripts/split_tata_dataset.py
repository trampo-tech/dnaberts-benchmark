from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split TATA datasets into train/val/test CSV files."
    )
    parser.add_argument(
        "--task",
        choices=["binary", "disambiguation", "all"],
        default="binary",
        help=(
            "binary: split tata-vs-background and tataless-vs-background, "
            "disambiguation: split tata-vs-tataless, all: split all datasets."
        ),
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/raw/tata_vs_tataless_1000bp.csv"),
        help="Input CSV for disambiguation mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/tata_vs_tataless"),
        help="Output directory for disambiguation mode.",
    )

    parser.add_argument(
        "--tata-vs-bg-csv",
        type=Path,
        default=Path("data/raw/tata_vs_background_1000bp.csv"),
        help="Input CSV for TATA promoter vs background.",
    )
    parser.add_argument(
        "--tataless-vs-bg-csv",
        type=Path,
        default=Path("data/raw/tataless_vs_background_1000bp.csv"),
        help="Input CSV for TATA-less promoter vs background.",
    )
    parser.add_argument(
        "--tata-vs-bg-outdir",
        type=Path,
        default=Path("data/processed/tata_vs_background"),
        help="Output directory for TATA vs background splits.",
    )
    parser.add_argument(
        "--tataless-vs-bg-outdir",
        type=Path,
        default=Path("data/processed/tataless_vs_background"),
        help="Output directory for TATA-less vs background splits.",
    )

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_one(input_csv: Path, output_dir: Path, seed: int, task_name: str):
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV for {task_name}: {input_csv}")

    df = pd.read_csv(input_csv)

    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column in {input_csv}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
    )

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"[{task_name}] train:",
        train_df.shape,
        train_df["label"].value_counts().to_dict(),
    )
    print(f"[{task_name}] val:", val_df.shape, val_df["label"].value_counts().to_dict())
    print(
        f"[{task_name}] test:", test_df.shape, test_df["label"].value_counts().to_dict()
    )
    print(f"[{task_name}] Saved splits under: {output_dir}")


def main():
    args = parse_args()

    if args.task in {"binary", "all"}:
        split_one(
            input_csv=args.tata_vs_bg_csv,
            output_dir=args.tata_vs_bg_outdir,
            seed=args.seed,
            task_name="tata_vs_background",
        )
        split_one(
            input_csv=args.tataless_vs_bg_csv,
            output_dir=args.tataless_vs_bg_outdir,
            seed=args.seed,
            task_name="tataless_vs_background",
        )

    if args.task in {"disambiguation", "all"}:
        split_one(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            seed=args.seed,
            task_name="tata_vs_tataless",
        )


if __name__ == "__main__":
    main()
