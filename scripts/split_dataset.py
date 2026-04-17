from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitJob:
    name: str
    input_csv: Path
    output_dir: Path


DATASET_PRESETS = {
    "promoter_binary": SplitJob(
        name="promoter_binary",
        input_csv=Path("data/raw/promoter_binary.csv"),
        output_dir=Path("data/processed"),
    ),
    "promoter_vs_genomic_negatives": SplitJob(
        name="promoter_vs_genomic_negatives",
        input_csv=Path("data/raw/promoter_vs_genomic_negatives_400bp.csv"),
        output_dir=Path("data/processed/promoter_vs_genomic_negatives"),
    ),
    "tata_vs_background": SplitJob(
        name="tata_vs_background",
        input_csv=Path("data/raw/tata_vs_background_1000bp.csv"),
        output_dir=Path("data/processed/tata"),
    ),
    "tataless_vs_background": SplitJob(
        name="tataless_vs_background",
        input_csv=Path("data/raw/tataless_vs_background_1000bp.csv"),
        output_dir=Path("data/processed/tataless_vs_background"),
    ),
    "tata_vs_tataless": SplitJob(
        name="tata_vs_tataless",
        input_csv=Path("data/raw/tata_vs_tataless_1000bp.csv"),
        output_dir=Path("data/processed/tata_vs_tataless"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one or more CSV datasets into train/val/test CSV files."
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=[*DATASET_PRESETS.keys(), "all"],
        help=(
            "Named dataset preset to split. Repeat the flag for multiple presets, "
            "or use 'all'. If omitted, the script splits --input-csv or defaults "
            "to the promoter_binary preset."
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Input CSV for a one-off split outside the named presets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory for a one-off split. For a single preset, this can "
            "override the default preset output directory."
        ),
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Label column used for stratified splitting.",
    )
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    sizes = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }
    if any(size <= 0 or size >= 1 for size in sizes.values()):
        raise ValueError("Split sizes must be between 0 and 1.")

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split sizes must sum to 1.0, received {total:.6f}."
        )


def infer_output_dir(input_csv: Path) -> Path:
    stem = input_csv.stem
    for suffix in ("_1000bp", "_400bp"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return Path("data/processed") / stem


def build_jobs(args: argparse.Namespace) -> list[SplitJob]:
    if args.datasets and args.input_csv is not None:
        raise ValueError("Use either --dataset or --input-csv, not both.")

    if args.datasets:
        selected = []
        for dataset in args.datasets:
            if dataset == "all":
                selected.extend(DATASET_PRESETS)
            else:
                selected.append(dataset)

        ordered_unique = list(dict.fromkeys(selected))
        if args.output_dir is not None and len(ordered_unique) > 1:
            raise ValueError(
                "--output-dir can only be used with a single preset or a custom --input-csv."
            )

        jobs = []
        for dataset in ordered_unique:
            preset = DATASET_PRESETS[dataset]
            jobs.append(
                SplitJob(
                    name=preset.name,
                    input_csv=preset.input_csv,
                    output_dir=args.output_dir or preset.output_dir,
                )
            )
        return jobs

    if args.input_csv is not None:
        return [
            SplitJob(
                name=args.input_csv.stem,
                input_csv=args.input_csv,
                output_dir=args.output_dir or infer_output_dir(args.input_csv),
            )
        ]

    preset = DATASET_PRESETS["promoter_binary"]
    return [
        SplitJob(
            name=preset.name,
            input_csv=preset.input_csv,
            output_dir=args.output_dir or preset.output_dir,
        )
    ]


def split_one(
    input_csv: Path,
    output_dir: Path,
    seed: int,
    task_name: str,
    label_col: str = "label",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV for {task_name}: {input_csv}")

    df = pd.read_csv(input_csv)
    if label_col not in df.columns:
        raise ValueError(f"Expected a '{label_col}' column in {input_csv}.")

    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=holdout_size,
        stratify=df[label_col],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / holdout_size,
        stratify=temp_df[label_col],
        random_state=seed,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(
        f"[{task_name}] train:",
        train_df.shape,
        train_df[label_col].value_counts().to_dict(),
    )
    print(
        f"[{task_name}] val:",
        val_df.shape,
        val_df[label_col].value_counts().to_dict(),
    )
    print(
        f"[{task_name}] test:",
        test_df.shape,
        test_df[label_col].value_counts().to_dict(),
    )
    print(f"[{task_name}] Saved splits under: {output_dir}")


def main() -> None:
    args = parse_args()
    validate_split_sizes(args.train_size, args.val_size, args.test_size)

    for job in build_jobs(args):
        split_one(
            input_csv=job.input_csv,
            output_dir=job.output_dir,
            seed=args.seed,
            task_name=job.name,
            label_col=args.label_col,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
        )


if __name__ == "__main__":
    main()
