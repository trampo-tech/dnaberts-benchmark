"""Prepare train/validation/test CSVs from a Hugging Face dataset.

Usage:
    python scripts/prepare_hf_dataset.py katarinagresova/Genomic_Benchmarks_human_nontata_promoters

This will create:
    data/processed/Genomic_Benchmarks_human_nontata_promoters/{train,validation,test}.csv
"""

import argparse
import os
import re
import sys

import pandas as pd
from datasets import DatasetDict, DownloadMode, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/validation/test CSVs from a Hugging Face dataset."
    )
    parser.add_argument(
        "dataset_id",
        help="Hugging Face dataset id (e.g., katarinagresova/Genomic_Benchmarks...)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task/config name for the dataset (e.g., regulatory_element_enhancer)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Optional sequence_length parameter passed to load_dataset",
    )
    return parser.parse_args()


def load_hf_dataset(dataset_id: str, task: str | None, sequence_length: int | None):
    load_kwargs = {"download_mode": DownloadMode.FORCE_REDOWNLOAD}
    if task:
        load_kwargs["name"] = task
    if sequence_length:
        load_kwargs["sequence_length"] = sequence_length

    try:
        return load_dataset(dataset_id, **load_kwargs)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

        task_args = f" --task {task}" if task else ""
        seq_args = f" --sequence-length {sequence_length}" if sequence_length else ""
        message = (
            "This dataset uses a legacy Hugging Face loading script, and the installed "
            "`datasets` version no longer supports scripted datasets.\n\n"
            "Workaround:\n"
            "  mkdir -p ~/.cache/huggingface/datasets/downloads\n"
            f"  uv run --with 'datasets==3.6.0' scripts/prepare_hf_dataset.py {dataset_id}{task_args}{seq_args}\n\n"
            "The cache directory is needed because this dataset script downloads reference files directly."
        )
        raise SystemExit(message) from exc


def detect_columns(d):
    features = d.features
    seq_col = None
    # try to find a DNA-like string column
    for col, feat in features.items():
        try:
            vals = d[col][:20]
        except Exception:
            vals = []
        if any(isinstance(v, str) and re.match(r"^[ACGTNacgtn]+$", v) for v in vals):
            seq_col = col
            break
    if seq_col is None:
        for col, feat in features.items():
            vals = d[col][:20]
            if any(isinstance(v, str) for v in vals):
                seq_col = col
                break
    # label detection
    label_col = None
    if "label" in features:
        label_col = "label"
    else:
        for col, feat in features.items():
            if getattr(feat, "names", None) is not None:
                label_col = col
                break
    if label_col is None:
        for col, feat in features.items():
            vals = d[col][:20]
            if any(isinstance(v, int) for v in vals):
                label_col = col
                break
    return seq_col, label_col


def main() -> None:
    args = parse_args()
    dataset_id = args.dataset_id
    task = args.task 
    print("Loading dataset:", dataset_id)

    ds = load_hf_dataset(dataset_id, args.task, args.sequence_length)
    print("Available splits:", list(ds.keys()))

    outdir = os.path.join("data", "processed", dataset_id.replace("/", "_") + "_" + task )
    os.makedirs(outdir, exist_ok=True)

    if "validation" not in ds:
        if "train" in ds and "test" in ds:
            print("Creating validation from train (10%)")
            split = ds["train"].train_test_split(test_size=0.1, seed=42)
            ds_train = split["train"]
            ds_val = split["test"]
            ds = DatasetDict(
                {"train": ds_train, "validation": ds_val, "test": ds["test"]}
            )
        elif "train" in ds:
            print("Splitting train into train/validation/test (80/10/10)")
            part = ds["train"].train_test_split(test_size=0.2, seed=42)
            rest = part["train"]
            test = part["test"]
            part2 = rest.train_test_split(test_size=0.1111111, seed=42)
            ds = DatasetDict(
                {"train": part2["train"], "validation": part2["test"], "test": test}
            )
        else:
            print("No train split found", file=sys.stderr)
            sys.exit(1)

    for split in ["train", "validation", "test"]:
        if split not in ds:
            print(f"Missing split {split}", file=sys.stderr)
            sys.exit(1)
        d = ds[split]
        seq_col, label_col = detect_columns(d)
        if seq_col is None or label_col is None:
            print(
                f"Could not auto-detect sequence/label columns for split {split}: features={d.features}",
                file=sys.stderr,
            )
            sys.exit(1)
        df = d.to_pandas()
        if seq_col != "sequence":
            df = df.rename(columns={seq_col: "sequence"})
        if label_col != "label":
            if df[label_col].dtype == object:
                df["label"] = pd.Categorical(df[label_col]).codes
            else:
                df["label"] = df[label_col].astype(int)
            if label_col in df.columns and label_col != "label":
                del df[label_col]
        else:
            if df["label"].dtype == object:
                df["label"] = pd.Categorical(df["label"]).codes
        df = df[["sequence", "label"]]
        fname = os.path.join(outdir, f"{split}.csv")
        df.to_csv(fname, index=False)
        print(f"Wrote {fname} (rows={len(df)})")

    print("Done. CSVs written to:", outdir)


if __name__ == "__main__":
    main()
