"""Unified dataset creation script.

Usage:
    python scripts/create_dataset.py --task promoter_binary
    python scripts/create_dataset.py --task genomic_negatives [--promoter-fa ...] [--negative-fa ...]
    python scripts/create_dataset.py --task tata_binary [--tata-fa ...] [--tataless-fa ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

REVERSE_COMPLEMENT_TABLE = str.maketrans("ACGTN", "TGCAN")
SEED = 42


def reverse_complement(seq: str) -> str:
    return seq.translate(REVERSE_COMPLEMENT_TABLE)[::-1]


def read_fasta_records(path: Path, window: int) -> list[tuple[str, str]]:
    rows = []
    for r in SeqIO.parse(str(path), "fasta"):
        seq = str(r.seq).upper()
        if len(seq) < window:
            continue
        seq = seq[:window]
        if "N" in seq:
            continue
        rows.append((r.id, seq))
    return rows


def records_to_df(records: list[tuple[str, str]], label: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"id": rid, "sequence": seq, "label": label} for rid, seq in records]
    )


def balance_and_shuffle(
    pos_df: pd.DataFrame, neg_df: pd.DataFrame, seed: int
) -> pd.DataFrame:
    n = min(len(pos_df), len(neg_df))
    pos_df = pos_df.sample(n=n, random_state=seed)
    neg_df = neg_df.sample(n=n, random_state=seed)
    return (
        pd.concat([pos_df, neg_df], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def save_csv(df: pd.DataFrame, path: Path, seed: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}  shape={df.shape}")
    print(f"  labels={df['label'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Task: promoter_binary
# ---------------------------------------------------------------------------

def task_promoter_binary(args: argparse.Namespace) -> None:
    POS_FA = Path("data/raw/epdnew_400bp.fa")
    NEG_FA = Path("data/raw/epdnew_biasaway_400bp.fa")
    CSV_OUT = Path("data/raw/promoter_binary.csv")
    WINDOW = 400

    print("Building promoter_binary CSV...")
    pos_records = read_fasta_records(POS_FA, WINDOW)
    neg_records = read_fasta_records(NEG_FA, WINDOW)
    pos_df = records_to_df(pos_records, label=1)
    neg_df = records_to_df(neg_records, label=0)
    print(f"  {len(pos_df)} positive, {len(neg_df)} negative samples")

    if pos_df.empty or neg_df.empty:
        raise RuntimeError("No valid sequences after filtering.")

    df = balance_and_shuffle(pos_df, neg_df, seed=args.seed)
    save_csv(df, CSV_OUT)


def maybe_balance(df: pd.DataFrame, seed: int, do_balance: bool) -> pd.DataFrame:
    if not do_balance:
        return df
    counts = df["label"].value_counts().to_dict()
    if len(counts) != 2:
        raise RuntimeError(f"Expected 2 classes, got: {counts}")
    n = min(counts.values())
    parts = [df[df["label"] == label].sample(n=n, random_state=seed) for label in sorted(counts)]
    return pd.concat(parts, ignore_index=True)


def task_genomic_negatives(args: argparse.Namespace) -> None:
    promoter_fa = args.promoter_fa
    negative_fa = args.negative_fa
    output_csv = args.output_csv or Path("data/raw/promoter_vs_genomic_negatives_400bp.csv")
    window = args.window

    for path, desc in [
        (promoter_fa, "promoter FASTA"),
        (negative_fa, "negative FASTA"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {desc}: {path}")

    print("Building promoter_vs_genomic_negatives from precomputed FASTA inputs...")
    positive_df = records_to_df(read_fasta_records(promoter_fa, window), label=1)
    negative_df = records_to_df(read_fasta_records(negative_fa, window), label=0)

    if positive_df.empty or negative_df.empty:
        raise RuntimeError("No valid sequences found after filtering.")

    print(f"  {len(positive_df)} positive, {len(negative_df)} negative samples")
    dataset_df = pd.concat([positive_df, negative_df], ignore_index=True)
    dataset_df = maybe_balance(dataset_df, seed=args.seed, do_balance=not args.no_balance)
    save_csv(dataset_df, output_csv, seed=args.seed)


# ---------------------------------------------------------------------------
# Task: tata_binary  (TATA-vs-BG and TATAless-vs-BG)
# ---------------------------------------------------------------------------

def task_tata_binary(args: argparse.Namespace) -> None:
    WINDOW = 1000

    tata_fa = args.tata_fa or Path("data/raw/epdnew_TATA_1000bp.fa")
    tataless_fa = args.tataless_fa or Path("data/raw/epdnew_TATAless_1000bp.fa")
    tata_neg_fa = args.tata_neg_fa or Path("data/raw/epdnew_TATA_biasaway_1000bp.fa")
    tataless_neg_fa = args.tataless_neg_fa or Path("data/raw/epdnew_TATAless_biasaway_1000bp.fa")

    for path, desc in [
        (tata_fa, "TATA promoter FASTA"),
        (tataless_fa, "TATA-less promoter FASTA"),
        (tata_neg_fa, "TATA background FASTA"),
        (tataless_neg_fa, "TATA-less background FASTA"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {desc}: {path}")

    tata_pos = records_to_df(read_fasta_records(tata_fa, WINDOW), label=1)
    tata_neg = records_to_df(read_fasta_records(tata_neg_fa, WINDOW), label=0)
    tataless_pos = records_to_df(read_fasta_records(tataless_fa, WINDOW), label=1)
    tataless_neg = records_to_df(read_fasta_records(tataless_neg_fa, WINDOW), label=0)

    for df, name in [(tata_pos, "TATA pos"), (tata_neg, "TATA neg"),
                     (tataless_pos, "TATAless pos"), (tataless_neg, "TATAless neg")]:
        if df.empty:
            raise RuntimeError(f"No valid {name} sequences after filtering.")

    print("Building tata_vs_background...")
    tata_vs_bg = balance_and_shuffle(tata_pos, tata_neg, seed=args.seed)
    save_csv(tata_vs_bg, Path("data/raw/tata_vs_background_1000bp.csv"))

    print("Building tataless_vs_background...")
    tataless_vs_bg = balance_and_shuffle(tataless_pos, tataless_neg, seed=args.seed)
    save_csv(tataless_vs_bg, Path("data/raw/tataless_vs_background_1000bp.csv"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create raw datasets for the DNA benchmark.")
    p.add_argument(
        "--task",
        required=True,
        choices=["promoter_binary", "genomic_negatives", "tata_binary"],
        help="Which dataset to build.",
    )
    p.add_argument("--seed", type=int, default=SEED)

    # genomic_negatives args
    p.add_argument("--promoter-fa", type=Path, default=Path("data/raw/epdnew_400bp.fa"))
    p.add_argument(
        "--negative-fa",
        type=Path,
        default=Path("data/raw/epdnew_biasaway_400bp.fa"),
    )
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--window", type=int, default=400)
    p.add_argument("--no-balance", action="store_true")

    # tata_binary args
    p.add_argument("--tata-fa", type=Path, default=None)
    p.add_argument("--tataless-fa", type=Path, default=None)
    p.add_argument("--tata-neg-fa", type=Path, default=None)
    p.add_argument("--tataless-neg-fa", type=Path, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    dispatch = {
        "promoter_binary": task_promoter_binary,
        "genomic_negatives": task_genomic_negatives,
        "tata_binary": task_tata_binary,
    }
    dispatch[args.task](args)


if __name__ == "__main__":
    main()
