from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build TATA-related datasets from EPDnew FASTA files, including "
            "binary promoter-vs-background tasks."
        )
    )
    parser.add_argument(
        "--tata-fa",
        type=Path,
        default=Path("data/raw/epdnew_TATA_1000bp.fa"),
        help="FASTA file with TATA-containing promoters.",
    )
    parser.add_argument(
        "--tataless-fa",
        type=Path,
        default=Path("data/raw/epdnew_TATAless_1000bp.fa"),
        help="FASTA file with TATA-less promoters.",
    )
    parser.add_argument(
        "--tata-neg-fa",
        type=Path,
        default=Path("data/raw/epdnew_TATA_biasaway_1000bp.fa"),
        help="FASTA file with background sequences matched to TATA promoters.",
    )
    parser.add_argument(
        "--tataless-neg-fa",
        type=Path,
        default=Path("data/raw/epdnew_TATAless_biasaway_1000bp.fa"),
        help="FASTA file with background sequences matched to TATA-less promoters.",
    )
    parser.add_argument(
        "--task",
        choices=["binary", "disambiguation", "all"],
        default="binary",
        help=(
            "binary: build TATA-vs-BG and TATA-less-vs-BG, "
            "disambiguation: build TATA-vs-TATA-less, all: build every dataset."
        ),
    )
    parser.add_argument(
        "--out-tata-vs-bg-csv",
        type=Path,
        default=Path("data/raw/tata_vs_background_1000bp.csv"),
        help="Output CSV for TATA promoter vs background.",
    )
    parser.add_argument(
        "--out-tataless-vs-bg-csv",
        type=Path,
        default=Path("data/raw/tataless_vs_background_1000bp.csv"),
        help="Output CSV for TATA-less promoter vs background.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1000,
        help="Keep first N bases from each sequence.",
    )
    parser.add_argument(
        "--dedup-mode",
        choices=["none", "gene"],
        default="none",
        help="none=keep all promoters, gene=keep one promoter per gene symbol.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable class balancing by downsampling to minority class size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_gene_symbol(promoter_name: str) -> str:
    # EPDnew promoter names often look like GENE_1, GENE_2, etc.
    return re.sub(r"_[0-9]+$", "", promoter_name)


def records_to_df(path: Path, window: int, source: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rec in SeqIO.parse(str(path), "fasta"):
        seq = str(rec.seq).upper()
        if len(seq) < window:
            continue
        seq = seq[:window]
        if "N" in seq:
            continue

        parts = rec.description.split()
        promoter_id = parts[0]
        promoter_name = parts[1] if len(parts) > 1 else promoter_id
        gene_symbol = parse_gene_symbol(promoter_name)

        rows.append(
            {
                "id": promoter_id,
                "promoter_name": promoter_name,
                "gene_symbol": gene_symbol,
                "sequence": seq,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def deduplicate_by_gene(df: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    if mode == "none":
        return df
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.drop_duplicates(subset=["gene_symbol"], keep="first").reset_index(
        drop=True
    )


def maybe_balance(df: pd.DataFrame, seed: int, do_balance: bool) -> pd.DataFrame:
    if not do_balance:
        return df

    counts = df["label"].value_counts().to_dict()
    if len(counts) != 2:
        raise RuntimeError(f"Expected 2 classes, got: {counts}")

    n = min(counts.values())
    parts = []
    for label in sorted(counts):
        parts.append(df[df["label"] == label].sample(n=n, random_state=seed))

    return pd.concat(parts, ignore_index=True)


def build_binary_dataset(
    pos_df: pd.DataFrame, neg_df: pd.DataFrame, seed: int, do_balance: bool
) -> pd.DataFrame:
    pos = pos_df.copy()
    neg = neg_df.copy()
    pos["label"] = 1
    neg["label"] = 0

    df = pd.concat([pos, neg], ignore_index=True)
    df = maybe_balance(df, seed=seed, do_balance=do_balance)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_dataset(df: pd.DataFrame, output_path: Path, name: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {name}: {output_path}")
    print(f"  shape={df.shape}")
    print(f"  labels={df['label'].value_counts().to_dict()}")
    print(f"  unique_genes={int(df['gene_symbol'].nunique())}")


def main():
    args = parse_args()

    if not args.tata_fa.exists():
        raise FileNotFoundError(f"Missing FASTA: {args.tata_fa}")
    if not args.tataless_fa.exists():
        raise FileNotFoundError(f"Missing FASTA: {args.tataless_fa}")

    if args.task in {"binary", "all"}:
        if not args.tata_neg_fa.exists():
            raise FileNotFoundError(
                f"Missing FASTA: {args.tata_neg_fa}. Use --tata-neg-fa to set the path."
            )
        if not args.tataless_neg_fa.exists():
            raise FileNotFoundError(
                f"Missing FASTA: {args.tataless_neg_fa}. Use --tataless-neg-fa to set the path."
            )

    tata_df = records_to_df(args.tata_fa, window=args.window, source="tata_promoter")
    tataless_df = records_to_df(
        args.tataless_fa, window=args.window, source="tataless_promoter"
    )

    if tata_df.empty:
        raise RuntimeError("No valid TATA sequences after filtering.")
    if tataless_df.empty:
        raise RuntimeError("No valid TATA-less sequences after filtering.")

    tata_df = deduplicate_by_gene(tata_df, mode=args.dedup_mode, seed=args.seed)
    tataless_df = deduplicate_by_gene(
        tataless_df, mode=args.dedup_mode, seed=args.seed
    )

    if args.task in {"binary", "all"}:
        tata_neg_df = records_to_df(
            args.tata_neg_fa, window=args.window, source="tata_background"
        )
        tataless_neg_df = records_to_df(
            args.tataless_neg_fa, window=args.window, source="tataless_background"
        )

        if tata_neg_df.empty:
            raise RuntimeError("No valid TATA background sequences after filtering.")
        if tataless_neg_df.empty:
            raise RuntimeError(
                "No valid TATA-less background sequences after filtering."
            )

        tata_vs_bg = build_binary_dataset(
            pos_df=tata_df,
            neg_df=tata_neg_df,
            seed=args.seed,
            do_balance=not args.no_balance,
        )
        tataless_vs_bg = build_binary_dataset(
            pos_df=tataless_df,
            neg_df=tataless_neg_df,
            seed=args.seed,
            do_balance=not args.no_balance,
        )

        save_dataset(
            tata_vs_bg,
            output_path=args.out_tata_vs_bg_csv,
            name="tata_vs_background",
        )
        save_dataset(
            tataless_vs_bg,
            output_path=args.out_tataless_vs_bg_csv,
            name="tataless_vs_background",
        )

if __name__ == "__main__":
    main()
