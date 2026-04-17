"""Unified dataset creation script.

Usage:
    python scripts/create_dataset.py --task promoter_binary
    python scripts/create_dataset.py --task genomic_negatives [--genome-fa ...] [--epd-bed ...]
    python scripts/create_dataset.py --task tata_binary [--tata-fa ...] [--tataless-fa ...]
"""

from __future__ import annotations

import argparse
import bisect
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pysam
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


# ---------------------------------------------------------------------------
# Task: genomic_negatives  (full genome-based hard negatives)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromoterRecord:
    record_id: str
    promoter_name: str
    gene_symbol: str
    sequence: str


@dataclass(frozen=True)
class TssRecord:
    chrom: str
    tss: int
    strand: str


def parse_gene_symbol(promoter_name: str) -> str:
    return re.sub(r"_[0-9]+$", "", promoter_name)


def read_positive_promoters(path: Path, window: int) -> list[PromoterRecord]:
    rows: list[PromoterRecord] = []
    for rec in SeqIO.parse(str(path), "fasta"):
        seq = str(rec.seq).upper()
        if len(seq) < window:
            continue
        seq = seq[:window]
        if "N" in seq:
            continue
        parts = rec.description.split()
        record_id = parts[0]
        promoter_name = parts[1] if len(parts) > 1 else record_id
        rows.append(
            PromoterRecord(
                record_id=record_id,
                promoter_name=promoter_name,
                gene_symbol=parse_gene_symbol(promoter_name),
                sequence=seq,
            )
        )
    return rows


def read_epd_bed(path: Path) -> dict[str, TssRecord]:
    mapping: dict[str, TssRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chrom, start, _end, promoter_name, _score, strand = line.split("\t")
            if promoter_name in mapping:
                raise ValueError(f"Duplicate promoter name in BED: {promoter_name}")
            mapping[promoter_name] = TssRecord(chrom=chrom, tss=int(start), strand=strand)
    return mapping


def build_tss_index(records: dict[str, TssRecord]) -> dict[str, list[int]]:
    per_chrom: dict[str, list[int]] = defaultdict(list)
    for record in records.values():
        per_chrom[record.chrom].append(record.tss)
    for chrom in per_chrom:
        per_chrom[chrom].sort()
    return dict(per_chrom)


def interval_from_relative_window(
    tss: int, strand: str, relative_start: int, relative_end: int,
) -> tuple[int, int]:
    if strand == "+":
        return tss + relative_start, tss + relative_end
    return tss - relative_end, tss - relative_start


def overlaps_promoter_buffer(
    chrom: str, start: int, end: int,
    tss_index: dict[str, list[int]], promoter_buffer: int,
) -> bool:
    tss_positions = tss_index.get(chrom, [])
    left = bisect.bisect_left(tss_positions, start - promoter_buffer)
    right = bisect.bisect_right(tss_positions, end + promoter_buffer)
    return left != right


def fetch_sequence(
    genome: pysam.FastaFile, chrom: str, start: int, end: int, strand: str,
) -> str | None:
    if start < 0:
        return None
    chrom_length = genome.get_reference_length(chrom)
    if end > chrom_length:
        return None
    seq = genome.fetch(chrom, start, end).upper()
    if len(seq) != end - start or "N" in seq:
        return None
    if strand == "-":
        return reverse_complement(seq)
    return seq


def build_positive_dataframe(records: list[PromoterRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": r.record_id,
                "promoter_name": r.promoter_name,
                "gene_symbol": r.gene_symbol,
                "sequence": r.sequence,
                "source": "promoter",
                "label": 1,
            }
            for r in records
        ]
    )


def build_flanking_negatives(
    positives: list[PromoterRecord],
    coord_map: dict[str, TssRecord],
    tss_index: dict[str, list[int]],
    genome: pysam.FastaFile,
    flank_shift: int,
    promoter_buffer: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    relative_windows = {
        "flanking_upstream_negative": (-299 - flank_shift, 101 - flank_shift),
        "flanking_downstream_negative": (-299 + flank_shift, 101 + flank_shift),
    }
    for record in positives:
        tss_record = coord_map.get(record.promoter_name)
        if tss_record is None:
            continue
        for source, (rel_start, rel_end) in relative_windows.items():
            start, end = interval_from_relative_window(
                tss=tss_record.tss, strand=tss_record.strand,
                relative_start=rel_start, relative_end=rel_end,
            )
            if overlaps_promoter_buffer(tss_record.chrom, start, end, tss_index, promoter_buffer):
                continue
            seq = fetch_sequence(genome, tss_record.chrom, start, end, tss_record.strand)
            if seq is None:
                continue
            rows.append({
                "id": f"{record.record_id}__{source}",
                "promoter_name": record.promoter_name,
                "gene_symbol": record.gene_symbol,
                "sequence": seq,
                "source": source,
                "label": 0,
            })
    return pd.DataFrame(rows)


def gc_content(seq: str) -> float:
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq) if seq else 0.0


def compute_promoter_gc_stats(records: list[PromoterRecord]) -> tuple[float, float]:
    gc_values = [gc_content(r.sequence) for r in records]
    mean_gc = sum(gc_values) / len(gc_values)
    return mean_gc, max(gc_values) - min(gc_values)


def build_intergenic_negatives(
    genome: pysam.FastaFile,
    tss_index: dict[str, list[int]],
    chromosomes: list[str],
    window: int,
    promoter_buffer: int,
    target_count: int,
    seed: int,
    attempt_multiplier: int,
    gc_target: float | None = None,
    gc_tolerance: float = 0.02,
) -> pd.DataFrame:
    rng = random.Random(seed)
    valid_chromosomes = [c for c in chromosomes if c in genome.references]
    if not valid_chromosomes:
        raise RuntimeError("None of the requested chromosomes exist in the genome FASTA.")

    do_gc_match = gc_target is not None and gc_tolerance > 0
    gc_lo = (gc_target - gc_tolerance) if do_gc_match else 0.0
    gc_hi = (gc_target + gc_tolerance) if do_gc_match else 1.0

    rows: list[dict[str, object]] = []
    max_attempts = max(target_count * attempt_multiplier, target_count)
    attempt = 0
    gc_rejected = 0

    while len(rows) < target_count and attempt < max_attempts:
        attempt += 1
        chrom = rng.choice(valid_chromosomes)
        chrom_length = genome.get_reference_length(chrom)
        if chrom_length <= window:
            continue
        start = rng.randint(0, chrom_length - window)
        end = start + window
        if overlaps_promoter_buffer(chrom, start, end, tss_index, promoter_buffer):
            continue
        strand = rng.choice(["+", "-"])
        seq = fetch_sequence(genome, chrom, start, end, strand)
        if seq is None:
            continue
        if do_gc_match:
            seq_gc = gc_content(seq)
            if seq_gc < gc_lo or seq_gc > gc_hi:
                gc_rejected += 1
                continue
        rows.append({
            "id": f"intergenic_seq_{len(rows)}",
            "promoter_name": f"intergenic_{chrom}_{start}_{end}",
            "gene_symbol": "INTERGENIC",
            "sequence": seq,
            "source": "intergenic_negative",
            "label": 0,
        })

    if do_gc_match:
        print(
            f"  GC-matching: target={gc_target:.3f} ± {gc_tolerance:.3f}, "
            f"rejected={gc_rejected}, accepted={len(rows)}"
        )
    if len(rows) < target_count:
        raise RuntimeError(
            f"Only sampled {len(rows)} intergenic negatives out of requested {target_count}. "
            f"({gc_rejected} rejected by GC filter). Try increasing --intergenic-attempt-multiplier "
            f"or relaxing --gc-tolerance."
        )
    return pd.DataFrame(rows)


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
    epd_bed = args.epd_bed
    genome_fa = args.genome_fa
    output_csv = args.output_csv or Path("data/raw/promoter_vs_genomic_negatives_400bp.csv")

    for path, desc in [
        (promoter_fa, "promoter FASTA"),
        (epd_bed, "EPDnew BED"),
        (genome_fa, "genome FASTA"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {desc}: {path}")

    fai = genome_fa.with_suffix(genome_fa.suffix + ".fai")
    if not fai.exists():
        raise FileNotFoundError(f"Missing FASTA index. Run: samtools faidx {genome_fa}")

    positives = read_positive_promoters(promoter_fa, window=args.window)
    if not positives:
        raise RuntimeError("No positive promoters found after filtering.")

    coord_map = read_epd_bed(epd_bed)
    missing = [r.promoter_name for r in positives if r.promoter_name not in coord_map]
    if missing:
        raise RuntimeError(f"Missing BED coords for {len(missing)} promoters. E.g.: {', '.join(missing[:5])}")

    tss_index = build_tss_index(coord_map)
    mean_gc, gc_range = compute_promoter_gc_stats(positives)
    print(f"Promoter GC stats: mean={mean_gc:.3f}, range={gc_range:.3f}")

    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

    with pysam.FastaFile(str(genome_fa)) as genome:
        positive_df = build_positive_dataframe(positives)
        flanking_df = build_flanking_negatives(
            positives, coord_map, tss_index, genome,
            flank_shift=args.flank_shift, promoter_buffer=args.promoter_buffer,
        )
        intergenic_target = max(int(round(len(positives) * args.intergenic_per_positive)), 1)
        intergenic_df = build_intergenic_negatives(
            genome, tss_index, chromosomes, args.window, args.promoter_buffer,
            intergenic_target, args.seed, args.intergenic_attempt_multiplier,
            gc_target=mean_gc if args.gc_tolerance > 0 else None,
            gc_tolerance=args.gc_tolerance,
        )

    if flanking_df.empty:
        raise RuntimeError("No flanking negatives generated. Loosen constraints.")

    negatives_df = pd.concat([flanking_df, intergenic_df], ignore_index=True)
    dataset_df = pd.concat([positive_df, negatives_df], ignore_index=True)
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
    p.add_argument("--epd-bed", type=Path, default=Path("data/raw/epdnew_human.bed"))
    p.add_argument("--genome-fa", type=Path, default=Path("data/raw/GRCh38.primary_assembly.genome.fa"))
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--window", type=int, default=400)
    p.add_argument("--flank-shift", type=int, default=800)
    p.add_argument("--promoter-buffer", type=int, default=500)
    p.add_argument("--intergenic-per-positive", type=float, default=1.0)
    p.add_argument("--intergenic-attempt-multiplier", type=int, default=30)
    p.add_argument("--gc-tolerance", type=float, default=0.02)
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
