import random
from pathlib import Path

import pandas as pd
from Bio import SeqIO

# ---------- CONFIG ----------
# inputs
DATA_DIR = Path("data")
POS_BED = DATA_DIR / "human_epdnew_positives.bed"
POS_FA = DATA_DIR / "human_positives.fa"  # your downloaded positive FASTA
GENOME_FA = DATA_DIR / "human_primary_annotations.fa"
GTF = DATA_DIR / "gencode_annotations.gtf"

# outputs
NEG_BED_OUT = DATA_DIR / "negatives.bed"
NEG_FA_OUT = DATA_DIR / "negatives.fa"
CSV_OUT = DATA_DIR / "promoter_binary.csv"

WINDOW = 400
SEED = 42
# ----------------------------

random.seed(SEED)


def load_genome(genome_fa):
    genome = {}
    for rec in SeqIO.parse(str(genome_fa), "fasta"):
        chrom = rec.id.split()[0]
        genome[chrom] = str(rec.seq).upper()
    return genome


def parse_bed(path):
    intervals = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            intervals.setdefault(chrom, []).append((start, end))
    return intervals


def parse_genes_from_gtf(gtf_path):
    genes = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end = (
                parts[0],
                parts[1],
                parts[2],
                int(parts[3]),
                int(parts[4]),
            )
            if feature != "gene":
                continue
            # GTF is 1-based inclusive; convert to BED-like 0-based half-open
            genes.setdefault(chrom, []).append((start - 1, end))
    return genes


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def overlaps_any(merged_intervals, start, end):
    # binary-search style linear fallback (fast enough for this size if merged)
    lo, hi = 0, len(merged_intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e = merged_intervals[mid]
        if end <= s:
            hi = mid - 1
        elif start >= e:
            lo = mid + 1
        else:
            return True
    return False


def sample_negative_windows(genome, exclude_by_chr, n_needed, window):
    chroms = [
        c for c in genome.keys() if len(genome[c]) >= window and c in exclude_by_chr
    ]
    weights = [len(genome[c]) for c in chroms]
    negatives = []
    seen = set()

    max_tries = n_needed * 200
    tries = 0
    while len(negatives) < n_needed and tries < max_tries:
        tries += 1
        chrom = random.choices(chroms, weights=weights, k=1)[0]
        clen = len(genome[chrom])
        start = random.randint(0, clen - window)
        end = start + window
        key = (chrom, start, end)
        if key in seen:
            continue
        if overlaps_any(exclude_by_chr[chrom], start, end):
            continue
        seq = genome[chrom][start:end]
        if "N" in seq:
            continue
        seen.add(key)
        negatives.append((chrom, start, end, seq))

    if len(negatives) < n_needed:
        raise RuntimeError(
            f"Could only sample {len(negatives)} negatives out of {n_needed}. Increase tries or relax constraints."
        )
    return negatives


def read_fasta_records(path):
    rows = []
    for r in SeqIO.parse(str(path), "fasta"):
        rows.append((r.id, str(r.seq).upper()))
    return rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading genome...")
    genome = load_genome(GENOME_FA)

    print("Reading positive BED...")
    pos_bed = parse_bed(POS_BED)
    n_pos = sum(len(v) for v in pos_bed.values())
    print("Positives in BED:", n_pos)

    print("Parsing genes from GTF...")
    genes = parse_genes_from_gtf(GTF)

    print("Building exclusion intervals (genes + positives)...")
    exclude_by_chr = {}
    for chrom in genome.keys():
        intervals = []
        intervals.extend(genes.get(chrom, []))
        intervals.extend(pos_bed.get(chrom, []))
        exclude_by_chr[chrom] = merge_intervals(intervals)

    print("Sampling negatives...")
    negatives = sample_negative_windows(genome, exclude_by_chr, n_pos, WINDOW)
    print("Negatives sampled:", len(negatives))

    print("Writing negatives.bed and negatives.fa...")
    with open(NEG_BED_OUT, "w") as bed_f, open(NEG_FA_OUT, "w") as fa_f:
        for i, (chrom, start, end, seq) in enumerate(negatives, 1):
            bed_f.write(f"{chrom}\t{start}\t{end}\tNEG_{i}\n")
            fa_f.write(f">NEG_{i}|{chrom}:{start}-{end}\n{seq}\n")

    print("Building CSV...")
    pos_records = read_fasta_records(POS_FA)
    # enforce 400 bp
    pos_df = pd.DataFrame(
        [
            {"id": rid, "sequence": seq[:WINDOW], "label": 1}
            for rid, seq in pos_records
            if len(seq) >= WINDOW and "N" not in seq[:WINDOW]
        ]
    )

    neg_df = pd.DataFrame(
        [
            {"id": f"NEG_{i + 1}", "sequence": seq, "label": 0}
            for i, (_, _, _, seq) in enumerate(negatives)
        ]
    )

    n = min(len(pos_df), len(neg_df))
    pos_df = pos_df.sample(n=n, random_state=SEED)
    neg_df = neg_df.sample(n=n, random_state=SEED)

    df = (
        pd.concat([pos_df, neg_df], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    df.to_csv(CSV_OUT, index=False)

    print(f"Saved: {NEG_BED_OUT}")
    print(f"Saved: {NEG_FA_OUT}")
    print(f"Saved: {CSV_OUT} with shape {df.shape}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
