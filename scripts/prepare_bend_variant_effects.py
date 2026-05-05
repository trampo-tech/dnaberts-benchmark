from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from pyfaidx import Fasta
from omegaconf import DictConfig


DEFAULT_INPUT_DIR = Path("data/raw/bend/variant_effects")
DEFAULT_OUTPUT_DIR = Path("data/processed/bend_variant_effects")


def _normalize_bed_columns(columns: list[object]) -> list[str]:
    aliases = {
        "#chrom": "chrom",
        "#chromosome": "chrom",
        "chromosome": "chrom",
    }
    normalized: list[str] = []
    for column in columns:
        name = str(column).strip().lower()
        normalized.append(aliases.get(name, name))
    return normalized


def _load_bed(path: Path, label_dtype: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", header=None, dtype=str, low_memory=False)
    if df.empty:
        raise SystemExit(f"Input BED is empty: {path}")

    header_candidates = set(_normalize_bed_columns(df.iloc[0].tolist()))
    has_header = {"chrom", "start", "end"}.issubset(header_candidates)
    if has_header:
        columns = _normalize_bed_columns(df.iloc[0].tolist())
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = columns
    else:
        if df.shape[1] < 6:
            raise SystemExit(f"Expected at least 6 BED columns in {path}, found {df.shape[1]}")
        df = df.iloc[:, :6].copy()
        df.columns = ["chrom", "start", "end", "ref", "alt", "label"]

    required = ["chrom", "start", "end", "ref", "alt", "label"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {path}: {', '.join(missing)}")

    df = df[required].copy()
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["ref"] = df["ref"].astype(str).str.upper()
    df["alt"] = df["alt"].astype(str).str.upper()
    if label_dtype == "float":
        df["label"] = df["label"].astype(float)
    else:
        df["label"] = df["label"].astype(int)
    return df


def _resolve_chromosome(genome: Fasta, chrom: str) -> str | None:
    if chrom in genome:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in genome:
        return chrom[3:]
    if not chrom.startswith("chr") and f"chr{chrom}" in genome:
        return f"chr{chrom}"
    return None


def _extract_sequences(
    genome: Fasta,
    chrom: str,
    start: int,
    end: int,
    ref: str,
    alt: str,
    window_size: int,
) -> tuple[str, str] | None:
    resolved_chrom = _resolve_chromosome(genome, chrom)
    if resolved_chrom is None:
        return None

    ref = ref.upper()
    alt = alt.upper()
    if len(ref) != len(alt):
        return None

    variant_start = int(start)
    variant_end = max(int(end), variant_start + len(ref))
    midpoint = variant_start + (variant_end - variant_start) // 2
    half_window = window_size // 2
    window_start = midpoint - half_window
    window_end = window_start + window_size
    if window_start < 0:
        return None

    chrom_length = len(genome[resolved_chrom])
    if window_end > chrom_length:
        return None

    ref_sequence = genome[resolved_chrom][window_start:window_end].seq.upper()
    if len(ref_sequence) != window_size:
        return None

    local_start = variant_start - window_start
    local_end = local_start + len(ref)
    if local_start < 0 or local_end > len(ref_sequence):
        return None

    if ref_sequence[local_start:local_end] != ref:
        return None

    alt_sequence = ref_sequence[:local_start] + alt + ref_sequence[local_end:]
    if len(alt_sequence) != window_size:
        return None
    return ref_sequence, alt_sequence


def _process_dataset(
    bed_path: Path,
    label_dtype: str,
    genome: Fasta,
    window_size: int,
    chromosome_filter: set[str] | None,
) -> pd.DataFrame:
    df = _load_bed(bed_path, label_dtype=label_dtype)
    if chromosome_filter:
        df = df[df["chrom"].isin(chromosome_filter)].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    skipped_mismatch = 0
    skipped_bounds = 0
    skipped_indel = 0
    skipped_chrom = 0

    for row in df.itertuples(index=False):
        if len(str(row.ref)) != len(str(row.alt)):
            skipped_indel += 1
            continue

        sequences = _extract_sequences(
            genome,
            chrom=str(row.chrom),
            start=int(row.start),
            end=int(row.end),
            ref=str(row.ref),
            alt=str(row.alt),
            window_size=window_size,
        )
        if sequences is None:
            resolved_chrom = _resolve_chromosome(genome, str(row.chrom))
            if resolved_chrom is None:
                skipped_chrom += 1
            else:
                midpoint = int(row.start) + (max(int(row.end), int(row.start) + len(str(row.ref))) - int(row.start)) // 2
                half_window = window_size // 2
                if midpoint - half_window < 0 or midpoint - half_window + window_size > len(genome[resolved_chrom]):
                    skipped_bounds += 1
                else:
                    skipped_mismatch += 1
            continue

        ref_sequence, alt_sequence = sequences
        rows.append(
            {
                "chrom": str(row.chrom),
                "pos": int(row.start),
                "ref_sequence": ref_sequence,
                "alt_sequence": alt_sequence,
                "ref_allele": str(row.ref),
                "alt_allele": str(row.alt),
                "label": row.label,
            }
        )

    print(
        f"Prepared {len(rows)} rows from {bed_path} "
        f"(skipped chrom={skipped_chrom}, bounds={skipped_bounds}, indel={skipped_indel}, mismatch={skipped_mismatch})"
    )
    return pd.DataFrame(rows)


def _parse_chromosomes(chromosomes: object) -> set[str] | None:
    if chromosomes is None:
        return None
    if isinstance(chromosomes, str):
        values = chromosomes.split(",")
    else:
        values = list(chromosomes)
    normalized = {str(chrom).strip() for chrom in values if str(chrom).strip()}
    return normalized or None


@hydra.main(
    version_base=None,
    config_path="../src/config",
    config_name="prepare_bend_variant_effects",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expression_bed = Path(cfg.datasets.expression.input_bed)
    disease_bed = Path(cfg.datasets.disease.input_bed)

    genome = Fasta(str(cfg.genome_fasta), sequence_always_upper=True, as_raw=False)
    chromosome_filter = _parse_chromosomes(cfg.chromosomes)

    expression_df = _process_dataset(
        expression_bed,
        label_dtype=str(cfg.datasets.expression.label_dtype),
        genome=genome,
        window_size=int(cfg.window_size),
        chromosome_filter=chromosome_filter,
    )
    disease_df = _process_dataset(
        disease_bed,
        label_dtype=str(cfg.datasets.disease.label_dtype),
        genome=genome,
        window_size=int(cfg.window_size),
        chromosome_filter=chromosome_filter,
    )

    expression_out = Path(cfg.datasets.expression.output_csv)
    disease_out = Path(cfg.datasets.disease.output_csv)
    expression_df.to_csv(expression_out, index=False)
    disease_df.to_csv(disease_out, index=False)
    print(f"Wrote {expression_out}")
    print(f"Wrote {disease_out}")


if __name__ == "__main__":
    main()