from __future__ import annotations

import json
from pathlib import Path

import hydra
import pandas as pd
from pyfaidx import Fasta
from omegaconf import DictConfig


DEFAULT_INPUT_DIR = Path("data/raw/bend/histone_modification")
DEFAULT_OUTPUT_DIR = Path("data/processed/bend_histone")
TRAIN_CHROMS = {f"chr{i}" for i in range(1, 17)}
VAL_CHROMS = {f"chr{i}" for i in range(17, 20)}
TEST_CHROMS = {"chr20", "chr21", "chr22", "chrX"}


def _resolve_chromosome(genome: Fasta, chrom: str) -> str | None:
    if chrom in genome:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in genome:
        return chrom[3:]
    if not chrom.startswith("chr") and f"chr{chrom}" in genome:
        return f"chr{chrom}"
    return None


def _normalize_chrom_name(chrom: str) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def _infer_split(chrom: str) -> str | None:
    normalized = _normalize_chrom_name(chrom)
    if normalized in TRAIN_CHROMS:
        return "train"
    if normalized in VAL_CHROMS:
        return "validation"
    if normalized in TEST_CHROMS:
        return "test"
    return None


def _candidate_label_columns(df: pd.DataFrame) -> list[str]:
    label_columns: list[str] = []
    for column in df.columns[3:]:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().mean() < 0.95:
            continue
        unique_values = set(series.dropna().astype(int).unique())
        if unique_values.issubset({0, 1}):
            label_columns.append(str(column))
    return label_columns


def _load_histone_bed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", header=None)
    if df.empty:
        return pd.DataFrame()

    header_candidates = {str(value).strip().lower() for value in df.iloc[0].tolist()}
    has_header = {"chrom", "start", "end"}.issubset(header_candidates)
    if has_header:
        columns = [str(value).strip().lower() for value in df.iloc[0].tolist()]
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = columns
        metadata = {"chrom", "start", "end", "name", "score", "strand", "split", "region_id", "window_idx"}
        label_columns = [column for column in df.columns if column not in metadata]
        if len(label_columns) < 18:
            label_columns = _candidate_label_columns(df)
    else:
        label_columns = _candidate_label_columns(df)
        if len(label_columns) < 18:
            label_columns = [str(column) for column in df.columns[3:21]]
        rename_map = {df.columns[0]: "chrom", df.columns[1]: "start", df.columns[2]: "end"}
        for index, column in enumerate(label_columns[:18]):
            rename_map[column] = f"label_{index}"
        df = df.rename(columns=rename_map)
        label_columns = [f"label_{index}" for index in range(min(18, len(label_columns)))]

    if len(label_columns) < 18:
        raise SystemExit(f"Could not identify 18 binary label columns in {path}")

    label_columns = list(label_columns[:18])
    keep_columns = ["chrom", "start", "end", *label_columns]
    missing = [column for column in keep_columns if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {path}: {', '.join(missing)}")

    out = df[keep_columns].copy()
    out["chrom"] = out["chrom"].astype(str)
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    for index, column in enumerate(label_columns):
        out[f"label_{index}"] = pd.to_numeric(out[column], errors="raise").astype(int)
        if column != f"label_{index}":
            del out[column]
    return out[["chrom", "start", "end", *[f"label_{index}" for index in range(18)]]]


def _load_all_histone_rows(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.rglob("*.bed*"))
    if not paths:
        raise SystemExit(f"No BED files found under {input_dir}")

    frames = []
    for path in paths:
        frame = _load_histone_bed(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise SystemExit(f"All BED files were empty under {input_dir}")
    return pd.concat(frames, ignore_index=True)


def _parse_chromosomes(chromosomes: object) -> set[str] | None:
    if chromosomes is None:
        return None
    if isinstance(chromosomes, str):
        values = chromosomes.split(",")
    else:
        values = list(chromosomes)
    normalized = {str(chrom).strip() for chrom in values if str(chrom).strip()}
    return normalized or None


@hydra.main(version_base=None, config_path="../src/config", config_name="prepare_bend_histone")
def main(cfg: DictConfig) -> None:
    window_size = int(cfg.window_size)
    label_span_bp = int(cfg.label_span_bp)
    stride = int(cfg.stride) if cfg.stride is not None else window_size
    if window_size % label_span_bp != 0:
        raise SystemExit("window-size must be divisible by label-span-bp")
    if stride % label_span_bp != 0:
        raise SystemExit("stride must be divisible by label-span-bp")

    bins_per_window = window_size // label_span_bp
    stride_bins = stride // label_span_bp
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    genome = Fasta(str(cfg.genome_fasta), sequence_always_upper=True, as_raw=False)

    df = _load_all_histone_rows(input_dir)
    chromosome_filter = _parse_chromosomes(cfg.chromosomes)
    if chromosome_filter:
        df = df[df["chrom"].isin(chromosome_filter)].reset_index(drop=True)

    df = df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    split_rows: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    window_counters: dict[str, int] = {}

    for chrom, chrom_df in df.groupby("chrom", sort=True):
        split_name = _infer_split(chrom)
        if split_name is None:
            continue
        resolved_chrom = _resolve_chromosome(genome, chrom)
        if resolved_chrom is None:
            print(f"[WARN] Skipping chromosome missing from FASTA: {chrom}")
            continue

        chrom_df = chrom_df.reset_index(drop=True)
        for row_index in range(0, len(chrom_df) - bins_per_window + 1, stride_bins):
            window = chrom_df.iloc[row_index : row_index + bins_per_window]
            expected_starts = [
                int(window.iloc[0]["start"]) + offset * label_span_bp
                for offset in range(bins_per_window)
            ]
            if window["start"].tolist() != expected_starts:
                continue

            window_start = int(window.iloc[0]["start"])
            window_end = window_start + window_size
            sequence = genome[resolved_chrom][window_start:window_end].seq.upper()
            if len(sequence) != window_size:
                continue

            labels = window[[f"label_{index}" for index in range(18)]].astype(int).values.tolist()
            label_payload: list[list[int]] | list[int]
            if len(labels) == 1:
                label_payload = labels[0]
            else:
                label_payload = labels

            window_idx = window_counters.get(chrom, 0)
            window_counters[chrom] = window_idx + 1
            split_rows[split_name].append(
                {
                    "region_id": _normalize_chrom_name(chrom),
                    "window_idx": window_idx,
                    "sequence": sequence,
                    "labels": json.dumps(label_payload),
                }
            )

    for split_name, rows in split_rows.items():
        out_path = output_dir / f"{split_name}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Wrote {out_path} (rows={len(rows)})")


if __name__ == "__main__":
    main()