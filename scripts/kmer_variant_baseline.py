from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a zero-shot k-mer L1 baseline on processed BEND variant CSVs."
    )
    parser.add_argument("--input-csv", required=True, help="Processed variant CSV path.")
    parser.add_argument(
        "--metric",
        required=True,
        choices=["spearman", "auroc"],
        help="Evaluation metric for the task.",
    )
    parser.add_argument("--k", type=int, default=6, help="k-mer size.")
    parser.add_argument(
        "--ref-col",
        default="ref_sequence",
        help="Column name for REF sequences.",
    )
    parser.add_argument(
        "--alt-col",
        default="alt_sequence",
        help="Column name for ALT sequences.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Column name for labels.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output path for per-variant baseline distances.",
    )
    return parser.parse_args()


def _build_kmer_vocab(k: int) -> dict[str, int]:
    bases = "ACGT"
    return {
        "".join(combo): index
        for index, combo in enumerate(itertools.product(bases, repeat=k))
    }


def _sequence_to_freqs(sequence: str, k: int, vocab: dict[str, int]) -> np.ndarray:
    counts: Counter[str] = Counter()
    sequence = sequence.upper()
    for index in range(len(sequence) - k + 1):
        kmer = sequence[index : index + k]
        if kmer in vocab:
            counts[kmer] += 1

    freqs = np.zeros(len(vocab), dtype=np.float32)
    total = sum(counts.values()) or 1
    for kmer, count in counts.items():
        freqs[vocab[kmer]] = count / total
    return freqs


def _evaluate(metric: str, labels: np.ndarray, distances: np.ndarray) -> float:
    if metric == "spearman":
        correlation = spearmanr(distances, labels).correlation
        return float(0.0 if correlation is None or np.isnan(correlation) else correlation)
    unique_labels = np.unique(labels)
    return float(roc_auc_score(labels, distances)) if unique_labels.size > 1 else 0.0


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    vocab = _build_kmer_vocab(args.k)

    distances = []
    for row in df.itertuples(index=False):
        ref_sequence = getattr(row, args.ref_col)
        alt_sequence = getattr(row, args.alt_col)
        ref_freqs = _sequence_to_freqs(str(ref_sequence), args.k, vocab)
        alt_freqs = _sequence_to_freqs(str(alt_sequence), args.k, vocab)
        distances.append(float(np.abs(ref_freqs - alt_freqs).sum()))

    distances_array = np.asarray(distances, dtype=np.float32)
    labels = df[args.label_col].to_numpy()
    score = _evaluate(args.metric, labels, distances_array)
    metric_name = "spearman_r" if args.metric == "spearman" else "auroc"
    print(f"{metric_name}: {score:.6f}")

    output_csv = args.output_csv
    if output_csv is None:
        input_name = Path(args.input_csv).stem
        output_csv = str(Path("reports") / f"kmer_variant_baseline_{input_name}.csv")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = df.copy()
    result_df["kmer_l1_distance"] = distances_array
    result_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()