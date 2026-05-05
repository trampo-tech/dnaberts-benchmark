from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


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


@hydra.main(version_base=None, config_path="../src/config", config_name="kmer_variant_baseline")
def main(cfg: DictConfig) -> None:
    df = pd.read_csv(cfg.input_csv)
    vocab = _build_kmer_vocab(int(cfg.k))

    distances = []
    for row in df.itertuples(index=False):
        ref_sequence = getattr(row, str(cfg.ref_col))
        alt_sequence = getattr(row, str(cfg.alt_col))
        ref_freqs = _sequence_to_freqs(str(ref_sequence), int(cfg.k), vocab)
        alt_freqs = _sequence_to_freqs(str(alt_sequence), int(cfg.k), vocab)
        distances.append(float(np.abs(ref_freqs - alt_freqs).sum()))

    distances_array = np.asarray(distances, dtype=np.float32)
    labels = df[str(cfg.label_col)].to_numpy()
    score = _evaluate(str(cfg.metric), labels, distances_array)
    metric_name = "spearman_r" if str(cfg.metric) == "spearman" else "auroc"
    print(f"{metric_name}: {score:.6f}")

    out_path = Path(cfg.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = df.copy()
    result_df["kmer_l1_distance"] = distances_array
    result_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()