"""Non-neural baseline: k-mer frequency features + Logistic Regression.

Reads the same train/val/test CSVs used by the deep-learning pipeline and
reports accuracy, F1, precision, recall, and ROC-AUC so results are directly
comparable.

Usage:
    uv run python scripts/baseline_kmer_logreg.py \
        --train-csv data/processed/tata_vs_background/train.csv \
        --val-csv   data/processed/tata_vs_background/val.csv \
        --test-csv  data/processed/tata_vs_background/test.csv

    # or with a different k:
    uv run python scripts/baseline_kmer_logreg.py \
        --train-csv data/processed/tataless_vs_background/train.csv \
        --val-csv   data/processed/tataless_vs_background/val.csv \
        --test-csv  data/processed/tataless_vs_background/test.csv \
        --kmer 6
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-mer LogReg baseline for DNA classification.")
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--val-csv", type=Path, required=True)
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--text-col", default="sequence")
    p.add_argument("--label-col", default="label")
    p.add_argument("--kmer", type=int, default=4, help="k-mer size (default 4 → 256 features)")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--leaderboard-csv",
        type=Path,
        default=Path("reports/benchmark_results.csv"),
        help="Append results to this leaderboard CSV.",
    )
    p.add_argument("--experiment-name", default=None, help="Experiment name for leaderboard. Inferred from train-csv if omitted.")
    return p.parse_args()


def build_kmer_vocab(k: int) -> dict[str, int]:
    bases = "ACGT"
    return {
        "".join(combo): idx
        for idx, combo in enumerate(itertools.product(bases, repeat=k))
    }


def sequences_to_kmer_freqs(sequences: list[str], k: int, vocab: dict[str, int]) -> np.ndarray:
    n_features = len(vocab)
    X = np.zeros((len(sequences), n_features), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq = seq.upper()
        counts: Counter[str] = Counter()
        for j in range(len(seq) - k + 1):
            kmer = seq[j : j + k]
            if kmer in vocab:
                counts[kmer] += 1
        total = sum(counts.values()) or 1
        for kmer, count in counts.items():
            X[i, vocab[kmer]] = count / total

    return X


def evaluate_split(model: LogisticRegression, X: np.ndarray, y: np.ndarray, prefix: str) -> dict[str, float]:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        f"{prefix}_accuracy": accuracy_score(y, preds),
        f"{prefix}_f1": f1_score(y, preds),
        f"{prefix}_precision": precision_score(y, preds),
        f"{prefix}_recall": recall_score(y, preds),
        f"{prefix}_roc_auc": roc_auc_score(y, probs),
    }
    return metrics


def main():
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    vocab = build_kmer_vocab(args.kmer)
    print(f"K-mer size: {args.kmer}  →  {len(vocab)} features")

    print("Extracting features...")
    X_train = sequences_to_kmer_freqs(train_df[args.text_col].tolist(), args.kmer, vocab)
    y_train = train_df[args.label_col].values

    X_val = sequences_to_kmer_freqs(val_df[args.text_col].tolist(), args.kmer, vocab)
    y_val = val_df[args.label_col].values

    X_test = sequences_to_kmer_freqs(test_df[args.text_col].tolist(), args.kmer, vocab)
    y_test = test_df[args.label_col].values

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("Training LogisticRegression...")
    started = time.perf_counter()
    clf = LogisticRegression(max_iter=args.max_iter, random_state=args.seed, solver="lbfgs")
    clf.fit(X_train, y_train)
    runtime = time.perf_counter() - started

    val_metrics = evaluate_split(clf, X_val, y_val, "validation")
    test_metrics = evaluate_split(clf, X_test, y_test, "test")

    print(f"\nTraining time: {runtime:.2f}s")
    print("\nValidation:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nTest:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Append to leaderboard
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = args.train_csv.parent.name

    model_name = f"baseline_kmer{args.kmer}_logreg"

    from services.benchmark_logger import append_leaderboard_row, build_run_id, get_git_commit, utc_timestamp

    summary = {
        "timestamp": utc_timestamp(),
        "run_id": build_run_id(experiment_name, model_name),
        "status": "completed",
        "error": None,
        "experiment": experiment_name,
        "model_name": model_name,
        "model_fallback": False,
        "model_fallback_reason": None,
        "seed": args.seed,
        "epochs": None,
        "train_bs": None,
        "eval_bs": None,
        "learning_rate": None,
        "max_length": None,
        "git_commit": get_git_commit(),
        "runtime_seconds": runtime,
        "train": {"train_runtime": runtime},
        "validation": val_metrics,
        "test": test_metrics,
        "output_dir": None,
        "device": "cpu",
    }

    append_leaderboard_row(str(args.leaderboard_csv), summary)
    print(f"\nAppended to {args.leaderboard_csv}")


if __name__ == "__main__":
    main()
