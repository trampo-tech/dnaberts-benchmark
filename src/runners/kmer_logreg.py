"""K-mer frequency + Logistic Regression baseline runner."""

from __future__ import annotations

import itertools
import time
from collections import Counter

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from services.benchmark_logger import (
    append_leaderboard_row,
    build_run_id,
    get_git_commit,
    save_run_summary,
    utc_timestamp,
)


def _build_kmer_vocab(k: int) -> dict[str, int]:
    bases = "ACGT"
    return {
        "".join(combo): idx
        for idx, combo in enumerate(itertools.product(bases, repeat=k))
    }


def _sequences_to_kmer_freqs(
    sequences: list[str], k: int, vocab: dict[str, int]
) -> np.ndarray:
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


def _evaluate_split(
    model: LogisticRegression, X: np.ndarray, y: np.ndarray, prefix: str
) -> dict[str, float]:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    return {
        f"{prefix}_accuracy": accuracy_score(y, preds),
        f"{prefix}_f1": f1_score(y, preds),
        f"{prefix}_precision": precision_score(y, preds),
        f"{prefix}_recall": recall_score(y, preds),
        f"{prefix}_roc_auc": roc_auc_score(y, probs),
    }


def run(cfg: DictConfig) -> None:
    k = int(cfg.model.kmer)
    max_iter = int(cfg.model.max_iter)
    seed = cfg.seed
    text_col = cfg.data.text_col
    label_col = cfg.data.label_col

    train_df = pd.read_csv(cfg.data.train_csv)
    val_df = pd.read_csv(cfg.data.val_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    vocab = _build_kmer_vocab(k)
    print(f"K-mer size: {k}  →  {len(vocab)} features")

    print("Extracting features...")
    X_train = _sequences_to_kmer_freqs(train_df[text_col].tolist(), k, vocab)
    y_train = train_df[label_col].values

    X_val = _sequences_to_kmer_freqs(val_df[text_col].tolist(), k, vocab)
    y_val = val_df[label_col].values

    X_test = _sequences_to_kmer_freqs(test_df[text_col].tolist(), k, vocab)
    y_test = test_df[label_col].values

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("Training LogisticRegression...")
    started = time.perf_counter()
    clf = LogisticRegression(max_iter=max_iter, random_state=seed, solver="lbfgs")
    clf.fit(X_train, y_train)
    runtime = time.perf_counter() - started

    val_metrics = _evaluate_split(clf, X_val, y_val, "validation")
    test_metrics = _evaluate_split(clf, X_test, y_test, "test")

    print(f"\nTraining time: {runtime:.2f}s")
    print("\nValidation:")
    for key, v in val_metrics.items():
        print(f"  {key}: {v:.4f}")
    print("\nTest:")
    for key, v in test_metrics.items():
        print(f"  {key}: {v:.4f}")

    model_name = cfg.model.name
    run_id = build_run_id(cfg.experiment_name, model_name)

    summary = {
        "timestamp": utc_timestamp(),
        "run_id": run_id,
        "status": "completed",
        "error": None,
        "experiment": cfg.experiment_name,
        "model_name": model_name,
        "model_fallback": False,
        "model_fallback_reason": None,
        "seed": seed,
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

    _ = save_run_summary(
        f"runs/{cfg.experiment_name}/{model_name}", summary
    )
    _ = append_leaderboard_row(cfg.train.leaderboard_csv, summary)
    print(f"\nAppended to {cfg.train.leaderboard_csv}")
