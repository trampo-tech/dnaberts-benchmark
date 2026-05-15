"""Verify EVO 2 model is loaded correctly 
original from: https://github.com/ArcInstitute/evo2/blob/main/evo2/test/test_evo2.py

Usage:
    uv run python scripts/verify_evo2.py
    uv run python scripts/verify_evo2.py --model_name evo2_1b_base
"""

from __future__ import annotations

import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
from importlib import resources
from pathlib import Path

from evo2 import Evo2
from evo2.utils import MODEL_NAMES

EXPECTED_METRICS = {
    "evo2_40b":       {"loss": 0.2159424,       "acc": 91.673},
    "evo2_20b":       {"loss": 0.2166748046875, "acc": 91.666},
    "evo2_7b":        {"loss": 0.3476563,       "acc": 86.346},
    "evo2_7b_base":   {"loss": 0.3520508,       "acc": 85.921},
    "evo2_1b_base":   {"loss": 0.501953125,     "acc": 79.556},
    "evo2_40b_base":  {"loss": 0.2149658,       "acc": 91.741},
}
EPS = 1e-3


def read_prompts() -> list[str]:
    """Read built-in test sequences shipped with the evo2 package."""
    with resources.files("evo2.test.data").joinpath("prompts.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [row[0] for row in reader]


def test_forward_pass(model, sequences: list[str]) -> tuple[list[float], list[float]]:
    losses = []
    accuracies = []

    for i, seq in enumerate(sequences):
        input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=int).to("cuda:0")

        with torch.inference_mode():
            logits, _ = model.model.forward(input_ids.unsqueeze(0))

        target_ids = input_ids[1:]
        pred_logits = logits[0, :-1, :]

        loss = F.cross_entropy(pred_logits, target_ids.long()).item()
        pred_tokens = torch.argmax(pred_logits, dim=-1)
        accuracy = (target_ids == pred_tokens).float().mean().item()

        losses.append(loss)
        accuracies.append(accuracy)

        status = "\033[31mWARN\033[0m" if accuracy < 0.5 else "\033[32mOK\033[0m"
        print(
            f"  Seq {i + 1}/{len(sequences)}  "
            f"loss={loss:.4f}  acc={accuracy:.2%}  [{status}]"
        )

    return accuracies, losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify EVO 2 model forward pass")
    parser.add_argument(
        "--model_name",
        choices=[m for m in MODEL_NAMES if m in EXPECTED_METRICS],
        default="evo2_1b_base",
        help="Model to verify",
    )
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    print(f"Loading {args.model_name} ...")
    model = Evo2(args.model_name)

    print("Running forward-pass verification ...")
    sequences = read_prompts()
    accuracies, losses = test_forward_pass(model, sequences)

    mean_loss = float(np.mean(losses))
    mean_acc = float(np.mean(accuracies)) * 100
    expected = EXPECTED_METRICS[args.model_name]

    print(f"\nMean loss:     {mean_loss:.6f}  (expected {expected['loss']:.6f})")
    print(f"Mean accuracy: {mean_acc:.3f}%  (expected {expected['acc']:.3f}%)")

    loss_ok = abs(mean_loss - expected["loss"]) < EPS
    acc_ok = abs(mean_acc - expected["acc"]) < 0.5  # 0.5% tolerance on accuracy

    if loss_ok and acc_ok:
        print("\nTest PASSED — model outputs match expected values.")
    else:
        print("\nTest FAILED — model outputs deviate from expected values.")
        if not loss_ok:
            print(
                "  Loss mismatch may indicate: wrong CUDA arch, missing TE, "
                "or corrupted checkpoint."
            )
        if not acc_ok:
            print("  Accuracy mismatch may indicate numerical issues.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
