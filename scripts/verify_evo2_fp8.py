"""Verify EVO 2 model is using FP8 via Transformer Engine.

Usage:
    uv run python scripts/verify_evo2_fp8.py
    uv run python scripts/verify_evo2_fp8.py --model_name evo2_7b
"""

from __future__ import annotations

import argparse
import torch
from evo2 import Evo2

TEST_SEQUENCE = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify EVO 2 model uses FP8 via Transformer Engine"
    )
    parser.add_argument(
        "--model_name",
        default="evo2_1b_base",
        help="Model to verify",
    )
    args = parser.parse_args()

    print(f"Loading {args.model_name} ...")
    model = Evo2(args.model_name)

    total_te_modules = sum(1 for m in model.model.modules() if hasattr(m, "fp8_meta"))
    print(f"TE-capable modules: {total_te_modules}")

    if total_te_modules == 0:
        print("\nFAIL: No Transformer Engine modules found — FP8 not available.")
        raise SystemExit(1)

    # Run a forward pass to activate FP8 lazily
    ids = torch.tensor(
        model.tokenizer.tokenize(TEST_SEQUENCE), dtype=int
    ).unsqueeze(0).to("cuda")
    with torch.no_grad():
        model.model.forward(ids)

    fp8_active = 0
    fp8_scales = []
    for name, m in model.model.named_modules():
        if getattr(m, "fp8", False):
            fp8_active += 1
            meta = m.fp8_meta.get("scaling_fwd")
            if meta and hasattr(meta, "scale"):
                s = meta.scale
                val = s.item() if s.numel() == 1 else float(s.flatten()[0])
                fp8_scales.append(val)

    print(f"FP8-active TE modules (after forward): {fp8_active}")

    if fp8_active == 0:
        print("\nFAIL: No FP8-active modules — FP8 is not engaged.")
        print("TE may be installed but the model fell back to bf16.")
        raise SystemExit(1)

    # Check that scales are meaningful (not all 1.0 = never activated)
    if fp8_scales:
        non_init = [s for s in fp8_scales if s != 1.0]
        print(f"  Scales with non-init values: {len(non_init)}/{len(fp8_scales)}")
        if non_init:
            print(f"  Sample scales: {non_init[:3]}")
        if len(non_init) < 1:
            print("\nWARN: All scales = 1.0 — FP8 may not be computing.")
    else:
        print("\nWARN: No scaling metadata found on FP8 modules.")

    print("\nPASS: FP8 is active on this model.")


if __name__ == "__main__":
    main()
