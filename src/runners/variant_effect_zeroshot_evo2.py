"""EVO 2 zero-shot variant-effect (BEND) runner.

Follows the same protocol as ``runners.variant_effect_zeroshot`` but
replaces HuggingFace AutoModel / AutoTokenizer with EVO 2 native APIs.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable

from config.resolver import make_evo2_variant_zeroshot_resolver
from services.benchmark_logger import (
    append_bend_leaderboard_row,
    build_run_id,
    get_git_commit,
    resolve_experiment_name,
    save_run_summary,
    utc_timestamp,
)
from services.metrics import numeric_metrics


# ---------------------------------------------------------------------------
# EVO 2 imports
# ---------------------------------------------------------------------------


def _import_evo2():
    try:
        from evo2 import Evo2

        return Evo2
    except ImportError:
        raise ImportError(
            "evo2 package not found. Install with: pip install evo2\n"
            "Note: flash-attn must be installed first: "
            "pip install flash-attn --no-build-isolation"
        )


# ---------------------------------------------------------------------------
# Environment & backbone loading
# ---------------------------------------------------------------------------


def _setup_environment() -> tuple[torch.device, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.type} ({device_name})")
    return device, device_name


def _load_evo2_backbone(cfg: DictConfig):
    Evo2 = _import_evo2()

    print(f"Loading EVO 2 model: {cfg.model.name} ...")
    evo2_model = Evo2(cfg.model.name)

    # Clone all parameters to remove "inference tensor" marking from
    # torch.inference_mode() used during checkpoint loading.
    for p in evo2_model.model.parameters():
        p.data = p.data.clone()

    for m in evo2_model.model.modules():
        if hasattr(m, "fp8_meta"):
            for key in ("scaling_fwd", "scaling_bwd"):
                if key in m.fp8_meta:
                    meta = m.fp8_meta[key]
                    for attr in ("scale", "amax_history"):
                        t = getattr(meta, attr, None)
                        if isinstance(t, torch.Tensor):
                            setattr(meta, attr, t.clone())

    tokenizer = evo2_model.tokenizer
    pad_token_id = int(tokenizer.pad_id)
    model = evo2_model.model

    return model, tokenizer, pad_token_id


# ---------------------------------------------------------------------------
# Tokenization (char-level, 1 token = 1 nucleotide)
# ---------------------------------------------------------------------------


def _tokenize_batch_evo2(
    tokenizer,
    sequences: list[str],
    token_max_length: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    all_ids: list[list[int]] = []
    all_masks: list[list[int]] = []

    for seq in sequences:
        seq = seq.upper()
        ids = tokenizer.tokenize(seq)
        if len(ids) > token_max_length:
            ids = ids[:token_max_length]
        mask = [1] * len(ids)
        pad_len = token_max_length - len(ids)
        ids = ids + [pad_token_id] * pad_len
        mask = mask + [0] * pad_len
        all_ids.append(ids)
        all_masks.append(mask)

    return {
        "input_ids": torch.tensor(all_ids),
        "attention_mask": torch.tensor(all_masks),
    }


# ---------------------------------------------------------------------------
# Embedding extraction (forward hook)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _embed_batch_evo2(
    model,
    tokenizer,
    sequences: list[str],
    token_max_length: int,
    pad_token_id: int,
    layer_name: str,
    device: torch.device,
) -> torch.Tensor:
    encoded = _tokenize_batch_evo2(
        tokenizer, sequences, token_max_length, pad_token_id
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")

    embeddings: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        embeddings["hidden"] = output.detach()

    layer = model.get_submodule(layer_name)
    handle = layer.register_forward_hook(hook_fn)
    try:
        model(input_ids)
    finally:
        handle.remove()

    hidden = embeddings.get("hidden")
    if hidden is None:
        raise RuntimeError(
            f"Embedding layer {layer_name!r} not captured. "
            f"Check layer_name in config."
        )

    if attention_mask is not None:
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        hidden = hidden * mask

    return hidden.cpu()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(metric_name: str, labels: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    metric_name = str(metric_name)
    if metric_name == "auroc":
        unique_labels = np.unique(labels)
        score = float(roc_auc_score(labels, distances)) if unique_labels.size > 1 else 0.0
        return {"auroc": score}

    raise ValueError(f"Unsupported BEND variant metric: {metric_name}. Expected 'auroc'.")


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def _setup_mlflow(cfg: DictConfig, run_id: str, params: dict[str, Any]):
    if not bool(getattr(cfg.train, "enable_mlflow", False)):
        return None

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(f"MLflow could not be imported: {exc}") from exc

    mlflow_tracking_uri = getattr(
        cfg.train,
        "mlflow_tracking_uri",
        f"file:{os.path.abspath(os.path.join(os.getcwd(), 'mlruns'))}",
    )
    mlflow_experiment = getattr(cfg.train, "mlflow_experiment", cfg.experiment_name)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
    mlflow.start_run(run_name=run_id)
    mlflow.log_params(params)
    return mlflow


def _finalize_mlflow(
    mlflow_client,
    summary: dict[str, Any],
    runtime_seconds: float,
    test_metrics: dict[str, float],
) -> None:
    if mlflow_client is None:
        return
    from typing import cast

    metrics = {
        **numeric_metrics(cast(dict[str, object], test_metrics)),
        "runtime_seconds": runtime_seconds,
        "n_variants": float(summary.get("n_variants", 0) or 0),
        "mean_cosine_distance": float(summary.get("mean_cosine_distance", 0.0) or 0.0),
    }
    if metrics:
        mlflow_client.log_metrics(metrics)
    mlflow_client.log_dict(summary, "run_summary.json")
    mlflow_client.end_run(status="FINISHED" if summary.get("status") == "completed" else "FAILED")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    resolver = make_evo2_variant_zeroshot_resolver(cfg)

    device, device_name = _setup_environment()
    experiment = resolve_experiment_name(cfg)
    run_id = build_run_id(experiment, cfg.model.name)

    variant_df = pd.read_csv(cfg.data.variant_csv)
    ref_col = str(cfg.data.ref_col)
    alt_col = str(cfg.data.alt_col)
    label_col = str(cfg.data.label_col)
    metric_name = str(cfg.data.metric)
    token_max_length = resolver.resolve(
        "token_max_length", type_fn=int, default=cfg.model.max_length
    )
    eval_bs = resolver.resolve("eval_bs", type_fn=int, default=cfg.train.eval_bs)
    layer_name = str(cfg.model.layer_name)

    model, tokenizer, pad_token_id = _load_evo2_backbone(cfg)
    model.to(device)
    model.eval()

    outdir = os.path.join(cfg.train.output_root, experiment, cfg.model.name.replace("/", "_"))
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    distance_csv = os.path.join(reports_dir, f"variant_effect_distances_{experiment}.csv")
    bend_csv = os.path.join(reports_dir, "benchmark_results_bend.csv")

    mlflow_client = _setup_mlflow(
        cfg,
        run_id,
        {
            "run_id": run_id,
            "experiment": experiment,
            "model_name": cfg.model.name,
            "token_max_length": token_max_length,
            "eval_bs": eval_bs,
            "layer_name": layer_name,
            "metric": metric_name,
            "seed": cfg.seed,
            "device": device_name,
        },
    )

    distances: list[float] = []
    started = time.perf_counter()
    status = "failed"
    error_message = None
    test_metrics: dict[str, float] = {}

    try:
        batch_starts = range(0, len(variant_df), eval_bs)
        for start in tqdm(batch_starts, desc="Variant effect (EVO2)", unit="batch"):
            batch = variant_df.iloc[start : start + eval_bs]
            ref_sequences = batch[ref_col].astype(str).tolist()
            alt_sequences = batch[alt_col].astype(str).tolist()
            batch_embeddings = _embed_batch_evo2(
                model,
                tokenizer,
                ref_sequences + alt_sequences,
                token_max_length,
                pad_token_id,
                layer_name,
                device,
            )
            split_idx = len(ref_sequences)
            ref_embeddings = batch_embeddings[:split_idx].reshape(split_idx, -1)
            alt_embeddings = batch_embeddings[split_idx:].reshape(len(alt_sequences), -1)
            batch_distances = 1.0 - F.cosine_similarity(alt_embeddings, ref_embeddings, dim=-1)
            distances.extend(batch_distances.numpy().astype(float).tolist())

        scores = np.asarray(distances, dtype=np.float32)
        labels = variant_df[label_col].to_numpy()
        test_metrics = _evaluate(metric_name, labels, scores)
        status = "completed"

        result_df = variant_df.copy()
        result_df["distance"] = scores
        result_df["run_id"] = run_id
        result_df.to_csv(distance_csv, index=False)
        print(f"Wrote {distance_csv}")
    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        runtime_seconds = float(time.perf_counter() - started)
        summary = {
            "timestamp": utc_timestamp(),
            "run_id": run_id,
            "status": status,
            "error": error_message,
            "experiment": experiment,
            "model_name": cfg.model.name,
            "model_fallback": False,
            "model_fallback_reason": None,
            "seed": cfg.seed,
            "epochs": None,
            "train_bs": None,
            "eval_bs": eval_bs,
            "learning_rate": None,
            "metric": metric_name,
            "max_length": token_max_length,
            "git_commit": get_git_commit(),
            "runtime_seconds": runtime_seconds,
            "n_variants": len(variant_df),
            "mean_cosine_distance": float(np.mean(distances)) if distances else 0.0,
            "train": {},
            "validation": {},
            "test": test_metrics,
            "output_dir": outdir,
            "device": device_name,
        }

        save_run_summary(outdir, summary)
        append_bend_leaderboard_row(bend_csv, summary)
        _finalize_mlflow(mlflow_client, summary, runtime_seconds, test_metrics)
