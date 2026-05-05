from __future__ import annotations

import importlib
import inspect
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoModel, AutoTokenizer

from modeling.compat import (
    block_triton_imports,
    disable_remote_flash_attention,
    fix_pad_token_id,
    legacy_remote_meta_init_disabled,
)
from services.benchmark_logger import (
    append_bend_leaderboard_row,
    build_run_id,
    get_git_commit,
    resolve_experiment_name,
    save_run_summary,
    utc_timestamp,
)
from services.metrics import numeric_metrics


def _import_optional_modules(modules: list[str] | None) -> None:
    for module_name in modules or []:
        importlib.import_module(module_name)


def _filter_forward_kwargs(module, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(module.forward)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs
    accepted = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in accepted}


def _setup_environment(cfg: DictConfig) -> tuple[torch.device, str]:
    if getattr(cfg.model, "disable_triton", False):
        block_triton_imports()

    _import_optional_modules(list(getattr(cfg.model, "import_modules", [])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.type} ({device_name})")
    return device, device_name


def _load_backbone(cfg: DictConfig, tokenizer) -> torch.nn.Module:
    revision = getattr(cfg.model, "revision", None)
    config = AutoConfig.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        revision=revision,
    )
    fix_pad_token_id(config, tokenizer)

    with legacy_remote_meta_init_disabled(cfg.model.name, cfg.model.trust_remote_code):
        model = AutoModel.from_pretrained(
            cfg.model.name,
            config=config,
            trust_remote_code=cfg.model.trust_remote_code,
            revision=revision,
        )

    if getattr(cfg.model, "disable_triton", False):
        disable_remote_flash_attention(model)

    return model


def _tokenize_batch(tokenizer, sequences: list[str], token_max_length: int):
    try:
        encoded = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=token_max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        return encoded, True
    except (NotImplementedError, TypeError, ValueError, RuntimeError):
        encoded = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=token_max_length,
            return_tensors="pt",
        )
        return encoded, False


def _center_indices(
    offset_mapping,
    attention_mask: torch.Tensor,
    target_position: int,
) -> torch.Tensor:
    if offset_mapping is None:
        valid_lengths = attention_mask.sum(dim=1)
        return torch.clamp((valid_lengths // 2).to(torch.long), min=0)

    if not isinstance(offset_mapping, torch.Tensor):
        offset_mapping = torch.as_tensor(offset_mapping)

    indices: list[int] = []
    valid_lengths = attention_mask.sum(dim=1).tolist()
    for batch_idx, valid_len in enumerate(valid_lengths):
        chosen_index = None
        for token_idx in range(int(valid_len)):
            start, end = offset_mapping[batch_idx, token_idx].tolist()
            if end <= start:
                continue
            if start <= target_position < end:
                chosen_index = token_idx
                break

        if chosen_index is None:
            chosen_index = max(0, int(valid_len // 2))
        indices.append(chosen_index)

    return torch.tensor(indices, dtype=torch.long)


@torch.no_grad()
def _embed_batch(
    model,
    tokenizer,
    sequences: list[str],
    token_max_length: int,
    device: torch.device,
    *,
    warned_about_offsets: bool,
) -> tuple[torch.Tensor, bool]:
    encoded, has_offsets = _tokenize_batch(tokenizer, sequences, token_max_length)
    if not has_offsets and not warned_about_offsets:
        print("[WARN] Tokenizer does not support offset mappings; falling back to seq_len // 2.")

    offset_mapping = encoded.pop("offset_mapping", None)
    attention_mask = encoded["attention_mask"]
    center_indices = _center_indices(offset_mapping, attention_mask, target_position=256)

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    model_inputs = _filter_forward_kwargs(model, model_inputs)

    outputs = model(**model_inputs)
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None and isinstance(outputs, (tuple, list)) and outputs:
        last_hidden_state = outputs[0]
    if last_hidden_state is None:
        raise ValueError("Backbone output does not contain last_hidden_state.")

    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    embeddings = last_hidden_state[batch_indices, center_indices.to(last_hidden_state.device)]
    return embeddings.cpu(), warned_about_offsets or not has_offsets


def _evaluate(metric_name: str, labels: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    metric_name = str(metric_name)
    if metric_name == "spearman":
        correlation = spearmanr(distances, labels).correlation
        if correlation is None or np.isnan(correlation):
            correlation = 0.0
        return {"spearman_r": float(correlation)}

    if metric_name == "auroc":
        unique_labels = np.unique(labels)
        score = float(roc_auc_score(labels, distances)) if unique_labels.size > 1 else 0.0
        return {"auroc": score}

    raise ValueError(f"Unsupported BEND metric: {metric_name}")


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
    metrics = {
        **numeric_metrics(test_metrics),
        "runtime_seconds": runtime_seconds,
        "n_variants": float(summary.get("n_variants", 0) or 0),
        "mean_cosine_distance": float(summary.get("mean_cosine_distance", 0.0) or 0.0),
    }
    if metrics:
        mlflow_client.log_metrics(metrics)
    mlflow_client.log_dict(summary, "run_summary.json")
    mlflow_client.end_run(status="FINISHED" if summary.get("status") == "completed" else "FAILED")


def run(cfg: DictConfig) -> None:
    device, device_name = _setup_environment(cfg)
    experiment = resolve_experiment_name(cfg)
    run_id = build_run_id(experiment, cfg.model.name)

    variant_df = pd.read_csv(cfg.data.variant_csv)
    ref_col = str(cfg.data.ref_col)
    alt_col = str(cfg.data.alt_col)
    label_col = str(cfg.data.label_col)
    metric_name = str(cfg.data.metric)
    token_max_length = int(
        getattr(cfg.data, "token_max_length", 0)
        or getattr(cfg.data, "max_length", 0)
        or cfg.model.max_length
    )
    eval_bs = int(getattr(cfg.data, "eval_bs", cfg.train.eval_bs))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        revision=getattr(cfg.model, "revision", None),
    )
    model = _load_backbone(cfg, tokenizer)
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
            "metric": metric_name,
            "seed": cfg.seed,
            "device": device_name,
        },
    )

    warned_about_offsets = False
    distances: list[float] = []
    started = time.perf_counter()
    status = "failed"
    error_message = None
    test_metrics: dict[str, float] = {}

    try:
        for start in range(0, len(variant_df), eval_bs):
            batch = variant_df.iloc[start : start + eval_bs]
            ref_embeddings, warned_about_offsets = _embed_batch(
                model,
                tokenizer,
                batch[ref_col].astype(str).tolist(),
                token_max_length,
                device,
                warned_about_offsets=warned_about_offsets,
            )
            alt_embeddings, warned_about_offsets = _embed_batch(
                model,
                tokenizer,
                batch[alt_col].astype(str).tolist(),
                token_max_length,
                device,
                warned_about_offsets=warned_about_offsets,
            )
            batch_distances = 1.0 - F.cosine_similarity(ref_embeddings, alt_embeddings, dim=-1)
            distances.extend(batch_distances.numpy().astype(float).tolist())

        scores = np.asarray(distances, dtype=np.float32)
        labels = variant_df[label_col].to_numpy()
        test_metrics = _evaluate(metric_name, labels, scores)
        status = "completed"

        result_df = variant_df.copy()
        result_df["cosine_distance"] = scores
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