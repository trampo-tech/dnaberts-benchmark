from __future__ import annotations

import importlib
import inspect
import os
import time
from typing import Any, cast

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

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

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
        try:
            model = AutoModel.from_pretrained(
                cfg.model.name,
                config=config,
                trust_remote_code=cfg.model.trust_remote_code,
                revision=revision,
            )
        except RuntimeError as exc:
            if "size mismatch" not in str(exc):
                raise

            mlm_model = AutoModelForMaskedLM.from_pretrained(
                cfg.model.name,
                config=config,
                trust_remote_code=cfg.model.trust_remote_code,
                revision=revision,
            )
            model = getattr(mlm_model, "esm", None)
            if model is None:
                raise

    if getattr(cfg.model, "disable_triton", False):
        disable_remote_flash_attention(model)

    return model


def _tokenize_batch(tokenizer, sequences: list[str], token_max_length: int):
    return tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=token_max_length,
        return_tensors="pt",
    )


@torch.no_grad()
def _embed_batch(
    model,
    tokenizer,
    sequences: list[str],
    token_max_length: int,
    device: torch.device,
) -> torch.Tensor:
    encoded = _tokenize_batch(tokenizer, sequences, token_max_length)
    attention_mask = encoded.get("attention_mask")

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

    if attention_mask is not None:
        masked_hidden_state = last_hidden_state * attention_mask.to(last_hidden_state.device).unsqueeze(-1)
        return masked_hidden_state.cpu()

    return last_hidden_state.cpu()


def _evaluate(metric_name: str, labels: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    metric_name = str(metric_name)
    if metric_name == "auroc":
        unique_labels = np.unique(labels)
        score = float(roc_auc_score(labels, distances)) if unique_labels.size > 1 else 0.0
        return {"auroc": score}

    raise ValueError(f"Unsupported BEND variant metric: {metric_name}. Expected 'auroc'.")


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
        **numeric_metrics(cast(dict[str, object], test_metrics)),
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

    distances: list[float] = []
    started = time.perf_counter()
    status = "failed"
    error_message = None
    test_metrics: dict[str, float] = {}

    try:
        batch_starts = range(0, len(variant_df), eval_bs)
        for start in tqdm(batch_starts, desc="Variant effect", unit="batch"):
            batch = variant_df.iloc[start : start + eval_bs]
            ref_sequences = batch[ref_col].astype(str).tolist()
            alt_sequences = batch[alt_col].astype(str).tolist()
            batch_embeddings = _embed_batch(
                model,
                tokenizer,
                ref_sequences + alt_sequences,
                token_max_length,
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