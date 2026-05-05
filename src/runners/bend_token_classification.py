from __future__ import annotations

import importlib
import inspect
import json
import os
import time
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from modeling.cnn_decoder import CNNDecoder
from modeling.compat import (
    block_triton_imports,
    disable_remote_flash_attention,
    fix_pad_token_id,
    legacy_remote_meta_init_disabled,
    resolve_hidden_size,
)
from services.benchmark_logger import (
    append_bend_leaderboard_row,
    build_run_id,
    get_git_commit,
    resolve_experiment_name,
    save_run_summary,
    utc_timestamp,
)
from services.metrics import compute_multilabel_auroc, numeric_metrics


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


class HistoneWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str, token_max_length: int):
        self.sequences = df[text_col].astype(str).str.upper().tolist()
        self.label_payloads = df[label_col].astype(str).tolist()
        self.tokenizer = tokenizer
        self.token_max_length = token_max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.sequences[index],
            truncation=True,
            padding="max_length",
            max_length=self.token_max_length,
        )
        labels = json.loads(self.label_payloads[index])
        if labels and isinstance(labels[0], int):
            labels = [labels]
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        encoded["labels"] = label_tensor
        return encoded


class HistoneCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        label_tensors = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        max_windows = max(label.shape[0] for label in label_tensors)
        num_labels = label_tensors[0].shape[1]
        labels = torch.zeros((len(label_tensors), max_windows, num_labels), dtype=torch.float32)
        label_mask = torch.zeros((len(label_tensors), max_windows), dtype=torch.bool)
        for index, label_tensor in enumerate(label_tensors):
            labels[index, : label_tensor.shape[0]] = label_tensor
            label_mask[index, : label_tensor.shape[0]] = True

        batch["labels"] = labels
        batch["label_mask"] = label_mask
        return batch


class BendTokenClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, decoder: CNNDecoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, **inputs: Any) -> torch.Tensor:
        model_inputs = _filter_forward_kwargs(self.backbone, inputs)
        outputs = self.backbone(**model_inputs)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None and isinstance(outputs, (tuple, list)) and outputs:
            last_hidden_state = outputs[0]
        if last_hidden_state is None:
            raise ValueError("Backbone output does not contain last_hidden_state.")
        return self.decoder(last_hidden_state)


def _setup_environment(cfg: DictConfig) -> tuple[torch.device, str]:
    if getattr(cfg.model, "disable_triton", False):
        block_triton_imports()
    _import_optional_modules(list(getattr(cfg.model, "import_modules", [])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.type} ({device_name})")
    return device, device_name


def _load_model(cfg: DictConfig, tokenizer, num_labels: int) -> BendTokenClassifier:
    revision = getattr(cfg.model, "revision", None)
    config = AutoConfig.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        revision=revision,
    )
    fix_pad_token_id(config, tokenizer)

    with legacy_remote_meta_init_disabled(cfg.model.name, cfg.model.trust_remote_code):
        backbone = AutoModel.from_pretrained(
            cfg.model.name,
            config=config,
            trust_remote_code=cfg.model.trust_remote_code,
            revision=revision,
        )

    if getattr(cfg.model, "disable_triton", False):
        disable_remote_flash_attention(backbone)

    decoder = CNNDecoder(
        input_size=resolve_hidden_size(backbone.config),
        hidden_size=int(cfg.model.cnn_hidden_size),
        kernel_size=int(cfg.model.cnn_kernel_size),
        output_size=num_labels,
        downsample_stride=int(cfg.model.cnn_downsample_stride),
    )
    return BendTokenClassifier(backbone=backbone, decoder=decoder)


def _masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape[:2]:
        raise ValueError(
            "Decoder output shape does not match label windows. "
            f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    mask = label_mask.unsqueeze(-1).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def _evaluate(model, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        label_mask = batch.pop("label_mask").to(device)
        model_inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(**model_inputs)
        probs = torch.sigmoid(logits)

        all_probs.append(probs[label_mask].cpu())
        all_labels.append(labels[label_mask].cpu())

    if not all_probs:
        return {"auroc_macro": 0.0}

    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return compute_multilabel_auroc(probs, labels)


def _save_best_checkpoint(best_dir: str, model: BendTokenClassifier, tokenizer, cfg: DictConfig) -> None:
    os.makedirs(best_dir, exist_ok=True)
    model.backbone.save_pretrained(best_dir) #type: ignore
    tokenizer.save_pretrained(best_dir)
    torch.save(model.decoder.state_dict(), os.path.join(best_dir, "cnn_decoder.pt"))
    torch.save(
        {
            "cnn_hidden_size": int(cfg.model.cnn_hidden_size),
            "cnn_kernel_size": int(cfg.model.cnn_kernel_size),
            "cnn_downsample_stride": int(cfg.model.cnn_downsample_stride),
        },
        os.path.join(best_dir, "cnn_decoder_config.pt"),
    )


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
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    if mlflow_client is None:
        return
    metrics = {
        **numeric_metrics(train_metrics),
        **numeric_metrics(validation_metrics),
        **numeric_metrics(test_metrics),
        "runtime_seconds": runtime_seconds,
    }
    if metrics:
        mlflow_client.log_metrics(metrics)
    mlflow_client.log_dict(summary, "run_summary.json")
    mlflow_client.end_run(status="FINISHED" if summary.get("status") == "completed" else "FAILED")


def run(cfg: DictConfig) -> None:
    device, device_name = _setup_environment(cfg)
    experiment = resolve_experiment_name(cfg)
    run_id = build_run_id(experiment, cfg.model.name)
    num_labels = int(cfg.data.num_labels)
    token_max_length = int(
        getattr(cfg.data, "token_max_length", 0)
        or getattr(cfg.data, "max_length", 0)
        or cfg.model.max_length
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        revision=getattr(cfg.model, "revision", None),
    )

    train_df = pd.read_csv(cfg.data.train_csv)
    val_df = pd.read_csv(cfg.data.val_csv)
    test_df = pd.read_csv(cfg.data.test_csv)

    train_dataset = HistoneWindowDataset(train_df, tokenizer, cfg.data.text_col, cfg.data.label_col, token_max_length)
    val_dataset = HistoneWindowDataset(val_df, tokenizer, cfg.data.text_col, cfg.data.label_col, token_max_length)
    test_dataset = HistoneWindowDataset(test_df, tokenizer, cfg.data.text_col, cfg.data.label_col, token_max_length)

    collator = HistoneCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(getattr(cfg.data, "train_bs", cfg.train.train_bs)),
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(getattr(cfg.data, "eval_bs", cfg.train.eval_bs)),
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(getattr(cfg.data, "eval_bs", cfg.train.eval_bs)),
        shuffle=False,
        collate_fn=collator,
    )

    model = _load_model(cfg, tokenizer, num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg.train.learning_rate), weight_decay=float(cfg.train.weight_decay))

    gradient_accumulation_steps = int(getattr(cfg.train, "gradient_accumulation_steps", 1) or 1)
    total_optimizer_steps = max(1, (len(train_loader) * int(cfg.train.epochs) + gradient_accumulation_steps - 1) // gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(getattr(cfg.train, "warmup_steps", 0) or 0),
        num_training_steps=total_optimizer_steps,
    )
    use_amp = bool(getattr(cfg.train, "fp16", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    outdir = os.path.join(cfg.train.output_root, experiment, cfg.model.name.replace("/", "_"))
    best_dir = os.path.join(outdir, "best_model")
    bend_csv = os.path.join(os.getcwd(), "reports", "benchmark_results_bend.csv")

    mlflow_client = _setup_mlflow(
        cfg,
        run_id,
        {
            "run_id": run_id,
            "experiment": experiment,
            "model_name": cfg.model.name,
            "token_max_length": token_max_length,
            "num_labels": num_labels,
            "learning_rate": cfg.train.learning_rate,
            "epochs": int(cfg.train.epochs),
            "train_bs": int(getattr(cfg.data, "train_bs", cfg.train.train_bs)),
            "eval_bs": int(getattr(cfg.data, "eval_bs", cfg.train.eval_bs)),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "weight_decay": cfg.train.weight_decay,
            "seed": cfg.seed,
            "device": device_name,
        },
    )

    started = time.perf_counter()
    status = "failed"
    error_message = None
    train_metrics: dict[str, float] = {}
    validation_metrics: dict[str, float] = {}
    test_metrics: dict[str, float] = {}
    best_val_score = float("-inf")

    try:
        for epoch in range(int(cfg.train.epochs)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            step_count = 0

            for step, batch in enumerate(train_loader, start=1):
                labels = batch.pop("labels").to(device)
                label_mask = batch.pop("label_mask").to(device)
                model_inputs = {key: value.to(device) for key, value in batch.items()}

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(**model_inputs)
                    loss = _masked_bce_loss(logits, labels, label_mask)
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
                if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                epoch_loss += float(loss.item()) * gradient_accumulation_steps
                step_count += 1

            train_metrics = {
                "train_loss": epoch_loss / max(step_count, 1),
                "train_runtime": float(time.perf_counter() - started),
                "epoch": float(epoch + 1),
            }
            validation_metrics = _evaluate(model, val_loader, device)
            print(f"Epoch {epoch + 1}: train_loss={train_metrics['train_loss']:.4f}, val_auroc_macro={validation_metrics.get('auroc_macro', 0.0):.4f}")

            current_val = float(validation_metrics.get("auroc_macro", 0.0))
            if current_val > best_val_score:
                best_val_score = current_val
                _save_best_checkpoint(best_dir, model, tokenizer, cfg)

        test_metrics = _evaluate(model, test_loader, device)
        status = "completed"
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
            "epochs": int(cfg.train.epochs),
            "train_bs": int(getattr(cfg.data, "train_bs", cfg.train.train_bs)),
            "eval_bs": int(getattr(cfg.data, "eval_bs", cfg.train.eval_bs)),
            "learning_rate": float(cfg.train.learning_rate),
            "max_length": token_max_length,
            "git_commit": get_git_commit(),
            "runtime_seconds": runtime_seconds,
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "output_dir": outdir,
            "device": device_name,
        }

        save_run_summary(outdir, summary)
        append_bend_leaderboard_row(bend_csv, summary)
        _finalize_mlflow(
            mlflow_client,
            summary,
            runtime_seconds,
            train_metrics,
            validation_metrics,
            test_metrics,
        )