"""EVO 2 runner — Hyena DNA language model with PEFT LoRA fine-tuning for sequence classification."""

from __future__ import annotations

import contextlib
import os
import time
from types import SimpleNamespace
from typing import Any

import evaluate
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    default_data_collator,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from runners.transformer import (
    BenchmarkTrainer,
    _build_training_args,
    _resolve_train_override,
)
from services.benchmark_logger import (
    append_leaderboard_row,
    build_run_id,
    get_git_commit,
    resolve_experiment_name,
    save_run_summary,
    utc_timestamp,
)
from services.metrics import (
    compute_metrics_from_logits,
    numeric_metrics,
)


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


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# EVO 2 sequence classifier (backbone + LoRA + linear head)
# ---------------------------------------------------------------------------


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class Evo2SequenceClassifier(nn.Module):
    def __init__(self, evo2_model, num_labels: int, layer_name: str, frozen_backbone: bool = False):
        super().__init__()
        self.evo2 = evo2_model

        # Register the PEFT-wrapped StripedHyena as an nn.Module child so that
        # Trainer.model.to(device) moves parameters to GPU.
        self.model = evo2_model.model

        self.num_labels = num_labels
        self.layer_name = layer_name
        self.frozen_backbone = frozen_backbone

        # The inner StripedHyena (with LoRA layers), bypassing PEFT forward
        if hasattr(self.model, "base_model") and hasattr(
            self.model.base_model, "model"
        ):
            self.inner_model = self.model.base_model.model
        else:
            self.inner_model = self.model

        hidden_size = self.inner_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.config = DummyConfig(
            num_labels=num_labels,
            hidden_size=hidden_size,
            model_type="evo2",
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # Bypass the HF-style PeftModel.forward which maps positional args
        # to keyword args (e.g. input_ids=input_ids) — incompatible with
        # StripedHyena.forward(x, ...).  Instead call the inner model
        # directly; LoRA layers are active because they are child modules.

        embeddings = {}

        def _hook_fn(layer_name):
            def hook(_, __, output):
                if isinstance(output, tuple):
                    output = output[0]
                # Only detach when the backbone is frozen (no LoRA).
                # With LoRA, keep the graph intact so gradients flow
                # through the adapter layers.
                embeddings[layer_name] = (
                    output.detach() if self.frozen_backbone else output
                )

            return hook

        handles = []
        try:
            layer = self.inner_model.get_submodule(self.layer_name)
            handles.append(layer.register_forward_hook(_hook_fn(self.layer_name)))

            # When the backbone is frozen (no LoRA), skip autograd tracking
            # inside the model to save VRAM.
            ctx = torch.no_grad() if self.frozen_backbone else contextlib.nullcontext()
            with ctx:
                _ = self.inner_model(input_ids)
        finally:
            for h in handles:
                h.remove()

        hidden = embeddings.get(self.layer_name)
        if hidden is None:
            raise RuntimeError(
                f"Embedding layer {self.layer_name!r} not captured. "
                f"Check layer_name in config."
            )

        # Mean-pool over sequence dimension with attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)

        logits = self.classifier(self.dropout(pooled.float()))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Metrics builder (mirrors transformer._build_metrics_fn)
# ---------------------------------------------------------------------------


def _build_metrics_fn(num_labels: int, average: str):
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_metrics_from_logits(
            logits,
            labels,
            acc_metric,
            f1_metric,
            prec_metric,
            rec_metric,
            num_labels=num_labels,
            average=average,
        )

    return compute_metrics


# ---------------------------------------------------------------------------
# MLflow helpers (mirrors transformer._setup_mlflow / _finalize_mlflow)
# ---------------------------------------------------------------------------


def _setup_mlflow(cfg: DictConfig, run_id: str, mlflow_params: dict[str, Any]):
    if not bool(getattr(cfg.train, "enable_mlflow", False)):
        return None

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow could not be imported. "
            f"Original error: {exc}"
        ) from exc

    mlflow_tracking_uri = getattr(
        cfg.train,
        "mlflow_tracking_uri",
        f"file:{os.path.abspath(os.path.join(os.getcwd(), 'mlruns'))}",
    )
    mlflow_experiment = getattr(
        cfg.train, "mlflow_experiment", cfg.experiment_name
    )
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
    mlflow.start_run(run_name=run_id)
    try:
        mlflow.log_params(mlflow_params)
    except mlflow.exceptions.MlflowException:
        pass

    return mlflow


def _finalize_mlflow(
    mlflow_client,
    summary: dict,
    status: str,
    runtime_seconds: float,
    train_metrics: dict,
    validation_metrics: dict,
    test_metrics: dict,
) -> None:
    if mlflow_client is None:
        return
    all_metrics = {
        **numeric_metrics(train_metrics),
        **numeric_metrics(validation_metrics),
        **numeric_metrics(test_metrics),
        "runtime_seconds": runtime_seconds,
    }
    if all_metrics:
        mlflow_client.log_metrics(all_metrics)
    mlflow_client.log_dict(summary, "run_summary.json")
    mlflow_client.end_run(
        status="FINISHED" if status == "completed" else "FAILED"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # -- Environment ----------------------------------------------------------
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    print(
        f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'} "
        f"({device_name})"
    )

    Evo2 = _import_evo2()

    # -- Resolve early values ------------------------------------------------
    task_name = getattr(cfg.data, "task", None)
    num_labels = int(
        getattr(cfg.data, "num_labels", None) or cfg.train.num_labels
    )

    # -- Load model (once, reuse for tokenization and training) --------------
    print(f"Loading EVO 2 model: {cfg.model.name} ...")
    evo2_model = Evo2(cfg.model.name)

    # Clone all parameters to remove "inference tensor" marking from
    # torch.inference_mode() used during checkpoint loading.
    for p in evo2_model.model.parameters():
        p.data = p.data.clone()

    tokenizer = evo2_model.tokenizer
    pad_token_id = int(tokenizer.pad_id)

    # -- Data: tokenize -------------------------------------------------------
    ds = load_dataset(
        "csv",
        data_files={
            "train": cfg.data.train_csv,
            "validation": cfg.data.val_csv,
            "test": cfg.data.test_csv,
        },
    )

    text_col = cfg.data.text_col
    label_col = cfg.data.label_col
    label_offset = int(getattr(cfg.data, "label_offset", 0))

    def tokenize(batch):
        seqs = [x.upper() for x in batch[text_col]]
        all_ids = []
        all_masks = []
        for seq in seqs:
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
            "input_ids": all_ids,
            "attention_mask": all_masks,
            "label": [int(x) - label_offset for x in batch[label_col]],
        }

    tokenized = ds.map(tokenize, batched=True, batch_size=128)

    keep_cols = {"input_ids", "attention_mask", "label"}
    for split in tokenized.keys():
        drop = [
            c for c in tokenized[split].column_names if c not in keep_cols
        ]
        tokenized[split] = tokenized[split].remove_columns(drop)

    # -- Model: LoRA + classifier head ---------------------------------------
    if getattr(cfg.model, "use_lora", False):
        from peft import LoraConfig, get_peft_model

        target_modules = str(cfg.model.lora_target_modules).split(",")
        lora_config = LoraConfig(
            r=int(cfg.model.lora_r),
            lora_alpha=int(cfg.model.lora_alpha),
            lora_dropout=float(cfg.model.lora_dropout),
            target_modules=target_modules,
            bias="none",
            task_type="SEQ_CLS",
        )
        # PEFT expects model.config.to_dict(); StripedHyena uses a dotdict
        saved_config = evo2_model.model.config
        evo2_model.model.config = None
        evo2_model.model = get_peft_model(evo2_model.model, lora_config)
        evo2_model.model.base_model.model.config = saved_config
    else:
        # Frozen backbone — only the classifier head is trained.
        for p in evo2_model.model.parameters():
            p.requires_grad = False

    model = Evo2SequenceClassifier(
        evo2_model=evo2_model,
        num_labels=num_labels,
        layer_name=cfg.model.layer_name,
        frozen_backbone=not bool(getattr(cfg.model, "use_lora", False)),
    )

    label_mode = str(getattr(cfg.model, "label_mode", "class_index"))
    if label_mode not in {"class_index", "one_hot"}:
        raise ValueError(
            f"Unsupported model.label_mode={label_mode}. "
            "Use class_index or one_hot."
        )

    # -- Metrics & output paths -----------------------------------------------
    metric_average = str(
        _resolve_train_override(cfg, "metric_average", "macro")
    )
    compute_metrics = _build_metrics_fn(num_labels, metric_average)
    experiment = resolve_experiment_name(cfg)
    outdir = os.path.join(
        cfg.train.output_root,
        experiment,
        cfg.model.name.replace("/", "_"),
    )
    run_id = build_run_id(experiment, cfg.model.name)

    # -- Training setup -------------------------------------------------------
    # Single override chain for every training param:
    #  1. cfg.model.tasks[task][key]   per-task EVO 2 value
    #  2. cfg.model.{key}              model-level EVO 2 default
    #  3. _resolve_train_override      gue.yaml → train.yaml
    def _resolve(key, type_fn=int, default=None):
        tasks = getattr(cfg.model, "tasks", {}) or {}
        if task_name and task_name in tasks and key in tasks[task_name]:
            return type_fn(tasks[task_name][key])
        model_val = getattr(cfg.model, key, None)
        if model_val is not None:
            return type_fn(model_val)
        return type_fn(_resolve_train_override(cfg, key, default))

    train_bs = _resolve("train_bs", int, cfg.train.train_bs)
    eval_bs = _resolve("eval_bs", int, cfg.train.eval_bs)
    gradient_accumulation_steps = _resolve("gradient_accumulation_steps", int, 1)
    token_max_length = _resolve("token_max_length", int, cfg.model.max_length)
    epochs = _resolve("epochs", int, cfg.train.epochs)
    learning_rate = _resolve("learning_rate", float, cfg.train.learning_rate)
    head_lr = _resolve("head_learning_rate", float, 0) or None
    warmup_steps = _resolve("warmup_steps", int, None) or None

    cfg.train.learning_rate = learning_rate
    cfg.train.gradient_accumulation_steps = gradient_accumulation_steps

    eval_steps = _resolve("eval_steps", int, None)
    if eval_steps:
        cfg.train.eval_steps = eval_steps

    save_steps = _resolve("save_steps", int, None)
    if save_steps:
        cfg.train.save_steps = save_steps

    if warmup_steps:
        cfg.train.warmup_steps = warmup_steps

    training_args = _build_training_args(
        cfg, outdir, run_id, train_bs, eval_bs, epochs
    )

    # EVO 2 weights are bf16 by default; use bf16 AMP instead of fp16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        training_args.fp16 = False
        training_args.bf16 = True

    # Disable safetensors — the column-split Wqkv permutation creates
    # non-contiguous parameters that safetensors cannot handle.
    training_args.save_safetensors = False

    # -- Trainer --------------------------------------------------------------
    callbacks = []
    early_stopping_patience = int(
        getattr(cfg.train, "early_stopping_patience", 0) or 0
    )
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=float(
                    getattr(cfg.train, "early_stopping_threshold", 0.0) or 0.0
                ),
            )
        )

    trainer = BenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        label_mode=label_mode,
        label_num_classes=num_labels,
        head_learning_rate=head_lr,
    )

    # -- MLflow ---------------------------------------------------------------
    model_load_info = SimpleNamespace(
        used_fallback=False, fallback_reason=None
    )

    if _is_main_process():
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
                "head_learning_rate": head_lr,
                "epochs": epochs,
                "train_bs": train_bs,
                "eval_bs": eval_bs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "weight_decay": cfg.train.weight_decay,
                "warmup_steps": (
                    int(warmup_steps) if warmup_steps is not None else None
                ),
                "metric_average": metric_average,
                "seed": cfg.seed,
                "lora_r": int(cfg.model.lora_r),
                "lora_alpha": int(cfg.model.lora_alpha),
                "layer_name": cfg.model.layer_name,
                "device": device_name,
            },
        )
    else:
        mlflow_client = None

    # -- Train & evaluate -----------------------------------------------------
    started = time.perf_counter()
    status = "failed"
    error_message = None
    train_metrics: dict[str, object] = {}
    validation_metrics: dict[str, object] = {}
    test_metrics: dict[str, object] = {}

    try:
        train_result = trainer.train()
        train_metrics = train_result.metrics
        validation_metrics = trainer.evaluate(
            eval_dataset=tokenized["validation"],
            metric_key_prefix="validation",
        )
        test_metrics = trainer.evaluate(
            eval_dataset=tokenized["test"],
            metric_key_prefix="test",
        )
        if _is_main_process():
            print("Validation:", validation_metrics)
            print("Test:", test_metrics)

        trainer.save_model(os.path.join(outdir, "best_model"))
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
            "model_fallback": model_load_info.used_fallback,
            "model_fallback_reason": model_load_info.fallback_reason,
            "seed": cfg.seed,
            "epochs": epochs,
            "train_bs": train_bs,
            "eval_bs": eval_bs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": cfg.train.learning_rate,
            "metric_average": metric_average,
            "token_max_length": token_max_length,
            "max_length": token_max_length,
            "git_commit": get_git_commit(),
            "runtime_seconds": runtime_seconds,
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "output_dir": outdir,
            "device": device_name,
        }

        if _is_main_process():
            _ = save_run_summary(outdir, summary)
            _ = append_leaderboard_row(cfg.train.leaderboard_csv, summary)
            _finalize_mlflow(
                mlflow_client,
                summary,
                status,
                runtime_seconds,
                train_metrics,
                validation_metrics,
                test_metrics,
            )
