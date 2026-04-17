"""HuggingFace Trainer runner for transformer-based DNA models (DNABERT2, Nucleotide Transformer)."""

from __future__ import annotations

import importlib
import os
import time
from typing import Any

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from modeling.train import load_model_for_sequence_classification
from services.benchmark_logger import (
    append_leaderboard_row,
    build_run_id,
    get_git_commit,
    save_run_summary,
    utc_timestamp,
)
from services.metrics import (
    compute_metrics_from_logits,
    numeric_metrics,
)


def _import_optional_modules(modules: list[str] | None):
    for module_name in modules or []:
        importlib.import_module(module_name)


def _preprocess_seq(seq: str) -> str:
    return seq.upper()


def _estimate_unk_ratio(
    tokenized_split,
    unk_token_id: int | None,
    special_token_ids: set[int],
    sample_size: int = 256,
) -> float | None:
    if unk_token_id is None:
        return None

    total_tokens = 0
    unk_tokens = 0
    n = min(len(tokenized_split), sample_size)

    for i in range(n):
        ids = tokenized_split[i]["input_ids"]
        for token_id in ids:
            if token_id in special_token_ids:
                continue
            total_tokens += 1
            if token_id == unk_token_id:
                unk_tokens += 1

    if total_tokens == 0:
        return None
    return unk_tokens / total_tokens


class BenchmarkTrainer(Trainer):
    def __init__(
        self,
        *args,
        label_mode: str = "class_index",
        label_num_classes: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_mode = label_mode
        self.label_num_classes = label_num_classes

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")

        if (
            labels is not None
            and self.label_mode == "one_hot"
            and getattr(labels, "ndim", 0) == 1
        ):
            num_classes = self.label_num_classes
            if num_classes is None:
                num_classes = int(
                    getattr(getattr(model, "config", None), "num_labels", 0) or 0
                )
            if num_classes <= 1:
                raise ValueError("label_mode=one_hot requires num_labels > 1")

            one_hot = F.one_hot(labels.to(torch.long), num_classes=num_classes).to(
                dtype=torch.float32
            )
            inputs = dict(inputs)
            inputs["labels"] = one_hot

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def run(cfg: DictConfig) -> None:
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'} ({device_name})")

    _import_optional_modules(list(getattr(cfg.model, "import_modules", [])))

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

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, trust_remote_code=cfg.model.trust_remote_code
    )

    def tokenize(batch):
        seqs = [_preprocess_seq(x) for x in batch[text_col]]
        out = tokenizer(seqs, truncation=True, max_length=cfg.model.max_length)
        out["label"] = [int(x) for x in batch[label_col]]
        return out

    tokenized = ds.map(tokenize, batched=True)

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    train_unk_ratio = _estimate_unk_ratio(
        tokenized["train"], tokenizer.unk_token_id, special_ids
    )
    if train_unk_ratio is not None and train_unk_ratio > 0.3:
        print(
            f"[WARN] High UNK token ratio in train split: {train_unk_ratio:.3f}. "
            "Check sequence_preprocess for this model."
        )

    keep_cols = {"input_ids", "attention_mask", "label", "token_type_ids"}
    for split in tokenized.keys():
        drop = [c for c in tokenized[split].column_names if c not in keep_cols]
        tokenized[split] = tokenized[split].remove_columns(drop)

    model, model_load_info = load_model_for_sequence_classification(
        cfg.model.name,
        num_labels=cfg.train.num_labels,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    label_mode = str(getattr(cfg.model, "label_mode", "class_index"))
    if label_mode not in {"class_index", "one_hot"}:
        raise ValueError(
            f"Unsupported model.label_mode={label_mode}. Use class_index or one_hot."
        )

    model_config: Any = getattr(model, "config", None)

    if model_config is not None:
        model_config.num_labels = int(cfg.train.num_labels)
        model_config.id2label = {
            i: f"LABEL_{i}" for i in range(int(cfg.train.num_labels))
        }
        model_config.label2id = {
            label: idx for idx, label in model_config.id2label.items()
        }

    configured_problem_type = getattr(cfg.model, "problem_type", None)
    if configured_problem_type is not None and model_config is not None:
        model_config.problem_type = str(configured_problem_type)

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_metrics_from_logits(
            logits, labels, acc_metric, f1_metric, prec_metric, rec_metric
        )

    outdir = os.path.join(
        cfg.train.output_root, cfg.experiment_name, cfg.model.name.replace("/", "_")
    )
    run_id = build_run_id(cfg.experiment_name, cfg.model.name)

    training_args_kwargs: dict[str, Any] = {
        "output_dir": outdir,
        "eval_strategy": cfg.train.eval_strategy,
        "save_strategy": cfg.train.save_strategy,
        "logging_steps": cfg.train.logging_steps,
        "learning_rate": cfg.train.learning_rate,
        "per_device_train_batch_size": cfg.train.train_bs,
        "per_device_eval_batch_size": cfg.train.eval_bs,
        "num_train_epochs": cfg.train.epochs,
        "weight_decay": cfg.train.weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": cfg.train.metric_for_best_model,
        "greater_is_better": True,
        "report_to": cfg.train.report_to,
        "fp16": cfg.train.fp16,
        "seed": cfg.seed,
        "run_name": run_id,
    }

    if str(cfg.train.eval_strategy) == "steps" and getattr(cfg.train, "eval_steps", None):
        training_args_kwargs["eval_steps"] = int(cfg.train.eval_steps)

    if str(cfg.train.save_strategy) == "steps" and getattr(cfg.train, "save_steps", None):
        training_args_kwargs["save_steps"] = int(cfg.train.save_steps)

    if getattr(cfg.train, "save_total_limit", None) is not None:
        training_args_kwargs["save_total_limit"] = int(cfg.train.save_total_limit)

    args = TrainingArguments(**training_args_kwargs)

    trainer_callbacks = []
    early_stopping_patience = int(getattr(cfg.train, "early_stopping_patience", 0) or 0)
    if early_stopping_patience > 0:
        trainer_callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=float(
                    getattr(cfg.train, "early_stopping_threshold", 0.0) or 0.0
                ),
            )
        )

    trainer = BenchmarkTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=trainer_callbacks,
        label_mode=label_mode,
        label_num_classes=int(cfg.train.num_labels),
    )

    mlflow_client = None
    enable_mlflow = bool(getattr(cfg.train, "enable_mlflow", False))
    if enable_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            message = (
                "MLflow could not be imported. This is often caused by incompatible dependencies. "
                f"Original error: {exc}"
            )
            raise RuntimeError(message) from exc
        else:
            mlflow_tracking_uri = getattr(
                cfg.train,
                "mlflow_tracking_uri",
                f"file:{os.path.abspath(os.path.join(os.getcwd(), 'mlruns'))}",
            )
            mlflow_experiment = getattr(
                cfg.train,
                "mlflow_experiment",
                cfg.experiment_name,
            )
            mlflow_client = mlflow
            mlflow_client.set_tracking_uri(mlflow_tracking_uri)
            mlflow_client.set_experiment(mlflow_experiment)
            mlflow_client.start_run(run_name=run_id)
            mlflow_client.log_params(
                {
                    "run_id": run_id,
                    "experiment": cfg.experiment_name,
                    "model_name": cfg.model.name,
                    "max_length": cfg.model.max_length,
                    "num_labels": cfg.train.num_labels,
                    "learning_rate": cfg.train.learning_rate,
                    "epochs": cfg.train.epochs,
                    "train_bs": cfg.train.train_bs,
                    "eval_bs": cfg.train.eval_bs,
                    "weight_decay": cfg.train.weight_decay,
                    "seed": cfg.seed,
                    "fallback_model_wrapper": model_load_info.used_fallback,
                    "device": device_name,
                }
            )

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
        print("Validation:", validation_metrics)
        print("Test:", test_metrics)

        trainer.save_model(os.path.join(outdir, "best_model"))
        tokenizer.save_pretrained(os.path.join(outdir, "best_model"))
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
            "experiment": cfg.experiment_name,
            "model_name": cfg.model.name,
            "model_fallback": model_load_info.used_fallback,
            "model_fallback_reason": model_load_info.fallback_reason,
            "seed": cfg.seed,
            "epochs": cfg.train.epochs,
            "train_bs": cfg.train.train_bs,
            "eval_bs": cfg.train.eval_bs,
            "learning_rate": cfg.train.learning_rate,
            "max_length": cfg.model.max_length,
            "git_commit": get_git_commit(),
            "runtime_seconds": runtime_seconds,
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "output_dir": outdir,
            "device": device_name,
        }

        _ = save_run_summary(outdir, summary)
        _ = append_leaderboard_row(cfg.train.leaderboard_csv, summary)

        if mlflow_client is not None:
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
