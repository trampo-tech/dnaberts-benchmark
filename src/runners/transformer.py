"""HuggingFace Trainer runner for transformer-based DNA models (DNABERT2, Nucleotide Transformer)."""

from __future__ import annotations

import importlib
import os
import time
from typing import Any

import evaluate
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from modeling.compat import block_triton_imports, disable_remote_flash_attention
from modeling.train import apply_lora, load_model_for_sequence_classification
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
        head_learning_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_mode = label_mode
        self.label_num_classes = label_num_classes
        self.head_learning_rate = head_learning_rate

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if self.head_learning_rate is None:
            return super().create_optimizer()

        head_params = []
        backbone_params = []
        head_names = {"classifier", "cls", "score", "pooler"}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(h in name for h in head_names):
                head_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": backbone_params, "lr": self.args.learning_rate},
                {"params": head_params, "lr": self.head_learning_rate},
            ],
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

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


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0) in distributed training."""
    return not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Helpers – each one owns a single phase of the pipeline
# ---------------------------------------------------------------------------


def _setup_environment(cfg: DictConfig) -> str:
    if getattr(cfg.model, "disable_triton", False):
        block_triton_imports()

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'} ({device_name})")

    _import_optional_modules(list(getattr(cfg.model, "import_modules", [])))
    return device_name


def _load_and_tokenize(cfg: DictConfig):
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
    num_labels = int(getattr(cfg.data, "num_labels", None) or cfg.train.num_labels)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        revision=getattr(cfg.model, "revision", None),
    )
    token_max_length = int(
        getattr(cfg.data, "token_max_length", 0)
        or getattr(cfg.data, "max_length", 0)
        or cfg.model.max_length
    )

    def tokenize(batch):
        seqs = [_preprocess_seq(x) for x in batch[text_col]]
        out = tokenizer(seqs, truncation=True, max_length=token_max_length)
        out["label"] = [int(x) - label_offset for x in batch[label_col]]
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

    return tokenized, tokenizer, num_labels, token_max_length


def _load_and_configure_model(cfg: DictConfig, num_labels: int, tokenizer: Any = None):
    model = load_model_for_sequence_classification(
        cfg.model.name,
        num_labels=num_labels,
        trust_remote_code=cfg.model.trust_remote_code,
        tokenizer=tokenizer,
        revision=getattr(cfg.model, "revision", None),
    )

    if getattr(cfg.model, "disable_triton", False):
        disable_remote_flash_attention(model)

    label_mode = str(getattr(cfg.model, "label_mode", "class_index"))
    if label_mode not in {"class_index", "one_hot"}:
        raise ValueError(
            f"Unsupported model.label_mode={label_mode}. Use class_index or one_hot."
        )

    model_config: Any = getattr(model, "config", None)
    if model_config is not None:
        model_config.num_labels = num_labels
        model_config.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        model_config.label2id = {
            label: idx for idx, label in model_config.id2label.items()
        }

    configured_problem_type = getattr(cfg.model, "problem_type", None)
    if configured_problem_type is not None and model_config is not None:
        model_config.problem_type = str(configured_problem_type)

    if getattr(cfg.model, "frozen_backbone", False):
        head_names = {"classifier", "cls", "score"}
        for name, param in model.named_parameters():
            if any(h in name for h in head_names):
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Frozen backbone: {trainable:,} / {total:,} trainable parameters")
    elif getattr(cfg.model, "use_lora", False):
        target_modules = str(cfg.model.lora_target_modules).split(",")
        model = apply_lora(
            model,
            r=int(cfg.model.lora_r),
            lora_alpha=int(cfg.model.lora_alpha),
            lora_dropout=float(cfg.model.lora_dropout),
            target_modules=target_modules,
        )

    return model, label_mode


def _resolve_train_override(cfg: DictConfig, key: str, default: Any = None) -> Any:
    direct_value = getattr(cfg.data, key, None)
    if direct_value is not None:
        return direct_value

    task_name = getattr(cfg.data, "task", None)
    task_registry = getattr(cfg.data, "tasks", None)
    if task_name and task_registry is not None and task_name in task_registry:
        task_cfg = task_registry[task_name]
        nested_value = getattr(task_cfg, key, None)
        if nested_value is not None:
            return nested_value

    return getattr(cfg.train, key, default)


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


def _build_training_args(
    cfg: DictConfig,
    outdir: str,
    run_id: str,
    train_bs: int,
    eval_bs: int,
    epochs: int,
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": outdir,
        "eval_strategy": cfg.train.eval_strategy,
        "save_strategy": cfg.train.save_strategy,
        "logging_steps": cfg.train.logging_steps,
        "learning_rate": cfg.train.learning_rate,
        "per_device_train_batch_size": train_bs,
        "per_device_eval_batch_size": eval_bs,
        "num_train_epochs": epochs,
        "weight_decay": cfg.train.weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": cfg.train.metric_for_best_model,
        "greater_is_better": True,
        "report_to": cfg.train.report_to,
        "fp16": cfg.train.fp16,
        "seed": cfg.seed,
        "run_name": run_id,
    }

    warmup_steps = getattr(cfg.train, "warmup_steps", None)
    if warmup_steps is not None:
        kwargs["warmup_steps"] = int(warmup_steps)
    else:
        warmup_ratio = float(getattr(cfg.train, "warmup_ratio", 0) or 0)
        if warmup_ratio > 0:
            kwargs["warmup_ratio"] = warmup_ratio

    if str(cfg.train.eval_strategy) == "steps" and getattr(
        cfg.train, "eval_steps", None
    ):
        kwargs["eval_steps"] = int(cfg.train.eval_steps)

    if str(cfg.train.save_strategy) == "steps" and getattr(
        cfg.train, "save_steps", None
    ):
        kwargs["save_steps"] = int(cfg.train.save_steps)

    if getattr(cfg.train, "gradient_accumulation_steps", None) is not None:
        kwargs["gradient_accumulation_steps"] = int(
            cfg.train.gradient_accumulation_steps
        )

    if getattr(cfg.train, "save_total_limit", None) is not None:
        kwargs["save_total_limit"] = int(cfg.train.save_total_limit)

    if getattr(cfg.train, "eval_accumulation_steps", None) is not None:
        kwargs["eval_accumulation_steps"] = int(cfg.train.eval_accumulation_steps)

    return TrainingArguments(**kwargs)


def _build_trainer(
    cfg: DictConfig,
    model,
    tokenizer,
    tokenized,
    training_args: TrainingArguments,
    compute_metrics,
    label_mode: str,
    num_labels: int,
    head_lr: float | None,
) -> BenchmarkTrainer:
    callbacks = []
    early_stopping_patience = int(getattr(cfg.train, "early_stopping_patience", 0) or 0)
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=float(
                    getattr(cfg.train, "early_stopping_threshold", 0.0) or 0.0
                ),
            )
        )

    return BenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        label_mode=label_mode,
        label_num_classes=num_labels,
        head_learning_rate=head_lr,
    )


def _setup_mlflow(
    cfg: DictConfig,
    run_id: str,
    mlflow_params: dict[str, Any],
):
    if not bool(getattr(cfg.train, "enable_mlflow", False)):
        return None

    try:
        import mlflow
    except ImportError as exc:
        message = (
            "MLflow could not be imported. This is often caused by incompatible dependencies. "
            f"Original error: {exc}"
        )
        raise RuntimeError(message) from exc

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
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
    mlflow.start_run(run_name=run_id)
    try:
        mlflow.log_params(mlflow_params)
    except mlflow.exceptions.MlflowException:
        pass  # HF Trainer may have already logged some params

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
    mlflow_client.end_run(status="FINISHED" if status == "completed" else "FAILED")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    # -- Environment ----------------------------------------------------------
    device_name = _setup_environment(cfg)

    # -- Data -----------------------------------------------------------------
    tokenized, tokenizer, num_labels, token_max_length = _load_and_tokenize(cfg)

    # -- Model ----------------------------------------------------------------
    model, label_mode = _load_and_configure_model(
        cfg, num_labels, tokenizer
    )

    # -- Metrics & output paths -----------------------------------------------
    metric_average = str(_resolve_train_override(cfg, "metric_average", "macro"))
    compute_metrics = _build_metrics_fn(num_labels, metric_average)
    experiment = resolve_experiment_name(cfg)
    outdir = os.path.join(
        cfg.train.output_root, experiment, cfg.model.name.replace("/", "_")
    )
    run_id = build_run_id(experiment, cfg.model.name)

    # -- Training setup -------------------------------------------------------
    train_bs = int(_resolve_train_override(cfg, "train_bs", cfg.train.train_bs))
    eval_bs = int(_resolve_train_override(cfg, "eval_bs", cfg.train.eval_bs))
    epochs = int(_resolve_train_override(cfg, "epochs", cfg.train.epochs))
    head_lr = float(_resolve_train_override(cfg, "head_learning_rate", 0) or 0) or None
    warmup_steps = _resolve_train_override(cfg, "warmup_steps", None)

    eval_steps = _resolve_train_override(cfg, "eval_steps", None)
    if eval_steps is not None:
        cfg.train.eval_steps = int(eval_steps)

    save_steps = _resolve_train_override(cfg, "save_steps", None)
    if save_steps is not None:
        cfg.train.save_steps = int(save_steps)

    gradient_accumulation_steps = _resolve_train_override(
        cfg, "gradient_accumulation_steps", 1
    )
    cfg.train.gradient_accumulation_steps = int(gradient_accumulation_steps)

    if warmup_steps is not None:
        cfg.train.warmup_steps = int(warmup_steps)

    training_args = _build_training_args(cfg, outdir, run_id, train_bs, eval_bs, epochs)
    trainer = _build_trainer(
        cfg,
        model,
        tokenizer,
        tokenized,
        training_args,
        compute_metrics,
        label_mode,
        num_labels,
        head_lr,
    )

    # -- MLflow ---------------------------------------------------------------
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
                "fallback_model_wrapper": False,
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
        if _is_main_process():
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
            "experiment": experiment,
            "model_name": cfg.model.name,
            "model_fallback": False,
            "model_fallback_reason": None,
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
