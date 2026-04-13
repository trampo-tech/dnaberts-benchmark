import importlib
import os
import random
import time

import evaluate
import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from modeling.train import load_model_for_sequence_classification
from services.benchmark_logger import (
    append_leaderboard_row,
    build_run_id,
    get_git_commit,
    save_run_summary,
    utc_timestamp,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        print("")
        pass


def import_optional_modules(modules: list[str] | None):
    for module_name in modules or []:
        importlib.import_module(module_name)


def preprocess_seq(seq: str, mode: str, cfg:DictConfig) -> str:
    s = seq.upper()
    if mode == "rna_t_to_u":
        return s.replace("T", "U")
    if mode == "kmer":
        kmer:int = cfg.model.kmer
        if len(s) < kmer:
            return s
        return " ".join(s[i : i + kmer] for i in range(len(s) - (kmer - 1)))
    if mode == "baseline_char":
        # simple baseline formatting: space-separated chars
        return " ".join(list(s))
    return s  # dna


def estimate_unk_ratio(
    tokenized_split,
    unk_token_id: int | None,
    special_token_ids: set[int],
    sample_size: int = 256,
) -> float | None:
    """ Estimates the ratio of unknown tokens given to the model, the lower the better (less unknown tokens)"""

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


def positive_class_probability(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        return 1.0 / (1.0 + np.exp(-logits))
    if logits.shape[-1] == 1:
        return 1.0 / (1.0 + np.exp(-logits[:, 0]))

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs[:, 1]


def numeric_metrics(metrics: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    print(
        f" Device selected by torch: {'cuda' if cuda_available else 'cpu'} ({device_name})"
    )

    import_optional_modules(list(getattr(cfg.model, "import_modules", [])))

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

    tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(
        cfg.model.name, trust_remote_code=cfg.model.trust_remote_code
    )  # ty:ignore[invalid-assignment]

    def tokenize(batch):
        seqs = [
            preprocess_seq(x, cfg.model.sequence_preprocess, cfg) for x in batch[text_col]
        ]
        out = tokenizer(seqs, truncation=True, max_length=cfg.model.max_length)
        out["label"] = batch[label_col]
        return out

    tokenized = ds.map(tokenize, batched=True)

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    train_unk_ratio = estimate_unk_ratio(
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

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        metrics = {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)[  # type: ignore
                "accuracy"
            ],
            "f1": f1_metric.compute(predictions=preds, references=labels)["f1"], # type: ignore
            "precision": prec_metric.compute(predictions=preds, references=labels)[ # type: ignore
                "precision"
            ], 
            "recall": rec_metric.compute(predictions=preds, references=labels)[ # type: ignore
                "recall"
            ], 
        }

        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and (logits.ndim == 1 or logits.shape[-1] <= 2):
            probs = positive_class_probability(np.asarray(logits))
            metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        else:
            metrics["roc_auc"] = 0.0

        return metrics

    outdir = os.path.join(
        cfg.train.output_root, cfg.experiment_name, cfg.model.name.replace("/", "_")
    )
    run_id = build_run_id(cfg.experiment_name, cfg.model.name)

    args = TrainingArguments(
        output_dir=outdir,
        eval_strategy=cfg.train.eval_strategy,
        save_strategy=cfg.train.save_strategy,
        logging_steps=cfg.train.logging_steps,
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.train_bs,
        per_device_eval_batch_size=cfg.train.eval_bs,
        num_train_epochs=cfg.train.epochs,
        weight_decay=cfg.train.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=True,
        report_to=cfg.train.report_to,
        fp16=cfg.train.fp16,
        seed=cfg.seed,
        run_name=run_id,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
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
            print(f"[WARN] {message}")
            mlflow_client = None
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
                    "sequence_preprocess": cfg.model.sequence_preprocess,
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
            eval_dataset=tokenized["validation"], metric_key_prefix="validation"
        )
        test_metrics = trainer.evaluate(
            eval_dataset=tokenized["test"], metric_key_prefix="test"
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
            mlflow_client.end_run(status="FINISHED" if status == "completed" else "FAILED")


if __name__ == "__main__":
    main()
