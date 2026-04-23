import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_experiment_name(cfg) -> str:
    name = str(cfg.experiment_name)
    if name == "gue" and getattr(getattr(cfg, "data", None), "task", None):
        return f"gue_{cfg.data.task}"
    return name


def build_run_id(experiment_name: str, model_name: str) -> str:
    model_alias = model_name.replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_name}__{model_alias}__{stamp}"


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def save_run_summary(output_dir: str, payload: dict[str, Any]) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path = out_path / "run_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def append_leaderboard_row(csv_path: str, summary: dict[str, Any]) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    validation = summary.get("validation", {})
    test = summary.get("test", {})
    train = summary.get("train", {})

    row = {
        "timestamp": summary.get("timestamp"),
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "experiment": summary.get("experiment"),
        "model_name": summary.get("model_name"),
        "model_fallback": summary.get("model_fallback"),
        "seed": summary.get("seed"),
        "epochs": summary.get("epochs"),
        "train_bs": summary.get("train_bs"),
        "eval_bs": summary.get("eval_bs"),
        "learning_rate": summary.get("learning_rate"),
        "max_length": summary.get("max_length"),
        "validation_f1": _to_float(validation.get("validation_f1")),
        "validation_accuracy": _to_float(validation.get("validation_accuracy")),
        "validation_roc_auc": _to_float(validation.get("validation_roc_auc")),
        "test_f1": _to_float(test.get("test_f1")),
        "test_accuracy": _to_float(test.get("test_accuracy")),
        "test_roc_auc": _to_float(test.get("test_roc_auc")),
        "train_runtime": _to_float(train.get("train_runtime")),
        "train_samples_per_second": _to_float(train.get("train_samples_per_second")),
        "device": summary.get("device"),
        "git_commit": summary.get("git_commit"),
    }

    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return path