import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_experiment_name(cfg) -> str:
    name = str(cfg.experiment_name)
    if name == "gue" and getattr(getattr(cfg, "data", None), "task", None):
        return f"gue_{cfg.data.task}"
    return name


def build_run_id(experiment_name: str, model_name: str) -> str:
    model_alias = model_name.replace("/", "_")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
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


def _csv_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def _normalize_fieldname(fieldname: Any) -> str:
    return str(fieldname or "").strip()


def _normalize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized_key = _normalize_fieldname(key)
        if not normalized_key or normalized_key == "test_spearman_r":
            continue
        existing_value = normalized.get(normalized_key)
        if normalized_key not in normalized or (existing_value in (None, "") and value not in (None, "")):
            normalized[normalized_key] = value
    return normalized


def _deduplicate_fieldnames(fieldnames: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for fieldname in fieldnames:
        canonical = _normalize_fieldname(fieldname)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        result.append(canonical)
    return result


def _merge_csv_columns(path: Path, row: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists():
        return _deduplicate_fieldnames(list(row.keys())), []

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        existing_rows = [_normalize_csv_row(existing_row) for existing_row in reader]
        fieldnames = _deduplicate_fieldnames(list(reader.fieldnames or []))

    for key in row:
        normalized_key = _normalize_fieldname(key)
        if normalized_key and normalized_key not in fieldnames:
            fieldnames.append(normalized_key)
    return fieldnames, existing_rows


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def append_bend_leaderboard_row(csv_path: str, summary: dict[str, Any]) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {}
    for key, value in summary.items():
        if key in {"train", "validation", "test"}:
            continue
        normalized_key = _normalize_fieldname(key)
        if not normalized_key:
            continue
        row[normalized_key] = _csv_value(value)

    for prefix, metrics in (
        ("train", summary.get("train", {})),
        ("val", summary.get("validation", {})),
        ("test", summary.get("test", {})),
    ):
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            normalized_key = str(key)
            for known_prefix in ("train_", "validation_", "test_", "val_"):
                if normalized_key.startswith(known_prefix):
                    normalized_key = normalized_key[len(known_prefix):]
                    break
            if prefix == "test" and normalized_key == "spearman_r":
                continue
            row[f"{prefix}_{normalized_key}"] = _csv_value(value)

    row = _normalize_csv_row(row)

    fieldnames, existing_rows = _merge_csv_columns(path, row)
    if existing_rows:
        existing_rows.append(row)
        _write_csv_rows(path, fieldnames, existing_rows)
        return path

    _write_csv_rows(path, fieldnames, [row])
    return path


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
        "validation_mcc": _to_float(validation.get("validation_mcc")),
        "validation_accuracy": _to_float(validation.get("validation_accuracy")),
        "validation_roc_auc": _to_float(validation.get("validation_roc_auc")),
        "test_f1": _to_float(test.get("test_f1")),
        "test_mcc": _to_float(test.get("test_mcc")),
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