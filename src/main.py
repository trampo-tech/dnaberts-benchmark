import os
import random

import evaluate
import hydra
import numpy as np
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def preprocess_seq(seq: str, mode: str) -> str:
    s = seq.upper()
    if mode == "rna_t_to_u":
        return s.replace("T", "U")
    if mode == "baseline_char":
        # simple baseline formatting: space-separated chars
        return " ".join(list(s))
    return s  # dna

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

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
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code
    )

    def tok(batch):
        seqs = [preprocess_seq(x, cfg.model.sequence_preprocess) for x in batch[text_col]]
        out = tokenizer(seqs, truncation=True, max_length=cfg.model.max_length)
        out["label"] = batch[label_col]
        return out

    tokenized = ds.map(tok, batched=True)
    keep_cols = {"input_ids", "attention_mask", "label", "token_type_ids"}
    for split in tokenized.keys():
        drop = [c for c in tokenized[split].column_names if c not in keep_cols]
        tokenized[split] = tokenized[split].remove_columns(drop)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.train.num_labels,
        trust_remote_code=cfg.model.trust_remote_code
    )

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1_metric.compute(predictions=preds, references=labels)["f1"],
            "precision": prec_metric.compute(predictions=preds, references=labels)["precision"],
            "recall": rec_metric.compute(predictions=preds, references=labels)["recall"],
        }

    outdir = os.path.join(
        cfg.train.output_root,
        cfg.experiment_name,
        cfg.model.name.replace("/", "_")
    )

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
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Validation:", trainer.evaluate(tokenized["validation"]))
    print("Test:", trainer.evaluate(tokenized["test"]))

    trainer.save_model(os.path.join(outdir, "best_model"))
    tokenizer.save_pretrained(os.path.join(outdir, "best_model"))

if __name__ == "__main__":
    main()
