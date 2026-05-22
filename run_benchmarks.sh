#!/bin/bash
set -e

echo "Running GUE tasks..."
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1 \
  model=dnabert2,nucleotide_transformer,bertbase \
  train.early_stopping_patience=0 \
  leaderboard_csv=reports/benchmark_full_runs.csv \
  save_predictions=true \
  tags=full_runs \
  seed=63194,21194

echo ""

echo "Running GUE tasks with frozen backbone..."
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1 \
  model=dnabert2,nucleotide_transformer,bertbase \
  train.early_stopping_patience=0 \
  model.frozen_backbone=true \
  leaderboard_csv=reports/benchmark_full_runs.csv \
  save_predictions=true \
  tags=full_runs \
  seed=63194,21194

echo ""



echo "Running BEND variant effects tasks (zero-shot)..."
uv run python src/main.py -m \
  data=bend_variant_expression,bend_variant_disease \
  model=dnabert2_zeroshot,nt_zeroshot,evo2_zeroshot \
  tags=full_runs \
  leaderboard_csv=reports/benchmark_full_runs.csv

echo ""

echo "All benchmarks completed!"
