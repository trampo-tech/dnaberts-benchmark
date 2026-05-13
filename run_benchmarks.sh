#!/bin/bash
set -e

echo "Running GUE tasks..."
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1 \
  model=dnabert2,nucleotide_transformer \ 
  leaderboard_csv=reports/benchmark_full_runs.csv \
  save_predictions=true \
  tags.benchmark_batch=full_runs \
  seed=63194,21194

echo ""

echo "Running GUE tasks with frozen backbone..."
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1 \
  model=dnabert2,nucleotide_transformer \
  model.frozen_backbone=true \
  leaderboard_csv=reports/benchmark_full_runs.csv \
  save_predictions=true \
  tags.benchmark_batch=full_runs \
  seed=63194,21194

echo ""



echo "Running BEND variant effects tasks (zero-shot)..."
uv run python src/main.py -m \
  data=bend_variant_expression,bend_variant_disease \
  model=dnabert2_zeroshot,nt_zeroshot \
  leaderboard_csv=reports/benchmark_full_runs.csv

echo ""

echo "All benchmarks completed!"
