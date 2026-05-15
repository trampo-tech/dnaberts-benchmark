#!/bin/bash
set -e

echo "Running evo2 1B with LoRA"
python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1 \
  model=evo2\
  train.early_stopping_patience=0 \
  leaderboard_csv=reports/benchmark_full_runs.csv \
  save_predictions=true \
  tags=full_runs \
  seed=63194,21194

echo "Running evo2 1B as a frozen backbone"
python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1 \
  model=evo2\
  model.frozen_backbone=true \
  train.early_stopping_patience=0 \
  leaderboard_csv=reports/benchmark_full_runs.csv \
  save_predictions=true \
  tags=full_runs \
  seed=63194,21194

echo "Running evo2 1B with full fine-tuning"
python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1 \
  model=evo2\
  model.use_lora=false\
  train.early_stopping_patience=0 \
  leaderboard_csv=reports/benchmark_full_runs.csv \
  save_predictions=true \
  tags=full_runs \
  seed=63194,21194
  