#!/bin/bash
set -e

echo "Running GUE tasks..."
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1 \
  model=dnabert2,nucleotide_transformer,evo2 \
  train.eval_accumulation_steps=0 \
  seed=63194,21194

echo ""



echo "Running BEND variant effects tasks (zero-shot)..."
uv run python src/main.py -m \
  data=bend_variant_expression,bend_variant_disease \
  model=dnabert2_zeroshot,nt_zeroshot \

echo ""

echo "All benchmarks completed!"
