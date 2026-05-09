```sh /home/blau/projects/dnaberts-benchmark/run_benchmarks.sh
#!/bin/bash
set -e

echo "Running GUE tasks..."
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,EPI_HUVEC,emp_H3K4me1,fungi_species_20 \
  model=dnabert2,nucleotide_transformer \
  seed=63194,21194

echo ""

echo "Running BEND variant effects tasks (zero-shot)..."
uv run python src/main.py -m \
  data=bend_variant_expression,bend_variant_disease \
  model=dnabert2_zeroshot,nt_zeroshot \
  seed=63194,21194

echo ""

# # ==============================================================================
# # BEND histone task – uses a custom PyTorch training loop (not DDP-compatible)
# # DNABERT-2 only.
# # ==============================================================================
# echo "Running BEND histone task (DNABERT-2 only)..."
# uv run python src/main.py -m \
#   data=bend_histone \
#   model=dnabert2_histone \
#   train=bend_default \
#   seed=63194,21194

# echo ""
echo "All benchmarks completed!"
