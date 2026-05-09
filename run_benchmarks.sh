#!/bin/bash
set -e

NUM_GPUS=$(uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)

if [ "$NUM_GPUS" -gt 1 ]; then
    ACCEL_ARGS=(--multi_gpu --num_processes "$NUM_GPUS" --mixed_precision fp16)
    echo "[INFO] Detected $NUM_GPUS GPUs → using accelerate DDP for GUE tasks"
elif [ "$NUM_GPUS" -eq 1 ]; then
    ACCEL_ARGS=(--mixed_precision fp16)
    echo "[INFO] Detected 1 GPU → using accelerate"
else
    ACCEL_ARGS=(--cpu)
    echo "[INFO] No GPU detected → running GUE tasks on CPU"
fi

echo ""

echo "Running GUE tasks..."
uv run accelerate launch "${ACCEL_ARGS[@]}" src/main.py -m \
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
