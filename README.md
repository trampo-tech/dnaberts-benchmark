# DNABerts Benchmarks

Multi-task benchmark for genomic foundation models using tasks from the
[Genome Understanding Evaluation (GUE)](https://arxiv.org/abs/2306.15006)
benchmark and selected BEND tasks, with Hydra-driven configuration, MLflow
tracking, and local leaderboard export.

<a id="readme-top"></a>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#supported-tasks">Supported Tasks</a></li>
    <li><a href="#models">Models</a></li>
    <li><a href="#quickstart">Quickstart</a></li>
    <li><a href="#gue-workflows">GUE Workflows</a></li>
    <li><a href="#bend-workflows">BEND Workflows</a></li>
    <li><a href="#tracking-outputs">Tracking Outputs</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#author">Author</a></li>
  </ol>
</details>

## Supported Tasks

The core GUE task set covers 7 tasks spanning different biological problems,
sequence lengths, species, and label counts. Per-task training parameters
(batch size, epochs, max token length) are defined in `src/config/data/gue.yaml`.

### GUE

| Task | Category | Seq Length | Labels | Train Size | Epochs | Batch Size |
|---|---|---|---|---|---|---|
| `prom_core_all` | Promoter detection | 70bp | 2 | 47k | 4 | 32 |
| `splice_reconstructed` | Splice site | 400bp | 3 | 36k | 5 | 32 |
| `human_tf_0` | TF binding (human) | 101bp | 2 | 32k | 3 | 32 |
| `mouse_0` | TF binding (mouse) | 101bp | 2 | 6k | 10 | 32 |
| `EPI_HUVEC` | Epigenetic marks | 3000bp | 2 | 10k | 10 | 4 |
| `emp_H3K4me1` | Histone modification | 500bp | 2 | 25k | 5 | 16 |
| `fungi_species_20` | Species classification | 10000bp | 20 | 8k | 10 | 1 |

### BEND

| Task | Config | Approach | Metric |
|---|---|---|---|
| Variant effects, expression | `data=bend_variant_expression` | Zero-shot cosine distance on REF vs ALT embeddings | Spearman rho |
| Variant effects, disease | `data=bend_variant_disease` | Zero-shot cosine distance on REF vs ALT embeddings | AUROC |
| Histone modification | `data=bend_histone` | Fine-tuned token classification with CNN decoder | Macro AUROC |

## Models

| Model | Config | Type |
|---|---|---|
| DNABERT-2 (117M) | `model=dnabert2` | Transformer |
| Nucleotide Transformer v2 (500M) | `model=nucleotide_transformer` | Transformer |
| DNABERT-2 zero-shot variant effect | `model=dnabert2_zeroshot` | Zero-shot backbone |
| Nucleotide Transformer zero-shot variant effect | `model=nt_zeroshot` | Zero-shot backbone |
| DNABERT-2 histone token classifier | `model=dnabert2_histone` | Transformer + CNN decoder |
| K-mer + Logistic Regression | `model=kmer_logreg` | Baseline |

## Quickstart

### 1. Install dependencies

```bash
uv sync
```

### 2. Download all GUE tasks

```bash
uv run task download-gue
```

### 3. Run a single GUE task

```bash
uv run python src/main.py data=gue data.task=splice_reconstructed
```

### 4. Launch MLflow UI

```bash
uv run task mlflow-ui
```

## GUE Workflows

### Run all supported GUE tasks (sequential multirun)

```bash
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,EPI_HUVEC,emp_H3K4me1,fungi_species_20
```

### Switch model

```bash
uv run python src/main.py data=gue data.task=emp_H3K4me1 model=nucleotide_transformer
```

### Override values from CLI

```bash
uv run python src/main.py data=gue data.task=fungi_species_20 train.learning_rate=1e-5
```

### Full sweep: all tasks × all models

```bash
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,EPI_HUVEC,emp_H3K4me1,fungi_species_20 \
  model=dnabert2,nucleotide_transformer,kmer_logreg
```

### Multiple seeds

```bash
uv run python src/main.py -m \
  data=gue \
  data.task=splice_reconstructed \
  seed=42,123,456
```


## BEND Workflows

BEND workflows require local raw data and an hg38 FASTA file. The prep scripts
expect a standard indexed FASTA; `pyfaidx` will create a `.fai` on first access
if needed. Default settings live in `src/config/download_bend.yaml`,
`src/config/prepare_bend_variant_effects.yaml`,
`src/config/prepare_bend_histone.yaml`, and
`src/config/kmer_variant_baseline.yaml`.

### 1. Download raw BEND files from ERDA

Download all required task groups:

```bash
uv run python scripts/download_bend.py
```
Genomes is included by default because it provides the base FASTA files.

```bash
uv run python scripts/download_bend.py
```

### 2. Prepare variant-effect datasets

```bash
uv run python scripts/prepare_bend_variant_effects.py \
  genome_fasta=data/raw/bend/data/genomes/GRCh38.primary_assembly.genome.fa
```

### 3. Run zero-shot variant-effect benchmarks

Expression task with DNABERT-2:

```bash
uv run python src/main.py model=dnabert2_zeroshot data=bend_variant_expression
```

Disease task with DNABERT-2:

```bash
uv run python src/main.py model=dnabert2_zeroshot data=bend_variant_disease
```

Expression task with Nucleotide Transformer:

```bash
uv run python src/main.py model=nt_zeroshot data=bend_variant_expression
```

The zero-shot runner writes per-variant cosine distances to
`reports/variant_effect_distances_<experiment>.csv`.

### 4. Run the k-mer zero-shot baseline

Expression baseline:

```bash
uv run python scripts/kmer_variant_baseline.py \
  data=bend_variant_expression
```

Disease baseline:

```bash
uv run python scripts/kmer_variant_baseline.py \
  data=bend_variant_disease
```

### 5. Prepare the histone dataset

```bash
uv run python scripts/prepare_bend_histone.py \
  genome_fasta=/path/to/GRCh38.primary_assembly.genome.fa
```

Useful options:

```bash
uv run python scripts/prepare_bend_histone.py \
  genome_fasta=/path/to/GRCh38.primary_assembly.genome.fa \
  window_size=1024 \
  stride=1024 \
  'chromosomes=[chr22]'
```

This writes:

- `data/processed/bend_histone/train.csv`
- `data/processed/bend_histone/validation.csv`
- `data/processed/bend_histone/test.csv`

### 6. Train the histone model

```bash
uv run python src/main.py model=dnabert2_histone data=bend_histone train=bend_default
```

`train=bend_default` is required for the BEND histone task because the metric
and checkpoint settings differ from the default GUE training config.

## Tracking Outputs

Each run writes:
- **Run summary JSON:** `runs/<experiment>/<model>/run_summary.json`
- **GUE leaderboard CSV (append):** `reports/benchmark_results.csv`
- **BEND leaderboard CSV (append):** `reports/benchmark_results_bend.csv`
- **BEND variant distances:** `reports/variant_effect_distances_<experiment>.csv`
- **MLflow run data:** `mlruns/`

GUE tracks F1, accuracy, MCC, and ROC-AUC. BEND tracks task-specific metrics
such as Spearman rho, AUROC, macro AUROC, and per-track AUROC.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data

### GUE

All GUE tasks are downloaded from `leannmlindsey/GUE` on Hugging Face. The
`download-gue` task in `pyproject.toml` calls `scripts/prepare_hf_dataset.py`
for each task, producing `data/processed/leannmlindsey_GUE_<task>/{train,validation,test}.csv`.

To add a new GUE task: add an entry to the `tasks` dict in `src/config/data/gue.yaml`,
then re-run the download script with the new task name added to the `TASKS` array.

### BEND

Supported BEND raw files are downloaded from ERDA into:

- `data/raw/bend/variant_effects/`
- `data/raw/bend/histone_modification/`

Processed outputs are written to:

- `data/processed/bend_variant_effects/{expression,disease}.csv`
- `data/processed/bend_histone/{train,validation,test}.csv`

The variant prep script builds centered REF and ALT windows directly from the
reference genome and skips variants with mismatched reference alleles or
out-of-bounds windows. The histone prep script expects BED-style rows with
`chrom`, `start`, `end`, and 18 binary label columns, then tiles contiguous
512 bp bins into trainable windows.

## Author

- [Matheus Girardi](matheusmgirari@gmail.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
