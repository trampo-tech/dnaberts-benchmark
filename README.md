# DNABerts Benchmarks

Multi-task benchmark for genomic foundation models using tasks from the
[Genome Understanding Evaluation (GUE)](https://arxiv.org/abs/2306.15006) benchmark,
with Hydra-driven configuration, MLflow tracking, and local leaderboard export.

<a id="readme-top"></a>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#benchmark-tasks">Benchmark Tasks</a></li>
    <li><a href="#models">Models</a></li>
    <li><a href="#quickstart">Quickstart</a></li>
    <li><a href="#running">Running</a></li>
    <li><a href="#tracking-outputs">Tracking Outputs</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#author">Author</a></li>
  </ol>
</details>

## Benchmark Tasks

The core task set covers 8 GUE tasks spanning different biological problems,
sequence lengths, species, and label counts. Per-task training parameters
(batch size, epochs, max token length) are defined in `src/config/data/gue.yaml`.

| Task | Category | Seq Length | Labels | Train Size | Epochs | Batch Size |
|---|---|---|---|---|---|---|
| `prom_core_all` | Promoter detection | 70bp | 2 | 47k | 5 | 32 |
| `splice_reconstructed` | Splice site | 400bp | 3 | 36k | 5 | 16 |
| `human_tf_0` | TF binding (human) | 101bp | 2 | 32k | 3 | 32 |
| `mouse_0` | TF binding (mouse) | 101bp | 2 | 6k | 10 | 16 |
| `EPI_K562` | Epigenetic marks | 3000bp | 2 | 10k | 5 | 8 |
| `emp_H3K4me1` | Histone modification | 500bp | 2 | 25k | 5 | 16 |
| `fungi_species_20` | Species classification | 10000bp | 20 | 8k | 10 | 8 |
| `virus_covid` | Virus classification | 999bp | 9 | 73k | 3 | 16 |

## Models

| Model | Config | Type |
|---|---|---|
| DNABERT-2 (117M) | `model=dnabert2` | Transformer |
| Nucleotide Transformer v2 (500M) | `model=nucleotide_transformer` | Transformer |
| K-mer + Logistic Regression | `model=kmer_logreg` | Baseline |

## Quickstart

### 1. Download all GUE tasks

```bash
uv run task download-gue
```

### 2. Run a single task

```bash
uv run python src/main.py data=gue data.task=splice_reconstructed
```

### 3. Run all tasks (sequential multirun)

```bash
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,EPI_K562,emp_H3K4me1,fungi_species_20,virus_covid
```

## Running

### Switch model

```bash
uv run python src/main.py data=gue data.task=virus_covid model=nucleotide_transformer
```

### Override values from CLI

```bash
uv run python src/main.py data=gue data.task=fungi_species_20 train.learning_rate=1e-5
```

### Full sweep: all tasks × all models

```bash
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,EPI_K562,emp_H3K4me1,fungi_species_20,virus_covid \
  model=dnabert2,nucleotide_transformer,kmer_logreg
```

### Multiple seeds

```bash
uv run python src/main.py -m \
  data=gue \
  data.task=splice_reconstructed \
  seed=42,123,456
```

> **Note:** Hydra multirun (`-m`) runs jobs **sequentially** by default.

## Tracking Outputs

Each run writes:
- **Run summary JSON:** `runs/<experiment>/<model>/run_summary.json`
- **Leaderboard CSV (append):** `reports/benchmark_results.csv`
- **MLflow run data:** `mlruns/`

Key tracked metrics: F1, accuracy, ROC-AUC.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Author

- [Matheus Girardi](matheusmgirari@gmail.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data

All GUE tasks are downloaded from `leannmlindsey/GUE` on Hugging Face. The download
script (`scripts/download_gue_tasks.sh`) calls `scripts/prepare_hf_dataset.py` for
each task, producing `data/processed/leannmlindsey_GUE_<task>/{train,validation,test}.csv`.

To add a new GUE task: add an entry to the `tasks` dict in `src/config/data/gue.yaml`,
then re-run the download script with the new task name added to the `TASKS` array.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
