# DNABerts Benchmarks

Benchmark pipeline for promoter classification with Hydra-driven model selection,
MLflow experiment tracking, and local leaderboard export.
<a id="readme-top"></a>
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#dnaberts-benchmarks">DNABerts Benchmarks</a>
    </li>
    <li><a href="#author">Author</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#setup-environment">Setup Environment</a></li>
        <li><a href="#running">Running</a></li>
      </ul>
    </li>
    <li><a href="#data">Data</a></li>
    <li><a href="#structure">Structure</a></li>
  </ol>
</details>

## How to run
```bash
uv run python src/main.py
```

This uses defaults from `src/config/config.yaml`:
- model: `dnabert`
- data: `promoter`
- train: `default`

## Switch model config

```bash
uv run python src/main.py model=bertbase
```

Available model configs:
- dnabert
- dnabert2
- rnabert
- nucleotide_transformer
- bertbase

## Override values from CLI

```bash
uv run python src/main.py model=dnabert2 train.train_bs=4 train.fp16=false train.epochs=5
```

## Multirun sweep

```bash
uv run python src/main.py -m model=dnabert2,rnabert,nucleotide_transformer,bertbase seed=42,123
```

## Tracking outputs

By default each run writes:
- run summary JSON: runs/<experiment>/<model>/run_summary.json
- leaderboard row append: reports/benchmark_results.csv
- MLflow run data: mlruns/

Key tracked metrics:
- f1
- accuracy
- roc_auc

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Author

- [Matheus Girardi](matheusmgirari@gmail.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Data

Default task: promoter vs background using:
- `data/raw/epdnew_400bp.fa`
- `data/raw/epdnew_biasaway_400bp.fa`

TATA-related positives:
- `data/raw/epdnew_TATA_1000bp.fa`
- `data/raw/epdnew_TATAless_1000bp.fa`

Background FASTA files (assumed already available):
- `data/raw/epdnew_TATA_biasaway_1000bp.fa`
- `data/raw/epdnew_TATAless_biasaway_1000bp.fa`

Build binary datasets (recommended for paper robustness):
- `tata_vs_background_1000bp.csv`
- `tataless_vs_background_1000bp.csv`

```bash
uv run python scripts/create_tata_dataset.py --task binary
```

Optionally also build TATA vs TATA-less disambiguation:

```bash
uv run python scripts/create_tata_dataset.py --task disambiguation
```

Split datasets:

```bash
uv run scripts/split_dataset.py --dataset tata_vs_background
uv run scripts/split_dataset.py --dataset tataless_vs_background
uv run scripts/split_dataset.py --input-csv data/raw/tata_vs_tataless_1000bp.csv --output-dir data/processed/tata_vs_tataless
```

Run training on new tasks (without replacing defaults):

```bash
uv run python src/main.py data=tata_promoter experiment_name=tata_vs_tataless
uv run python src/main.py data=tata_vs_background experiment_name=tata_vs_background
uv run python src/main.py data=tataless_vs_background experiment_name=tataless_vs_background
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
