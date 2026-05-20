# DNABerts Benchmarks

Benchmark multi-tarefa para modelos fundacionais de genômica usando tarefas do
[Genome Understanding Evaluation (GUE)](https://arxiv.org/abs/2306.15006) e
tarefas selecionadas do BEND, com configuração via Hydra, tracking com MLflow, e
exportação local de leaderboard.

<a id="readme-top"></a>

<details open="open">
  <summary>Índice</summary>
  <ol>
    <li><a href="#tarefas-suportadas">Tarefas Suportadas</a></li>
    <li><a href="#modelos">Modelos</a></li>
    <li><a href="#quickstart">Quickstart</a></li>
    <li><a href="#suite-completa">Suite Completa de Benchmarks</a></li>
    <li><a href="#workflows-gue">Workflows GUE</a></li>
    <li><a href="#workflows-bend">Workflows BEND</a></li>
    <li><a href="#outputs-de-tracking">Outputs de Tracking</a></li>
    <li><a href="#mlflow-tags">MLflow Tags</a></li>
    <li><a href="#dados">Dados</a></li>
    <li><a href="#autor">Autor</a></li>
  </ol>
</details>

## Tarefas Suportadas

O conjunto principal do GUE cobre 5 tarefas abrangendo diferentes problemas
biológicos, comprimentos de sequência, espécies e número de rótulos. Os
parâmetros de treino por tarefa (batch size, épocas, comprimento máximo de
tokens) estão definidos em `src/config/data/gue.yaml`.

### GUE

| Tarefa | Categoria | Seq | Labels | Treino | Épocas | Batch |
|---|---|---|---|---|---|---|
| `prom_core_all` | Detecção de promotores | 70bp | 2 | 47k | 4 | 32 |
| `splice_reconstructed` | Splice site | 400bp | 3 | 36k | 5 | 32 |
| `human_tf_0` | TF binding (humano) | 101bp | 2 | 32k | 3 | 32 |
| `mouse_0` | TF binding (mouse) | 101bp | 2 | 6k | 10 | 32 |
| `emp_H3K4me1` | Modificação de histonas | 500bp | 2 | 25k | 5 | 16 |

### BEND

| Tarefa | Config | Abordagem | Métrica |
|---|---|---|---|
| Variant effects, expression | `data=bend_variant_expression` | Distância cosseno zero-shot entre embeddings REF vs ALT | Spearman rho |
| Variant effects, disease | `data=bend_variant_disease` | Distância cosseno zero-shot entre embeddings REF vs ALT | AUROC |

## Modelos

| Modelo | Config | Tipo |
|---|---|---|
| DNABERT-2 (117M) | `model=dnabert2` | Transformer |
| Nucleotide Transformer v2 (500M) | `model=nucleotide_transformer` | Transformer |
| DNABERT-2 zero-shot | `model=dnabert2_zeroshot` | Backbone zero-shot |
| Nucleotide Transformer zero-shot | `model=nt_zeroshot` | Backbone zero-shot |
| EVO 2 (1B base) | `model=evo2` | Hyena + LoRA |
| K-mer + Logistic Regression | `model=kmer_logreg` | Baseline |

> O EVO 2 também suporta `model.frozen_backbone=true` como alternativa ao LoRA,
> congelando todo o backbone e treinando apenas a cabeça de classificação.

## Quickstart

### 1. Instalar dependências

```bash
uv sync
```

> **EVO 2 apenas:** `flash-attn` deve ser instalado separadamente antes de
> executar benchmarks com EVO 2 (veja a [seção EVO 2](#evo-2) abaixo).

### 2. Baixar todas as tarefas GUE

```bash
uv run task download-gue
```

### 3. Executar uma única tarefa GUE

```bash
uv run python src/main.py data=gue data.task=splice_reconstructed
```

### 4. Abrir interface MLflow

```bash
uv run task mlflow-ui
```

## Suite Completa

O script `run_benchmarks.sh` executa a suite completa de benchmarks:

- **GUE padrão**: 5 tarefas × 3 modelos × 2 seeds (`63194`, `21194`)
- **GUE com backbone congelado**: mesmas varreduras com `model.frozen_backbone=true`
- **BEND zero-shot**: 2 tarefas × 2 modelos zero-shot
- Habilita `save_predictions=true`, `tags.benchmark_batch=full_runs` e grava o
  leaderboard em `reports/benchmark_full_runs.csv`

```bash
bash run_benchmarks.sh
```

## Workflows GUE

### Executar todas as tarefas GUE suportadas (multirun sequencial)

```bash
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1
```

### Alternar modelo

```bash
uv run python src/main.py data=gue data.task=emp_H3K4me1 model=nucleotide_transformer
```

### Congelar o backbone (apenas cabeça treinável)

```bash
uv run python src/main.py data=gue data.task=prom_core_all model.frozen_backbone=true
```

### Sobrescrever valores via CLI

```bash
uv run python src/main.py data=gue data.task=splice_reconstructed train.learning_rate=1e-5
```

### Varredura completa: todas as tarefas × todos os modelos

```bash
uv run python src/main.py -m \
  data=gue \
  data.task=prom_core_all,splice_reconstructed,human_tf_0,mouse_0,emp_H3K4me1 \
  model=dnabert2,nucleotide_transformer,kmer_logreg
```

### Múltiplas seeds

```bash
uv run python src/main.py -m \
  data=gue \
  data.task=splice_reconstructed \
  seed=42,123,456
```

### Salvar predições do conjunto de teste

```bash
uv run python src/main.py data=gue data.task=splice_reconstructed save_predictions=true
```

Os CSVs de predição são gravados em `reports/predictions/<run_id>.csv` com
colunas: `run_id`, `true_label`, `predicted_label`, `prob_class_0`...`prob_class_N`.

### Alterar o caminho do leaderboard

```bash
uv run python src/main.py data=gue data.task=prom_core_all leaderboard_csv=reports/minha_varredura.csv
```

O default é `reports/benchmark_results.csv`, configurável em `config.yaml`.

### EVO 2

EVO 2 é um modelo de linguagem de DNA baseado em Hyena do Arc Institute,
carregado via pacote PyPI `evo2` (não HuggingFace Transformers). Utiliza um
tokenizador `CharLevelTokenizer` em nível de byte onde 1 token ≈ 1 nucleotídeo,
tornando as sequências mais longas em contagem de tokens do que tokenizadores
BPE como o do DNABERT-2. Por padrão, LoRA é aplicado às camadas MLP
(`mlp.l1`, `mlp.l2`, `mlp.l3`, `out_filter_dense`) enquanto o resto permanece
congelado. Alternativamente, é possível usar `model.frozen_backbone=true` para
congelar todo o backbone — nesse caso apenas a cabeça linear de classificação é
treinada (sem LoRA).
 

**Diferenças em relação aos outros modelos:**

- **Não é um modelo HF** — carregado via `evo2.Evo2("evo2_1b_base")`, não via
  `AutoModel`. O loop de treino ignora o `DataCollatorWithPadding` padrão e usa
  sequências pré-padded com `default_data_collator`.
- **Flash-Attention obrigatório** — `evo2` depende internamente de `flash-attn`.
- **Transformer Engine** - Para o modelo 1B (40B e 20B também) é necessário ter uma GPU com suporte para fp8 via Transformer Engine.

Para instalação veja a documentação presente em `docs/evo2-ada-lovelace-setup.md` que explica detalhadamente o processo de instalação.

## Workflows BEND

Os workflows BEND exigem dados brutos locais e um arquivo FASTA hg38. Os
scripts de preparação esperam um FASTA indexado padrão; o `pyfaidx` criará um
`.fai` no primeiro acesso se necessário. As configurações padrão estão em
`src/config/download_bend.yaml`, `src/config/prepare_bend_variant_effects.yaml`
e `src/config/kmer_variant_baseline.yaml`.

### 1. Baixar arquivos BEND brutos do ERDA

```bash
uv run python scripts/download_bend.py
```

Genomes está incluído por padrão pois fornece os arquivos FASTA base.

### 2. Preparar datasets de variant-effect

```bash
uv run python scripts/prepare_bend_variant_effects.py \
  genome_fasta=data/raw/bend/data/genomes/GRCh38.primary_assembly.genome.fa
```

### 3. Executar benchmarks zero-shot de variant-effect

Tarefa de expression com DNABERT-2:

```bash
uv run python src/main.py model=dnabert2_zeroshot data=bend_variant_expression
```

Tarefa de disease com DNABERT-2:

```bash
uv run python src/main.py model=dnabert2_zeroshot data=bend_variant_disease
```

Tarefa de expression com Nucleotide Transformer:

```bash
uv run python src/main.py model=nt_zeroshot data=bend_variant_expression
```

O runner zero-shot grava distâncias cosseno por variante em
`reports/variant_effect_distances_<experiment>.csv`. O arquivo fica maior que 100mb portanto nenhum é salvo no github.

### 4. Executar o baseline zero-shot com k-mer

Baseline expression:

```bash
uv run python scripts/kmer_variant_baseline.py \
  data=bend_variant_expression
```

Baseline disease:

```bash
uv run python scripts/kmer_variant_baseline.py \
  data=bend_variant_disease
```

## Outputs de Tracking

Cada execução grava:

| Output | Caminho | Descrição |
|---|---|---|
| Leaderboard GUE (append) | `reports/benchmark_results.csv` | Métricas agregadas por run (F1, MCC, accuracy, ROC-AUC) |
| Leaderboard BEND (append) | `reports/benchmark_results_bend.csv` | Métricas de variant-effect |
| Predições de teste | `reports/predictions/<run_id>.csv` | Quando `save_predictions=true` — labels, predições e probabilidades por amostra |
| Dados de execução MLflow | `mlruns/` | Parâmetros, métricas e artefatos |
| Sumário da run | `runs/<exp>/<model>/run_summary.json` | Sumário completo em JSON |

> O caminho do leaderboard GUE é configurável via `leaderboard_csv` no `config.yaml` ou via CLI: `leaderboard_csv=reports/outro.csv`. O script `run_benchmarks.sh` usa `reports/benchmark_full_runs.csv`.


## MLflow Tags

As runs do MLflow recebem automaticamente as tags `frozen_backbone` e
`model_type` apropriadamente. Tags adicionais podem ser definidas via config ou CLI:

```bash
uv run python src/main.py data=gue data.task=prom_core_all tags.minha_tag=valor
```

Múltiplas tags:

```bash
uv run python src/main.py -m \
  data=gue data.task=splice_reconstructed \
  tags.benchmark_batch=full_runs tags.notes=teste_lr
```

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Dados

### GUE

Todas as tarefas GUE são baixadas de `leannmlindsey/GUE` no Hugging Face. A
tarefa `download-gue` no `pyproject.toml` chama `scripts/prepare_hf_dataset.py`
para cada tarefa, gerando
`data/processed/leannmlindsey_GUE_<task>/{train,validation,test}.csv`.

Para adicionar uma nova tarefa GUE: adicione uma entrada no dicionário `tasks`
em `src/config/data/gue.yaml` e execute novamente o script de download com o
nome da nova tarefa no array `TASKS`.

### BEND

Os arquivos BEND brutos suportados são baixados do ERDA para:

- `data/raw/bend/variant_effects/`
- `data/raw/bend/data/genomes/`

Os outputs processados são gravados em:

- `data/processed/bend_variant_effects/{expression,disease}.csv`

O script de preparação de variantes constrói janelas centradas REF e ALT
diretamente do genoma de referência e ignora variantes com alelos de referência
incompatíveis ou janelas fora dos limites.

## Autores

- Matheus Girardi
- Gabriel Bau
- Andrei Silva
- Artur Pandolfo
- Evandro Diniz

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>
