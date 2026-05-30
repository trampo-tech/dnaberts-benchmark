================================================================================
                         DNABERTS BENCHMARK
     Benchmarking de Modelos de Fundação Genômica (GUE + BEND)
================================================================================

Autores: Matheus Girardi, Gabriel Bau, Andrei Silva, Artur Pandolfo, Evandro Diniz
Licença: MIT

Este projeto avalia modelos de linguagem para DNA nos benchmarks GUE
(Genome Understanding Evaluation) e BEND (zero-shot variant-effect prediction),
com suporte a 6 configurações de modelo: DNABERT-2, Nucleotide Transformer v2,
EVO 2, K-mer + Regressão Logística, BERT-base-uncased e variantes zero-shot.


1. CÓDIGO-FONTE (src/)
--------------------------------------------------------------------------------
Arquitetura Hydra-based com runners especializados por tipo de modelo.

  src/main.py
    Ponto de entrada. Mapeia model.type para o runner correspondente,
    define seeds e inicia a execução.

  src/runners/transformer.py
    Treino supervisionado GUE via HuggingFace Trainer para DNABERT-2,
    Nucleotide Transformer e BERT-base-uncased. Suporta LoRA e backbone
    congelado. Custom BenchmarkTrainer com otimizador dual-LR.

  src/runners/evo2.py
    Treino supervisionado GUE para EVO 2. Tokenizador char-level,
    extração de embeddings via forward hooks, suporte bf16 e FP8.

  src/runners/kmer_logreg.py
    Baseline GUE com regressão logística (sklearn) sobre frequência de
    k-mers (k=4 padrão).

  src/runners/variant_effect_zeroshot.py
    Zero-shot BEND: extrai embeddings REF/ALT de backbones HF e calcula
    distância cosseno para predição de efeito de variantes.

  src/runners/variant_effect_zeroshot_evo2.py
    Mesmo protocolo acima adaptado para EVO 2 (APIs nativas).

  src/services/benchmark_logger.py
    Utilitários de logging: geração de timestamps e run_ids, salvamento
    de sumários JSON, predições CSV e leaderboard (GUE e BEND).

  src/services/metrics.py
    Cálculo de métricas a partir de logits: acurácia, F1, MCC, precisão,
    recall e ROC-AUC. Usa evaluate + sklearn.

  src/modeling/train.py
    Carregamento de modelos HuggingFace para classificação de sequências
    e aplicação de LoRA via PEFT.

  src/modeling/compat.py
    Correções de compatibilidade: bloqueio de Triton legado, desabilitação
    de Flash Attention remoto, ajuste de pad_token_id e hidden_size.


2. CONFIGURAÇÕES (src/config/)
--------------------------------------------------------------------------------
Configurações Hydra organizadas por domínio, com resolução de parâmetros
por cadeia de prioridade (src/config/resolver.py).

  src/config/config.yaml
    Configuração raiz. Define modelo, dados, treino e parâmetros globais
    (experiment_name, leaderboard_csv, seed=42, tags).

  src/config/model/                 (9 arquivos YAML)
    dnabert2.yaml                   DNABERT-2-117M, max_length=2048
    nucleotide_transformer.yaml     NT v2 500M, max_length=2048, LoRA r=8
    evo2.yaml                       EVO 2 1B, max_length=8192, LoRA r=16
    kmer_logreg.yaml                Baseline k=4, max_iter=1000
    bertbase.yaml                   BERT-base-uncased, max_length=512
    dnabert2_zeroshot.yaml          DNABERT-2 zero-shot, max_length=128
    nt_zeroshot.yaml                NT zero-shot, max_length=128
    evo2_zeroshot.yaml              EVO 2 zero-shot, max_length=8192

  src/config/data/                  (3 arquivos YAML)
    gue.yaml                        5 tarefas GUE com parâmetros por tarefa
    bend_variant_expression.yaml    Dados BEND - expressão
    bend_variant_disease.yaml       Dados BEND - doença

  src/config/train/                 (2 arquivos YAML)
    default.yaml                    Treino GUE: lr=3e-5, early_stopping=7,
                                    fp16, métrica alvo=f1, MLflow ativo
    bend_default.yaml               Treino BEND: métrica alvo=auroc_macro


3. SCRIPTS DE BENCHMARK
--------------------------------------------------------------------------------
Scripts shell que executam as suítes completas de avaliação.

  run_benchmarks.sh
    Suite completa: 5 tarefas GUE x 3 modelos x 2 seeds + BEND zero-shot
    x 2 tarefas x 2 modelos. Gera reports/benchmark_full_runs.csv (144
    execuções). Utiliza uv run + Hydra multirun (-m).

  run_evo2_full.sh
    Suite EVO 2: treino GUE com LoRA, backbone congelado e fine-tuning
    completo, mais BEND zero-shot. Utiliza python diretamente.


4. SCRIPTS DE DADOS (scripts/)
--------------------------------------------------------------------------------
Preparação e verificação de dados. As tasks abaixo são definidas em
pyproject.toml e executadas via uv run task <nome>.

  uv run task download-gue
    Baixa datasets GUE do HuggingFace (leannmlindsey/GUE) e gera CSVs
    de treino/validação/teste em data/processed/.
    Chama: scripts/prepare_hf_dataset.py

  uv run task download-bend
    Baixa arquivos brutos BEND (variant effects + genomas) do ERDA
    para data/raw/bend/.
    Chama: scripts/download_bend.py

  uv run task download-bend-variants
    Constrói sequências REF/ALT janeladas a partir dos arquivos BED
    e genoma de referência FASTA.
    Chama: scripts/prepare_bend_variant_effects.py

  uv run task kmer-variant-baseline
    Baseline BEND usando distância L1 entre frequências de k-mers.
    Chama: scripts/kmer_variant_baseline.py

  scripts/create_dataset.py
    Cria datasets brutos a partir de arquivos FASTA (promotores,
    negativos genômicos, TATA binário).

  scripts/split_dataset.py
    Divide dataset bruto em treino/validação/teste com split
    estratificado.

  scripts/verify_evo2.py
    Verifica forward pass do EVO 2 contra valores esperados de loss
    e acurácia.

  scripts/verify_evo2_fp8.py
    Verifica suporte a FP8 via Transformer Engine no EVO 2.


5. RESULTADOS (reports/)
--------------------------------------------------------------------------------
Arquivos com métricas e predições de todas as execuções.

  reports/benchmark_results.csv
    Leaderboard GUE: 109 execuções com métricas por tarefa/modelo/seed.
    Colunas: F1, MCC, acurácia, ROC-AUC (validação e teste), runtime,
    dispositivo, git_commit.

  reports/benchmark_results_bend.csv
    Leaderboard BEND: 6 execuções zero-shot com AUROC, número de
    variantes e distância cosseno média.

  reports/benchmark_full_runs.csv
    Resultado da suite completa (144 execuções), mesmo formato do
    benchmark_results.csv, gerado por run_benchmarks.sh.

  reports/predictions/
    141 arquivos CSV com probabilidades por amostra. Nomenclatura:
    {experimento}__{modelo}__{timestamp}.csv. Colunas: run_id,
    true_label, predicted_label, prob_class_0..prob_class_N.


6. RASTREAMENTO (mlruns/)
--------------------------------------------------------------------------------
Arquivos de rastreamento do MLflow (~93k artefatos), com métricas por
passo de treino, parâmetros dos modelos e sumários JSON de cada execução.
Distribuídos em 4 experimentos distintos (GUE, HF Trainer, BEND, Default).

Para visualizar a interface:
    uv run task mlflow-ui
    # Abre em http://127.0.0.1:5000


7. DOCUMENTAÇÃO
--------------------------------------------------------------------------------
  README.md
    Documento principal do projeto. Explica como utilizar os códigos:
    instalação de dependências com uv, estrutura de diretórios,
    comandos para execução de benchmarks, uso do Hydra para
    configuração e varredura de hiperparâmetros, e interpretação
    dos resultados.

  docs/evo2-ada-lovelace-setup.md
    Guia detalhado de instalação do EVO 2 para GPUs Ada Lovelace
    (RTX 4090). Cobre flash-attn com patch, toolchain conda vs pip,
    suporte FP8 via Transformer Engine e erros comuns.
