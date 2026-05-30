================================================================================
                         DNABERTS BENCHMARK
     Benchmarking de Modelos de Fundação Genômica (GUE + BEND)
================================================================================

Autores: Matheus Girardi, Gabriel Bau, Andrei Silva, Artur Pandolfo, Evandro Diniz

Este projeto avalia modelos de linguagem para DNA em um subset de tarefas dos benchmarks GUE
(Genome Understanding Evaluation) e BEND (zero-shot variant-effect prediction),
com suporte a 6 configurações de modelo: DNABERT-2, Nucleotide Transformer v2,
EVO 2, K-mer + Regressão Logística, BERT-base-uncased e variantes zero-shot.


1. CÓDIGO-FONTE (src/)
--------------------------------------------------------------------------------
A arquitetura com é centralizada para Hydra com runners por tipo de modelo.

  src/main.py
    Ponto de entrada.

  src/runners/transformer.py
    Treino supervisionado GUE via HuggingFace Trainer para DNABERT-2,
    Nucleotide Transformer e BERT-base-uncased. Suporta LoRA, backbone
    congelado e fine-tuning completo.

  src/runners/evo2.py
    Treino supervisionado GUE para EVO 2. Tokenizador char-level,
    extração de embeddings via forward hooks, suporte para FP8.

  src/runners/variant_effect_zeroshot.py
    Zero-shot BEND: extrai embeddings REF/ALT de backbones HuggingFace e calcula
    distância cosseno para predição de efeito de variantes.

  src/runners/variant_effect_zeroshot_evo2.py
    Mesmo protocolo acima adaptado para EVO 2.

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
    Alguma correções de compatibilidade: bloqueio de Triton legado, desabilitação
    de Flash Attention remoto, ajuste de pad_token_id e hidden_size.


2. CONFIGURAÇÕES (src/config/)
--------------------------------------------------------------------------------
Configurações Hydra organizadas por domínio, com resolução de parâmetros
por prioridade definida em (src/config/resolver.py).

  src/config/config.yaml
    Configuração raiz. Define modelo, dados, treino e parâmetros globais
    (experiment_name, leaderboard_csv, seed=42, tags).

  src/config/model/
    dnabert2.yaml                   DNABERT-2-117M
    nucleotide_transformer.yaml     NT v2 500M
    evo2.yaml                       EVO 2 1B
    bertbase.yaml                   BERT-base-uncased
    dnabert2_zeroshot.yaml          DNABERT-2 zero-shot
    nt_zeroshot.yaml                NT zero-shot
    evo2_zeroshot.yaml              EVO 2 zero-shot

  src/config/data/
    gue.yaml                        5 tarefas GUE com parâmetros por tarefa
    bend_variant_expression.yaml    Dados BEND - expressão
    bend_variant_disease.yaml       Dados BEND - doença

  src/config/train/
    default.yaml                    Defaults que são sobrescritos por tarefa GUE
    bend_default.yaml               Treino BEND


3. SCRIPTS DE BENCHMARK
--------------------------------------------------------------------------------
Scripts shell que executam as suítes completas de avaliação.

  run_benchmarks.sh
    Suite completa: 5 tarefas GUE x 3 modelos x 3 seeds + BEND zero-shot
    x 2 tarefas x 2 modelos. Escreve resultados em reports/benchmark_full_runs.csv.
    Utiliza uv run + Hydra multirun (-m).

  run_evo2_full.sh
    Suite EVO 2: treino GUE com LoRA, backbone congelado e fine-tuning
    completo, mais BEND zero-shot. Utiliza python diretamente devido ao environment diferente para o Evo2.


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

  scripts/graphs.py
    Gera figuras comparativas (PNG) a partir dos CSVs de resultados.
    Produz 9 gráficos organizados em 5 visões: macro (acurácia vs.
    tempo), estratégia de fine-tuning por modelo, detalhamento EVO 2,
    heatmap GUE por tarefa x modelo, e resultados BEND.
    Saída: reports/figures/visao*.png


5. RESULTADOS (reports/)
--------------------------------------------------------------------------------
Arquivos com métricas e predições de todas as execuções.

  reports/benchmark_results.csv
    Leaderboard GUE de execuções gerais.

  reports/benchmark_results_bend.csv
    Leaderboard BEND de execuções gerais.

  reports/benchmark_full_runs.csv
    Resultado da suite completa, mesmo formato do
    benchmark_results.csv, gerado por run_benchmarks.sh.

  reports/predictions/
    Predições de todas as execuções do run_benchmarks.sh,
    um arquivo CSV por execução.

  reports/gue_summary.csv
    Sumário agregado das execuções GUE com médias por tarefa, modelo base e tipo de treinamento.

  reports/figures/
    9 figuras PNG (300 DPI) geradas por scripts/graphs.py:
      visao1_macro_f1.png / visao1_macro_mcc.png
        Dispersão: métrica média vs. tempo de treino por modelo/tipo.
      visao2_estrategia_f1.png / visao2_estrategia_mcc.png
        Barras comparando Sem FT vs. Com FT por modelo.
      visao3_micro_evo2_f1.png / visao3_micro_evo2_mcc.png
        EVO 2 detalhado por tarefa: Sem FT vs. Full FT vs. LoRA.
      visao4_geral_heatmap_f1.png / visao4_geral_heatmap_mcc.png
        Heatmap: melhor métrica por modelo × tarefa GUE.
      visao5_bend.png
        AUROC zero-shot por modelo nas tarefas BEND.


6. RASTREAMENTO (mlruns/)
--------------------------------------------------------------------------------
Arquivos de rastreamento do MLflow, com métricas por
passo de treino, parâmetros dos modelos e sumários JSON de cada execução.



7. DOCUMENTAÇÃO
--------------------------------------------------------------------------------
  README.md
    Documento principal do projeto para uso do mesmo. Explica como utilizar os códigos:
    instalação de dependências com uv, estrutura de diretórios,
    comandos para execução de benchmarks, uso do Hydra para
    configuração e varredura de hiperparâmetros, e interpretação
    dos resultados.

  docs/evo2-ada-lovelace-setup.md
    Guia detalhado de instalação do EVO 2 para GPUs Ada Lovelace
    (RTX 4090). Cobre flash-attn com patch, toolchain conda vs pip,
    suporte FP8 via Transformer Engine e erros comuns.
