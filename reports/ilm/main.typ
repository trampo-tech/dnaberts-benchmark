#import "@preview/ilm:2.0.0": *

#set text(lang: "pt")

#show: ilm.with(
  title: [Benchmark de Modelos de\ Fundação Genômicos],
  date: datetime(year: 2026, month: 05, day: 12),
  author: "Matheus Girardi, Gabriel Bau",
  abstract: [
  ],
  table-of-contents: outline(title: "Índice"),
)

= Visão Geral

O projeto consiste em um benchmark multi-tarefa para avaliação de modelos de fundação genômicos, utilizando tarefas do benchmark GUE #footnote[Zhou et al., "DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genomes", 2024.] e tarefas selecionadas do BEND #footnote[Marin et al., "BEND: Benchmarking DNA Language Models on Biologically Meaningful Tasks", 2024.]. 

O pipeline é gerenciado via Hydra, com tracking de experimentos no MLflow e exportação automática de leaderboards em CSV.

Atualmente, dois modelos foram avaliados em todas as tarefas GUE e nas tarefas de Variant Effect do BEND, totalizando 24 execuções nas RTX 4090 (5 tarefas GUE #sym.times 2 modelos #sym.times 2 sementes + 2 tarefas BEND #sym.times 2 modelos). As execuções foram realizadas com dois seeds distintos (21194 e 63194) para avaliar a estabilidade dos resultados.

Fundamentalmente o que fizemos não é grande novidade, apenas recriamos resultados até então. O fator novidade seria o comparativo com o modelo Evo2, que em seu paper original não se comparou aos modelos que utilizamos aqui e nem utilizou tasks da comunidade.

Ademais, outro ponto é a adição de um novo dataset providenciado pela equipe do professor Roberto H., que traria dados não vistos antes com validação de especialistas da área genômica.

= Modelos

== DNABERT-2 (117M)

O DNABERT-2 #footnote[Zhou et al., "DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genomes", 2024.] é um modelo transformer encoder-only baseado na arquitetura BERT, treinado com o objetivo de masked language modeling (MLM) em sequências de DNA multi-espécies. Utiliza o tokenizador Byte Pair Encoding (BPE) com vocabulário de 4096 tokens para lidar eficientemente com sequências longas, substituindo a tokenização por k-mers utilizada na versão anterior. Conta com 117 milhões de parâmetros, 12 camadas transformer e dimensão de hidden size de 768. Em nosso benchmark, utilizamos fine-tuning completo do modelo em todas as tarefas.

== Nucleotide Transformer v2 (500M multi-espécies)

O Nucleotide Transformer v2 #footnote[Dalla-Torre et al., "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics", 2023.] pertence a uma família de modelos transformer treinados em genomas de mais de 850 espécies. A versão utilizada (500M multi-species) possui 24 camadas, hidden size de 1024 e 500 milhões de parâmetros. Diferentemente do DNABERT-2, o tokenizador opera em nível de nucleotídeo único (6-mers não sobrepostos). Em nosso benchmark, utilizamos fine-tuning via Low-Rank Adaptation (LoRA) com r=8, alpha=16 e dropout de 0.05, aplicado nas projeções query e value das camadas de atenção, seguindo a configuração do paper original.

= Tarefas Implementadas

== GUE (Genome Understanding Evaluation)

As 5 tarefas do benchmark GUE selecionadas abrangem diferentes problemas biológicos, comprimentos de sequência e números de classes:

#figure(
  table(
    columns: 5,
    table.header[*Tarefa*][*Categoria*][*Seq*][*Classes*][*Épocas*],
    [`prom_core_all`], [Detecção de promotores], [70 bp], [2], [4],
    [`splice_reconstructed`], [Sítio de splicing], [400 bp], [3], [5],
    [`human_tf_0`], [Ligação de TF (humano)], [101 bp], [2], [3],
    [`mouse_0`], [Ligação de TF (camundongo)], [101 bp], [2], [10],
    [`emp_H3K4me1`], [Marcas epigenéticas (levedura)], [500 bp], [2], [5],
  ),
  caption: [Tarefas GUE implementadas e suas configurações de treinamento.]
) <tab:gue_tasks>

=== Descrição das Tarefas

- *Core promoter detection (`prom_core_all`, humano)*: Predição da região promotora central (core promoter), a região mais próxima ao transcription start site (TSS). A janela de contexto é de -34 a +35 bp ao redor do TSS, tornando-a mais desafiadora devido ao contexto reduzido. O dataset combina promotores TATA e não-TATA do Eukaryotic Promoter Database (EPDnew). Sequências negativas são construídas com conteúdo GC equivalente fora de regiões promotoras.

- *Splice site prediction (`splice_reconstructed`, humano)*: Predição de sítios doadores e aceptores de splicing no genoma humano. O dataset original (Wang et al., 2019) contém sequências de 400 bp extraídas do genoma de referência humano Ensembl GRCh38. Como modelos existentes atingem performance quase perfeita no dataset original, a versão reconstruída adiciona exemplos adversariais iterativamente (falsos positivos do hold-out set) para aumentar a dificuldade.

- *Transcription factor binding (`human_tf_0`)*: Predição de sítios de ligação de fatores de transcrição (TF) no genoma humano. Utiliza dados dos experimentos ENCODE ChIP-seq (690 experimentos, 161 TFs em 91 linhagens celulares humanas). Extrai-se uma região de 101 bp ao redor do centro de cada pico como classe positiva, com sequências negativas não sobrepostas de mesmo comprimento e conteúdo GC. 

- *Transcription factor binding (`mouse_0`)*: Análogo ao `human_tf_0`, utilizando dados de ENCODE ChIP-seq de camundongo (78 experimentos). Sequências negativas são geradas por dinucleotide shuffling preservando frequências relativas. Demais configurações idênticas à versão humana.

- *Epigenetic marks prediction (`emp_H3K4me1`, levedura)*: Predição de marcas epigenéticas em levedura, especificamente a modificação de histona H3K4me1. Modificações epigenéticas influenciam a expressão gênica sem alterar a sequência de DNA. Os 10 datasets originais foram divididos em treino/validação/teste na proporção 8:1:1.

Todas as tarefas utilizam fine-tuning completo para o DNABERT-2 e LoRA para o Nucleotide Transformer v2 (500M multi-espécies). O treinamento emprega AdamW com learning rate de 3e-5 e fp16 mixed precision.

== BEND (Variant Effects)

As tarefas de efeito de variante do BEND utilizam a abordagem zero-shot: calcula-se a distância de cosseno entre os embeddings das sequências de referência (REF) e alternativa (ALT). Quanto maior a distância, maior o efeito previsto da variante.

#figure(
  table(
    columns: 4,
    table.header[*Tarefa*][*Config*][*Métrica*][*Variantes*],
    [Expressão], [`bend_variant_expression`], [AUROC], [105.263],
    [Doença], [`bend_variant_disease`], [AUROC], [295.495],
  ),
  caption: [Tarefas de variante do BEND implementadas.]
) <tab:bend_tasks>

=== Descrição das Tarefas

- *Noncoding variant effects — Expression (`bend_variant_expression`)*: Predição de efeito de variantes de nucleotídeo único (SNPs) na expressão gênica. Utiliza o dataset DeepSEA (Zhou & Troyanskaya, 2015) com SNPs funcionais (eQTLs do GRASP) e SNPs de fundo genético (1000 Genomes Project). São 98.221 variantes sem efeito e 8.000 com efeito. O contexto de embedding é de 512 bp ao redor da variante.

- *Noncoding variant effects — Disease (`bend_variant_disease`)*: Predição de patogenicidade de SNPs não-codificantes a partir do ClinVar. Contém 274.399 variantes benignas e 21.524 patogênicas. Filtram-se variantes codificantes e mitocondriais. O contexto de embedding também é de 512 bp.

= Comparação de Hiperparâmetros

As principais discrepâncias comparando com os códigos presentes no Github são:

- *Learning rate do NT*: Utilizamos 3e-5 (mesmo valor do DNABERT-2),
  enquanto o script oficial usa 1e-4.
- *Batch size*: Nosso benchmark utiliza batch sizes maiores (32 vs. 8),
  resultando em menos passos de gradiente por época. Combinado com early
  stopping, isso pode levar a treinamento insuficiente em datasets pequenos.
- *Early stopping*: Os scripts de referência não utilizam early stopping,
  treinando pelo número fixo de épocas. Nossa configuração pode
  interromper o treinamento prematuramente, especialmente nas tarefas
  `mouse_0` e `emp_H3K4me1`.

= Resultados

Os resultados abaixo comparam o melhor valor obtido entre as duas sementes (21194 e 63194) na RTX 4090 com os valores de referência da literatura. As métricas reportadas são MCC (Matthews Correlation Coefficient) para tarefas GUE e AUROC para tarefas BEND. As referências para o DNABERT-2 são do artigo original do modelo; para o Nucleotide Transformer, utiliza-se o
NT-2500M-multi (modelo de maior capacidade da família NT) como referência.

== Tarefas GUE (MCC)

#figure(
  table(
    columns: 6,
    table.header[*Tarefa*][*DNABERT-2*][*DNABERT-2 (Ref.)*][*NT v2 500M*][*NT 2500M (Ref.)*][*#sym.delta*],
    [`prom_core_all`],   [0.6741], [0.6937], [0.6916], [0.7033], [#sym.delta = -0.0196 / -0.0117],
    [`splice_reconstructed`], [0.8643], [0.8499], [0.8998], [0.8935], [#sym.delta = +0.0144 / +0.0063],
    [`human_tf_0`],      [0.6930], [0.7199], [0.6785], [0.6664], [#sym.delta = -0.0269 / +0.0121],
    [`mouse_0`],         [0.6348], [0.5676], [0.6275], [0.6331], [#sym.delta = +0.0672 / -0.0056],
    [`emp_H3K4me1`],     [0.5095], [0.5052], [0.5334], [0.5530], [#sym.delta = +0.0043 / -0.0196],
  ),
  caption: [Resultados GUE na RTX 4090 (melhor semente) vs. referência da literatura. O #sym.delta indica DNABERT-2 (Nosso #sym.minus Ref.) / NT (Nosso #sym.minus Ref.).]
) <tab:gue_results>

== Tarefas BEND — Efeitos de Variante (AUROC)

#figure(
  table(
    columns: 4,
    table.header[*Tarefa*][*DNABERT-2*][*DNABERT-2 (Ref.)*][*#sym.delta*],
    [Expressão], [0.4903], [0.49], [+0.0003],
    [Doença],    [0.5440], [0.51], [+0.0340],
  ),
  caption: [Resultados BEND (variant effects) com DNABERT-2 vs. referência. Resultados do Nucleotide Transformer pendentes.]
) <tab:bend_results>

= Próximos Passos

== Modelo Evo2

O Evo2 #footnote[Brixi et al., "Evo 2: Genome modeling and design across all domains of life", 2025.] é um modelo de fundação genômico desenvolvido pelo Arc Institute em colaboração com a NVIDIA, treinado em mais de 128 mil genomas abrangendo todos os domínios da vida. Diferentemente dos modelos transformer tradicionais, o Evo2 utiliza a arquitetura Striped Hyena, que combina operadores de convolução híbridos (Hyena) com atenção multi-head para processamento eficiente de sequências extremamente longas (até 1 milhão de nucleotídeos). O modelo conta com versões de 1B, 7B, 20B e 40B de parâmetros, sendo que as versões de 1B e 7B são mais viavéis dado nosso hardware. Outro detalhe importante é a diferença no tokenizador, que opera em nível de nucleotídeo único.
=== Adaptação LoRA

Implementamos uma adaptação LoRA para o Evo2 baseada nas discussões e código do BioNEMO #footnote[BioNEMO Framework, NVIDIA. https://github.com/NVIDIA/bionemo-framework/issues/884]. A integração não é trivial pois as camadas lineares do Evo2 utilizam Transformer Engine (TELinear, classes `te.Linear`) em vez de `nn.Linear` padrão, o que impõe limitações ao wrapper PEFT da HuggingFace. A solução atual contorna essa limitação utilizando hooks manuais ou reimplementação parcial das camadas para compatibilidade com LoRA.

=== Resultados Preliminares

--- TODO: Adicionar resultados preliminares do Evo2 nas tarefas GUE (MCC) assim que disponíveis. ---

=== Possíveis riscos

O Evo2 é significativamente maior que os outros modelos utilizados (1B--7B parâmetros vs. 117M--500M). A utilização de LoRA é vital para viabilizar o treinamento em hardware de nível consumidor. Ainda é possível que ocorram erros de integração entre o PEFT e o Transformer Engine. Uma direção alternativa seria comparar o Evo2 em formato zero-shot treinando apenas o classificador linear (probe), limitando o custo computacional ao embedding das sequências.

== Execuções sem Fine-Tuning

Na última reunião (11/05) foi levantada a questão de executarmos os modelos sem fine-tuning para estabelecer uma baseline comparativa. Considerando o estado atual da pipeline, esta é uma tarefa de implementação simples, custando apenas o tempo de execução.


== Adição de Novo Dataset

Discutiu-se a inclusão de um novo dataset proveniente da equipe para enriquecer o benchmark com uma tarefa adicional. Esta seção serve como esqueleto para acomodar esta tarefa futura.

=== Descrição da Tarefa

--- TODO: Preencher descrição da tarefa (natureza do dataset, espécie, objetivo biológico). ---

=== Formato dos Dados

--- TODO: Descrever formato das sequências (comprimento, tipo de sequência), número de classes, tamanho do dataset (treino/validação/teste). ---

=== Métricas e Baseline

--- TODO: Definir métrica de avaliação e estabelecer baseline (performance trivial, performance de modelo simples). ---

=== Possíveis riscos

O principal risco é o fato de não termos referência da dificuldade da tarefa para modelos de fundação genômicos, o que dificulta a calibração de hiperparâmetros e expectativas de performance.

= Resumo do Status

#figure(
  table(
    columns: 4,
    table.header[*Modelo*][*GUE*][*BEND*][w/o Fine Tuning],
    [DNABERT-2 (117M)],       [#text(fill: green)[#sym.checkmark] 5/5], [#text(fill: green)[#sym.checkmark] Variantes], [#text(fill: red)[#sym.circle] Pendente],
    [NT v2 (500M multi)],     [#text(fill: green)[#sym.checkmark] 5/5], [#text(fill: green)[#sym.checkmark] Variantes], [#text(fill: red)[#sym.circle] Pendente],
    [Evo2],                   [#text(fill: red)[#sym.circle] Pendente], [#text(fill: red)[#sym.circle] Pendente], [#text(fill: red)[#sym.circle] Pendente],
  ),
  caption: [Status atual de cobertura do benchmark por modelo e categoria de tarefa.]
) <tab:status>
