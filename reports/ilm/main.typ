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

O projeto #text(style: "italic")[dnaberts-benchmark] é um framework multi-tarefa para avaliação de modelos de fundação genômicos, utilizando tarefas do benchmark GUE #footnote[Zhou et al., "DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genomes", 2024.] e tarefas selecionadas do BEND #footnote[Marin et al., "BEND: Benchmarking DNA Language Models on Biologically Meaningful Tasks", 2024.]. O pipeline é gerenciado via Hydra, com tracking de experimentos no MLflow e exportação automática
de leaderboards em CSV.

Atualmente, dois modelos foram avaliados em todas as tarefas GUE e nas tarefas de Variant Effect do BEND, totalizando 28 execuções nas RTX 4090 (7 tarefas GUE #sym.times 2 modelos #sym.times 2 sementes). As execuções foram realizadas com dois seeds distintos (21194 e 63194) para avaliar a estabilidade dos resultados.

= Última reunião

Analisamos outros benchmarks e como foram feitos:
Fundamentalmente o que fizemos não é grande novidade, apenas recriamos resultados até então. O fator novidade seria o comparativo com o modelo Evo2, que em seu paper original não se comparou aos modelos que utlizamos aqui e nem utilizou tasks da comunidade.
Ademais, se adicionarmos o novo dataset conforme conversado teríamos outro fator novo. Acredito que temos um potencial interessante de comparar e contrastar modelos, no entanto agora temos uma limitação de tempo e isso que nos preocupa. Discussões sobre seguem ao final deste relatório.
Falando um pouco da metodologia que estamos utilizando, nossas métricas seguem outros benchmarks na área. As tarefas são provenientes de benchmarks publicamente disponíveis.


Succintamente, o objetivo aqui é definirmos uma rota para chegarmos em um resultado satisfatório para a disciplina.

= Tarefas Implementadas

== GUE (Genome Understanding Evaluation)

As 7 tarefas do benchmark GUE abrangem diferentes problemas biológicos, comprimentos de sequência e números de classes:

#figure(
  table(
    columns: 5,
    table.header[*Tarefa*][*Categoria*][*Seq*][*Classes*][*Épocas*],
    [`prom_core_all`], [Detecção de promotores], [70 bp], [2], [4],
    [`splice_reconstructed`], [Sítio de splicing], [400 bp], [3], [5],
    [`human_tf_0`], [Ligação de TF (humano)], [101 bp], [2], [3],
    [`mouse_0`], [Ligação de TF (camundongo)], [101 bp], [2], [10],
    [`EPI_HUVEC`], [Marcas epigenéticas], [3000 bp], [2], [10],
    [`emp_H3K4me1`], [Modificação de histonas], [500 bp], [2], [5],
    [`fungi_species_20`], [Classificação de espécies], [10000 bp], [20], [10],
  ),
  caption: [Tarefas GUE implementadas e suas configurações de treinamento.]
) <tab:gue_tasks>

Todas as tarefas utilizam fine-tuning completo para o DNABERT-2 e LoRA para o Nucleotide Transformer v2 (500M multi-espécies). O treinamento emprega AdamW com learning rate de 3e-5, early stopping com paciência de 10 avaliações e fp16 mixed precision.

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
  `mouse_0`, `emp_H3K4me1` e `EPI_HUVEC`.


  === Early Stopping

  #figure(
    table(
      columns: 3,
      table.header[*Tarefa*][*DNABERT-2*][*NT v2 500M*],
      [`prom_core_all`], [3 / 4], [4 / 4],
      [`splice_reconstructed`], [5 / 5], [5 / 5],
      [`mouse_0`], [10 / 10], [10 / 10],
      [`human_tf_0`], [3 / 3], [3 / 3],
      [`EPI_HUVEC`], [5 / 10], [5 / 10],
      [`emp_H3K4me1`], [4 / 5], [4 / 5],
      [`fungi_species_20`], [3 / 10], [2 / 10],
    ),
    caption: [Numero de epocas executadas por tarefa antes da parada do treinamento (epocas executadas / total configurado).]
  ) <tab:early_stopping>


No entanto, no artigo original essas foram as informações passadas:

#blockquote[
This section presents the hyperparameters we used in the fine-tuning stage on each model. Table 7 shows the number of training steps we used for each task. We use AdamW (Loshchilov & Hutter, 2019) as optimizer. We keep most of the other hyperparameters the same for all the models across all the datasets, including a batch size of 32, a warmup step of 50, and a weight decay of 0.01. For DNABERT and DNABERT-2, we perform standard fine-tuning with a learning rate of 3e-5, while for the Nucleotide Transformers, we perform parameter efficient fine-tuning (PEFT) using Low-Rank Adaptation (LoRA) with a learning rate of 1e-4, a LoRA alpha of 16, a LoRA dropout of 0.05, and a LoRA r of 8. The hyperparameters are selected based on grid searches over commonly used ones in preliminary experiments [...]
]

= Resultados

Os resultados abaixo comparam o melhor valor obtido entre as duas sementes (21194 e 63194) na RTX 4090 com os valores de referência da literatura. As métricas reportadas são MCC (Matthews Correlation Coefficient) para tarefas GUE e AUROC para tarefas BEND. As referências para o DNABERT-2 são do artigo original do modelo; para o Nucleotide Transformer, utiliza-se o
NT-2500M-multi (modelo de maior capacidade da família NT) como referência.

== Tarefas GUE (MCC)

#figure(
  table(
    columns: 6,
    table.header[*Tarefa*][*DNABERT-2*][*DNABERT-2 (Ref.)*][*NT v2 500M*][*NT 2500M (Ref.)*][*#sym.delta*],
    [`prom_core_all`],   [0.6680], [0.6937], [0.6916], [0.7033], [#sym.delta = -0.0257 / -0.0117],
    [`splice_reconstructed`], [0.8563], [0.8499], [0.8998], [0.8935], [#sym.delta = +0.0064 / +0.0063],
    [`human_tf_0`],      [0.6860], [0.7199], [0.6785], [0.6664], [#sym.delta = -0.0339 / +0.0121],
    [`mouse_0`],         [0.6348], [0.5676], [0.6275], [0.6331], [#sym.delta = +0.0672 / -0.0056],
    [`emp_H3K4me1`],     [0.4898], [0.5052], [0.5334], [0.5530], [#sym.delta = -0.0154 / -0.0196],
    [`EPI_HUVEC`],       [0.1465], [—],      [0.3151], [—],      [Ver discussão abaixo],
    [`fungi_species_20`],[0.8558], [0.9304], [0.9245], [0.9285], [#sym.delta = -0.0746 / -0.0040],
  ),
  caption: [Resultados GUE na RTX 4090 (melhor semente) vs. referência da literatura. O #sym.delta indica DNABERT-2 (Nosso #sym.minus Ref.) / NT (Nosso #sym.minus Ref.).]
) <tab:gue_results>

== Tarefas BEND — Efeitos de Variante (AUROC)

#figure(
  table(
    columns: 5,
    table.header[*Tarefa*][*DNABERT-2*][*DNABERT-2 (Ref.)*][*#sym.delta*][*Nota*],
    [Expressão], [0.4903], [0.49], [+0.0003], [Resultado alinhado com a literatura],
    [Doença],    [0.5440], [0.51], [+0.0340], [Ligeiramente acima da referência],
  ),
  caption: [Resultados BEND (variant effects) na RTX 4090 vs. referência.]
) <tab:bend_results>

= Discussão: Tarefa EPI

A tarefa `EPI_HUVEC` (marcas epigenéticas em células HUVEC, 3000 bp) apresentou desempenho consideravelmente abaixo do esperado, com MCC máximo de 0.3151 (NT) e 0.1465 (DNABERT-2). No seed 21194, o DNABERT-2 chegou a obter MCC = 0.0, indicando colapso total da classificação. As tarefas não conseguem converger mesmo com todas as épocas. (Mostrar os gráficos de treinamento `gue_EPI_HUVEC__zhihan1996_DNABERT-2-117M__20260512T132922Z`)

Tivemos problemas com resultados de outras tarefas do conjunto Enhancer Promoter Interaction, a `EPI_HUVEC` é a segunda tarefa que tentamos fazer funcionar. A principal problemática é que os autores do paper original não publicaram nenhum hiperparâmetro (Ver tabela de hiperparâmetros para comparativo) das tarefas do dataset GUE+ (Tarefas EPI, Fungi e Covid). Outros usuários constataram dificuldades em replicar resultados das tarefas EPI e Covid (https://github.com/MAGICS-LAB/DNABERT_2/issues/99 e https://github.com/MAGICS-LAB/DNABERT_2/issues/103).

A questão que fica é como devemos proceder nessa tarefa? É uma tarefa interessante dado seu comprimento e natureza, no entano acreditamos que insistir nela talvez seja mau uso de nosso tempo, algo que não temos muito.

= Próximos Passos

== Modelo Evo2 nas tarefas GUE

Adicionar o modelo Evo2 (Arc Institute)ao benchmark já foi realizado com uma implementação LoRA adaptada de https://github.com/NVIDIA/bionemo-framework/issues/884 utilizando PEFT, não é 100% igual devido à limitações do wrapper PEFT em cima de camadas Lineares da Transformer Engine (TELinear e não nn.Linear)
Agora falta executá-lo em todas as 7 tarefas GUE. Este modelo utiliza uma arquitetura diferente (Striped Hyena) e foi treinado em uma escala massiva de genomas.

=== Possíveis riscos
O modelo evo2 é significativamente maior que os outros modelos que estamos lidando, a utilização de LoRA é vital para podermos treinar algo e ainda é possível que ocorram erros entre a integração do PEFT com o modelo. Assim, uma direção alternativa que podemos tomar seria comparar esse modelo em um formato zeroshot nas tarefas treinando apenas o classificador, uma vez que nosos benchmark é limitado a hardware de nível consumidor.

== Modelos anteriores na tarefa de Histone (BEND)

Executar DNABERT-2 e Nucleotide Transformer na tarefa de modificação de
histonas do BEND (`bend_histone`). A implementação atual foi adaptada da original do paper BEND e utiliza uma CNN como header. Para comparações com os outros modelos acreditamos melhor trocar essa head por uma camada linear simples como nas outras tarefas.

=== Possíveis riscos
Essa adaptação pode levar a uma pior performance dos modelos do que foi reportado pelos criadores do BEND

== Modelo Evo2 nas tarefas BEND

Após a integração do Evo2 no pipeline GUE, estender o suporte para as
tarefas BEND:

- *Variant effects (zero-shot)*: Adaptar o runner `variant_effect_zeroshot.py`
  para extrair embeddings do Evo2 e calcular distâncias de cosseno.
- *Histone modification*: Criar config `model/evo2_histone.yaml` e executar
  fine-tuning com decodificador linear.
- Executar o sweep com 2 sementes e comparar com os resultados atuais.


== Adição do dataset
Na última reunião e em discussões subsequentes foi comentado que poderíamos adicionar um novo dataset para execuções, proveniente da equipe. Algumas dúvidas para resolvermos:
- Qual a natureza desse dataset?
- Quantos ajustes teriam de ser feitos para este dataset ter formato sequência:label. Ou seja o quão cru esse dataset é?
- Esse dataset já foi testado com algum modelo? Para termos uma noção do quão difícil ele é para os modelos

==== Possíveis riscos
Para a adição do dataset o maior problema seria se precisarmos realizar uma grande quantidade de ajustes para formatá-lo, uma vez que nos falta conhecimentos da área.

== Riscos gerais
Conforme mencionado pelo professor Andrey, é possível que a máquina que estamos utilizando para rodar os experimentos seja ocupada por outra pessoa repentinamente. Assim, gostaria de entender melhor como seria para acessarmos a máquina do professor Scalabrin e que tipo de preparações teríamos de ter. Gostaríamos de ter ela como um plano B de emergência, mas não queremos importunar o professor.

== Execuções sem fine-tuning
Na última reunião (11/05) o professor Marco perguntou se havíamos executado os modelos sem fine-tuning, não fizemos. No entanto, considerando o atual estado da pipeline seria uma tarefa bem simples de ser feita custando praticamente só o tempo de execução. Seria uma proposta para obter mais resultados, acredito que compute não seria tanto problema já que apenas o HEAD linear seria treinado.

= Resumo do Status

#figure(
  table(
    columns: 3,
    table.header[*Modelo*][*GUE*][*BEND*],
    [DNABERT-2 (117M)],       [#text(fill: green)[#sym.checkmark] 7/7], [#text(fill: green)[#sym.checkmark] Variantes #text(fill: red)[#sym.circle] Histonas],
    [NT v2 (500M multi)],     [#text(fill: green)[#sym.checkmark] 7/7], [#text(fill: green)[#sym.checkmark] Variantes #text(fill: red)[#sym.circle] Histonas],
    [Evo2],                   [#text(fill: red)[#sym.circle] Pendente], [#text(fill: red)[#sym.circle] Pendente],
  ),
  caption: [Status atual de cobertura do benchmark por modelo e categoria de tarefa.]
) <tab:status>
