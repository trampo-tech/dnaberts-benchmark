import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# =====================================================
# CONFIGURAÇÃO VISUAL IEEE
# =====================================================

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 300

os.makedirs("reports/figures", exist_ok=True)

# =====================================================
# CARREGAR CSVs
# =====================================================

df_full = pd.read_csv("reports/benchmark_full_runs.csv")

# =====================================================
# LIMPAR TASKS
# =====================================================

df_full["task"] = (
    df_full["experiment"]
    .astype(str)
    .str.lower()
    .str.replace("_frozen", "", regex=False)
    .str.replace("_lora", "", regex=False)
)

# =====================================================
# EXTRAIR MODELO E TIPO DE TREINO
# =====================================================

def extract_model(run_id):
    run_id = str(run_id).lower()
    if "dnabert" in run_id:
        return "DNABERT2"
    if "evo2" in run_id:
        return "EVO2"
    if "nucleotide" in run_id:
        return "Nucleotide Transformer"
    if "bert-base" in run_id or "bert_base" in run_id:
        return "BERT Base"
    return None

def extract_training_type(run_id):
    run_id = str(run_id).lower()
    if "lora" in run_id:
        return "LoRA"
    if "frozen" in run_id or "finetune" in run_id or "full" in run_id:
        return "Full FT"
    return "No FT"

df_full["base_model"]    = df_full["run_id"].apply(extract_model)
df_full["training_type"] = df_full["run_id"].apply(extract_training_type)
df_full = df_full[df_full["base_model"].notna()]

# =====================================================
# AGREGAR: MÉDIA E DESVIO POR TASK / MODELO / TREINO
# =====================================================

grouped_gue = (
    df_full.groupby(["task", "base_model", "training_type"])["test_mcc"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "test_mcc", "std": "test_mcc_std"})
)

grouped_gue.to_csv("reports/gue_summary.csv", index=False)

# =====================================================
# CONFIGURAÇÕES VISUAIS
# =====================================================

palette = {
    "DNABERT2":               "#4C72B0",
    "EVO2":                   "#55A868",
    "Nucleotide Transformer":  "#DD8452",
    "BERT Base":               "#C44E52",
}

hatch_map = {
    "No FT":   "",       # sólido
    "Full FT": "////",   # listras diagonais
    "LoRA":    "xxxx",   # cruzado
}

model_order    = ["BERT Base", "DNABERT2", "Nucleotide Transformer", "EVO2"]
training_order = ["No FT", "Full FT", "LoRA"]

tasks = sorted(grouped_gue["task"].unique())

_label_overrides = {
    "gue_emp_h3k4me1":          "EMP\nH3K4me1",
    "gue_human_tf_0":           "Human TF",
    "gue_mouse_0":              "Mouse TF",
    "gue_prom_core_all":        "Promoter\nCore",
    "gue_splice_reconstructed": "Splice",
}
task_label_map = {
    t: _label_overrides.get(t, t.replace("gue_", "").replace("_", " "))
    for t in tasks
}

# =====================================================
# CONSTRUIR GRÁFICO 

n_tasks  = len(tasks)
models   = model_order
n_models = len(models)

bar_width   = 0.18
group_gap   = 0.05
group_width = n_models * bar_width + group_gap

x_positions = np.arange(n_tasks) * (group_width + 0.1)

fig, ax = plt.subplots(figsize=(7, 3.5))

for m_idx, model in enumerate(models):
    for t_idx, task in enumerate(tasks):
        subset = grouped_gue[
            (grouped_gue["base_model"] == model) &
            (grouped_gue["task"] == task)
        ]

        if subset.empty:
            continue

        row = subset.sort_values("test_mcc", ascending=False).iloc[0]

        x     = x_positions[t_idx] + m_idx * bar_width
        hatch = hatch_map.get(row["training_type"], "")
        color = palette[model]
        err   = row["test_mcc_std"] if not pd.isna(row["test_mcc_std"]) else 0

        ax.bar(
            x, row["test_mcc"],
            width=bar_width * 0.9,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
            yerr=err,
            error_kw=dict(
                ecolor="dimgray",
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
            ),
            zorder=3,
        )

        # Valor em cima da barra (acima da barra de erro)
        label_y = row["test_mcc"] + err + 0.015
        ax.text(
            x, label_y,
            f"{row['test_mcc']:.2f}",
            ha="center", va="bottom",
            fontsize=4.5,
            rotation=90,
            color="black",
            zorder=4,
        )

# =====================================================
# EIXOS E LABELS
# =====================================================

task_labels  = [task_label_map[t] for t in tasks]
tick_centers = x_positions + (n_models * bar_width) / 2 - bar_width / 2

ax.set_xticks(tick_centers)
ax.set_xticklabels(task_labels, fontsize=6.5, ha="center", rotation=0)
ax.set_ylabel("MCC", fontsize=8)
ax.set_xlabel("GUE Task", fontsize=8)
ax.set_title("GUE Benchmark — Model Comparison", fontsize=9, fontweight="bold")
ax.set_ylim(0, grouped_gue["test_mcc"].max() + grouped_gue["test_mcc_std"].max() + 0.12)
ax.tick_params(axis="y", labelsize=7)
ax.yaxis.grid(True, linewidth=0.5, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

# =====================================================
# LEGENDA DUPLA: Modelos (cor) + Fine-tuning (hatch)
# =====================================================

model_handles = [
    mpatches.Patch(facecolor=palette[m], edgecolor="black", linewidth=0.6, label=m)
    for m in model_order
]

ft_handles = [
    mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.6,
                   hatch=hatch_map[t], label=t)
    for t in ["No FT", "Full FT", "LoRA"]
]

leg1 = ax.legend(
    handles=model_handles,
    title="Model", fontsize=6, title_fontsize=6.5,
    loc="upper left", framealpha=0.9
)
ax.add_artist(leg1)

ax.legend(
    handles=ft_handles,
    title="Training", fontsize=6, title_fontsize=6.5,
    loc="upper right", framealpha=0.9
)

plt.tight_layout()

plt.savefig("reports/figures/gue_benchmark.png", dpi=300, bbox_inches="tight")
plt.savefig("reports/figures/gue_benchmark.pdf", bbox_inches="tight")
plt.close()

print("Arquivo gerado:")
print("  reports/figures/gue_benchmark.png")
print("  reports/figures/gue_benchmark.pdf")
print()
print("===== RESULTADOS GUE =====")
print(grouped_gue.to_string(index=False))