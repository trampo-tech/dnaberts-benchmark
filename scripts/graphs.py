import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np

# =====================================================
# CONFIGURAÇÃO VISUAL
# =====================================================

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 300
os.makedirs("reports/figures", exist_ok=True)

# =====================================================
# CARREGAR CSVs
# =====================================================

df      = pd.read_csv("reports/benchmark_full_runs.csv")
df_bend = pd.read_csv("reports/benchmark_results_bend.csv")

# =====================================================
# EXTRAIR MODELO E TIPO DE TREINO — GUE
# =====================================================

def extract_model(run_id):
    r = str(run_id).lower()
    if "dnabert" in r:    return "DNABERT2"
    if "evo2" in r:       return "EVO2"
    if "nucleotide" in r: return "NTv2"
    if "bert-base" in r:  return "BERT Base"
    return None

def extract_training_type(experiment):
    e = str(experiment).lower()
    if e.endswith("_lora"):   return "LoRA"
    if e.endswith("_frozen"): return "Sem FT"
    return "Full FT"

df["base_model"]    = df["run_id"].apply(extract_model)
df["training_type"] = df["experiment"].apply(extract_training_type)
df["task"]          = (
    df["experiment"].astype(str).str.lower()
    .str.replace("_frozen", "", regex=False)
    .str.replace("_lora",   "", regex=False)
)
df = df[df["base_model"].notna()]

# =====================================================
# EXTRAIR MODELO — BEND (sempre Sem FT)
# =====================================================

df_bend["base_model"]    = df_bend["run_id"].apply(extract_model)
df_bend["training_type"] = "Sem FT"
df_bend["task"]          = df_bend["experiment"].astype(str).str.lower()
df_bend = df_bend[df_bend["base_model"].notna()]

# =====================================================
# CONFIGURAÇÕES COMPARTILHADAS
# =====================================================

palette = {
    "DNABERT2":  "#4C72B0",
    "EVO2":      "#55A868",
    "NTv2":      "#DD8452",
    "BERT Base": "#C44E52",
}

hatch_map = {
    "Sem FT":  "",
    "Full FT": "////",
    "LoRA":    "xxxx",
}

model_params = {
    "BERT Base": 110,
    "DNABERT2":  117,
    "NTv2":      500,
    "EVO2":     1000,
}

model_order     = ["BERT Base", "DNABERT2", "NTv2", "EVO2"]
model_order_bend = ["DNABERT2", "NTv2", "EVO2"]

tasks_gue = [
    "gue_emp_h3k4me1",
    "gue_human_tf_0",
    "gue_mouse_0",
    "gue_prom_core_all",
    "gue_splice_reconstructed",
]

task_label_map = {
    "gue_emp_h3k4me1":          "EMP\nH3K4me1",
    "gue_human_tf_0":           "TF Humano",
    "gue_mouse_0":              "TF Murino",
    "gue_prom_core_all":        "Promotor\nCore",
    "gue_splice_reconstructed": "Splice",
}

bend_label_map = {
    "bend_variant_disease":    "Variante\nDoença",
    "bend_variant_expression": "Variante\nExpressão",
}

# =====================================================
# LOOP DE MÉTRICAS (MCC e F1) — GUE
# =====================================================

for metric in ["mcc", "f1"]:
    metric_col = f"test_{metric}"
    metric_label = "MCC" if metric == "mcc" else "F1-Score"
    suffix = f"_{metric}"
    
    print(f"\n--- Processando métrica: {metric_label} ---")

    # AGREGAR — GUE
    agg = (
        df.groupby(["task", "base_model", "training_type"])
        .agg(
            val_mean=(metric_col, "mean"),
            val_std =(metric_col, "std"),
            runtime =("train_runtime", "mean"),
        )
        .reset_index()
    )

    macro = (
        agg[agg["task"].isin(tasks_gue)]
        .groupby(["base_model", "training_type"])
        .agg(
            val_macro  =("val_mean", "mean"),
            runtime_avg=("runtime",  "mean"),
        )
        .reset_index()
    )

    # =====================================================
    # VISÃO 1 — MACRO: Valor médio vs Tempo (scatter)
    # =====================================================
    fig, ax = plt.subplots(figsize=(6, 4))

    for _, row in macro.iterrows():
        model    = row["base_model"]
        training = row["training_type"]
        x        = row["runtime_avg"] / 60
        y        = row["val_macro"]
        marker   = "o" if training == "Sem FT" else ("s" if training == "Full FT" else "^")

        ax.scatter(
            x, y, s=80, color=palette[model],
            marker=marker, edgecolors="black",
            linewidths=0.6, zorder=3, alpha=0.9,
        )

    label_offsets = {
        ("BERT Base", "Sem FT"):   ( 6,  6),
        ("BERT Base", "Full FT"):  ( 6, -14),
        ("DNABERT2",  "Sem FT"):   (-52,  6),
        ("DNABERT2",  "Full FT"):  ( 6, -14),
        ("NTv2",      "Sem FT"):   ( 6,  6),
        ("NTv2",      "LoRA"):     ( 6,  6),
        ("EVO2",      "Sem FT"):   (-52,  6),
        ("EVO2",      "Full FT"):  ( 6, -14),
        ("EVO2",      "LoRA"):     ( 6,  6),
    }

    for _, row in macro.iterrows():
        model    = row["base_model"]
        training = row["training_type"]
        x        = row["runtime_avg"] / 60
        y        = row["val_macro"]
        dx, dy   = label_offsets.get((model, training), (6, 4))

        ax.annotate(
            f"{model} / {training}",
            xy=(x, y), xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.2, color="black",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4) if abs(dx) > 10 else None,
        )

    model_handles = [
        mpatches.Patch(facecolor=palette[m], edgecolor="black", linewidth=0.6, label=m)
        for m in model_order
    ]
    shape_handles = [
        mlines.Line2D([], [], marker="o", color="gray", linestyle="None", markersize=5, label="Sem FT"),
        mlines.Line2D([], [], marker="s", color="gray", linestyle="None", markersize=5, label="Full FT"),
        mlines.Line2D([], [], marker="^", color="gray", linestyle="None", markersize=5, label="LoRA"),
    ]

    leg1 = ax.legend(handles=model_handles, title="Modelo",     fontsize=6, title_fontsize=6.5, loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles,        title="Treinamento", fontsize=6, title_fontsize=6.5, loc="lower left")

    ax.set_xlabel("Tempo Médio de Treino por Tarefa (min)", fontsize=8)
    ax.set_ylabel(f"{metric_label} Médio nas Tarefas GUE",  fontsize=8)
    ax.set_title("Acurácia vs. Custo Computacional", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.yaxis.grid(True, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"reports/figures/visao1_macro{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Visão 1 — Macro salva ({metric})")

    # =====================================================
    # VISÃO 2 — ESTRATÉGIA: Sem FT vs melhor FT por modelo
    # =====================================================
    ft_priority = {"DNABERT2": "Full FT", "NTv2": "LoRA", "BERT Base": "Full FT", "EVO2": "Full FT"}

    records = []
    for model in model_order:
        for task in tasks_gue:
            sub   = agg[(agg["base_model"] == model) & (agg["task"] == task)]
            no_ft = sub[sub["training_type"] == "Sem FT"]

            if model == "EVO2":
                ft = sub[sub["training_type"].isin(["Full FT", "LoRA"])].sort_values("val_mean", ascending=False).head(1)
            else:
                ft = sub[sub["training_type"] == ft_priority[model]]

            if not no_ft.empty:
                records.append({"model": model, "task": task, "training": "Sem FT",  "val": no_ft.iloc[0]["val_mean"]})
            if not ft.empty:
                records.append({"model": model, "task": task, "training": "Com FT", "val": ft.iloc[0]["val_mean"]})

    strat_df  = pd.DataFrame(records)
    strat_avg = strat_df.groupby(["model", "training"])["val"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x       = np.arange(len(model_order))
    width   = 0.32
    offsets = [-width/2, width/2]
    labels  = ["Sem FT", "Com FT"]
    hatches = ["", "////"]

    for i, (label, hatch) in enumerate(zip(labels, hatches)):
        vals = []
        for model in model_order:
            row = strat_avg[(strat_avg["model"] == model) & (strat_avg["training"] == label)]
            vals.append(row.iloc[0]["val"] if not row.empty else 0)

        bars = ax.bar(
            x + offsets[i], vals,
            width=width * 0.9,
            color=[palette[m] for m in model_order],
            hatch=hatch, edgecolor="black", linewidth=0.6,
            zorder=3, label=label,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=5, rotation=90, zorder=4
            )

    for i, model in enumerate(model_order):
        r0 = strat_avg[(strat_avg["model"] == model) & (strat_avg["training"] == "Sem FT")]
        r1 = strat_avg[(strat_avg["model"] == model) & (strat_avg["training"] == "Com FT")]
        if r0.empty or r1.empty: continue
        gain = r1.iloc[0]["val"] - r0.iloc[0]["val"]
        ax.annotate(
            f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}",
            xy=(x[i], max(r0.iloc[0]["val"], r1.iloc[0]["val"]) + 0.07),
            ha="center", fontsize=5.5,
            color="green" if gain >= 0 else "red", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=7)
    ax.set_ylabel(f"{metric_label} Médio nas Tarefas GUE", fontsize=8)
    ax.set_title(f"Impacto do Fine-Tuning — Quanto o FT ajuda? ({metric_label})", fontsize=9, fontweight="bold")
    ax.set_ylim(0, strat_avg["val"].max() + 0.25)
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.grid(True, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    ft_legend = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="",     label="Sem FT"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="////", label="Com FT"),
    ]
    model_handles = [
        mpatches.Patch(facecolor=palette[m], edgecolor="black", linewidth=0.6, label=m)
        for m in model_order
    ]
    leg1 = ax.legend(handles=model_handles, title="Modelo",      fontsize=6, title_fontsize=6.5, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=ft_legend,            title="Treinamento", fontsize=6, title_fontsize=6.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"reports/figures/visao2_estrategia{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Visão 2 — Estratégia salva ({metric})")

    # =====================================================
    # VISÃO 3 — MICRO: EVO2 — Sem FT vs Full FT vs LoRA
    # =====================================================
    evo2 = agg[(agg["base_model"] == "EVO2") & (agg["task"].isin(tasks_gue))].copy()
    evo2["task_label"] = evo2["task"].map(task_label_map)
    tasks_sorted = [task_label_map[t] for t in tasks_gue]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    bar_width = 0.22
    offsets   = np.array([-bar_width, 0, bar_width])
    x         = np.arange(len(tasks_sorted))

    for i, tt in enumerate(["Sem FT", "Full FT", "LoRA"]):
        vals, errs = [], []
        for tl in tasks_sorted:
            row = evo2[(evo2["task_label"] == tl) & (evo2["training_type"] == tt)]
            if row.empty:
                vals.append(0); errs.append(0)
            else:
                vals.append(row.iloc[0]["val_mean"])
                errs.append(row.iloc[0]["val_std"] if not pd.isna(row.iloc[0]["val_std"]) else 0)

        bars = ax.bar(
            x + offsets[i], vals,
            width=bar_width * 0.9, color="#55A868",
            hatch=hatch_map[tt], edgecolor="black", linewidth=0.6,
            yerr=errs, error_kw=dict(ecolor="dimgray", elinewidth=0.8, capsize=2, capthick=0.8),
            zorder=3, label=tt,
        )
        for bar, val, err in zip(bars, vals, errs):
            if val == 0: continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + err + 0.012,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=5, rotation=90, zorder=4
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks_sorted, fontsize=6.5, ha="center")
    ax.set_ylabel(metric_label, fontsize=8)
    ax.set_title(f"EVO2 — Sem FT vs Full FT vs LoRA ({metric_label})", fontsize=9, fontweight="bold")
    ax.set_ylim(0, evo2["val_mean"].max() + evo2["val_std"].max() + 0.15)
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.grid(True, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    ft_handles = [
        mpatches.Patch(facecolor="#55A868", edgecolor="black", hatch=hatch_map[t], label=t)
        for t in ["Sem FT", "Full FT", "LoRA"]
    ]
    ax.legend(handles=ft_handles, title="Treinamento", fontsize=6, title_fontsize=6.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"reports/figures/visao3_micro_evo2{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Visão 3 — Micro EVO2 salva ({metric})")

    # =====================================================
    # VISÃO 4 — GERAL: Heatmap GUE task × modelo
    # =====================================================
    heat_records = []
    for model in model_order:
        for task in tasks_gue:
            sub = agg[(agg["base_model"] == model) & (agg["task"] == task)]
            if sub.empty:
                heat_records.append({"model": model, "task": task_label_map.get(task, task), "val": np.nan, "training": ""})
            else:
                best = sub.sort_values("val_mean", ascending=False).iloc[0]
                heat_records.append({
                    "model": model, "task": task_label_map.get(task, task),
                    "val": best["val_mean"], "training": best["training_type"],
                })

    heat_df    = pd.DataFrame(heat_records)
    heat_pivot = heat_df.pivot(index="task", columns="model", values="val") \
        .reindex(index=[task_label_map[t] for t in tasks_gue], columns=model_order)
    annot_pivot = heat_df.pivot(index="task", columns="model", values="training") \
        .reindex(index=[task_label_map[t] for t in tasks_gue], columns=model_order)

    annot_matrix = heat_pivot.copy().astype(str)
    for task in annot_matrix.index:
        for model in annot_matrix.columns:
            v  = heat_pivot.loc[task, model]
            tt = annot_pivot.loc[task, model]
            annot_matrix.loc[task, model] = "—" if pd.isna(v) else f"{v:.2f}\n({tt})"

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(
        heat_pivot.astype(float), annot=annot_matrix, fmt="",
        cmap="YlGn", linewidths=0.5, linecolor="white", ax=ax,
        annot_kws={"size": 6.5}, cbar_kws={"label": metric_label, "shrink": 0.8},
        vmin=0, vmax=1,
    )
    ax.set_title(f"Melhor {metric_label} por Modelo × Tarefa GUE (todas as estratégias)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Modelo",       fontsize=8)
    ax.set_ylabel("Tarefa GUE",   fontsize=8)
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=7, rotation=0)

    plt.tight_layout()
    plt.savefig(f"reports/figures/visao4_geral_heatmap{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Visão 4 — Heatmap GUE salva ({metric})")


# =====================================================
# VISÃO 5 — BEND: AUROC por modelo e tarefa (Sem FT)
# =====================================================

agg_bend = (
    df_bend.groupby(["task", "base_model"])
    .agg(
        auroc_mean=("test_auroc", "mean"),
        auroc_std =("test_auroc", "std"),
    )
    .reset_index()
)

tasks_bend   = sorted(agg_bend["task"].unique())
tasks_bend_l = [bend_label_map.get(t, t) for t in tasks_bend]

fig, ax = plt.subplots(figsize=(6, 3.5))

n_tasks   = len(tasks_bend)
n_models  = len(model_order_bend)
bar_width = 0.22
x         = np.arange(n_tasks)
offsets   = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * bar_width

for m_idx, model in enumerate(model_order_bend):
    vals, errs = [], []
    for task in tasks_bend:
        row = agg_bend[(agg_bend["base_model"] == model) & (agg_bend["task"] == task)]
        if row.empty:
            vals.append(0); errs.append(0)
        else:
            vals.append(row.iloc[0]["auroc_mean"])
            errs.append(row.iloc[0]["auroc_std"] if not pd.isna(row.iloc[0]["auroc_std"]) else 0)

    bars = ax.bar(
        x + offsets[m_idx], vals,
        width=bar_width * 0.9,
        color=palette[model],
        edgecolor="black", linewidth=0.6,
        yerr=errs, error_kw=dict(ecolor="dimgray", elinewidth=0.8, capsize=2, capthick=0.8),
        zorder=3, label=model,
    )
    for bar, val, err in zip(bars, vals, errs):
        if val == 0: continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.012,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=6, rotation=90, zorder=4
        )

ax.set_xticks(x)
ax.set_xticklabels(tasks_bend_l, fontsize=8, ha="center")
ax.set_ylabel("AUROC", fontsize=8)
ax.set_title("BEND — Avaliação Zero-Shot (Sem Fine-Tuning)", fontsize=9, fontweight="bold")
ax.set_ylim(0, agg_bend["auroc_mean"].max() + agg_bend["auroc_std"].max() + 0.15)
ax.tick_params(axis="y", labelsize=7)
ax.yaxis.grid(True, linewidth=0.5, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

model_handles = [
    mpatches.Patch(facecolor=palette[m], edgecolor="black", linewidth=0.6, label=m)
    for m in model_order_bend
]
ax.legend(handles=model_handles, title="Modelo", fontsize=6, title_fontsize=6.5, loc="upper right")

# Linha de referência AUROC = 0.5 (aleatório)
ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else n_tasks - 0.1,
        0.51, "Aleatório", fontsize=5.5, color="red", va="bottom", ha="right")

plt.tight_layout()
plt.savefig("reports/figures/visao5_bend.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Visão 5 — BEND salva (AUROC)")

print("\nTodos os arquivos gerados com sucesso em reports/figures/ !")