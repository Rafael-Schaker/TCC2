import json
from pathlib import Path

import matplotlib.pyplot as plt

# ---------- Config ----------
CWD = Path(".").resolve()
OUT = (CWD / "kpi_charts"); OUT.mkdir(parents=True, exist_ok=True)
PLOT_PATH  = OUT / "macro_bars.png"

# Métrica macro a plotar: 'f1', 'precision', 'recall' ou 'mcc'
METRIC_TO_PLOT = "f1"  # equivalente ao "macro_f1" nos arquivos

EVAL_FILES = {
    "treinado_s/Limite": "eval_results__meta_llama_3_1_8b_Q4_K_M_latest.json",
    "Base_s/Limite": "eval_results__llama3_1_8b.json",
    "Treinado_c/Limite": "eval_results_wcap__meta_llama_3_1_8b_Q4_K_M_latest.json",
    "Base_c/Limit": "eval_results_wcap__llama3_1_8b.json",
}

MODEL_MAP = {
    "meta-llama-3.1-8b.Q4_K_M:latest": "Treinado",
    "llama3.1:8b": "Base",
}

# ---------- Helpers ----------
def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

# ---------- Carrega macros ----------
rows = []
for run, fname in EVAL_FILES.items():
    p = CWD / fname
    if not p.exists():
        print(f"[WARN] arquivo não encontrado: {p}")
        continue

    d = jload(p)
    macro = d.get("macro_avg", {}) or {}
    model_raw = d.get("model_name", "") or ""
    model_label = MODEL_MAP.get(model_raw, model_raw or "desconhecido")

    val = macro.get(METRIC_TO_PLOT)
    if val is None:
        print(f"[WARN] métrica '{METRIC_TO_PLOT}' ausente em: {p}")
        continue

    rows.append({
        "run": run,
        "model_label": model_label,
        "value": float(val),
    })

if not rows:
    print("[WARN] nada para plotar (sem dados).")
else:
    # ---------- Prepara dados para o gráfico ----------
    x_labels = [r["model_label"] for r in rows]  # somente o nome do modelo embaixo da coluna
    y_vals   = [r["value"] for r in rows]

    # Cores sóbrias
    COLOR_TRAINED = "#5C80BC"
    COLOR_BASE    = "#D6C28B"

    def tipo_cor(model_label: str) -> str:
        ml = (model_label or "").lower()
        if "trained" in ml:
            return COLOR_TRAINED
        return COLOR_BASE

    colors = [tipo_cor(lbl) for lbl in x_labels]

    # ---------- Plot ----------
    x = range(len(x_labels))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, y_vals, color=colors, edgecolor="none")

    # Identificação do modelo embaixo de cada coluna
    plt.xticks(list(x), x_labels, rotation=0, fontsize=10)

    # Valor do macro em cima da respectiva coluna
    y_max = max(y_vals) if y_vals else 1.0
    y_offset = y_max * 0.02
    for xi, bar, val in zip(x, bars, y_vals):
        plt.text(xi, bar.get_height() + y_offset, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=10)

    # Rótulos discretos
    metric_name = {
        "f1": "Macro F1",
        "precision": "Macro Precision",
        "recall": "Macro Recall",
        "mcc": "Macro MCC",
    }.get(METRIC_TO_PLOT, METRIC_TO_PLOT)

    plt.title(f"{metric_name} por Modelo", fontsize=12)
    plt.ylabel(metric_name, fontsize=11)
    plt.xlabel("Modelo", fontsize=11)

    # Legenda mínima Base x Treinado
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_BASE, label="Base"),
        Patch(facecolor=COLOR_TRAINED, label="Treinado"),
    ]
    plt.legend(handles=legend_elements, frameon=False, loc="best", fontsize=10)

    # Aparência limpa
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    print(f"[OK] Gráfico salvo em: {PLOT_PATH}")
