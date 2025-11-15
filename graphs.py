import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ===== Config mínima =====
CWD = Path(".").resolve()
OUT = (CWD / "kpi_charts"); OUT.mkdir(parents=True, exist_ok=True)

EVAL_FILES = {
    "treinado_s/Limite": "eval_results__meta_llama_3_1_8b_Q4_K_M_latest.json",
    "Base_s/Limite": "eval_results__llama3_1_8b.json",
    "Treinado_c/Limite": "eval_results_wcap__meta_llama_3_1_8b_Q4_K_M_latest.json",
    "Base_c/Limit": "eval_results_wcap__llama3_1_8b.json",
}
RESULT_FILES = ["results.json", "results_wcap.json"]

MODEL_MAP = {
    "meta-llama-3.1-8b.Q4_K_M:latest": "Treinado",
    "llama3.1:8b": "Base",
}

COLOR_TRAINED = "#5C80BC"  # azul fosco
COLOR_BASE    = "#DB8036"  # cinza azulado

# ===== Helpers curtos =====
def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def fnum(x):
    try:
        return float(x)
    except Exception:
        return None

def tipo(model_label: str, run_key: str = "") -> str:
    ml = (model_label or "").lower()
    rk = (run_key or "").lower()
    if "trained" in ml or "treinad" in rk:
        return "Treinado"
    return "Base"

def cor(tipo_str: str) -> str:
    return COLOR_TRAINED if tipo_str == "Treinado" else COLOR_BASE

# ===== 1) Gráficos MACRO (F1/Precision/Recall/MCC) =====
macro_rows = []
for run_key, fname in EVAL_FILES.items():
    p = CWD / fname
    if not p.exists():
        continue
    d = jload(p)
    macro = d.get("macro_avg", {}) or {}
    model_raw = d.get("model_name", "") or ""
    label = MODEL_MAP.get(model_raw, model_raw or "desconhecido")
    macro_rows.append({
        "run": run_key,
        "model_label": label,
        "macro_f1": fnum(macro.get("f1")),
        "macro_precision": fnum(macro.get("precision")),
        "macro_recall": fnum(macro.get("recall")),
        "macro_mcc": fnum(macro.get("mcc")),
    })

macro_df = pd.DataFrame(macro_rows)

def plot_macro(metric_key: str, metric_name: str):
    if macro_df.empty or metric_key not in macro_df.columns:
        return
    df = macro_df.dropna(subset=[metric_key]).copy()
    if df.empty:
        return

    x_labels = df["model_label"].tolist()
    y_vals   = df[metric_key].astype(float).tolist()
    colors   = [cor(tipo(ml, rk)) for ml, rk in zip(df["model_label"], df["run"])]

    x = np.arange(len(x_labels))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, y_vals, color=colors, edgecolor="none")
    plt.xticks(x, x_labels, rotation=0, fontsize=10)

    # eixo 0–1 fixo + rótulo “seguro” abaixo de 1.0
    ax = plt.gca()
    ax.set_ylim(0.0, 1.0)
    for xi, bar, val in zip(x, bars, y_vals):
        pos = min(val + 0.02, 0.98)
        plt.text(xi, pos, f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.ylabel(f"{metric_name} (0–1)", fontsize=11)
    plt.xlabel("Modelo", fontsize=11)

    plt.legend(
        handles=[Patch(facecolor=COLOR_BASE, label="Base"),
                 Patch(facecolor=COLOR_TRAINED, label="Treinado")],
        frameon=False, loc="best", fontsize=10
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    plt.title(f"{metric_name} por Modelo", fontsize=12)
    plt.tight_layout()
    outp = OUT / f"{metric_key}_bars.png"
    plt.savefig(outp, dpi=200)
    plt.close()
    print(f"[OK] Gráfico salvo: {outp}")

for key, name in [
    ("macro_f1", "Macro F1"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("macro_mcc", "Macro MCC"),
]:
    plot_macro(key, name)

# ===== 2) Gráfico Tokens vs Latency =====
runs = []
for fname in RESULT_FILES:
    p = CWD / fname
    if not p.exists():
        continue
    data = jload(p)
    items = data["results"] if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [])
    for it in items:
        if not isinstance(it, dict): 
            continue
        model_raw = str(it.get("model", "") or "")
        label = MODEL_MAP.get(model_raw, model_raw or "desconhecido")

        tok = it.get("total_tokens")
        if tok is None:
            tok = (it.get("prompt_tokens") or 0) + (it.get("completion_tokens") or 0)
        lat = it.get("latency_s")

        tok = fnum(tok); lat = fnum(lat)
        if tok is not None and lat is not None and tok > 0 and lat >= 0:
            runs.append((label, tok, lat))

if runs:
    labels, X, Y = zip(*runs)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    groups = {}
    for lbl, x, y in zip(labels, X, Y):
        groups.setdefault(lbl, {"x": [], "y": []})
        groups[lbl]["x"].append(x)
        groups[lbl]["y"].append(y)

    plt.figure(figsize=(10, 6))
    for lbl, dct in groups.items():
        plt.scatter(dct["x"], dct["y"], label=lbl, alpha=0.9)

    if len(X) >= 2 and np.std(X) > 0:
        m, b = np.polyfit(X, Y, 1)
        x_line = np.linspace(X.min(), X.max(), 100)
        plt.plot(x_line, m * x_line + b, linestyle="--")
        r = float(np.corrcoef(X, Y)[0, 1])
    else:
        r = 0.0

    plt.title(f"Tokens vs Latency (all runs)\nOverall correlation ≈ {r:.3f}")
    plt.xlabel("Total tokens")
    plt.ylabel("Latency (s)")
    plt.legend()
    plt.tight_layout()
    outp = OUT / "tokens_vs_latency.png"
    plt.savefig(outp, dpi=200)
    plt.close()
    print(f"[OK] Gráfico salvo: {outp}")
else:
    print("[WARN] sem dados válidos para Tokens vs Latency.")
