import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CWD = Path(".").resolve()
OUT = (CWD / "kpi_charts"); OUT.mkdir(parents=True, exist_ok=True)

EVAL_FILES = {
    "trained_no_wcap": "eval_results__meta_llama_3_1_8b_Q4_K_M_latest.json",
    "baseline_no_wcap": "eval_results__llama3_1_8b.json",
    "trained_wcap": "eval_results_wcap__meta_llama_3_1_8b_Q4_K_M_latest.json",
    "baseline_wcap": "eval_results_wcap__llama3_1_8b.json",
}
RESULT_FILES = ["results.json", "results_wcap.json"]

# keep your label mapping exactly as defined
MODEL_MAP = {
    "meta-llama-3.1-8b.Q4_K_M:latest": "llama3.1:trained",
    "llama3.1:8b": "llama3.1:base",
}

jload = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
num = lambda df, cols: df.assign(**{c: pd.to_numeric(df.get(c), errors="coerce") for c in cols})

def bar(series, labels, title, ylabel, outname):
    ax = series.plot(kind="bar", figsize=(6,4))
    ax.set(title=title, ylabel=ylabel, xlabel="Run")
    ax.tick_params(axis="x", rotation=45)
    for i, p in enumerate(ax.patches):
        ax.annotate(labels.iloc[i], (p.get_x()+p.get_width()/2, p.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT/outname, dpi=160); plt.close()
    print(f"[OK] saved: {OUT/outname}")

print(f"[INFO] working dir: {CWD}")

# ----- macro metrics -----
rows = []
for run, fname in EVAL_FILES.items():
    p = CWD / fname
    print(f"[CHK] {fname} -> {'OK' if p.exists() else 'NOT FOUND'}")
    if not p.exists(): continue
    d = jload(p)
    macro, model_raw = d.get("macro_avg", {}), d.get("model_name", "")
    rows.append({
        "run": run,
        "model": MODEL_MAP.get(model_raw, model_raw or "llama3.1:base"),
        "macro_precision": macro.get("precision"),
        "macro_recall": macro.get("recall"),
        "macro_f1": macro.get("f1"),
        "macro_mcc": macro.get("mcc"),
    })

if rows:
    df = pd.DataFrame(rows)
    df = num(df, ["macro_precision","macro_recall","macro_f1","macro_mcc"])
    df = (df.assign(xlabel=lambda x: x["run"].str.replace("_"," "))
            .sort_values("run").set_index("xlabel"))
    charts = [
        ("macro_f1","Macro F1 by run","Macro F1","macro_f1_by_run.png"),
        ("macro_precision","Macro Precision by run","Macro Precision","macro_precision_by_run.png"),
        ("macro_recall","Macro Recall by run","Macro Recall","macro_recall_by_run.png"),
        ("macro_mcc","Macro MCC by run","Macro MCC","macro_mcc_by_run.png"),
    ]
    for col, title, ylab, fname in charts:
        bar(df[col], df["model"], title, ylab, fname)
else:
    print("[WARN] no eval_results*.json loaded, skipping macro charts")

# ----- tokens vs latency -----
runs = []
for fname in RESULT_FILES:
    p = CWD / fname
    print(f"[CHK] {fname} -> {'OK' if p.exists() else 'NOT FOUND'}")
    if not p.exists(): continue
    r = jload(p)
    runs += (r["results"] if isinstance(r, dict) and "results" in r else (r if isinstance(r, list) else []))

if runs:
    rdf = pd.json_normalize(runs)
    for c in ["latency_s","total_tokens","prompt_tokens","completion_tokens","model"]:
        if c not in rdf.columns: rdf[c] = np.nan
    rdf = num(rdf, ["latency_s","total_tokens","prompt_tokens","completion_tokens"])
    rdf.loc[rdf["total_tokens"].isna(), "total_tokens"] = rdf["prompt_tokens"].fillna(0)+rdf["completion_tokens"].fillna(0)
    rdf = rdf.dropna(subset=["total_tokens","latency_s"]).copy()
    rdf["model_label"] = rdf["model"].map(MODEL_MAP).fillna(rdf["model"]).replace({
        "meta-llama-3.1-8b.Q4_K_M:latest": "llama3.1:trained",
        "llama3.1:8b": "llama3.1:base",
    })

    plt.figure(figsize=(6,4))
    for m, dsub in rdf.groupby("model_label"):
        plt.scatter(dsub["total_tokens"], dsub["latency_s"], alpha=0.8, label=str(m))
    x, y = rdf["total_tokens"].values, rdf["latency_s"].values
    corr = float("nan")
    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        m, b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 100)
        plt.plot(xs, m*xs+b, linestyle="--", linewidth=1)
        corr = np.corrcoef(x, y)[0, 1]
    plt.xlabel("Total tokens"); plt.ylabel("Latency (s)")
    plt.title(f"Tokens vs Latency (all runs)\nOverall correlation ~= {corr:.3f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT/"tokens_vs_latency_all.png", dpi=160); plt.close()
    print(f"[OK] saved: {OUT/'tokens_vs_latency_all.png'}")
else:
    print("[WARN] no results*.json loaded, skipping tokens vs latency")

print(f"Done. Output dir: {OUT}")
