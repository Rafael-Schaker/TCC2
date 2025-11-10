import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- Config ----------
CWD = Path(".").resolve()
OUT = (CWD / "kpi_charts"); OUT.mkdir(parents=True, exist_ok=True)
XLSX_PATH = OUT / "kpi_charts_data.xlsx"

EVAL_FILES = {
    "trained_no_wcap": "eval_results__meta_llama_3_1_8b_Q4_K_M_latest.json",
    "baseline_no_wcap": "eval_results__llama3_1_8b.json",
    "trained_wcap": "eval_results_wcap__meta_llama_3_1_8b_Q4_K_M_latest.json",
    "baseline_wcap": "eval_results_wcap__llama3_1_8b.json",
}
RESULT_FILES = ["results.json", "results_wcap.json"]

MODEL_MAP = {
    "meta-llama-3.1-8b.Q4_K_M:latest": "llama3.1:trained",
    "llama3.1:8b": "llama3.1:base",
}

# ---------- Helpers ----------
def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def num(df: pd.DataFrame, cols):
    return df.assign(**{c: pd.to_numeric(df.get(c), errors="coerce") for c in cols})

def sanitize_sheet_name(name: str) -> str:
    n = re.sub(r'[:\\/\?\*\[\]]', '_', str(name))
    return n[:31] if len(n) > 31 else n

# ---------- Macro (eval_results*.json) ----------
macro_rows = []
for run, fname in EVAL_FILES.items():
    p = CWD / fname
    if not p.exists():
        continue
    d = jload(p)
    macro, model_raw = d.get("macro_avg", {}), d.get("model_name", "")
    macro_rows.append({
        "run": run,
        "model_raw": model_raw,
        "model_label": MODEL_MAP.get(model_raw, model_raw or "llama3.1:base"),
        "macro_precision": macro.get("precision"),
        "macro_recall": macro.get("recall"),
        "macro_f1": macro.get("f1"),
        "macro_mcc": macro.get("mcc"),
        "source_file": fname,
    })

macro_df = pd.DataFrame(macro_rows)
if not macro_df.empty:
    macro_df = num(macro_df, ["macro_precision","macro_recall","macro_f1","macro_mcc"])

# ---------- Runs (results*.json) ----------
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
        it = dict(it)
        it["_source_file"] = fname
        runs.append(it)

runs_df = pd.json_normalize(runs) if runs else pd.DataFrame()

# garantir colunas mínimas
for c in ["latency_s","total_tokens","prompt_tokens","completion_tokens","model",
          "scenario_id","condition","request_id","_source_file"]:
    if c not in runs_df.columns:
        runs_df[c] = np.nan

# numéricos e total_tokens
if not runs_df.empty:
    runs_df = num(runs_df, ["latency_s","total_tokens","prompt_tokens","completion_tokens"])
    runs_df.loc[runs_df["total_tokens"].isna(), "total_tokens"] = \
        runs_df["prompt_tokens"].fillna(0) + runs_df["completion_tokens"].fillna(0)
    runs_df["model_label"] = runs_df["model"].map(MODEL_MAP).fillna(runs_df["model"])

# ---------- Salvar Excel ----------
sheets = {}
if not macro_df.empty:
    sheets["macro"] = macro_df
if not runs_df.empty:
    cols = ["_source_file","model","model_label","scenario_id","condition",
            "total_tokens","prompt_tokens","completion_tokens","latency_s","request_id"]
    sheets["runs"] = runs_df[cols].sort_values(["_source_file","model_label","scenario_id"], na_position="last")

if sheets:
    # tenta xlsxwriter; se não tiver, usa engine padrão
    try:
        with pd.ExcelWriter(XLSX_PATH, engine="xlsxwriter") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=sanitize_sheet_name(name), index=False)
    except Exception:
        with pd.ExcelWriter(XLSX_PATH) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=sanitize_sheet_name(name), index=False)
    print(f"[OK] Excel salvo em: {XLSX_PATH}")
else:
    print("[WARN] nada para escrever no Excel.")
