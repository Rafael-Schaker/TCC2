import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ========================
# Config
# ========================
CWD = Path(".").resolve()
OUT = (CWD / "kpi_charts"); OUT.mkdir(parents=True, exist_ok=True)

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

MAIN_XLSX   = OUT / "kpi_charts_data.xlsx"        # macro + runs
TOKLAT_XLSX = OUT / "tokens_latency_data.xlsx"    # APENAS dados p/ gráfico Tokens vs Latency

# ========================
# Helpers
# ========================
def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def num(df: pd.DataFrame, cols):
    return df.assign(**{c: pd.to_numeric(df.get(c), errors="coerce") for c in cols})

def sanitize_sheet_name(name: str) -> str:
    n = re.sub(r'[:\\/\?\*\[\]]', '_', str(name))
    return n[:31] if len(n) > 31 else n

def save_excel_or_csv(sheets: dict, xlsx_path: Path, out_dir: Path):
    """Tenta salvar em Excel; se faltar engine, salva CSVs."""
    if not sheets:
        print("[WARN] nada para escrever.")
        return
    # tenta xlsxwriter -> openpyxl -> CSV fallback
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            engine = None

    if engine:
        try:
            with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
                for name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sanitize_sheet_name(name), index=False)
            print(f"[OK] Excel salvo em: {xlsx_path} (engine={engine})")
            return
        except Exception as e:
            print(f"[WARN] falha ao salvar Excel com engine={engine}: {e}. Indo para CSV...")

    # CSV fallback
    for name, df in sheets.items():
        csv_path = out_dir / f"{xlsx_path.stem}__{name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[OK] CSV salvo: {csv_path}")

def corr_slope_intercept(x: np.ndarray, y: np.ndarray):
    """Retorna (corr, slope, intercept) com NaN se não houver pontos suficientes."""
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 2:
            return float("nan"), float("nan"), float("nan")
        corr = float(np.corrcoef(x, y)[0, 1])
        slope, inter = np.polyfit(x, y, 1)
        return corr, float(slope), float(inter)
    except Exception:
        return float("nan"), float("nan"), float("nan")

# ========================
# Carrega macro (eval_results*.json)
# ========================
macro_rows = []
for run, fname in EVAL_FILES.items():
    p = CWD / fname
    print(f"[CHK] {fname} -> {'OK' if p.exists() else 'NOT FOUND'}")
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

# ========================
# Carrega runs (results*.json)
# ========================
runs = []
for fname in RESULT_FILES:
    p = CWD / fname
    print(f"[CHK] {fname} -> {'OK' if p.exists() else 'NOT FOUND'}")
    if not p.exists(): 
        continue
    data = jload(p)
    items = data["results"] if isinstance(data, dict) and "results" in data else (data if isinstance(data, list) else [])
    for it in items:
        if isinstance(it, dict):
            row = dict(it)
            row["_source_file"] = fname
            runs.append(row)

runs_df = pd.json_normalize(runs) if runs else pd.DataFrame()

# garante colunas mínimas
min_cols = ["latency_s","total_tokens","prompt_tokens","completion_tokens","model",
            "scenario_id","condition","request_id","_source_file"]
for c in min_cols:
    if c not in runs_df.columns: runs_df[c] = np.nan

if not runs_df.empty:
    runs_df = num(runs_df, ["latency_s","total_tokens","prompt_tokens","completion_tokens"])
    # total_tokens = prompt + completion se vier vazio
    runs_df.loc[runs_df["total_tokens"].isna(), "total_tokens"] = \
        runs_df["prompt_tokens"].fillna(0) + runs_df["completion_tokens"].fillna(0)
    runs_df["model_label"] = runs_df["model"].map(MODEL_MAP).fillna(runs_df["model"])

# ========================
# Monta planilhas
# ========================
sheets_main = {}
if not macro_df.empty:
    sheets_main["macro"] = macro_df

if not runs_df.empty:
    cols = ["_source_file","model","model_label","scenario_id","condition",
            "total_tokens","prompt_tokens","completion_tokens","latency_s","request_id"]
    sheets_main["runs"] = runs_df[cols].sort_values(["_source_file","model_label","scenario_id"], na_position="last")

# Planilha específica para Tokens vs Latency
sheets_toklat = {}
if not runs_df.empty:
    tok = runs_df.copy()
    tok = tok.dropna(subset=["total_tokens","latency_s"])
    tok = tok.assign(
        total_tokens=pd.to_numeric(tok["total_tokens"], errors="coerce"),
        latency_s=pd.to_numeric(tok["latency_s"], errors="coerce"),
    )
    tok = tok.dropna(subset=["total_tokens","latency_s"])

    # somente as colunas necessárias para criar o scatter no Excel
    tok_cols = ["_source_file","model","model_label","scenario_id","condition",
                "total_tokens","prompt_tokens","completion_tokens","latency_s","request_id"]
    tok = tok[tok_cols].sort_values(["_source_file","model_label","scenario_id"], na_position="last")
    sheets_toklat["tokens_latency"] = tok

    # (opcional) resumo de correlação por modelo e geral
    rows_corr = []
    # por modelo
    for m, g in tok.groupby("model_label"):
        corr, slope, inter = corr_slope_intercept(g["total_tokens"].values, g["latency_s"].values)
        rows_corr.append({"model_label": m, "n": len(g), "corr": corr, "slope": slope, "intercept": inter})
    # overall
    corr, slope, inter = corr_slope_intercept(tok["total_tokens"].values, tok["latency_s"].values)
    rows_corr.append({"model_label": "_overall_", "n": len(tok), "corr": corr, "slope": slope, "intercept": inter})
    sheets_toklat["corr_summary"] = pd.DataFrame(rows_corr)

# ========================
# Salva arquivos
# ========================
if sheets_main:
    save_excel_or_csv(sheets_main, MAIN_XLSX, OUT)
else:
    print("[WARN] planilhas principais vazias (macro/runs).")

if sheets_toklat:
    save_excel_or_csv(sheets_toklat, TOKLAT_XLSX, OUT)
else:
    print("[WARN] sem dados para tokens vs latency.")

print(f"Done. Output dir: {OUT}")
