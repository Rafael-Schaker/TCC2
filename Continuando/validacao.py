import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# =========================
# Helpers de texto
# =========================

def _strip_reasoning(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", text)
    text = re.sub(r"\n\s*```\s*$", "", text)
    text = re.sub(r"(?i)(^|\n)\s*(input|output)\s*:\s*", "\n", text)
    return text.strip()

def _first_json_array_block(text: str) -> Optional[str]:
    """Retorna a string do primeiro bloco que parece um array JSON."""
    m = re.search(r"\[\s*{.*}\s*\]", text, flags=re.DOTALL)
    return m.group(0) if m else None

def _strip_accents(s: str) -> str:
    """Remove acentos/diacríticos de uma string (para normalização de CHAVES)."""
    if not isinstance(s, str):
        return s
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

# =========================
# Parsing & Validação das respostas dos modelos
# =========================

def tentar_parse_json(texto: str) -> Optional[List[dict]]:
    """
    Tenta parsear a resposta do modelo como JSON (array):
      1) Limpa cercas/ruídos
      2) Tenta JSON direto
      3) Tenta extrair primeiro array JSON
    """
    texto = _strip_code_fences(_strip_reasoning(texto))
    # 1) json direto
    try:
        parsed = json.loads(texto)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # 2) primeiro array no texto
    bloco = _first_json_array_block(texto)
    if not bloco:
        return None
    try:
        return json.loads(bloco)
    except Exception:
        return None

def _normalize_item_keys(obj: dict) -> dict:
    """Normaliza CHAVES do item (remove acentos, lower, strip)."""
    normalized = {}
    for k, v in obj.items():
        nk = _strip_accents(str(k)).strip().lower()
        normalized[nk] = v
    return normalized

def _coerce_item_types(item: dict) -> dict:
    """Garante tipos básicos esperados."""
    out = dict(item)
    out["kpi"] = str(out.get("kpi", "") or "")
    out["importancia"] = str(out.get("importancia", "") or "")
    out["formula"] = str(out.get("formula", "") or "")
    mets = out.get("metricas", [])
    if mets is None:
        mets = []
    if not isinstance(mets, list):
        mets = [mets]
    out["metricas"] = [str(x) for x in mets]
    return out

def validar_saida(parsed: Optional[List[dict]]) -> Dict[str, Any]:
    """
    Score simples (0–100), tolerante a acentos nas CHAVES originais:
      - JSON válido (40)
      - todos campos presentes (30)
      - fórmulas não vazias (30)
    Retorna também a lista normalizada.
    """
    result = {
        "valid_json": False,
        "n_kpis": 0,
        "all_fields_present": False,
        "has_formulas": False,
        "adherence_score": 0,
        "normalized": None,
    }

    if not isinstance(parsed, list) or len(parsed) == 0:
        return result

    norm_list = []
    fields_ok = True
    formulas_ok = True

    for raw in parsed:
        if not isinstance(raw, dict):
            fields_ok = False
            continue
        item = _normalize_item_keys(raw)
        item = _coerce_item_types(item)
        if not all(k in item for k in ("kpi", "importancia", "formula", "metricas")):
            fields_ok = False
        if not item.get("formula", "").strip():
            formulas_ok = False
        norm_list.append(item)

    result["valid_json"] = True
    result["n_kpis"] = len(norm_list)
    result["all_fields_present"] = fields_ok
    result["has_formulas"] = formulas_ok
    result["normalized"] = norm_list

    score = 0
    if result["valid_json"]:
        score += 40
    if fields_ok:
        score += 30
    if formulas_ok:
        score += 30
    result["adherence_score"] = score
    return result

# =========================
# Carregamento dos resultados (APENAS benchmark_results.json)
# =========================

def _load_records_from_json_allow_jsonl(path: str) -> List[dict]:
    """
    Lê SOMENTE o arquivo 'benchmark_results.json'.
    - Primeiro tenta como ARRAY JSON.
    - Se falhar, tenta interpretar o MESMO arquivo como JSONL (um objeto por linha).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # 1) Tenta array JSON puro
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError("Conteúdo do .json não é um ARRAY JSON.")
    except json.JSONDecodeError:
        # 2) Interpreta como JSONL (um objeto por linha)
        recs: List[dict] = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    recs.append(obj)
            except Exception:
                # ignora linhas inválidas
                continue
        if recs:
            return recs
        # se não conseguiu, propaga erro claro
        raise json.JSONDecodeError(
            "benchmark_results.json não é um array JSON nem JSONL válido.",
            doc=text,
            pos=0
        )

def load_records(base_dir: Optional[str] = None) -> Tuple[List[dict], str]:
    """Carrega APENAS benchmark_results.json do diretório informado (ou atual)."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    p_json = os.path.join(base_dir, "benchmark_results copy.json")
    recs = _load_records_from_json_allow_jsonl(p_json)
    return recs, p_json

# =========================
# Pós-validação
# =========================

def avaliar_registros(regs: List[dict]) -> pd.DataFrame:
    """Aplica parsing/validação por registro e retorna DataFrame com métricas + scores."""
    linhas = []
    for r in regs:
        scenario = r.get("scenario") or {}
        scenario_id = scenario.get("id") or r.get("scenario_id") or ""
        model = r.get("model") or ""
        latency_s = r.get("latency_s")
        prompt_tokens = r.get("prompt_tokens")
        completion_tokens = r.get("completion_tokens")
        total_tokens = r.get("total_tokens")
        content = r.get("content") or ""

        parsed = tentar_parse_json(content) if content else None
        qual = validar_saida(parsed)

        linhas.append({
            "scenario_id": scenario_id,
            "model": model,
            "latency_s": latency_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "valid_json": qual["valid_json"],
            "n_kpis": qual["n_kpis"],
            "fields_ok": qual["all_fields_present"],
            "has_formulas": qual["has_formulas"],
            "adherence_score": qual["adherence_score"],
        })

    df = pd.DataFrame(linhas)
    # Tipos numéricos seguros
    for col in ["latency_s", "prompt_tokens", "completion_tokens", "total_tokens", "adherence_score", "n_kpis"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def resumo_console(df: pd.DataFrame) -> None:
    if df.empty:
        print("(validacao) Nenhum registro válido encontrado.")
        return

    sort_cols = ["adherence_score", "total_tokens", "latency_s"]
    ascending = [False, True, True]
    top_por_cenario = (
        df.sort_values(sort_cols, ascending=ascending)
          .groupby("scenario_id", as_index=False)
          .first()[["scenario_id", "model", "adherence_score", "total_tokens", "latency_s"]]
    )

    print("\n[validacao] Top 1 por cenario:")
    for _, r in top_por_cenario.iterrows():
        print(f"- {r['scenario_id']}: {r['model']} | score={r['adherence_score']} | "
              f"tokens={r['total_tokens']} | latency={r['latency_s']}s")

    medias = (
        df.groupby("model", as_index=False)[
            ["latency_s", "prompt_tokens", "completion_tokens", "total_tokens", "adherence_score", "n_kpis"]
        ]
        .mean(numeric_only=True)
        .sort_values(["adherence_score", "total_tokens", "latency_s"], ascending=[False, True, True])
    )

    print("\n[validacao] Medias por modelo (todas as execucoes):")
    for _, r in medias.iterrows():
        print(f"- {r['model']}: score_medio={r['adherence_score']:.1f} | "
              f"n_kpis_med={r['n_kpis']:.1f} | "
              f"lat_media={r['latency_s']:.3f}s | tokens_tot_med={r['total_tokens']:.1f} "
              f"(prompt={r['prompt_tokens']:.1f}, completion={r['completion_tokens']:.1f})")

def exportar_csv(df: pd.DataFrame, base_dir: Optional[str] = None) -> str:
    """Exporta um CSV com as métricas + scores. Retorna o caminho."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base_dir, "benchmark_results_scored.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out

# =========================
# CLI
# =========================

def main():
    try:
        regs, origem = load_records()
        print(f"(validacao) Carregado: {origem} | registros={len(regs)}")
    except Exception as e:
        print(f"(validacao) ERRO ao carregar resultados: {type(e).__name__}: {e}")
        return

    df = avaliar_registros(regs)
    if df.empty:
        print("(validacao) Sem linhas para avaliar.")
        return

    resumo_console(df)
    out_csv = exportar_csv(df)
    print(f"\n(validacao) CSV com scores salvo em: {out_csv}")

if __name__ == "__main__":
    main()
