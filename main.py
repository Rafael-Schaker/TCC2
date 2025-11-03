import dataclasses
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple

import ollama
from langchain_core.prompts import ChatPromptTemplate

# =========================
# CONFIG
# =========================

MODELOS = [
    "meta-llama-3.1-8b.Q4_K_M:latest",
    "llama3.1:8b",
]

SCENARIOS = [
    {"id": "attr_streaming_mix",          "nicho": "Mídia/Publisher & Streaming",     "area": "Atribuição & Mix de Canais",     "objetivo": "otimizar o mix de canais para contribuição real e elevar ROAS em 15%"},
    {"id": "ads_telecom_cac",             "nicho": "Telecom & Internet",              "area": "Aquisição de Tráfego",            "objetivo": "reduzir o CAC em 20% mantendo volume de novos clientes"},
    {"id": "brand_fintech_roas",          "nicho": "Fintech",                          "area": "Marca & Awareness",               "objetivo": "aumentar o ROAS em 30% sem perder volume de impressões"},
    {"id": "crm_ong_donativos",           "nicho": "Governamental / ONG",              "area": "CRM & Mensageria",                "objetivo": "crescer doações recorrentes em 25% e reduzir churn de doadores em 15%"},
    {"id": "pricing_ondemand_ticket",     "nicho": "Apps on-demand",                   "area": "Pricing & Receita",               "objetivo": "elevar o ticket médio em 12% sem reduzir a taxa de conversão"},
    {"id": "onboarding_imob_d7",          "nicho": "Imobiliário / Portais de leads",   "area": "Ativação & Onboarding",           "objetivo": "melhorar a ativação no D7 em 5 p.p."},
    {"id": "retencao_saasb2b_churn",      "nicho": "SaaS B2B",                         "area": "Retenção",                         "objetivo": "reduzir o churn mensal em 2 p.p. mantendo NPS >= 60"},
    {"id": "growth_saasb2c_trialpaid",    "nicho": "SaaS B2C",                         "area": "App Growth & Mobile",             "objetivo": "aumentar conversão de trial para pago em 25% com payback < 6 meses"},
    {"id": "comun_marketplace_ret30",     "nicho": "Marketplace",                       "area": "Conteúdo & Comunidade",           "objetivo": "elevar retenção 30 dias para 45%"},
    {"id": "cs_saude_csat_fcr",           "nicho": "Saúde Digital / Telemedicina",     "area": "Suporte & Experiência do Cliente","objetivo": "aumentar CSAT para 90% e reduzir recontatos (FCR +15 p.p.)"},
    {"id": "app_educacao_conv",           "nicho": "Educação",                          "area": "App Growth & Mobile",             "objetivo": "aumentar conversão de onboarding em 20% e reduzir bounce em 10%"},
    {"id": "omni_telecom_sla",            "nicho": "Telecom & Internet",               "area": "Omnichannel & Offline Impact",    "objetivo": "atribuir vendas offline e manter SLA de atendimento < 2 min no chat"},
    {"id": "devops_auto_uptime",          "nicho": "marketplace automotivo",           "area": "engenharia/DevOps",               "objetivo": "elevar uptime para 99,95% e reduzir MTTR em 20%"},
    {"id": "risco_mobilidade_fraude",     "nicho": "mobilidade urbana",                "area": "compliance & risco",              "objetivo": "reduzir o tempo de resolução de fraudes em 25% mantendo FPR < 3%"},
    {"id": "crm_agritech_reativacao",     "nicho": "agritech de insumos",              "area": "CRM/lifecycle",                    "objetivo": "elevar reativação 30d em 15% e reduzir unsubscribe em 20%"},
    {"id": "ops_logtech_otif",            "nicho": "logtech (3PL)",                     "area": "operações/logística",              "objetivo": "aumentar OTIF em 8 p.p. e reduzir custo por pedido em 12%"},
    {"id": "prod_moda_entregab",          "nicho": "e-commerce de moda",               "area": "produto/experiência",             "objetivo": "elevar entregabilidade de e-mail para 98% e reduzir bounces < 1%"},
    {"id": "crm_hotel_forecast",          "nicho": "hospitalidade (hotéis)",           "area": "CRM/lifecycle",                    "objetivo": "melhorar acurácia de previsão de demanda para > 85%"},
    {"id": "prod_eventos_rsvp",           "nicho": "plataforma de eventos",            "area": "produto/experiência",             "objetivo": "aumentar taxa de RSVP confirmado em 20% via e-mail/push"},
    {"id": "people_conteudo_turnover",    "nicho": "plataforma de conteúdo",           "area": "RH/People",                        "objetivo": "reduzir turnover em 3 p.p. mantendo time-to-hire <= 30 dias"},
    {"id": "perfsec_saas_mrr",            "nicho": "SaaS de cibersegurança",           "area": "marketing de performance",         "objetivo": "crescer MRR em 10% elevando MQL rate em 15% com CPL estável"},
    {"id": "sales_edcorp_cac",            "nicho": "educação corporativa",             "area": "vendas (inside sales)",           "objetivo": "reduzir CAC em 15% mantendo win rate >= 25%"},
    {"id": "sales_gaming_checkout",       "nicho": "gaming mobile",                     "area": "vendas (inside sales)",           "objetivo": "aumentar conversão de checkout em 15% sem reduzir ARPPU"},
    {"id": "crm_solar_churn90",           "nicho": "energia solar distribuída",        "area": "CRM/lifecycle",                    "objetivo": "reduzir churn em 90 dias em 2 p.p. e elevar CTR em 20%"},
    {"id": "ads_imob_cpl",                "nicho": "Imobiliário / Portais de leads",   "area": "Aquisição de Tráfego",            "objetivo": "reduzir CPL em 20% preservando taxa de qualificação >= 35%"},
]

TEMPERATURE = 0.3
SEED = 423
KEEP_ALIVE = "10m"
NUM_CTX = 4096
MAX_NEW_TOKENS = 2048

def _base_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

BASE_DIR = _base_dir()
RESULTS_UNLIMITED_JSON = os.path.join(BASE_DIR, "results.json")
RESULTS_LIMITED_JSON   = os.path.join(BASE_DIR, "results_wcap.json")

# =========================
# Utils
# =========================

def log(msg: str):
    print(msg, flush=True)

def to_jsonable(obj):
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, bytes):
            return obj.decode("utf-8", "replace")
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if isinstance(obj, Mapping):
            return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            return [to_jsonable(x) for x in obj]
    except Exception:
        pass
    return repr(obj)

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def extract_first_json_array(text: str) -> str:
    """Extrai o primeiro array JSON `[...]` como TEXTO (usa pilha de colchetes)."""
    if not isinstance(text, str) or not text:
        return ""
    s = _strip_code_fences(text)

    start = s.find("[")
    if start == -1:
        return s  # devolve texto original se não achar array
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return s  # sem fechamento adequado -> devolve original

def extract_json_objects(s: str) -> List[str]:
    """
    Extrai substrings de objetos JSON balanceados `{...}` sem usar regex recursiva.
    Trata strings e escapes para não quebrar em chaves dentro de strings.
    """
    if not isinstance(s, str) or not s:
        return []
    s = _strip_code_fences(s)

    objs: List[str] = []
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    objs.append(s[start:i+1])
                    start = -1
    return objs

def _remove_accents(txt: str) -> str:
    return unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")

def _norm_key(k: str) -> str:
    k2 = _remove_accents(k).lower().strip()
    if k2 in {"kpi"}:
        return "kpi"
    if k2 in {"importancia", "importância"}:
        return "importancia"
    if k2 in {"formula", "fórmula"}:
        return "formula"
    if k2 in {"metricas", "métricas", "metrica", "métrica"}:
        return "metricas"
    return k2

def _as_list_of_str(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        out = []
        for v in val:
            if isinstance(v, (str, int, float, bool)):
                out.append(str(v))
        return out
    if isinstance(val, (str, bytes)):
        s = val.decode("utf-8", "replace") if isinstance(val, bytes) else str(val)
        parts = re.split(r"[,\|;]+", s)
        return [p.strip() for p in parts if p.strip()]
    return [str(val)]

def _normalize_item(obj: Any) -> Dict[str, Any]:
    """Normaliza um item para {kpi, importancia, formula, metricas} (sem acentos nas chaves)."""
    if not isinstance(obj, Mapping):
        return {}
    norm = {}
    for k, v in obj.items():
        nk = _norm_key(str(k))
        norm[nk] = v

    kpi = str(norm.get("kpi", "") or "").strip()
    imp = str(norm.get("importancia", "") or "").strip()
    frm = str(norm.get("formula", "") or "").strip()
    mets = _as_list_of_str(norm.get("metricas", []))

    if not kpi:
        return {}
    return {"kpi": kpi, "importancia": imp, "formula": frm, "metricas": mets}

def _json_loads_try(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None

def sanitize_output(raw_content: str, limit4: bool = False) -> Tuple[str, List[Dict[str, Any]], bool, List[str]]:
    """
    Garante um array JSON válido:
    - tenta carregar o primeiro `[...]`;
    - se falhar, varre por objetos `{...}` balanceados;
    - normaliza chaves e garante metricas como lista.
    """
    issues: List[str] = []
    s = _strip_code_fences(raw_content or "")
    array_text = extract_first_json_array(s)

    items: List[Dict[str, Any]] = []
    valid = False

    # 1) tentativa direta
    arr = _json_loads_try(array_text)
    if isinstance(arr, list):
        for obj in arr:
            it = _normalize_item(obj)
            if it:
                items.append(it)
        if items:
            valid = True
        else:
            issues.append("empty_array_after_normalize")
    else:
        issues.append("array_parse_failed")

        # 2) fallback: objetos balanceados
        objs_text = extract_json_objects(s)
        if not objs_text:
            issues.append("no_objects_found")
        else:
            salvaged = 0
            for ot in objs_text:
                obj = _json_loads_try(ot)
                if isinstance(obj, Mapping):
                    it = _normalize_item(obj)
                    if it:
                        items.append(it)
                        salvaged += 1
            if salvaged == 0:
                issues.append("unrecoverable_objects")
            else:
                valid = True

    # 3) respeitar limite (modo limited)
    if limit4 and len(items) > 4:
        items = items[:4]

    out_text = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return out_text, items, valid, issues

# =========================
# Prompts
# =========================

SYSTEM_PROMPT = (
    "Você é um especialista em KPIs.\n"
    "Regras de saída:\n"
    "- Responda SOMENTE com um ARRAY JSON bruto (sem markdown, sem cercas ``` e sem rótulos como 'Input:'/'Output:').\n"
    "- Use EXATAMENTE estas chaves por item: kpi, importancia, formula, metricas (sem acentos nas chaves).\n"
    "- Todos os VALORES TEXTUAIS devem estar em Português do Brasil.\n"
    "- A lista deve ser concisa (apenas os KPIs essenciais ao contexto).\n"
    "- Fórmulas curtas, claras e computáveis.\n"
)

USER_PROMPT_LIMITED = (
    "Contexto:\n"
    "- nicho: {nicho}\n"
    "- area: {area}\n"
    "- objetivo: {objetivo}\n\n"
    "Tarefa: recomende KPIs até 4 adequados ao contexto.\n\n"
    "Formato OBRIGATÓRIO (array JSON bruto, sem markdown):\n"
    "[{{\"kpi\":\"...\",\"importancia\":\"...\",\"formula\":\"...\",\"metricas\":[\"...\"]}}, ...]"
)

USER_PROMPT_UNLIMITED = (
    "Contexto:\n"
    "- nicho: {nicho}\n"
    "- area: {area}\n"
    "- objetivo: {objetivo}\n\n"
    "Tarefa: recomende KPIs adequados ao contexto.\n\n"
    "Formato OBRIGATÓRIO (array JSON bruto, sem markdown):\n"
    "[{{\"kpi\":\"...\",\"importancia\":\"...\",\"formula\":\"...\",\"metricas\":[\"...\"]}}, ...]"
)

def montar_mensagens(scenario: Dict[str, Any], limited: bool) -> Tuple[str, str]:
    template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_LIMITED if limited else USER_PROMPT_UNLIMITED),
    ])
    msgs = template.format_messages(
        nicho=scenario["nicho"],
        area=scenario["area"],
        objetivo=scenario["objetivo"],
    )
    return msgs[0].content, msgs[1].content

# =========================
# Execução Ollama
# =========================

def _model_disponivel(nome: str) -> bool:
    try:
        data = ollama.list()
        modelos = [m.get("name") for m in (data.get("models") or [])]
        return any((nome == m or nome in (m or "")) for m in modelos)
    except Exception:
        return True  # se der erro ao listar, tenta usar assim mesmo

def run_ollama_chat(model: str, system_msg: str, user_msg: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        options={
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": NUM_CTX,
            "num_predict": MAX_NEW_TOKENS,
        },
        keep_alive=KEEP_ALIVE,
        stream=False,
    )
    latency_s = time.perf_counter() - t0

    try:
        content = (resp.get("message", {}) or {}).get("content", "") or ""
    except AttributeError:
        try:
            content = getattr(getattr(resp, "message", None), "content", "") or ""
        except Exception:
            content = ""

    def _iget(o, key, default=0):
        try:
            return int(getattr(o, key, None) or o.get(key) or default)
        except Exception:
            return default

    prompt_tok = _iget(resp, "prompt_eval_count", 0)
    compl_tok  = _iget(resp, "eval_count", 0)
    total_tok  = prompt_tok + compl_tok

    return {
        "content": content,
        "latency_s": float(latency_s),
        "prompt_tokens": prompt_tok,
        "completion_tokens": compl_tok,
        "total_tokens": total_tok,
        "raw": to_jsonable(resp),
    }

# =========================
# Main
# =========================

def safe_dump(path: str, data: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

def main():
    safe_dump(RESULTS_UNLIMITED_JSON, [])
    safe_dump(RESULTS_LIMITED_JSON, [])
    log(f"[Init] results (unlimited): {RESULTS_UNLIMITED_JSON}")
    log(f"[Init] results (limited)  : {RESULTS_LIMITED_JSON}")

    request_id = str(uuid.uuid4())
    results_unlimited: List[Dict[str, Any]] = []
    results_limited:   List[Dict[str, Any]] = []

    for sc in SCENARIOS:
        log(f"\n[SCENARIO] {sc['id']}")
        for model in MODELOS:
            # -------- UNLIMITED --------
            try:
                system_msg, user_msg = montar_mensagens(sc, limited=False)
                prompt_hash = hashlib.sha256((system_msg + "\n" + user_msg).encode("utf-8")).hexdigest()[:12]
                run = run_ollama_chat(model, system_msg, user_msg)
                out_text, items, valid, issues = sanitize_output(run["content"], limit4=False)
                rec = {
                    "request_id": request_id,
                    "scenario_id": sc["id"],
                    "model": model,
                    "condition": "unlimited",
                    "prompt_hash": prompt_hash,
                    "latency_s": run["latency_s"],
                    "prompt_tokens": run["prompt_tokens"],
                    "completion_tokens": run["completion_tokens"],
                    "total_tokens": run["total_tokens"],
                    "output": out_text,
                    "parsed_items": items,
                    "valid": valid,
                    "issues": issues,
                }
                log(f"[OK] unlimited -> {model} (valid={valid}, items={len(items)})")
            except Exception as e:
                rec = {
                    "request_id": request_id,
                    "scenario_id": sc.get("id"),
                    "model": model,
                    "condition": "unlimited",
                    "error": f"{type(e).__name__}: {e}",
                }
                log(f"[ERR] unlimited -> {model}: {rec['error']}")
            results_unlimited.append(rec)
            safe_dump(RESULTS_UNLIMITED_JSON, results_unlimited)

            # -------- LIMITED (até 4) --------
            try:
                system_msg, user_msg = montar_mensagens(sc, limited=True)
                prompt_hash = hashlib.sha256((system_msg + "\n" + user_msg).encode("utf-8")).hexdigest()[:12]
                run = run_ollama_chat(model, system_msg, user_msg)
                out_text, items, valid, issues = sanitize_output(run["content"], limit4=True)
                rec = {
                    "request_id": request_id,
                    "scenario_id": sc["id"],
                    "model": model,
                    "condition": "limited",
                    "prompt_hash": prompt_hash,
                    "latency_s": run["latency_s"],
                    "prompt_tokens": run["prompt_tokens"],
                    "completion_tokens": run["completion_tokens"],
                    "total_tokens": run["total_tokens"],
                    "output": out_text,
                    "parsed_items": items,
                    "valid": valid,
                    "issues": issues,
                }
                log(f"[OK] limited   -> {model} (valid={valid}, items={len(items)})")
            except Exception as e:
                rec = {
                    "request_id": request_id,
                    "scenario_id": sc.get("id"),
                    "model": model,
                    "condition": "limited",
                    "error": f"{type(e).__name__}: {e}",
                }
                log(f"[ERR] limited   -> {model}: {rec['error']}")
            results_limited.append(rec)
            safe_dump(RESULTS_LIMITED_JSON, results_limited)

    log("\n[SUMMARY]")
    log(f"  Unlimited records: {len(results_unlimited)}  -> {RESULTS_UNLIMITED_JSON}")
    log(f"  Limited   records: {len(results_limited)}    -> {RESULTS_LIMITED_JSON}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        raise
