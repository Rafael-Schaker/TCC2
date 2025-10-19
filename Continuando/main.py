import dataclasses
import hashlib
import json
import os
import time
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
    "gpt-oss:20b",
    "qwen3:4b",
    "gemma3:4b",
    "deepseek-r1:7b",
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ATENÇÃO: nome conforme solicitado (sem o 'r')
RESULTS_JSONL = os.path.join(BASE_DIR, "benchmark_results.json")

# =========================
# Utils
# =========================

def to_jsonable(obj):
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, bytes):
            return obj.decode("utf-8", "replace")
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, "model_dump"):      # Pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"):            # Pydantic v1
            return obj.dict()
        if isinstance(obj, Mapping):
            return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            return [to_jsonable(x) for x in obj]
    except Exception:
        pass
    return repr(obj)

# =========================
# Prompts (PT-BR + formato limpo)
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

USER_PROMPT_TEMPLATE = (
    "Contexto:\n"
    "- nicho: {nicho}\n"
    "- area: {area}\n"
    "- objetivo: {objetivo}\n\n"
    "Tarefa: recomende KPIs até 4 adequados ao contexto.\n\n"
    "Formato OBRIGATÓRIO (array JSON bruto, sem markdown):\n"
    "[{{\"kpi\":\"...\",\"importancia\":\"...\",\"formula\":\"...\",\"metricas\":[\"...\"]}}, ...]"
)

def montar_mensagens(scenario: Dict[str, Any]) -> Tuple[str, str]:
    template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
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

def run_ollama_chat(model: str, system_msg: str, user_msg: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    # Sem options.format="json" para evitar content vazio em alguns modelos.
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

    # Conteúdo textual (compatível com dict/objeto)
    content = ""
    try:
        content = (resp.get("message", {}) or {}).get("content", "") or ""
    except AttributeError:
        try:
            content = getattr(getattr(resp, "message", None), "content", "") or ""
        except Exception:
            content = ""

    # Tokens
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

def main():
    request_id = str(uuid.uuid4())

    with open(RESULTS_JSONL, "w", encoding="utf-8") as _f:
        pass

    for sc in SCENARIOS:
        system_msg, user_msg = montar_mensagens(sc)
        prompt_hash = hashlib.sha256((system_msg + "\n" + user_msg).encode("utf-8")).hexdigest()[:12]

        for model in MODELOS:
            try:
                run = run_ollama_chat(model, system_msg, user_msg)
                registro = {
                    "request_id": request_id,
                    "scenario": sc,
                    "model": model,
                    "system_msg": system_msg,
                    "user_msg": user_msg,
                    "prompt_hash": prompt_hash,
                    "latency_s": run["latency_s"],
                    "prompt_tokens": run["prompt_tokens"],
                    "completion_tokens": run["completion_tokens"],
                    "total_tokens": run["total_tokens"],
                    "content": run["content"],     # bruto do modelo
                    "raw": to_jsonable(run["raw"]),
                }
            except Exception as e:
                registro = {
                    "request_id": request_id,
                    "scenario": sc,
                    "model": model,
                    "system_msg": system_msg,
                    "user_msg": user_msg,
                    "prompt_hash": prompt_hash,
                    "error": f"{type(e).__name__}: {e}",
                }

            with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
