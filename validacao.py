import json
import re
import unicodedata
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# =========================
# Config
# =========================
MODEL_FILES = ["results_wcap.json", "results.json"]
DEBUG = True  # coloque False para silenciar logs

def norm(txt: Any) -> str:
    """Normaliza acento/caixa e remove não-alfa-num -> 'dau/mau' => 'dau mau'."""
    if not isinstance(txt, str): 
        return ""
    t = unicodedata.normalize("NFKD", txt).encode("ascii","ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


# =========================
# COLE AQUI O SEU UNIVERSO COMPLETO (o mesmo que você já usa)
# =========================
UNIVERSO: Set[str] =  {  
    "Taxa de Retorno sobre Atribuição",
    "Taxa de Conversão por Canal",
    "Custo por Atribuição por Canal",
    "Taxa de Retorno sobre Investimento",
    "CAC",
    "AOV",
    "Taxa de Conversão",
    "Taxa de Reativação 30d",
    "Taxa de Abertura",
    "Taxa de Cliques (CTR)",
    "Taxa de Conversão por Peça",
    "ROI",
    "CPA",
    "Incremental Revenue",
    "Attribution Share",
    "CPL",
    "Win Rate",
    "MQL Rate",
    "CPM",
    "CTR",
    "Share of Voice",
    "Taxa de doacao recorrente",
    "Churn de doadores",
    "LTV do doador",
    "Custo por doador recorrente",
    "ARPU",
    "Abandono de carrinho",
    "Ativacao D7",
    "Conclusao de onboarding",
    "Tempo ate primeira acao",
    "Retencao D7",
    "Churn de clientes",
    "Retencao 30d",
    "NPS",
    "Churn de MRR",
    "Conversao trial pago",
    "Payback CAC",
    "DAU MAU",
    "Engajamento composto",
    "CSAT",
    "FCR",
    "Tempo medio de atendimento",
    "Tempo de primeira resposta",
    "Bounce rate",
    "Tempo de onboarding",
    "SLA de atendimento",
    "Tempo de espera no chat",
    "Atribuicao de vendas offline",
    "Uptime",
    "MTTR",
    "Change failure rate",
    "Deployment frequency",
    "Tempo de resolucao de fraudes",
    "FPR",
    "TPR",
    "Valor recuperado por fraude",
    "Reativacao 30d",
    "Unsubscribe rate",
    "Open Rate",
    "OTIF",
    "Custo por pedido",
    "Lead time de pedido",
    "Fill rate",
    "Entregabilidade de e-mail",
    "Bounce rate de e mail",
    "Spam complaint rate",
    "Acuracia de previsao",
    "MAPE",
    "Taxa de no show",
    "Taxa de ocupacao",
    "RSVP confirmado",
    "Tempo para confirmar RSVP",
    "Turnover",
    "Time to hire",
    "Offer acceptance rate",
    "Custo por contratacao",
    "MRR",
    "Average deal size",
    "Duracao do ciclo de vendas",
    "Conversao de checkout",
    "ARPPU",
    "Abandono de checkout",
    "Tempo de checkout",
    "Churn 90d",
    "Retencao 90d",
    "Opt in rate",
    "Conversao de landing page",
    "Taxa de qualificacao",
    "Taxa de Retenção por Canal",
    "Taxa de Desempenho de Canal",
    "Custo por Conversão",
    "Retorno sobre Investimento",
    "Taxa de Retenção de ROI",
    "Participação de Atribuição",
    "Taxa de Conversão por Impressões",
    "Custo de Atribuição por Canal",
    "Receita Incremental por Canal",
    "Retorno sobre Investimento (ROI)",
    "Taxa de Retorno sobre Atribuição (ROAS)",
    "Custo por Conversão (CPA)",
    "Participação de Atribuição (Attribution Share)",
    "Receita Incremental (Incremental Revenue)",
    "Taxa de Conversão por Atribuição",
    "Taxa de Conversão por Canal (CVR Canal)",
    "Custo por Conversão por Canal (CPC)",
    "Taxa de Conversão Geral (CVR)",
    "Custo por Aquisição (CPA)",
    "Receita por Aquisição",
    "Taxa de Conversão por Clique",
    "Custo Médio por Atribuição",
    "Custo por Atribuição",
    "Receita Incremental",
    "ROAS",
    "Custo por Atribuição (CPA atribuída)",
    "Taxa de Conversão por Atribuição (CVR atribuída)",
    "Taxa de Conversão por Atribuição de Canal",
    "Custo por Conversão por Atribuição",
    "Custo por Atribuição de Canal",
    "Taxa de Conversão por Atribuição por Canal",
    "Retorno sobre Investimento (ROI) por Canal",
    "Taxa de Conversão por Impressões (CVR/Impressões)",
    "Custo por Aquisição (CPA) por Canal",
    "Taxa de Conversão por Clique (CVR/Clicks)",
    "Receita por Aquisição (RPA)",
    "Custo Médio por Atribuição (CMA)",
    "Taxa de conversão por canal",
    "Custo por atribuição por canal",
    "Retorno sobre investimento",
    "Participação de Atribuição (Attribution Share)",
    "Receita incremental",
    "Retorno sobre Atribuição (ROAS)",
    "Custo por Aquisição (CPA)",
    "Custo por conversão",
    "Taxa de conversão geral",
    "Taxa de conversão por clique",
    "Custo por aquisição (CPA)",
    "Custo por aquisição",
    "Taxa de conversão por visualizações",
    "Taxa de retenção 30 dias",
    "Engajamento composto (CE)",
    "Taxa de conversão de checkout",
    "Tempo médio de checkout",
    "Abandono de checkout",
    "Valor médio por usuário pagante (ARPPU)",
    "Tempo de primeira resposta",
    "Tempo total de resolução",
    "Taxa de satisfação do cliente (CSAT)",
    "Taxa de resolução no primeiro contato (FCR)",
    "Tempo médio de atendimento (AHT)",
    "Taxa de satisfação do cliente",
    "Deduplicação de envios",
    "Taxa de Entregabilidade",
    "Taxa de Reclamação de Spam",
    "Taxa de Bounce (E-mail)",
    "Open Rate (Taxa de Abertura)",
    "Spam Complaint Rate",
    "Bounce Rate (E-mail)",
    "Email Deliverability Rate",
    "Taxa de abertura (Open Rate)",
    "Taxa de reclamação (Spam Complaint Rate)",
    "Taxa de bounce (Bounce Rate)",
    "Taxa de entregabilidade (Deliverability Rate)",
    "Ativação D7",
    "Conclusão de Onboarding",
    "Tempo até Primeira Ação",
    "Retenção D7",
    "Churn de Clientes",
    "Retenção 30d",
    "NPS (Net Promoter Score)",
    "Churn de MRR",
    "Conversão Trial → Pago",
    "Payback de CAC",
    "DAU/MAU",
    "Engajamento Composto",
    "Tempo Médio de Atendimento (AHT)",
    "Tempo de Primeira Resposta (FRT)",
    "First Contact Resolution (FCR)",
    "Satisfação do Cliente (CSAT)",
    "Tempo de Espera no Chat",
    "SLA de Atendimento",
    "Atribuição de Vendas Offline",
    "Tempo de Resolução de Fraudes",
    "Taxa de Falsos Positivos (FPR)",
    "Taxa de Verdadeiros Positivos (TPR)",
    "Valor Recuperado por Fraude",
    "Taxa de Reativação 30d",
    "Taxa de Descadastramento (Unsubscribe rate)",
    "Taxa de Abertura (Open Rate)",
    "On-time In-Full (OTIF)",
    "Custo por Pedido",
    "Lead Time de Pedido",
    "Fill Rate",
    "Acurácia de Previsão",
    "MAPE (Erro Percentual Absoluto Médio)",
    "Taxa de No-show",
    "Taxa de Ocupação",
    "RSVP Confirmado",
    "Tempo para Confirmar RSVP",
    "Tempo de contratação (Time to hire)",
    "Aceitação de proposta (Offer acceptance rate)",
    "Custo por contratação",
    "Retenção 90d",
    "Opt-in Rate",
    "Taxa de Qualificação",
    "Taxa de Conversão de Landing Page",
    "Taxa de Conversão de Trial para Pago",
    "Taxa de Conversão de Onboarding",
    "Taxa de Conversão do Funil",
    "Taxa de Conversão geral (CVR)",
    "Taxa de Conversão por Peça (CVR criativo)",
    "Taxa de Conversão por Sessões",
    "Taxa de Abertura de E-mail",
    "Taxa de Cliques (Click-through Rate)",
    "Receita por Usuário (ARPU)",
    "Abandono de Carrinho",
    "Taxa de Reativação",
    "Tempo de Onboarding",
    "Tempo até primeira ação",
    "Retenção D30",
    "Índice NPS",
    "Churn de Receita (MRR)",
    "Conversão Trial-Pago",
    "Payback de CAC (meses)",
    "Relação DAU/MAU",
    "Índice de Engajamento Composto",
    "SLA",
    "Chat Wait Time",
    "Offline Sales Attribution",
    "Disponibilidade",
    "Mean Time To Recovery (MTTR)",
    "Change Failure Rate",
    "Deployment Frequency",
    "Fraud Resolution Time",
    "False Positive Rate (FPR)",
    "True Positive Rate (TPR)",
    "Fraud Recovered Value",
    "Reativação 30d",
    "Taxa de descadastro (Unsubscribe)",
    "Abertura (Open Rate)",
    "On-time In Full (OTIF)",
    "Cost per Order",
    "Order Lead Time",
    "Forecast Accuracy",
    "Mean Absolute Percentage Error (MAPE)",
    "No-show Rate",
    "Occupation Rate",
    "RSVP Confirmed Rate",
    "Time to Confirm RSVP",
    "Employee Turnover",
    "Time to Hire",
    "Offer Acceptance Rate",
    "Cost per Hire",
    "Monthly Recurring Revenue (MRR)",
    "Average Deal Size",
    "Sales Cycle Length",
    "Checkout Conversion Rate",
    "Average Revenue Per Paying User (ARPPU)",
    "Checkout Abandonment Rate",
    "Average Checkout Time",
    "Customer Churn 90d",
    "Customer Retention 90d",
    "Opt-in Rate (Consentimento)",
    "Landing Page Conversion Rate",
    "Qualification Rate",
    "Taxa de retenção mensal (churn reverso)",
    "Tempo médio para primeira compra",
    "Taxa de reengajamento de push",
    "Frequência média de compra",
    "Taxa de clique em notificações",
    "ROAS por canal",
    "CPA por canal",
    "Receita incremental por canal",
    "Taxa de conversão por dispositivo",
    "Taxa de conversão por região",
    "Taxa de conversão por campanha",
    "Taxa de conversão de e-mail",
    "Taxa de conversão de push",
    "Taxa de conversão de SMS",
    "Taxa de conversão atribuída",
    "Taxa de retenção por campanha",
    "Taxa de retenção por canal",
    "Taxa de ativação",
    "Taxa de churn",
    "Taxa de upgrade",
    "Taxa de downgrade",
    "Tempo médio de resolução",
    "Tempo médio de primeira resposta",
    "Satisfação pós-atendimento",
    "Taxa de autoatendimento resolvido",
    "Tempo de espera (fila) do chat",
    "Cumprimento de SLA",
    "Conversão atribuída",
    "Taxa de conversão de prova gratuita",
    "Tempo médio até conversão",
    "Ciclo médio de vendas (dias)",
    "Taxa de follow-up efetivo",
    "Taxa de qualificação de leads",
    "Taxa de resposta de e-mails de vendas",
    "Taxa de conexão em cold calls",
    "Conversão por etapa do funil",
    "Taxa de abandono de carrinho",
    "Valor médio do pedido (AOV)",
    "MRR (Monthly Recurring Revenue)",
    "Taxa de conversão de checkout (CR)",
    "Tempo médio de checkout (s)",
    "Abandono de checkout (CR_abandono)",
    "ARPPU (Average Revenue Per Paying User)",
    "Taxa de churn em 90 dias",
    "Retenção em 90 dias",
    "Taxa de opt-in (consentimento)",
    "Conversão de landing page",
    "Taxa de qualificação (SQL rate)",
    "Taxa de Conversões",
    "Taxa de conversão de onboarding (CR Onboarding)",
    "Taxa de conversão do funil (CR Funil)",
    "Taxa de conversão geral (CR Geral)",
    "Conversões",
    "Impressionamentos",
    "Número de Atribuições",
    "Custo Total",
    "novos_clientes",
    "receita_atribuida",
    "pedidos",
    "Receita atribuída",
    "Número de pedidos",
    "Taxa de retorno sobre atribuição",
    "Taxa de conversão",
    "Custo por atribuição",
    "Retorno sobre investimento total",
    "Participação de atribuição",
    "Receita incremental total",
    "Taxa de conversão por atribuição",
    "Taxa de conversão por cliques",
    "Taxa de conversão por visualizações (CVR/Impressões)",
    "Taxa de conversão de sessão",
    "Taxa de conversão de peça",
    "Tempo médio de atendimento",
    "Tempo médio de primeira resposta",
    "Taxa de resolução no primeiro contato",
    "Taxa de satisfação",
    "Taxa de abertura",
    "Taxa de descadastramento",
    "Entregabilidade",
    "Reclamações de spam",
    "Taxa de bounce",
    "Tempo de resolução de fraudes",
    "Taxa de falsos positivos",
    "Taxa de verdadeiros positivos",
    "Valor recuperado",
    "Lead time",
    "Custo por pedido (CPO)",
    "Precisão da previsão",
    "MAPE (%)",
    "No-show (%)",
    "Ocupação (%)",
    "RSVP confirmado (%)",
    "Tempo para confirmar (min)",
    "Índice de turnover",
    "Tempo para contratar (dias)",
    "Aceitação de oferta (%)",
    "Custo por contratação (R$)",
    "Tamanho médio do negócio",
    "Duração do ciclo de vendas (dias)",
    "Conversão de checkout (%)",
    "Tempo de checkout (s)",
    "Abandono de checkout (%)",
    "ARPPU (R$)",
}

def expand_universe(universe_raw: Set[str]) -> Set[str]:
    """Gera universo normalizado + extrai siglas entre parênteses + aliases úteis."""
    out: Set[str] = set()
    for s in universe_raw:
        ns = norm(s)
        if ns:
            out.add(ns)
        # siglas entre parênteses (ex.: "... (CPC)" -> "cpc")
        for abbr in re.findall(r"\(([A-Za-z0-9\-/]+)\)", s or ""):
            na = norm(abbr)
            if na:
                out.add(na)
    # aliases comuns
    out.update({
        "cpc","cpa","cpl","ctr","cpm","roas","roi","mrr","arpu","arppu","mape","fcr","csat","nps",
        # DAU/MAU pode aparecer de formas diferentes
        "dau mau","daumau"
    })
    return out

UNIV_NORM: Set[str] = expand_universe(UNIVERSO)

# --- ALIASES (regex -> canônico) ---
ALIASES = [
    (r"^disponibilidade$", "uptime"),
    (r"^tempo medio (?:para )?recupera.*$", "mttr"),
    (r"^mttr.*$", "mttr"),

    (r".*\broas\b.*|^taxa de retorno sobre atribuicao roas$", "roas"),
    (r".*\broi\b.*|^retorno sobre investimento$", "roi"),

    (r"^custo por mil impress.*$", "cpm"),
    (r"^custo por aquisicao cpa$", "cpa"),
    (r"^custo por lead.*$", "cpl"),
    (r"^custo por cliques?$|^custo por clique$", "cpc"),

    (r"^mql.*rate.*$|^taxa de conversao mql$", "mql rate"),
    (r"^win rate$", "win rate"),

    (r"^taxa de satisfacao .*csat.*$|^csat$", "csat"),
    (r"^nps.*$", "nps"),

    (r"^taxa de rsvp confirmad.*$", "rsvp confirmado"),

    (r"^entregabilidade de emails?$|^entregabilidade de e mail$", "entregabilidade de e mail"),
    (r"^open rate.*$|^taxa de abertura.*$", "taxa de abertura"),
    (r"^spam complaint.*$|^taxa de reclam.*spam.*$", "spam complaint rate"),
    # Se quiser ser conservador com 'taxa de bounce', remova esta linha:
    (r"^taxa de bounce(?!.*site).*$", "bounce rate de e mail"),

    (r"^taxa de retencao 30.*$|^retencao 30 dias$", "retencao 30d"),
    (r"^taxa de retencao 90.*$|^retencao 90 dias$", "retencao 90d"),
    (r"^churn 90.*$|^taxa de churn em 90 dias$", "churn 90d"),

    (r"^ativac[aã]o d ?7$", "ativacao d7"),
    (r"^taxa de conversao do onboarding$", "taxa de conversao de onboarding"),

    (r"^custo de atribuic[aã]o por canal$", "custo por atribuicao por canal"),
    (r"^taxa de atribuic[aã]o de vendas? offline$", "atribuicao de vendas offline"),
]

ALIAS_REGEX = [(re.compile(pat), norm(dst)) for pat, dst in ALIASES]

def canonicalize(n: str, sid_n: str = "") -> str:
    """Converte variantes/sinônimos em rótulos canônicos já presentes no universo."""
    if not n:
        return n
    for rx, dst in ALIAS_REGEX:
        if rx.match(n):
            return dst
    # Casos de contains simples
    if "retorno sobre atribuicao" in n and "roas" in n:
        return "roas"
    return n


# =========================
# Gabarito
# =========================
GAB_PATH = Path("gabarito_cenarios.json")
if GAB_PATH.exists():
    GABARITO_RAW = json.loads(GAB_PATH.read_text(encoding="utf-8"))
else:
    # fallback mínimo só para evitar crash se o arquivo não existir
    GABARITO_RAW = {
        "attr_streaming_mix": [{"kpi":"ROAS"},{"kpi":"CPA"},{"kpi":"Attribution Share"},{"kpi":"Incremental Revenue"}]
    }

def _get_ci(d: dict, key: str, default=""):
    """Case-insensitive dict getter."""
    k_low = key.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == k_low:
            return v
    return default

def normalize_gabarito(gab_raw: dict) -> Dict[str, Set[str]]:
    gab_norm: Dict[str, Set[str]] = {}
    for k, v in gab_raw.items():
        if not isinstance(v, list):
            continue
        items: Set[str] = set()
        for it in v:
            if isinstance(it, dict):
                val = (
                    _get_ci(it, "kpi") or _get_ci(it, "nome") or
                    _get_ci(it, "metric") or _get_ci(it, "name") or ""
                )
            else:
                val = it
            n = norm(val)
            if n:
                items.add(n)
        # restringe ao universo normalizado
        items &= UNIV_NORM
        if items:
            gab_norm[norm(k)] = items
    return gab_norm

GAB_BY_NORM = normalize_gabarito(GABARITO_RAW)

# =========================
# Extração de Scenario e KPIs (apenas o necessário)
# =========================
def get_scenario_id(rec: dict) -> Tuple[str, str]:
    cand = rec.get("scenario_id") or rec.get("scenario") or rec.get("id") or rec.get("slug") or rec.get("name")
    if isinstance(cand, dict):
        sid_raw = cand.get("id") or cand.get("slug") or cand.get("name") or ""
    else:
        sid_raw = cand if isinstance(cand, str) else ""
    return sid_raw, norm(sid_raw)

def extract_kpis(rec: dict) -> List[str]:
    """
    Suporta:
      - parsed_items: [{kpi: "..."}]
      - output: "[{...},{...}]" (string com JSON array)
    """
    # 1) Preferência: parsed_items
    arr = rec.get("parsed_items")
    if isinstance(arr, list):
        out = []
        for item in arr:
            if isinstance(item, dict):
                val = (_get_ci(item, "kpi") or _get_ci(item, "nome") or
                       _get_ci(item, "metric") or _get_ci(item, "name"))
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
        if out:
            return out

    # 2) Fallback: JSON array dentro do "output" (string)
    raw = rec.get("output")
    if isinstance(raw, str) and raw.strip():
        m = re.search(r"\[[\s\S]*?\]", raw)  # pega primeiro array JSON
        if m:
            try:
                arr2 = json.loads(m.group(0))
                if isinstance(arr2, list):
                    out = []
                    for item in arr2:
                        if isinstance(item, dict):
                            val = (_get_ci(item, "kpi") or _get_ci(item, "nome") or
                                   _get_ci(item, "metric") or _get_ci(item, "name"))
                            if isinstance(val, str) and val.strip():
                                out.append(val.strip())
                        elif isinstance(item, str) and item.strip():
                            out.append(item.strip())
                    if out:
                        return out
            except Exception:
                pass

    return []  # nada extraído

# =========================
# Métricas
# =========================
def compare_sets(pred: Set[str], gold: Set[str], universe_norm: Set[str]) -> dict:
    p = set(pred) & universe_norm
    g = set(gold) & universe_norm

    TP = len(p & g)
    FP = len(p - g)
    FN = len(g - p)
    TN = max(len(universe_norm) - (TP + FP + FN), 0)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    denom     = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    mcc       = ((TP*TN - FP*FN) / (denom ** 0.5)) if denom > 0 else 0.0
    npv       = TN / (TN + FN) if (TN + FN) else 0.0
    fdr       = FP / (TP + FP) if (TP + FP) else 0.0

    return {
        "TP":TP, "FP":FP, "FN":FN, "TN":TN,
        "precision":precision, "recall":recall, "f1":f1,
        "denom": float(denom), "mcc":mcc, "npv":npv, "fdr":fdr
    }

# =========================
# I/O
# =========================
def load_results(path: str) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Formato não suportado em {path}: esperado list ou dict['results']")

def safe_model_name(model: str) -> str:
    if not isinstance(model, str) or not model:
        model = "unknown_model"
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")

# =========================
# Avaliação
# =========================
def evaluate_file_grouped_by_model(file_path: str) -> dict:
    rows = load_results(file_path)
    models = {}
    for r in rows:
        model_raw = r.get("model") or "unknown_model"
        M = models.setdefault(model_raw, {
            "model_name": model_raw,
            "preds_by_sid_norm": defaultdict(set),
            "sid_norm_to_raw": {},
            "seen_any_sid": set(),
        })

        sid_raw, sid_n = get_scenario_id(r)
        if not sid_n:
            continue

        M["seen_any_sid"].add(sid_n)
        M["sid_norm_to_raw"].setdefault(sid_n, sid_raw)

        names = extract_kpis(r)
        n_all = [norm(n) for n in names if norm(n)]
        n_all = [canonicalize(x, sid_n) for x in n_all]
        kept  = {n for n in n_all if n in UNIV_NORM}
        if DEBUG and n_all and not kept:
            print(f"[DEBUG] Predicoes fora do universo em '{sid_raw}' / {model_raw}: {n_all[:6]}")
        if kept:
            M["preds_by_sid_norm"][sid_n].update(kept)

    reports = {}
    for mkey, M in models.items():
        by_scenario = {}
        skipped = []
        sum_counts = {"TP":0, "FP":0, "FN":0, "TN":0}
        sum_avgs   = {"precision":0.0, "recall":0.0, "f1":0.0, "denom":0.0, "mcc":0.0, "npv":0.0, "fdr":0.0}
        ncen = 0

        for sid_n in sorted(M["seen_any_sid"]):
            sid_raw = M["sid_norm_to_raw"].get(sid_n, sid_n)
            gold = {canonicalize(g, sid_n) for g in (GAB_BY_NORM.get(sid_n) or set())}
            if not gold:
                skipped.append(sid_raw)
                continue

            pred = set(M["preds_by_sid_norm"].get(sid_n, set()))
            if DEBUG:
                print(f"[DBG] {sid_raw} :: pred={sorted(list(pred))} | gold={sorted(list(gold))}")

            res = compare_sets(pred, gold, UNIV_NORM)
            ncen += 1
            by_scenario[sid_raw] = {**res, "pred":sorted(pred), "gold":sorted(gold), "sid_norm":sid_n}
            # somatórios
            for k in ("TP","FP","FN","TN"):
                sum_counts[k] += int(res[k])
            for k in ("precision","recall","f1","denom","mcc","npv","fdr"):
                sum_avgs[k] += float(res[k])

        macro = {k: (sum_avgs[k]/ncen) if ncen else 0.0 for k in sum_avgs}
        reports[mkey] = {
            "file": file_path,
            "model_name": M["model_name"],
            "by_scenario": by_scenario,
            "macro_avg": macro,
            "totals": sum_counts,
            "evaluated_scenarios": len(by_scenario),
            "skipped_scenarios": skipped,
        }
    return reports

def main():
    # Sanidade: gabarito deve caber no universo
    gold_all = set().union(*GAB_BY_NORM.values()) if GAB_BY_NORM else set()
    if gold_all - UNIV_NORM:
        print(f"[AVISO] {len(gold_all - UNIV_NORM)} KPIs do gabarito fora do universo. Ex.: {list(gold_all - UNIV_NORM)[:10]}")

    for fp in MODEL_FILES:
        p = Path(fp)
        if not p.exists():
            print(f"[AVISO] Arquivo não encontrado: {fp}")
            continue

        print(f"\n==== Arquivo: {fp} ====")
        reports = evaluate_file_grouped_by_model(fp)

        for _, rep in reports.items():
            print(f"\n== Modelo: {rep['model_name']} ==")
            for sid, r in sorted(rep["by_scenario"].items()):
                print(f"- {sid}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} MCC={r['mcc']:.3f} "
                      f"NPV={r['npv']:.3f} FDR={r['fdr']:.3f} "
                      f"(TP={r['TP']}, FP={r['FP']}, FN={r['FN']}, TN={r['TN']}, Den={r['denom']:.0f})")
            ma = rep["macro_avg"]; tot = rep["totals"]
            print(f"\nResumo (media por cenario): Precision={ma['precision']:.3f} | Recall={ma['recall']:.3f} | "
                  f"F1={ma['f1']:.3f} | MCC={ma['mcc']:.3f} | NPV={ma['npv']:.3f} | FDR={ma['fdr']:.3f}")
            print(f"Totais: TP={tot['TP']} FP={tot['FP']} FN={tot['FN']} TN={tot['TN']}")

            out_path = Path(f"eval_{Path(fp).stem}__{safe_model_name(rep['model_name'])}.json")
            out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] Salvo: {out_path.resolve()}")

if __name__ == "__main__":
    main()
