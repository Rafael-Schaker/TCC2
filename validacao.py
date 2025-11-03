import difflib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# =========================
# Config
# =========================
DEBUG = False  # False para silenciar logs

# =========================
# Helpers
# =========================
def norm(txt: Any) -> str:
    if not isinstance(txt, str):
        return ""
    t = unicodedata.normalize("NFKD", txt).encode("ascii","ignore").decode("ascii")
    t = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    return re.sub(r"\s+", " ", t)

STOPWORDS = {
    "taxa","indice","de","do","da","das","dos","por","para","no","na","nos","nas",
    "em","o","a","os","as","geral","rate","score","percentual","percent","mediana",
    "media","medio","promoter","net","cliente","clientes","usuario","usuarios",
    "confirmado","confirmada","confirmados","confirmadas","aberto","abertura","open",
    "click","clique","clicks","cliques","true","false","value","valores","tempo","medio",
    "médio","min","segundos","s","dias","dia","meses","mes","horas","hora","d7","d30","mrr","kpi"
}
def tokens_for(s: str) -> Set[str]:
    return {w for w in norm(s).split() if w and w not in STOPWORDS}

def load_json_lenient(path: str):
    p = Path(path)
    if not p.exists():
        if DEBUG: print(f"[WARN] Arquivo não encontrado: {path}")
        return None
    txt = p.read_text(encoding="utf-8", errors="ignore")
    try:
        return json.loads(txt)
    except Exception:
        txt2 = re.sub(r",\s*(\]|\})", r"\1", txt)  # vírgulas sobrando
        try:
            return json.loads(txt2)
        except Exception as e:
            print(f"[ERR] Could not parse JSON at {path}: {e}")
            return None

# =========================
# Universo (lido do JSON)
# =========================
univ_raw = load_json_lenient("universo.json")
if isinstance(univ_raw, list):
    UNIVERSE_RAW: Set[str] = set(map(str, univ_raw))
else:
    UNIVERSE_RAW = set()

def expand_universe(universe_raw: Set[str]) -> Tuple[Set[str], Dict[str,str], Dict[str,Set[str]]]:
    out_norm: Set[str] = set()
    norm_to_canon: Dict[str, str] = {}
    token_index: Dict[str, Set[str]] = {}
    for s in universe_raw:
        ns = norm(s)
        if not ns:
            continue
        out_norm.add(ns)
        norm_to_canon.setdefault(ns, s)
        token_index[ns] = tokens_for(ns)
        # Siglas entre parênteses, ex.: "... (CPC)"
        for abbr in re.findall(r"\(([A-Za-z0-9\-/]+)\)", s):
            na = norm(abbr)
            if na:
                out_norm.add(na)
                norm_to_canon.setdefault(na, abbr)
                token_index[na] = tokens_for(na)
    # Abreviações comuns
    for a in ["cpc","cpa","cpl","ctr","cpm","roas","roi","mrr","arpu","arppu","mape","fcr","csat","nps","dau mau","daumau"]:
        out_norm.add(a); norm_to_canon.setdefault(a, a.upper()); token_index[a] = tokens_for(a)
    return out_norm, norm_to_canon, token_index

UNIV_NORM, NORM2CANON, UNIV_TOKENS = expand_universe(UNIVERSE_RAW)

# --- ALIASES (regex -> canônico normalizado) ---
ALIASES = [
    (r"^disponibilidade$", "uptime"),
    (r"^taxa de uptime$", "uptime"),
    (r"^uptime$", "uptime"),

    (r"^tempo medio (?:para )?recupera.*$", "mttr"),
    (r"^tempo medio (?:para )?resolucao.*$", "mttr"),
    (r"^tempo medio (?:para )?restaurar.*$", "mttr"),
    (r"^mttr.*$", "mttr"),

    (r".*\broas\b.*|^taxa de retorno sobre atribuicao(?: roas)?$", "roas"),
    (r".*\broi\b.*|^retorno sobre investimento$", "roi"),

    (r"^custo por mil impress.*$", "cpm"),
    (r"^custo por aquisicao(?: cpa)?$", "cpa"),
    (r"^custo por conversao$", "cpa"),
    (r"^custo por lead.*$", "cpl"),
    (r"^custo por cliques?$|^custo por clique$", "cpc"),

    (r"^mql.*rate.*$|^taxa de conversao mql$", "mql rate"),
    (r"^win rate$", "win rate"),

    (r"^nps.*$|^net promoter score$", "nps"),
    (r"^taxa de satisfacao .*csat.*$|^csat$|^taxa de satisfacao do cliente(?: \(csat\))?$", "csat"),
    (r"^taxa de resolucao no primeiro contato.*$|^first contact resolution.*$|^fcr$", "fcr"),

    (r"^taxa de rsvp confirmad.*$|^rsvp confirmed.*$", "rsvp confirmado"),

    (r"^entregabilidade de emails?$|^entregabilidade de e mail.*$|^email deliverability.*$", "entregabilidade de e mail"),
    (r"^open rate.*$|^taxa de abertura.*$", "taxa de abertura"),
    (r"^spam complaint.*$|^taxa de reclama.*spam.*$", "spam complaint rate"),

    # Bounce
    (r"^taxa de bounce.*(email|e mail).*$", "bounce rate de e mail"),
    (r"^taxa de bounce.*$", "bounce rate"),

    # Retenção/Churn
    (r"^taxa de retencao 30.*$|^retencao 30 dias$", "retencao 30d"),
    (r"^taxa de retencao 90.*$|^retencao 90 dias$", "retencao 90d"),
    (r"^churn 90.*$|^taxa de churn em 90 dias$", "churn 90d"),
    (r"^churn de receita.*$|^churn de mrr$", "churn de mrr"),

    # Onboarding/Ativação
    (r"^ativac[aã]o d ?7$", "ativacao d7"),
    (r"^taxa de conversao do onboarding$", "taxa de conversao de onboarding"),

    # Atribuição/Omni
    (r"^custo de atribuic[aã]o por canal$", "custo por atribuicao por canal"),
    (r"^custo por atribuic[aã]o por canal$", "custo por atribuicao por canal"),
    (r"^taxa de atribuic[aã]o de vendas? offline$", "atribuicao de vendas offline"),
    (r"^taxa de conversao por impress.*$", "taxa de conversao por impressoes"),

    # DAU/MAU + Checkout
    (r"^dau[/ ]?mau$", "dau mau"),
    (r"^checkout conversion rate$", "conversao de checkout"),
    (r"^checkout abandonment.*$", "abandono de checkout"),
    (r"^average revenue per paying user.*$", "arppu"),

    # Conversão geral → conversão
    (r"^taxa de conversao geral$", "taxa de conversao"),
]
ALIAS_REGEX = [(re.compile(pat), norm(dst)) for pat, dst in ALIASES]

def _debug_map(src: str, dst: str):
    if DEBUG and src != dst:
        print(f"[MAP] '{src}' -> '{dst}'")

def canonicalize_one(n: str) -> str:
    if not n:
        return ""
    # 1) aliases diretos
    for rx, dst in ALIAS_REGEX:
        if rx.match(n):
            _debug_map(n, dst)
            return dst
    # 2) se já está no universo
    if n in UNIV_NORM:
        return n
    # 3) similaridade por tokens (Jaccard + bônus de subconjunto)
    t = tokens_for(n)
    if not t:
        return n
    best, best_score = None, 0.0
    for u in UNIV_NORM:
        ut = UNIV_TOKENS.get(u, set())
        if not ut:
            continue
        inter = len(t & ut); union = len(t | ut)
        jacc = inter / union if union else 0.0
        subset_boost = 0.1 if (t <= ut or ut <= t) else 0.0
        score = jacc + subset_boost
        if score > best_score:
            best_score, best = score, u
    if best and best_score >= 0.60:
        _debug_map(n, best)
        return best
    # 4) fallback por string similarity
    close = difflib.get_close_matches(n, list(UNIV_NORM), n=1, cutoff=0.85)
    if close:
        _debug_map(n, close[0])
        return close[0]
    return n

# =========================
# Gabarito
# =========================
gab_raw = load_json_lenient("gabarito_cenarios.json") or {}

def _get_ci(d: dict, key: str, default=""):
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
                val = (_get_ci(it,"kpi") or _get_ci(it,"nome") or _get_ci(it,"metric") or _get_ci(it,"name") or "")
            else:
                val = it
            can = canonicalize_one(norm(val))
            if can in UNIV_NORM:
                items.add(can)
        if items:
            gab_norm[norm(k)] = items
    return gab_norm

GAB_BY_NORM = normalize_gabarito(gab_raw)

# =========================
# Predições
# =========================
def load_results(path: str) -> List[dict]:
    data = load_json_lenient(path)
    if data is None:
        return []
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    return []

def get_scenario_id(rec: dict) -> Tuple[str,str]:
    cand = rec.get("scenario_id") or rec.get("scenario") or rec.get("id") or rec.get("slug") or rec.get("name")
    if isinstance(cand, dict):
        sid_raw = cand.get("id") or cand.get("slug") or cand.get("name") or ""
    else:
        sid_raw = cand if isinstance(cand, str) else ""
    return sid_raw, norm(sid_raw)

def extract_kpis(rec: dict) -> List[str]:
    arr = rec.get("parsed_items")
    if isinstance(arr, list):
        out = []
        for item in arr:
            if isinstance(item, dict):
                val = (_get_ci(item,"kpi") or _get_ci(item,"nome") or _get_ci(item,"metric") or _get_ci(item,"name"))
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
        if out:
            return out
    raw = rec.get("output")
    if isinstance(raw, str) and raw.strip():
        m = re.search(r"\[[\s\S]*?\]", raw)  # primeiro array JSON
        if m:
            try:
                arr2 = json.loads(m.group(0))
                if isinstance(arr2, list):
                    out = []
                    for item in arr2:
                        if isinstance(item, dict):
                            val = (_get_ci(item,"kpi") or _get_ci(item,"nome") or _get_ci(item,"metric") or _get_ci(item,"name"))
                            if isinstance(val, str) and val.strip():
                                out.append(val.strip())
                        elif isinstance(item, str) and item.strip():
                            out.append(item.strip())
                    if out:
                        return out
            except Exception:
                pass
    return []

# =========================
# Métricas
# =========================
def compare_sets(pred: Set[str], gold: Set[str], universe_norm: Set[str]) -> dict:
    p = set(pred) & universe_norm
    g = set(gold) & universe_norm
    TP = len(p & g); FP = len(p - g); FN = len(g - p)
    TN = max(len(universe_norm) - (TP + FP + FN), 0)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    denom     = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    mcc       = ((TP*TN - FP*FN) / (denom ** 0.5)) if denom > 0 else 0.0
    npv       = TN / (TN + FN) if (TN + FN) else 0.0
    fdr       = FP / (TP + FP) if (TP + FP) else 0.0

    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,
            "precision":precision,"recall":recall,"f1":f1,
            "denom": float(denom),"mcc":mcc,"npv":npv,"fdr":fdr}

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
        norms = [canonicalize_one(norm(n)) for n in names]
        kept  = {x for x in norms if x in UNIV_NORM}
        if DEBUG and norms and not kept:
            print(f"[DEBUG] Predicoes fora do universo em '{sid_raw}' / {model_raw}: {norms[:6]}")
        if kept:
            M["preds_by_sid_norm"][sid_n].update(kept)

    reports = {}
    for mkey, M in models.items():
        by_scenario = {}
        skipped = []
        sum_counts = {"TP":0,"FP":0,"FN":0,"TN":0}
        sum_avgs   = {"precision":0.0,"recall":0.0,"f1":0.0,"denom":0.0,"mcc":0.0,"npv":0.0,"fdr":0.0}
        ncen = 0

        for sid_n in sorted(M["seen_any_sid"]):
            sid_raw = M["sid_norm_to_raw"].get(sid_n, sid_n)
            gold = set(GAB_BY_NORM.get(sid_n) or set())
            if not gold:
                skipped.append(sid_raw); continue

            pred = set(M["preds_by_sid_norm"].get(sid_n, set()))
            if DEBUG:
                print(f"[DBG] {sid_raw} :: pred={sorted(list(pred))} | gold={sorted(list(gold))}")

            res = compare_sets(pred, gold, UNIV_NORM)
            ncen += 1
            by_scenario[sid_raw] = {**res, "pred":sorted(pred), "gold":sorted(gold), "sid_norm":sid_n}

            for k in ("TP","FP","FN","TN"): sum_counts[k] += int(res[k])
            for k in ("precision","recall","f1","denom","mcc","npv","fdr"): sum_avgs[k] += float(res[k])

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
    # Coleta apenas os arquivos existentes na pasta atual
    files = [name for name in ("results.json", "results_wcap.json") if Path(name).exists()]
    if not files:
        print("[WARN] Nenhum arquivo de resultados encontrado (results.json / results_wcap.json).")
        return

    for fp in files:
        print(f"\n==== Avaliando: {fp} ====")
        rep = evaluate_file_grouped_by_model(fp)
        for _, summary in rep.items():
            model_key = re.sub(r"[^a-zA-Z0-9]+", "_", summary["model_name"]).strip("_") or "unknown"
            outp = Path(f"eval_{Path(fp).stem}__{model_key}.json")
            outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] Salvo: {outp}")

if __name__ == "__main__":
    main()

