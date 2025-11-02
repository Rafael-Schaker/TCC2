# validate_results.py
import json
import re
import unicodedata
from collections import defaultdict
from math import sqrt

# === Cole aqui o dict Python do gabarito (o bloco "gabarito_cenarios" acima) ===
GABARITO_CENARIOS = {...}  # <- SUBSTITUA pelo dict do gabarito (mesma estrutura)

def norm(txt: str) -> str:
    if not isinstance(txt, str): return ""
    t = unicodedata.normalize("NFKD", txt).encode("ascii","ignore").decode("ascii")
    t = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    return t

def extract_kpis_from_record(rec: dict) -> list[str]:
    """
    Tenta extrair nomes de KPIs de diferentes formatos:
    - rec["output_json"] já parseado (lista com objetos {kpi,...})
    - rec["json"] string com um array JSON
    - rec["output"] / rec["response"] texto com bullets "1. Nome"
    Retorna lista de strings (nomes de KPI).
    """
    # 1) caminho direto: lista JSON já estruturada
    for key in ("output_json","prediction_json","result_json"):
        if isinstance(rec.get(key), list):
            return [str(x.get("kpi","")).strip() for x in rec[key] if isinstance(x, dict)]
    # 2) string JSON
    for key in ("json","output","response","text"):
        raw = rec.get(key)
        if isinstance(raw, str):
            # tenta achar primeiro array JSON
            m = re.search(r"\[[\s\S]*\]", raw)
            if m:
                try:
                    arr = json.loads(m.group(0))
                    if isinstance(arr, list):
                        return [str(x.get("kpi","")).strip() for x in arr if isinstance(x, dict)]
                except Exception:
                    pass
            # fallback: bullets "1. KPI" / linhas com "• Fórmula:"
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            names = []
            for ln in lines:
                m1 = re.match(r"^\d+\.\s*(.+)$", ln)
                if m1:
                    names.append(m1.group(1).strip())
                elif "• Fórmula" in ln and names and names[-1].endswith(":"):
                    # não necessário aqui; manter simples
                    pass
            if names: return names
    return []

def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # suporta lista simples ou dict com 'results'
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Formato não suportado em {path}")

def compare_sets(pred: set[str], gold: set[str], universo: set[str]) -> dict:
    TP = len(pred & gold)
    FP = len(pred - gold)
    FN = len(gold - pred)
    # Para TN, usamos um universo fixo de KPIs (catálogo)
    TN = max(len(universo) - (TP + FP + FN), 0)
    precision = TP / (TP + FP) if (TP+FP) else 0.0
    recall    = TP / (TP + FN) if (TP+FN) else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    denom = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    mcc = ((TP*TN - FP*FN) / sqrt(denom)) if denom>0 else 0.0
    npv = TN / (TN + FN) if (TN+FN) else 0.0
    fdr = FP / (TP + FP) if (TP+FP) else 0.0
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,
            "precision":precision,"recall":recall,"f1":f1,
            "mcc":mcc,"npv":npv,"fdr":fdr}

def main():
    # Universo: todos KPIs do catálogo
    # Se preferir, derive do seu catálogo carregando o arquivo.
    UNIVERSO = {
        # nomes normalizados dos KPIs do catálogo (mesmos do bloco do catálogo)
        "taxa de conversao","cac","cpl","roas","aov","arpu","mrr","ltv cac",
        "churn de clientes","retencao 30d","ctr","opt in rate","reativacao 30d",
        "bounce rate site","payback cac","revenue growth","cash conversion cycle",
        "inventory turnover","fill rate","lead time de pedido","otif","deployment frequency",
        "lead time for changes","change failure rate","mttr","disponibilidade",
        "error budget burn","build success rate","defect escape rate","ticket deflection",
        "backlog aberto","average deal size","win rate","quota attainment",
        "upsell cross sell rate","revenue per message","engajamento composto",
        "taxa de abertura","mql rate"
    }

    files = ["results_wcap.json","result.json"]
    agregado = defaultdict(list)

    for fpath in files:
        try:
            rows = load_results(fpath)
        except Exception as e:
            print(f"[AVISO] Não consegui ler {fpath}: {e}")
            continue

        por_cenario = defaultdict(list)
        for r in rows:
            sid = r.get("scenario_id") or r.get("scenario") or r.get("id")
            if not sid: continue
            pred_names = [norm(n) for n in extract_kpis_from_record(r) if n]
            por_cenario[sid].extend([p for p in pred_names if p])

        # consolida por cenário (set)
        metrics_sum = {"TP":0,"FP":0,"FN":0,"TN":0,"precision":0,"recall":0,"f1":0,"mcc":0,"npv":0,"fdr":0}
        ncen = 0

        print(f"\n==== Arquivo: {fpath} ====")
        for sid, preds in por_cenario.items():
            gold = {norm(x["kpi"]) for x in GABARITO_CENARIOS.get(sid, [])}
            if not gold: 
                # cenário sem gabarito — pula para não enviesar
                continue
            pred = set(preds)
            res = compare_sets(pred, gold, UNIVERSO)
            ncen += 1
            for k in metrics_sum: metrics_sum[k] += res[k] if k in res else 0
            print(f"- {sid}: P={res['precision']:.2f} R={res['recall']:.2f} F1={res['f1']:.2f} MCC={res['mcc']:.2f} (TP={res['TP']}, FP={res['FP']}, FN={res['FN']})")

        if ncen:
            media = {k:(metrics_sum[k]/ncen if k not in ("TP","FP","FN","TN") else metrics_sum[k]) for k in metrics_sum}
            print(f"\nResumo {fpath} (média por cenário):")
            print(f"Precision={media['precision']:.3f} | Recall={media['recall']:.3f} | F1={media['f1']:.3f} | MCC={media['mcc']:.3f} | NPV={media['npv']:.3f} | FDR={media['fdr']:.3f}")
            print(f"Totais: TP={media['TP']} FP={media['FP']} FN={media['FN']} TN={media['TN']}")
        else:
            print("Nenhum cenário válido com gabarito encontrado neste arquivo.")

if __name__ == "__main__":
    main()
