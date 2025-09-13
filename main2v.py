import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # diretório do script
GERENTES_CSV = os.path.join(BASE_DIR, "Gerentes.csv")
KPIS_CSV = os.path.join(BASE_DIR, "Kpi.csv")
METRICAS_CSV = os.path.join(BASE_DIR, "Metricas.csv")

# Ajuste a lista aos modelos que você tem (use `ollama list`)
MODELOS = [
    "gpt-oss:20b",
    #"qwen3:30b",
    #"gemma3:27b",
    #"deepseek-r1:32b",
    # Exemplos leves, se quiser comparar:
    "qwen3:4b",
    "gemma3:4b",
    "deepseek-r1:7b",
]

TEMPERATURE = 0.2
KEEP_ALIVE = "10m"

AREA_ALVO = "Vendas, Marketing e Suporte"  # “área de vendas” nos seus CSVs

# =========================
# Helpers
# =========================
def _parse_id_list(val) -> List[int]:
    if pd.isna(val):
        return []
    s = str(val)
    toks = re.split(r"[;,]\s*|\s+", s.strip())
    out = []
    for t in toks:
        if not t:
            continue
        try:
            out.append(int(t))
        except ValueError:
            # ignora tokens não numéricos
            pass
    return out

def _read_csv_robusto(path: str) -> pd.DataFrame:
    """
    Leitura tolerante a BOM e variação de separador.
    - Tenta auto-inferência de separador (engine='python', sep=None).
    - Cai para separador padrão (vírgula) se necessário.
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig", engine="python", sep=None)
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")

def _colmap_lower(df: pd.DataFrame) -> Dict[str, str]:
    # mapa {nome_normalizado: nome_original}
    return {str(c).strip().lower(): c for c in df.columns}

def carregar_metricas(path: str) -> Dict[int, str]:
    df = _read_csv_robusto(path)
    cols = _colmap_lower(df)
    id_col = cols.get("id")
    nome_col = cols.get("nome") or cols.get("name")
    if not id_col or not nome_col:
        raise ValueError(
            f"Metricas.csv deve conter colunas 'Id' e 'nome'. "
            f"Encontradas: {list(df.columns)}"
        )
    metricas: Dict[int, str] = {}
    for _, r in df.iterrows():
        try:
            mid = int(r[id_col])
            nome = str(r[nome_col]).strip()
            if nome:
                metricas[mid] = nome
        except Exception:
            continue
    return metricas

def carregar_kpis(path: str) -> Dict[int, Tuple[str, List[int]]]:
    df = _read_csv_robusto(path)
    cols = _colmap_lower(df)
    id_col = cols.get("id")
    nome_col = cols.get("nome") or cols.get("name")
    ids_m_col = cols.get("id_metricas_utilizadas")
    if not id_col or not nome_col or not ids_m_col:
        raise ValueError(
            f"Kpi.csv deve conter 'Id', 'Nome', 'id_metricas_utilizadas'. "
            f"Encontradas: {list(df.columns)}"
        )
    kpis: Dict[int, Tuple[str, List[int]]] = {}
    for _, r in df.iterrows():
        try:
            kid = int(r[id_col])
            nome = str(r[nome_col]).strip()
            mids = _parse_id_list(r[ids_m_col])
            if nome:
                kpis[kid] = (nome, mids)
        except Exception:
            continue
    return kpis

def construir_contexto_compacto(metricas: Dict[int, str], kpis: Dict[int, Tuple[str, List[int]]]) -> str:
    """Glossário + tabela KPI->métricas (nomes). Mantém curto para caber no contexto."""
    glossario = (
        "Definições rápidas:\n"
        "- Métrica: medição atômica (ex.: impressões, cliques, custo, pedidos, receita).\n"
        "- KPI: indicador calculado a partir de métricas que reflete objetivo de negócio "
        "(ex.: CTR, CPC, CPA, ROAS, Taxa de Conversão).\n"
        "Quando sugerir um KPI, inclua uma justificativa curta e a fórmula com as métricas necessárias.\n"
    )

    kpis_view = []
    for kid, (nome, mids) in kpis.items():
        kpis_view.append({
            "id": kid,
            "nome": nome,
            "metricas": [metricas.get(m, f"id:{m}") for m in mids]
        })

    base = {
        "metricas": [{"id": mid, "nome": mname} for mid, mname in metricas.items()],
        "kpis": kpis_view,
    }
    return glossario + "\nBase (compacta):\n" + json.dumps(base, ensure_ascii=False, indent=2)

def llm(model: str) -> OllamaLLM:
    return OllamaLLM(model=model, temperature=TEMPERATURE, keep_alive=KEEP_ALIVE)

def perguntar_recomendacao_kpis(contexto: str, model: str) -> str:
    """
    Pergunta única:
    - Sou gerente da área de vendas.
    - Recomende 1 a 3 KPIs.
    - Para cada KPI: {nome, importancia, formula, metricas}
    - Saída obrigatória em JSON (lista de objetos).
    """
    template = ChatPromptTemplate.from_messages([
        ("system",
         "Você é especialista em KPIs de Marketing. "
         "Use SOMENTE o CONTEXTO fornecido. Se algo não estiver no contexto, seja conservador.\n\n"
         "=== CONTEXTO ===\n{contexto}\n=== FIM CONTEXTO ===\n\n"
         "Responda APENAS em JSON válido (sem comentários, sem texto fora do JSON). "
         "Formato: [ {{\"kpi\": \"...\", \"importancia\": \"...\", \"formula\": \"...\", \"metricas\": [\"...\"]}} , ... ]"),
        ("human",
         "Sou gerente da área de vendas (\"Vendas, Marketing e Suporte\"). "
         "Recomende 1 a 3 KPIs adequados. Para cada KPI, inclua:\n"
         "- kpi: nome do KPI\n- importancia: por que é importante (1–2 frases)\n"
         "- formula: fórmula de cálculo (se conhecida no contexto)\n- metricas: nomes das métricas necessárias")
    ])
    chain = template | llm(model)
    return chain.invoke({"contexto": contexto})


def tentar_parse_json(texto: str) -> Optional[List[dict]]:
    """Tenta extrair JSON do retorno (se o modelo colocou algo extra)."""
    # Remove possíveis blocos de raciocínio
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL | re.IGNORECASE).strip()
    # 1) Se a resposta é puro JSON, tenta direto
    try:
        return json.loads(texto)
    except Exception:
        pass
    # 2) Caso contrário, tenta capturar o primeiro array JSON
    m = re.search(r"\[\s*{.*}\s*\]", texto, flags=re.DOTALL)
    if not m:
        return None
    bloco = m.group(0)
    try:
        return json.loads(bloco)
    except Exception:
        return None

def main():
    # 1) Carrega CSVs e monta contexto
    metricas = carregar_metricas(METRICAS_CSV)
    kpis = carregar_kpis(KPIS_CSV)
    contexto = construir_contexto_compacto(metricas, kpis)

    print(f"Pergunta: Recomendações de KPI para a área '{AREA_ALVO}'\n")

    # 2) Consulta vários modelos e compara a saída
    for model in MODELOS:
        print(f"=== Modelo: {model} ===")
        try:
            bruto = perguntar_recomendacao_kpis(contexto, model)
            parsed = tentar_parse_json(str(bruto))
            if parsed is None:
                print("(Aviso) Retorno não-JSON detectado; mostrando bruto:\n", str(bruto).strip(), "\n")
                continue

            # Mostra resumido por KPI (mantendo a estrutura da pergunta)
            for item in parsed[:3]:
                kpi = (item.get("kpi") or "").strip()
                importancia = (item.get("importancia") or "").strip()
                formula = (item.get("formula") or "").strip()
                metricas_list = item.get("metricas") or []
                metricas_fmt = ", ".join([str(x) for x in metricas_list])
                print(f"- KPI: {kpi}")
                print(f"  Importância: {importancia}")
                print(f"  Fórmula: {formula}")
                print(f"  Métricas: {metricas_fmt}")
            print()
        except Exception as e:
            print(f"ERRO: {type(e).__name__}: {e}\n")

if __name__ == "__main__":
    main()
