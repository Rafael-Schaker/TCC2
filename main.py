import re
import time
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# ================== CONFIG ==================
MODELOS: List[str] = [
    "gpt-oss:20b",
    #"qwen3:4b",
    #"gemma3:4b",
    #"deepseek-r1:7b",
    "qwen3:30b",
    "gemma3:27b",
    "deepseek-r1:32b",
]

TEMPERATURE = 0.2
KEEP_ALIVE = "10m"   # opcional: mantém o modelo carregado entre chamadas

PERGUNTA = "Escolha uma cor."
TEMPLATE = (
    "Responda com APENAS UMA cor em português (uma única palavra). "
    "Não explique, não use frases nem tags.\n"
    "Pergunta: {pergunta}\n"
    "Resposta:"
)
prompt = ChatPromptTemplate.from_template(TEMPLATE)
# ============================================

def remove_think(texto: str) -> str:
    """Remove blocos <think>...</think> (se existirem) apenas para facilitar a extração da resposta."""
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL | re.IGNORECASE)

def extrair_primeira_palavra(texto: str) -> str:
    """Extrai a primeira palavra com letras (inclui acentos). Caso não exista, retorna o texto aparado."""
    m = re.search(r"[A-Za-zÀ-ÿ]+", texto)
    return m.group(0) if m else texto.strip()

def medir_stream_full(chain, entrada: dict) -> Tuple[str, float, float]:
    """
    Faz streaming sem limitar/cortar:
      - TTFT = tempo até o primeiro token chegar
      - total = tempo até o término natural da geração
    Retorna (texto_completo, ttft_s, total_s).
    """
    inicio = time.perf_counter()
    ttft: Optional[float] = None
    buf = ""

    for chunk in chain.stream(entrada):
        now = time.perf_counter()
        if ttft is None:
            ttft = now - inicio
        buf += str(chunk)

    total = time.perf_counter() - inicio
    return buf, (ttft if ttft is not None else total), total

def consultar_modelo(modelo: str, pergunta: str) -> Tuple[str, Optional[str], Optional[float], Optional[float], Optional[str]]:
    try:
        llm = OllamaLLM(
            model=modelo,
            temperature=TEMPERATURE,
            keep_alive=KEEP_ALIVE,
        )
        chain = prompt | llm
        bruto, ttft, total = medir_stream_full(chain, {"pergunta": pergunta})

        texto = remove_think(str(bruto))
        resposta = extrair_primeira_palavra(texto)
        return modelo, resposta, ttft, total, None
    except Exception as e:
        return modelo, None, None, None, f"{type(e).__name__}: {e}"

def fmt_s(x: Optional[float]) -> str:
    return f"{x:.3f}s" if isinstance(x, float) else "-"

def main():
    print(f"Pergunta: {PERGUNTA}\n")
    header = f"{'Modelo':16} | {'Resposta':10} | {'TTFT':10} | {'Total':10}"
    print(header)
    print("-" * len(header))

    for m in MODELOS:
        modelo, resposta, ttft, total, erro = consultar_modelo(m, PERGUNTA)
        if erro:
            print(f"{modelo:16} | {'ERRO':10} | {'-':10} | {'-':10}")
        else:
            print(f"{modelo:16} | {resposta or '':10} | {fmt_s(ttft):10} | {fmt_s(total):10}")

if __name__ == "__main__":
    main()
