import re
import time
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Modelos instalados no seu Ollama
MODELOS: List[str] = [
    "gpt-oss:20b",
    "qwen3:30b",
    "gemma3:27b",
    "deepseek-r1:32b",
]

TEMPERATURE = 0.2

TEMPLATE = (
    "Instrução: responda com apenas UMA cor em português, "
    "em UMA palavra (sem frases, sem explicações).\n\n"
    "Pergunta: {pergunta}\n\n"
    "Resposta:"
)
prompt = ChatPromptTemplate.from_template(TEMPLATE)

def strip_reasoning(texto: str) -> str:
    """Remove blocos <think>...</think> se existirem e espaços extras."""
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL | re.IGNORECASE).strip()

def medir_stream(chain, entrada: dict) -> Tuple[Optional[str], float, float]:
    """
    Executa o chain em streaming e mede:
    - TTFT (tempo até o 1º token)
    - Tempo total (até terminar)
    Retorna (texto_final, ttft_s, total_s).
    Em caso de erro, retorna (None, 0.0, 0.0) e a exceção será tratada fora.
    """
    inicio = time.perf_counter()
    ttft: Optional[float] = None
    acumulado = ""

    for chunk in chain.stream(entrada):
        if ttft is None:
            ttft = time.perf_counter() - inicio
        acumulado += str(chunk)

    total = time.perf_counter() - inicio
    return acumulado, (ttft if ttft is not None else total), total

def testar_modelo(nome_modelo: str) -> Tuple[str, Optional[str], Optional[float], Optional[float], Optional[str]]:
    """
    Testa um modelo e retorna:
      (modelo, resposta_1_palavra, ttft_s, total_s, erro)
    Se erro != None, os demais campos (resposta/tempos) podem vir None.
    """
    try:
        llm = OllamaLLM(model=nome_modelo, temperature=TEMPERATURE)
        chain = prompt | llm
        entrada = {"pergunta": "Escolha uma cor."}

        texto_stream, ttft, total = medir_stream(chain, entrada)
        texto_final = strip_reasoning(texto_stream or "")
        resposta = texto_final.split()[0] if texto_final else ""
        return nome_modelo, resposta, ttft, total, None

    except Exception as e:
        return nome_modelo, None, None, None, f"{type(e).__name__}: {e}"

def formatar_segundos(s: Optional[float]) -> str:
    return f"{s:.3f}s" if isinstance(s, float) else "-"

def main():
    print("=== Benchmark LangChain + Ollama ===")
    print("Pergunta: 'Escolha uma cor.' (responder 1 palavra em PT-BR)\n")

    resultados = []
    for m in MODELOS:
        print(f"-> Modelo: {m}")
        modelo, resposta, ttft, total, erro = testar_modelo(m)
        if erro:
            print(f"   ERRO: {erro}\n")
        else:
            print(f"   Cor: {resposta}")
            print(f"   TTFT: {formatar_segundos(ttft)} | Total: {formatar_segundos(total)}\n")
        resultados.append((modelo, resposta, ttft, total, erro))

    # Resumo final
    print("=== Resumo ===")
    for modelo, resposta, ttft, total, erro in resultados:
        if erro:
            print(f"{modelo:16} | ERRO: {erro}")
        else:
            print(f"{modelo:16} | Cor: {resposta:10} | TTFT: {formatar_segundos(ttft)} | Total: {formatar_segundos(total)}")

if __name__ == "__main__":
    main()
