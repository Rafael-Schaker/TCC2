import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
chave_api = os.getenv("OPENAI_API_KEY")
mensagem = [
    SystemMessage("Responda em ingles."),
    HumanMessage("Ola chatGPT, qual a sua cor favorita?"),
]

modelo = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = modelo 

resposta = chain.invoke(mensagem)

print(resposta.content)