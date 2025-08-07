# .venv\Scripts\activate

# pip install -r requirements/requirements_openIA.txt
# ou
# pip install python-dotenv langchain langchain-openai langchain-community langchain-chroma chromadb openai pypdf

from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

CAMINHO_DB = "db"
PROMPT_TEMPLATE = """
Reponda a pergunta do usuário:
(pergunta: {pergunta})

com base nessas informações abaixo:
(informações: {base_conhecimento})

"""

def perguntar():
    pergunta = input("Digite sua pergunta: ")

    funcao_embedding = OpenAIEmbeddings()
    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=funcao_embedding    
    )

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=4)

    if len(resultados) == 0 or resultados[0][1] < 0.5:
        print("Não sei.")
        return
    
    textos_resultados = []

    for resultado in resultados:
        texto = resultado[0].page_content
        textos_resultados.append(texto)

    base_conhecimento = "\n\n----\n\n".join(textos_resultados)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt.invoke({"pergunta":pergunta , "base_conhecimento":base_conhecimento})

    modelo = ChatOpenAI(model="gpt-3.5-turbo")
    texto_resposta = modelo.invoke(prompt).content
    print(texto_resposta)

perguntar()