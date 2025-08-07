# .venv\Scripts\activate

# pip install -r requirements/requirements_gratuito.txt
# ou
# pip install langchain langchain-community langchain-chroma chromadb ython-dotenv pypdf sentence-transformers transformers accelerate torch safetensors

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

CAMINHO_DB = "db"
PROMPT_TEMPLATE = """
Responda a pergunta do usuário:
(pergunta: {pergunta})

com base nessas informações abaixo:
(informações: {base_conhecimento})
"""

def perguntar():
    pergunta = input("Digite sua pergunta: ")

    # Embeddings gratuitos
    funcao_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=CAMINHO_DB,
        embedding_function=funcao_embedding    
    )

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=4)

    if len(resultados) == 0 or resultados[0][1] < 0.5:
        print("Não sei.")
        return
    
    textos_resultados = [resultado[0].page_content for resultado in resultados]
    base_conhecimento = "\n\n----\n\n".join(textos_resultados)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).invoke({
        "pergunta": pergunta,
        "base_conhecimento": base_conhecimento
    }).to_string()

    # Modelo local gratuito (usa HuggingFace Hub)
    llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=500)
    resposta = llm(prompt)[0]["generated_text"]

    print(resposta)

perguntar()
