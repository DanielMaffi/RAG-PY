# .venv\Scripts\activate

# pip install -r requirements_gratuito.txt    
# ou 
# pip install langchain langchain-community chromadb langchain-chroma python-dotenv pypdf sentence-transformers langchain-huggingface

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

PASTA_BASE = "base"

def Criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    documentos = carregador.load()
    return documentos

def dividir_chunks(documentos):
    separador = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    return separador.split_documents(documentos)

def vetorizar_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="db"
    )
    print(f"Banco de dados criado com {len(chunks)} chunks.")

Criar_db()
