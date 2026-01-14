from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from app.core.settings import settings

def get_vectorstore() -> Chroma:
    """
    Returns a persistent Chroma vector store.
    Uses OpenAI embeddings for MVP.
    """
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    client = PersistentClient(path=settings.chroma_persist_dir)

    vs = Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
    )
    return vs
