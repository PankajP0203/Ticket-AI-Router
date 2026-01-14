from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.kb.chroma_store import get_vectorstore

def ingest_text_docs(
    docs: List[Dict[str, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Dict[str, int]:
    """
    docs: [{ "doc_id": "billing_refund_policy", "text": "...", "source": "internal_kb" }, ...]
    """
    vs = get_vectorstore()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts, metadatas, ids = [], [], []
    for d in docs:
        doc_id = d["doc_id"]
        text = d["text"]
        source = d.get("source", "unknown")

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"doc_id": doc_id, "source": source, "chunk_index": i})
            ids.append(f"{doc_id}::chunk::{i}")

    # Add to Chroma
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    return {"documents": len(docs), "chunks": len(texts)}
