from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import uuid

from app.graph.graph import build_graph
from app.kb.ingest import ingest_text_docs
from app.kb.chroma_store import get_vectorstore

app = FastAPI(title="Ticket AI Router (MVP)")
graph = build_graph()

class TicketIn(BaseModel):
    subject: str
    description: str
    customer_metadata: Dict[str, Any] = Field(default_factory=dict)

class KBIngestDoc(BaseModel):
    doc_id: str
    text: str
    source: str = "internal_kb"

class KBIngestRequest(BaseModel):
    docs: List[KBIngestDoc]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/kb/ingest")
def kb_ingest(req: KBIngestRequest):
    payload = [d.model_dump() for d in req.docs]
    return ingest_text_docs(payload)

@app.get("/kb/search")
def kb_search(q: str, k: int = 5):
    vs = get_vectorstore()
    results = vs.similarity_search_with_score(q, k=k)
    out = []
    for doc, score in results:
        out.append({
            "content": doc.page_content[:600],
            "score": float(score),
            "metadata": doc.metadata
        })
    return {"query": q, "results": out}

@app.post("/tickets/run")
def run_ticket(ticket: TicketIn):
    ticket_id = str(uuid.uuid4())

    init_state = {
        "ticket": {
            "id": ticket_id,
            "subject": ticket.subject,
            "description": ticket.description,
            "customer_metadata": ticket.customer_metadata,
        },
        "cleaned_text": "",
        "routing": {"predicted_team": "Other", "confidence": 0.0, "reason": ""},
        "retrieval": {"query": "", "documents": []},
        "decision": {"action": "CLARIFY", "reason": "Not decided"},
        "response": {"draft": "", "citations": [], "confidence": 0.0},
        "trace": [],
    }

    final_state = graph.invoke(init_state)

    return {
        "ticket_id": ticket_id,
        "routing": final_state["routing"],
        "decision": final_state["decision"],
        "retrieval": {
            "query": final_state["retrieval"]["query"],
            "top_docs": final_state["retrieval"]["documents"][:3],
        },
        "response": final_state["response"],
        "cleaned_preview": final_state["cleaned_text"][:400],
        "trace": final_state["trace"],
    }
