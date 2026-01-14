import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from app.core.settings import settings
from app.graph.state import TicketGraphState

_llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    model="gpt-4o-mini",
    temperature=0.0,
)

CATEGORIES = [
    "Billing",
    "Account/KYC",
    "Technical",
    "Shipping",
    "Refunds",
    "Fraud/Risk",
    "Other",
]

def _add_trace(state: TicketGraphState, node: str, node_input: Dict[str, Any], node_output: Dict[str, Any]) -> None:
    state["trace"].append({"node": node, "input": node_input, "output": node_output})

def intake_parser(state: TicketGraphState) -> TicketGraphState:
    """
    Minimal intake:
    - normalize whitespace
    - strip long signatures (basic heuristic)
    """
    raw = f"{state['ticket']['subject']}\n\n{state['ticket']['description']}".strip()
    text = re.sub(r"[ \t]+", " ", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # crude signature stripping: cut off after common signoff markers if present
    signoffs = ["regards,", "thanks,", "thank you,", "sincerely,"]
    lower = text.lower()
    cutoff = None
    for s in signoffs:
        idx = lower.find("\n" + s)
        if idx != -1:
            cutoff = idx
            break
    cleaned = text[:cutoff].strip() if cutoff else text

    state["cleaned_text"] = cleaned
    _add_trace(state, "intake_parser", {"raw_len": len(raw)}, {"cleaned_len": len(cleaned)})
    return state

def router_classifier(state: TicketGraphState) -> TicketGraphState:
    """
    Minimal router:
    - LLM classifies into one of the categories with confidence + rationale
    - Output is structured
    """
    prompt = f"""
You are a support ticket routing classifier.

Pick exactly ONE category from:
{", ".join(CATEGORIES)}

Return JSON only with keys:
predicted_team (string),
confidence (number from 0 to 1),
reason (short string).

Ticket:
{state["cleaned_text"]}
""".strip()

    res = _llm.invoke(prompt).content

    # Safe parse: very small "good enough" JSON extraction
    import json
    try:
        parsed = json.loads(res)
    except Exception:
        # fallback if model adds text around JSON
        m = re.search(r"\{.*\}", res, flags=re.S)
        parsed = json.loads(m.group(0)) if m else {"predicted_team": "Other", "confidence": 0.3, "reason": "Parse fallback"}

    team = parsed.get("predicted_team", "Other")
    if team not in CATEGORIES:
        team = "Other"

    routing = {
        "predicted_team": team,
        "confidence": float(parsed.get("confidence", 0.3)),
        "reason": str(parsed.get("reason", ""))[:200],
    }

    state["routing"] = routing
    _add_trace(state, "router_classifier", {"categories": CATEGORIES}, routing)
    return state
