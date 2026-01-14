from typing import TypedDict, Optional, Dict, Any, List

class Ticket(TypedDict):
    id: str
    subject: str
    description: str
    customer_metadata: Dict[str, Any]

class Routing(TypedDict):
    predicted_team: str
    confidence: float
    reason: str

class TicketGraphState(TypedDict):
    ticket: Ticket
    cleaned_text: str
    routing: Routing
    trace: List[Dict[str, Any]]
