from langgraph.graph import StateGraph, END
from app.graph.state import TicketGraphState
from app.graph.nodes import (
    intake_parser,
    router_classifier,
    retriever,
    resolver_decider,
    response_generator,
    quality_guard,
)

def build_graph():
    g = StateGraph(TicketGraphState)

    g.add_node("intake_parser", intake_parser)
    g.add_node("router_classifier", router_classifier)
    g.add_node("retriever", retriever)
    g.add_node("resolver_decider", resolver_decider)
    g.add_node("response_generator", response_generator)
    g.add_node("quality_guard", quality_guard)

    g.set_entry_point("intake_parser")
    g.add_edge("intake_parser", "router_classifier")
    g.add_edge("router_classifier", "retriever")
    g.add_edge("retriever", "resolver_decider")
    g.add_edge("resolver_decider", "response_generator")
    g.add_edge("response_generator", "quality_guard")
    g.add_edge("quality_guard", END)

    return g.compile()
