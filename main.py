import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from loguru import logger

from agents.agent1_intake import process_ticket
from agents.agent2_classifier import classify_ticket
from agents.agent3_router import route_ticket
from agents.agent4_resolver import get_resolution
from agents.agent5_escalation import escalate_ticket

# ── State Definition ──────────────────────────────────────────
class TicketState(TypedDict):
    # Input
    raw_title:       str
    raw_description: str
    priority:        str
    ticket_id:       int

    # Agent 1 output
    clean_title:       Optional[str]
    clean_description: Optional[str]
    pii_detected:      Optional[list]

    # Agent 2 output
    category:          Optional[str]
    confidence:        Optional[float]
    reasoning:         Optional[str]

    # Agent 3 output
    department:           Optional[str]
    escalation_contact:   Optional[str]
    needs_escalation:     Optional[bool]

    # Agent 4 output
    resolution:    Optional[str]
    sources:       Optional[list]
    rag_confidence:Optional[float]

    # Agent 5 output
    final_action:      Optional[str]
    escalate:          Optional[bool]
    quality_score:     Optional[int]
    final_reason:      Optional[str]

# ── Agent Node Functions ───────────────────────────────────────
def node_intake(state: TicketState) -> TicketState:
    logger.info("═══ Agent 1: Intake ═══")
    result = process_ticket({
        "id":          state["ticket_id"],
        "title":       state["raw_title"],
        "description": state["raw_description"],
        "priority":    state["priority"]
    })
    return {
        **state,
        "clean_title":       result["title"],
        "clean_description": result["description"],
        "pii_detected":      result["pii_detected"]
    }

def node_classifier(state: TicketState) -> TicketState:
    logger.info("═══ Agent 2: Classifier ═══")
    result = classify_ticket(
        state["clean_title"],
        state["clean_description"]
    )
    return {
        **state,
        "category":   result["category"],
        "confidence": result["confidence"],
        "reasoning":  result.get("reasoning", "")
    }

def node_router(state: TicketState) -> TicketState:
    logger.info("═══ Agent 3: Router ═══")
    result = route_ticket(
        category=state["category"],
        priority=state["priority"],
        confidence=state["confidence"],
        ticket_id=state["ticket_id"]
    )
    return {
        **state,
        "department":         result["department"],
        "escalation_contact": result["escalation_contact"],
        "needs_escalation":   result["needs_escalation"]
    }

def node_resolver(state: TicketState) -> TicketState:
    logger.info("═══ Agent 4: RAG Resolver ═══")
    result = get_resolution(
        title=state["clean_title"],
        description=state["clean_description"],
        category=state["category"],
        ticket_id=state["ticket_id"]
    )
    return {
        **state,
        "resolution":     result["resolution"],
        "sources":        result.get("sources", []),
        "rag_confidence": result.get("confidence", 0.8)
    }

def node_escalation(state: TicketState) -> TicketState:
    logger.info("═══ Agent 5: Escalation ═══")
    result = escalate_ticket(
        title=state["clean_title"],
        description=state["clean_description"],
        category=state["category"],
        priority=state["priority"],
        confidence=state["confidence"],
        resolution=state["resolution"],
        department=state["department"],
        escalation_contact=state["escalation_contact"],
        ticket_id=state["ticket_id"]
    )
    return {
        **state,
        "final_action":  result["action"],
        "escalate":      result["escalate"],
        "quality_score": result.get("quality_score", 3),
        "final_reason":  result["reason"]
    }

# ── Build Graph ────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("intake",     node_intake)
    graph.add_node("classifier", node_classifier)
    graph.add_node("router",     node_router)
    graph.add_node("resolver",   node_resolver)
    graph.add_node("escalation", node_escalation)

    graph.set_entry_point("intake")
    graph.add_edge("intake",     "classifier")
    graph.add_edge("classifier", "router")
    graph.add_edge("router",     "resolver")
    graph.add_edge("resolver",   "escalation")
    graph.add_edge("escalation", END)

    return graph.compile()

# ── Process Single Ticket ──────────────────────────────────────
def process(
    title: str,
    description: str,
    priority: str = "Medium",
    ticket_id: int = 1
) -> dict:
    app = build_graph()

    initial_state = TicketState(
        raw_title=title,
        raw_description=description,
        priority=priority,
        ticket_id=ticket_id,
        clean_title=None,
        clean_description=None,
        pii_detected=None,
        category=None,
        confidence=None,
        reasoning=None,
        department=None,
        escalation_contact=None,
        needs_escalation=None,
        resolution=None,
        sources=None,
        rag_confidence=None,
        final_action=None,
        escalate=None,
        quality_score=None,
        final_reason=None
    )

    result = app.invoke(initial_state)
    return result

# ── Main Test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  SENTINELDESK — Full Pipeline Test")
    print("═"*60)

    result = process(
        title="VPN not connecting after password reset",
        description="User john.smith@company.com cannot connect to corporate VPN since password was reset. IP 192.168.1.100. Authentication timeout error appearing.",
        priority="High",
        ticket_id=1
    )

    print("\n" + "═"*60)
    print("  FINAL RESULT")
    print("═"*60)
    print(f"  Category:      {result['category']}")
    print(f"  Confidence:    {result['confidence']}")
    print(f"  Department:    {result['department']}")
    print(f"  PII Detected:  {result['pii_detected']}")
    print(f"  Resolution:    {result['resolution'][:100]}...")
    print(f"  Action:        {result['final_action']}")
    print(f"  Quality Score: {result['quality_score']}/5")
    print(f"  Escalate:      {result['escalate']}")
    print("═"*60)