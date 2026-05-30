"""
LangGraph orchestration — the multi-agent pipeline as a compiled StateGraph.

This is the *literal* LangGraph version of pipeline/orchestrator.py: each agent
is a graph node reading/writing a shared TypedDict state, with a conditional
edge that lets the safety gate short-circuit straight to the end. Same logic,
same components — wrapped in the framework named in the design.

The dependency-free Pipeline remains the robust fallback for environments
without langgraph (e.g. a locked-down demo box); build_graph() raises a clear
message if langgraph isn't installed, rather than failing obscurely.

    graph = build_graph(scorer, resolver, kg)
    result = graph.invoke({"title": t, "description": d, "reasoning": []})
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..classifier import DeterministicScorer, EdgeCaseResolver
from ..classifier.explainability import explain_decision
from ..safety.safety_layer import safety_check


class GraphState(TypedDict, total=False):
    title: str
    description: str
    safe: bool
    category: str
    confidence: float
    method: str
    resolution: str
    outcome: str
    reasoning: list[str]


def build_graph(scorer: DeterministicScorer, resolver: EdgeCaseResolver,
                kg=None, auto_gate: float = 0.80):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "langgraph is not installed (`pip install langgraph`). The "
            "dependency-free pipeline.Pipeline provides the same workflow without it."
        ) from e

    def intake_safety(state: GraphState) -> dict:
        r = safety_check(state["title"], state.get("description", ""))
        if r.bypass_llm:
            return {"safe": False, "category": r.department, "method": "safety_gate",
                    "outcome": "safety_escalated",
                    "reasoning": state.get("reasoning", []) +
                    [f"safety: matched '{r.matched_category}' -> escalate, bypass LLM"]}
        return {"safe": True,
                "reasoning": state.get("reasoning", []) + [f"safety: clear ({r.latency_ms:.2f} ms)"]}

    def classify_route_resolve(state: GraphState) -> dict:
        chain = explain_decision(state["title"], state.get("description", ""), scorer, resolver, kg)
        mismatch = chain.resolution_category is not None and chain.resolution_category != chain.category
        reasoning = state.get("reasoning", []) + chain.steps
        if chain.escalated:
            outcome, resolution = "escalated", None
            reasoning.append("judge: resolver could not decide -> escalate")
        elif mismatch:
            outcome, resolution = "escalated", None
            reasoning.append(f"judge: routed {chain.category} but fix matches "
                             f"{chain.resolution_category} -> inconsistent, escalate")
        elif chain.resolution and chain.confidence >= auto_gate:
            outcome, resolution = "auto_resolved", chain.resolution
            reasoning.append(f"judge: confident ({chain.confidence:.0%}) + consistent fix -> auto-resolve")
        else:
            outcome, resolution = "escalated", None
            reasoning.append("judge: no confident resolution -> escalate with context")
        return {"category": chain.category, "confidence": chain.confidence,
                "method": chain.method, "resolution": resolution,
                "outcome": outcome, "reasoning": reasoning}

    builder = StateGraph(GraphState)
    builder.add_node("intake_safety", intake_safety)
    builder.add_node("classify_route_resolve", classify_route_resolve)
    builder.set_entry_point("intake_safety")
    builder.add_conditional_edges(
        "intake_safety",
        lambda s: "end" if not s.get("safe", True) else "continue",
        {"end": END, "continue": "classify_route_resolve"},
    )
    builder.add_edge("classify_route_resolve", END)
    return builder.compile()
