"""
End-to-end multi-agent orchestrator.

Threads a single shared PipelineState through the agent stages, turning the
separate components into one runnable workflow:

  Agent 1  Intake/Safety   -> Stage-0 regex gate; a high-stakes match escalates
                              immediately and bypasses every downstream stage.
  Agent 2  Classifier      -> deterministic scorer + SVM (confidence).
  Agent 3  Router          -> EdgeCaseResolver: confidence gate, then the
                              scorer/kNN/SVM agreement ladder + LLM tiebreak.
  Agent 4  RAG Resolver    -> Graph-RAG symptom->cause->resolution traversal.
  Agent 5  Escalation Judge-> auto-resolve if confident + a resolution exists,
                              else escalate to a human.

This is the structure of a LangGraph graph: nodes that read/write a shared
state object. Wrapping it in actual LangGraph is a thin adapter (each method
below becomes a node, PipelineState becomes the graph state) — kept dependency
-free here so the end-to-end demo is robust.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..classifier import DeterministicScorer, EdgeCaseResolver
from ..classifier.explainability import explain_decision
from ..safety.safety_layer import safety_check


@dataclass
class PipelineState:
    title: str
    description: str
    safe: bool = True
    category: str | None = None
    confidence: float = 0.0
    method: str | None = None
    resolution: str | None = None
    outcome: str = "pending"          # "auto_resolved" | "escalated" | "safety_escalated"
    reasoning: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.title} {self.description}".strip()


class Pipeline:
    def __init__(
        self,
        scorer: DeterministicScorer,
        resolver: EdgeCaseResolver,
        graph=None,
        auto_resolve_gate: float = 0.80,
    ) -> None:
        self.scorer = scorer
        self.resolver = resolver
        self.graph = graph
        self.auto_gate = auto_resolve_gate

    def run(self, title: str, description: str = "") -> PipelineState:
        s = PipelineState(title=title, description=description)

        # Agent 1 — Intake/Safety
        safety = safety_check(title, description)
        if safety.bypass_llm:
            s.safe = False
            s.category = safety.department
            s.method = "safety_gate"
            s.outcome = "safety_escalated"
            s.reasoning.append(f"safety: matched '{safety.matched_category}' -> escalate, bypass LLM")
            return s
        s.reasoning.append(f"safety: clear ({safety.latency_ms:.2f} ms)")

        # Agents 2-4 — classify, route, resolve (unified reasoning chain)
        chain = explain_decision(title, description, self.scorer, self.resolver, self.graph)
        s.category, s.confidence, s.method = chain.category, chain.confidence, chain.method
        s.resolution = chain.resolution
        s.reasoning.extend(chain.steps)

        # Agent 5 — Escalation judge
        mismatch = (
            chain.resolution_category is not None
            and chain.resolution_category != s.category
        )
        if chain.escalated:
            s.outcome = "escalated"
            s.reasoning.append("judge: resolver could not decide -> escalate to human")
        elif mismatch:
            # routing and the retrieved fix point at different teams: never hand out
            # a contradictory resolution — escalate with both for a human to reconcile.
            s.resolution = None
            s.outcome = "escalated"
            s.reasoning.append(
                f"judge: routed to {s.category} but resolution matches "
                f"{chain.resolution_category} -> inconsistent, escalate (no auto-resolve)"
            )
        elif s.resolution and s.confidence >= self.auto_gate:
            s.outcome = "auto_resolved"
            s.reasoning.append(f"judge: confident ({s.confidence:.0%}) and a consistent resolution exists -> auto-resolve")
        else:
            s.outcome = "escalated"
            s.reasoning.append("judge: routed, but no confident resolution -> escalate with context")
        return s
