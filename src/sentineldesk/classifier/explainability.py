"""
Explainability engine — the unified reasoning chain for a decision.

The pieces are already grounded individually (the scorer records every term that
fired, the resolver records its routing trace, the graph records its traversal
path). This assembles them into one human-readable explanation of *why* a ticket
was routed where it was and *why* a resolution was suggested — nothing invented,
every claim points at a real matched term, voter, or graph edge.

    chain = explain_decision(title, desc, scorer, resolver, graph)
    print(chain.render())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .explain import explain as explain_score
from .resolver import Decision, EdgeCaseResolver
from .scorer import DeterministicScorer


@dataclass
class ReasoningChain:
    category: str
    method: str
    confidence: float
    steps: list[str] = field(default_factory=list)
    resolution: str | None = None
    resolution_category: str | None = None
    escalated: bool = False

    def render(self) -> str:
        head = (f"ROUTE -> {self.category}  (via {self.method}, "
                f"confidence {self.confidence:.0%}"
                + (", ESCALATED to human" if self.escalated else "") + ")")
        body = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.steps))
        tail = f"\nRESOLUTION: {self.resolution}" if self.resolution else ""
        return f"{head}\n{body}{tail}"


def explain_decision(
    title: str,
    description: str,
    scorer: DeterministicScorer,
    resolver: EdgeCaseResolver,
    graph=None,
) -> ReasoningChain:
    text = f"{title} {description}".strip()

    # 1. grounded lexical reasoning (which terms fired)
    sr = scorer.classify(title, description)
    lexical = explain_score(sr)

    # 2. the routing decision + its trace
    decision: Decision = resolver.resolve(title, description)

    steps: list[str] = [f"lexical signal: {lexical}"]
    steps += [f"routing: {s}" for s in decision.trace]
    if decision.votes:
        steps.append("voters -> " + ", ".join(f"{k}={v}" for k, v in decision.votes.items()))

    # 3. resolution via graph traversal (if a graph is supplied)
    resolution = None
    resolution_category = None
    if graph is not None:
        gr = graph.query(text)
        if gr.root_cause:
            steps.append(
                f"resolution path: symptoms {gr.symptom_hits} -> "
                f"root cause '{gr.root_cause}' -> retrieved fix"
            )
            resolution = gr.resolution
            resolution_category = gr.category
        else:
            steps.append("resolution: no matching cause in the graph -> hand off")

    return ReasoningChain(
        category=decision.category,
        method=decision.method,
        confidence=decision.confidence,
        steps=steps,
        resolution=resolution,
        resolution_category=resolution_category,
        escalated=decision.escalated,
    )
