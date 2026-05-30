"""
Edge-case resolver — the agentic classification loop.

This is the layer that turns SentinelDesk from a single classifier into a
reason-act-decide agent. It is *confidence-gated*: the cheap SVM handles the
clear majority untouched, and only low-confidence tickets enter the ladder.

Ladder (cheap -> expensive, stop at the first confident decision):
  0. SVM confidence gate. High confidence -> route, done. (~the 98%.)
  1. Gather voters: deterministic scorer (weights domain-UNIQUE terms 3x/2x,
     so a strong domain term outranks an incidental "user/access" mention),
     SVM, and a kNN vote over the corpus.
  2. Agreement gate. If a majority of voters agree -> route with that label.
  3. LLM tiebreak. Voters disagree -> the LLM reasons over the candidates and
     the evidence each cited, and picks. This is the visible ReAct step.
  4. Escalate. Still unresolved / LLM abstains -> hand to a human.

Every decision carries a `trace`: the ordered reasoning steps, so the route is
explainable end to end.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..llm.client import LLMClient
from .knn import KNNVoter
from .scorer import DeterministicScorer
from .svm import SVMClassifier


@dataclass
class Decision:
    category: str
    method: str               # "confident_svm" | "agreement" | "llm_tiebreak" | "escalated"
    confidence: float
    trace: list[str] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)  # voter -> its pick
    escalated: bool = False


class EdgeCaseResolver:
    def __init__(
        self,
        scorer: DeterministicScorer,
        svm: SVMClassifier,
        knn: KNNVoter,
        llm: LLMClient | None = None,
        confidence_gate: float = 0.80,
    ) -> None:
        self.scorer = scorer
        self.svm = svm
        self.knn = knn
        self.llm = llm
        self.gate = confidence_gate

    def resolve(self, title: str, description: str = "") -> Decision:
        text = f"{title} {description}".strip()
        trace: list[str] = []

        # 0. SVM confidence gate — the clear majority never enters the ladder.
        svm_label, svm_conf = self.svm.predict(text)
        trace.append(f"SVM: {svm_label} (confidence {svm_conf:.2f})")
        if svm_conf >= self.gate:
            trace.append(f"confidence >= {self.gate:.2f} -> route directly")
            return Decision(svm_label, "confident_svm", svm_conf, trace,
                            {"svm": svm_label})

        trace.append(f"confidence < {self.gate:.2f} -> entering edge-case ladder")

        # 1. Gather the three voters.
        sr = self.scorer.classify(title, description)
        kv = self.knn.vote(text)
        votes = {"svm": svm_label, "scorer": sr.category, "knn": kv.category}
        trace.append(
            f"voters -> scorer: {sr.category} (margin {sr.margin:.1f}), "
            f"kNN: {kv.category} ({kv.confidence:.2f}), SVM: {svm_label}"
        )

        # 2. Agreement gate — majority of the three.
        tally = Counter(votes.values())
        winner, agree = tally.most_common(1)[0]
        if agree >= 2:
            conf = agree / 3.0
            trace.append(f"{agree}/3 voters agree -> {winner}")
            return Decision(winner, "agreement", conf, trace, votes)

        # 3. LLM tiebreak — voters disagree; reason over the candidates.
        trace.append("voters disagree (3-way split)")
        if self.llm is not None:
            choice = self._llm_tiebreak(text, sr, votes)
            if choice in votes.values():
                trace.append(f"LLM tiebreak -> {choice}")
                return Decision(choice, "llm_tiebreak", 0.5, trace, votes)
            trace.append(f"LLM returned '{choice}' (not a candidate) -> escalate")

        # 4. Escalate.
        trace.append("unresolved -> escalate to human")
        return Decision(svm_label, "escalated", svm_conf, trace, votes, escalated=True)

    def _llm_tiebreak(self, text: str, sr, votes: dict[str, str]) -> str:
        candidates = sorted(set(votes.values()))
        evidence = ", ".join(f"{sig}={terms}" for sig, terms in sr.matched.items() if terms)
        prompt = (
            "You are routing an IT support ticket to exactly one team. "
            f"The candidate teams are: {', '.join(candidates)}.\n"
            f"Ticket: {text}\n"
            f"Lexical evidence found: {evidence or 'none'}\n"
            "Reply with ONLY the exact team name from the candidate list that best "
            "matches the ROOT cause (not an incidental mention)."
        )
        resp = self.llm.complete(prompt, temperature=0.0).strip()
        # tolerate the model echoing extra words: match a candidate as a substring
        for c in candidates:
            if c.lower() in resp.lower():
                return c
        return resp
