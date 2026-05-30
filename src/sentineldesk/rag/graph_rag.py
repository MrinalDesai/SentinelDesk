"""
Graph RAG (minimal, real) — symptom -> root cause -> resolution traversal.

This is a genuine knowledge graph with typed nodes and weighted edges, not a
similarity lookup in disguise. It reuses artifacts you already have:

  - SYMPTOM -> ROOT_CAUSE edges come from the concept vocabulary: every layman
    phrasing ("websites won't load") points to its canonical cause term
    ("dns resolution"), plus corpus co-occurrence reinforces the weight.
  - ROOT_CAUSE -> RESOLUTION edges come from the corpus: the resolutions seen on
    tickets exhibiting that cause.

A query traverses symptoms -> the best-supported root cause -> its resolution,
returning the path so the answer is explainable. Because it walks to a specific
resolution instead of stuffing many retrieved tickets into a prompt, it hands
the LLM a short, targeted context (the token-efficiency motivation) — though we
make no fixed reduction claim.

Pure stdlib (dict adjacency); swap in networkx or a graph DB later without
changing the traversal.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..corpus import LabeledTicket
from ..vocabulary.concepts import ConceptVocabulary

# Boilerplate stubs that look like a resolution but carry no actual remediation.
_STUB_PATTERNS = (
    "please follow these steps",
    "follow the steps below",
    "follow these steps",
    "see below",
    "as follows",
    "steps to resolve",
    "to resolve the issue",
    "contact the it department",
    "contact support",
    "tbd",
    "n/a",
)
# Verbs / markers that signal an actionable resolution.
_ACTION_MARKERS = (
    "check", "restart", "reboot", "isolate", "disconnect", "roll back", "rollback",
    "reset", "update", "patch", "increase", "expand", "clear", "remove", "add",
    "verify", "configure", "apply", "replace", "kill", "flush", "rotate", "review",
    "1.", "2.", "step 1", "- ",
)


def resolution_quality(text: str | None) -> int:
    """
    Score a candidate resolution 0..3. 0 = unusable (empty/stub). Higher = more
    actionable. Used both to reject stubs and to pick the best vetted fix.
    """
    if not text:
        return 0
    body = str(text).strip().strip("[]'\"").strip()
    low = body.lower()
    # a stub is short and/or is just boilerplate preamble with nothing after it
    stub = any(p in low for p in _STUB_PATTERNS)
    if len(body) < 25:
        return 0
    if stub and len(body) < 60:          # boilerplate with no real content following
        return 0
    score = 1
    if len(body) >= 60:
        score += 1
    if any(m in low for m in _ACTION_MARKERS):
        score += 1
    return score


def best_resolution(counter: "Counter | None") -> str | None:
    """Pick the highest-quality resolution for a cause; None if none clear the gate."""
    if not counter:
        return None
    scored = [(resolution_quality(r), cnt, r) for r, cnt in counter.items()]
    scored = [s for s in scored if s[0] > 0]              # drop stubs/empties
    if not scored:
        return None
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)  # quality, then frequency
    return scored[0][2]


@dataclass
class GraphResult:
    root_cause: str | None
    category: str | None
    resolution: str | None
    path: list[str] = field(default_factory=list)     # symptom(s) -> cause -> resolution
    symptom_hits: list[str] = field(default_factory=list)
    candidates: list[tuple[str, int]] = field(default_factory=list)  # (cause, support)
    resolution_quality: int = 0


class KnowledgeGraph:
    def __init__(self, vocab: ConceptVocabulary) -> None:
        self.vocab = vocab
        # symptom surface form -> canonical root cause
        self.symptom_to_cause: dict[str, str] = {}
        # root cause -> its category
        self.cause_category: dict[str, str] = {}
        # symptom -> {cause: weight} (corpus-reinforced; usually one cause per symptom)
        self.symptom_cause_w: dict[str, Counter] = defaultdict(Counter)
        # root cause -> {resolution: count}
        self.cause_resolutions: dict[str, Counter] = defaultdict(Counter)

    @classmethod
    def build(cls, tickets: list[LabeledTicket], vocab: ConceptVocabulary) -> "KnowledgeGraph":
        g = cls(vocab)
        # 1. symptom -> cause edges from the concept vocabulary
        for category, groups in vocab.groups.items():
            for group in groups:
                canonical = group[0]
                g.cause_category[canonical] = category
                for surface in group:               # canonical + its synonyms are all symptoms
                    g.symptom_to_cause[surface.lower()] = canonical
        # 2. reinforce symptom->cause weights + build cause->resolution from the corpus
        for t in tickets:
            text = t.text.lower()
            causes_here = set()
            for symptom, cause in g.symptom_to_cause.items():
                if symptom in text:
                    g.symptom_cause_w[symptom][cause] += 1
                    causes_here.add(cause)
            if t.resolution:
                for cause in causes_here:
                    g.cause_resolutions[cause][t.resolution] += 1
        return g

    @property
    def stats(self) -> dict:
        return {
            "symptom_nodes": len(self.symptom_to_cause),
            "root_cause_nodes": len(self.cause_category),
            "resolution_nodes": sum(len(r) for r in self.cause_resolutions.values()),
            "symptom_cause_edges": sum(len(v) for v in self.symptom_cause_w.values()),
            "causes_with_vetted_resolution": sum(
                1 for c in self.cause_resolutions if best_resolution(self.cause_resolutions[c])
            ),
        }

    def query(self, text: str) -> GraphResult:
        low = text.lower()
        # hop 1: which symptoms fire?
        hits = [s for s in self.symptom_to_cause if s in low]
        if not hits:
            return GraphResult(None, None, None, ["no symptom matched"], [], [])
        # hop 2: traverse symptoms -> root causes, accumulate support
        support: Counter = Counter()
        for s in hits:
            support[self.symptom_to_cause[s]] += 1
        ranked = support.most_common()
        top_cause, _ = ranked[0]
        category = self.cause_category.get(top_cause)
        # hop 3: traverse root cause -> best VETTED resolution (stubs/empties filtered)
        res_counter = self.cause_resolutions.get(top_cause)
        resolution = best_resolution(res_counter)
        quality = resolution_quality(resolution)
        sym_for_cause = [s for s in hits if self.symptom_to_cause[s] == top_cause]
        res_label = (
            (resolution[:60] + "..." if len(resolution) > 60 else resolution)
            if resolution else "no vetted resolution on file -> escalate to human"
        )
        path = [
            f"symptoms{sym_for_cause}",
            f"root cause: {top_cause}",
            f"resolution: {res_label}",
        ]
        return GraphResult(top_cause, category, resolution, path, hits, ranked, quality)
