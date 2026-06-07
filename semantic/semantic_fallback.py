#!/usr/bin/env python3
"""
SentinelDesk — Semantic Edge-Case Fallback  (isolated addendum)

Where it sits in the escalation ladder:

    lexical (fast, most tickets)
        -> voter ladder (ambiguous)
        -> *** SEMANTIC FALLBACK (extreme edge cases) ***   <-- this module
        -> LLM tiebreak (rare)
        -> human (last resort)

It activates ONLY for extreme edge cases — tickets the lexical layer can't get
purchase on (no/low signal, out-of-vocabulary phrasing). For those, it embeds the
ticket, finds its nearest semantic neighbours in the labelled corpus, and routes by
their consensus. This is where embeddings genuinely earn their place — not on the
common path, which stays deterministic and auditable.

Honest design notes:
  • In-memory cosine search over corpus embeddings. For a few-thousand-ticket corpus
    this is plenty; **Qdrant is the documented scale-up seam** (swap InMemoryIndex for
    a QdrantIndex with the same interface — not run here).
  • The embedder is an interface. On your machine, LocalEmbedder uses a real
    sentence-transformer (default: BAAI/bge-small / bge-m3 family). In environments
    without the model, HashEmbedder is a deterministic stand-in so the *logic* is
    testable — it is clearly NOT a real semantic model and is labelled as such.
  • The core deterministic pipeline is UNCHANGED; this is an additive tier.

Run the demo:
    pip install sentence-transformers      # on your machine
    python semantic/semantic_fallback.py --in data/real_3000.csv
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentineldesk.corpus import load_tickets_csv                  # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary    # noqa: E402


# ---------- embedder interface ----------
class Embedder(Protocol):
    name: str
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """Real sentence-transformer embeddings (runs on your machine)."""
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # imported lazily
        self.model = SentenceTransformer(model)
        self.name = f"sentence-transformer:{model}"
    def encode(self, texts):
        return [list(map(float, v)) for v in self.model.encode(texts, normalize_embeddings=True)]


class HashEmbedder:
    """Deterministic stand-in for environments without the real model.
    NOT a semantic model — hashed bag-of-words into a fixed vector. Lets the
    fallback *logic* be tested; real semantics require LocalEmbedder."""
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = "hash-stub (NOT a real embedding model)"
    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


# ---------- in-memory index (Qdrant is the scale-up seam) ----------
@dataclass
class Neighbour:
    category: str
    score: float
    text: str


class InMemoryIndex:
    """Cosine NN over corpus embeddings. Same interface a QdrantIndex would expose."""
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.vecs: list[list[float]] = []
        self.meta: list[tuple[str, str]] = []  # (category, text)

    def build(self, tickets):
        texts = [f"{t.title} {t.description}".strip() for t in tickets]
        self.vecs = self.embedder.encode(texts)
        self.meta = [(t.category, txt) for t, txt in zip(tickets, texts)]
        return self

    def search(self, text: str, k: int = 5) -> list[Neighbour]:
        q = self.embedder.encode([text])[0]
        scored = [(_cos(q, v), cat, txt) for v, (cat, txt) in zip(self.vecs, self.meta)]
        scored.sort(reverse=True)
        return [Neighbour(cat, round(s, 3), txt) for s, cat, txt in scored[:k]]


# ---------- the fallback decision ----------
@dataclass
class SemanticDecision:
    category: str
    confidence: float
    neighbours: list[Neighbour]
    fired: bool
    note: str


class SemanticFallback:
    """Fires only for extreme edge cases. Returns a routed category by neighbour
    consensus, or signals 'still unclear -> escalate'."""
    def __init__(self, index: InMemoryIndex, cv: ConceptVocabulary, k: int = 5,
                 min_consensus: float = 0.6, min_score: float = 0.35):
        self.index, self.cv, self.k = index, cv, k
        self.min_consensus, self.min_score = min_consensus, min_score

    def is_extreme_edge(self, scorer_result) -> bool:
        """An extreme edge case = lexical layer found essentially nothing."""
        no_signal = not any(scorer_result.matched.get(s) for s in ("unique", "freq", "layman"))
        return scorer_result.is_edge_case and (no_signal or scorer_result.confidence < 0.35)

    def route(self, title: str, description: str) -> SemanticDecision:
        text = self.cv.normalize(f"{title} {description}".strip())
        nbrs = self.index.search(text, self.k)
        if not nbrs or nbrs[0].score < self.min_score:
            return SemanticDecision("", 0.0, nbrs, False,
                "no semantically similar ticket above threshold -> escalate to human")
        tally = Counter(n.category for n in nbrs)
        top, count = tally.most_common(1)[0]
        consensus = count / len(nbrs)
        if consensus < self.min_consensus:
            return SemanticDecision("", round(consensus, 2), nbrs, False,
                "neighbours disagree -> escalate to human")
        return SemanticDecision(top, round(consensus, 2), nbrs, True,
            f"routed by {count}/{len(nbrs)} semantic neighbours")


# ---------- demo ----------
def demo(tickets, embedder: Embedder):
    cv = ConceptVocabulary.from_layman_map()
    from sentineldesk.classifier.scorer import DeterministicScorer, VocabModel
    scorer = DeterministicScorer(VocabModel.build(tickets))
    idx = InMemoryIndex(embedder).build(tickets)
    fb = SemanticFallback(idx, cv)

    print(f"embedder: {embedder.name}")
    print(f"index: {len(idx.vecs)} tickets  |  Qdrant = documented scale-up seam\n")

    # deliberately low-signal / out-of-vocabulary phrasings (extreme edge cases)
    probes = [
        ("", "the thingy in the corner room keeps making everyone's screens freeze up"),
        ("", "stuff stopped working after the weekend update, nobody can get in"),
        ("", "it's doing that thing again where the spinny wheel never stops"),
    ]
    for title, desc in probes:
        sr = scorer.classify(title, desc)
        extreme = fb.is_extreme_edge(sr)
        print(f"TICKET: {desc!r}")
        print(f"  lexical: {sr.category} ({sr.confidence:.2f}), edge={sr.is_edge_case}, "
              f"extreme-edge={extreme}")
        if extreme:
            d = fb.route(title, desc)
            print(f"  -> SEMANTIC FALLBACK fired: "
                  + (f"{d.category} ({d.confidence}) — {d.note}" if d.fired else d.note))
            for n in d.neighbours[:3]:
                print(f"       ~ {n.score}  [{n.category}]  {n.text[:60]}")
        else:
            print("  -> lexical handled it; fallback not needed")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "data" / "real_3000.csv"))
    ap.add_argument("--real-model", action="store_true",
                    help="use the real sentence-transformer (needs the package + model)")
    args = ap.parse_args()
    path = Path(args.inp)
    if not path.exists():
        print(f"data not found: {path}"); sys.exit(1)
    tickets = load_tickets_csv(path)
    try:
        emb = LocalEmbedder() if args.real_model else HashEmbedder()
    except Exception as e:
        print(f"(real model unavailable: {e}\n falling back to hash stub for logic demo)")
        emb = HashEmbedder()
    demo(tickets, emb)
