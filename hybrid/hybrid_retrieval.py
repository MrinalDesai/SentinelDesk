#!/usr/bin/env python3
"""
SentinelDesk — Hybrid Retrieval  (real, lightweight, isolated)

Combines, for real:
  • BM25            — lexical ranking (rank-bm25)
  • Semantic search — sentence-transformer embeddings (BGE family, e.g. bge-m3)
                      stored & queried in QDRANT running in embedded/in-memory mode
                      (real Qdrant engine, no separate server; clusters at scale)
  • Graph RAG       — the existing symptom -> cause -> vetted-resolution graph
  • Hybrid merge    — Reciprocal Rank Fusion (RRF) over the rankers

Every component genuinely runs. The only honesty caveats:
  • Qdrant runs in IN-MEMORY mode here (real engine; a hosted Qdrant cluster is the
    scale path — same client code, just a URL).
  • Semantic search needs the embedding model on your machine (sentence-transformers).
    A HashEmbedder stub exists ONLY to test the wiring where the model isn't present;
    it is clearly labelled and is NOT a real semantic model.
  • The deterministic core pipeline is UNCHANGED; this is an additive retrieval path.

Run:
    pip install rank-bm25 qdrant-client sentence-transformers
    python hybrid/hybrid_retrieval.py --in data/real_3000.csv --real-model
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "semantic", "optimizer"):
    sys.path.insert(0, str(ROOT / _sub))

from sentineldesk.corpus import load_tickets_csv                  # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary    # noqa: E402
from sentineldesk.rag import KnowledgeGraph                       # noqa: E402
from semantic_fallback import HashEmbedder, LocalEmbedder         # noqa: E402

from rank_bm25 import BM25Okapi                                   # noqa: E402
from qdrant_client import QdrantClient                            # noqa: E402
from qdrant_client.models import Distance, VectorParams, PointStruct  # noqa: E402


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]+", s.lower())


@dataclass
class Ranked:
    category: str
    text: str
    score: float


# ---------- 1. BM25 ----------
class BM25Retriever:
    def __init__(self, tickets):
        self.meta = [(t.category, f"{t.title} {t.description}".strip()) for t in tickets]
        self.bm25 = BM25Okapi([_tok(txt) for _, txt in self.meta])
        self.name = "BM25 (rank-bm25)"
    def search(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(_tok(query))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Ranked(self.meta[i][0], self.meta[i][1], float(scores[i])) for i in idx]


# ---------- 2. Semantic search backed by Qdrant (in-memory) ----------
class QdrantSemanticRetriever:
    def __init__(self, tickets, embedder, cache_path=None):
        self.embedder = embedder
        self.client = QdrantClient(location=":memory:")          # real engine, embedded
        self.coll = "tickets"
        texts = [f"{t.title} {t.description}".strip() for t in tickets]
        vecs = self._embed_cached(texts, cache_path)
        dim = len(vecs[0])
        if self.client.collection_exists(self.coll):
            self.client.delete_collection(self.coll)
        self.client.create_collection(self.coll,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        self.client.upsert(self.coll, points=[
            PointStruct(id=str(uuid.uuid4()), vector=v,
                        payload={"category": t.category, "text": txt})
            for t, txt, v in zip(tickets, texts, vecs)])
        self.name = f"Qdrant(in-memory) + {embedder.name}"

    def _embed_cached(self, texts, cache_path):
        """Embed once, cache to disk; reload instantly on later runs.
        Cache is keyed on (embedder name + corpus size + a content hash)."""
        if cache_path is None:
            return self.embedder.encode(texts)
        import json, hashlib
        key = hashlib.md5((self.embedder.name + str(len(texts)) +
                           "".join(texts)[:5000]).encode()).hexdigest()
        cache = Path(cache_path)
        if cache.exists():
            try:
                blob = json.loads(cache.read_text())
                if blob.get("key") == key:
                    print(f"  (loaded {len(blob['vecs'])} cached embeddings — fast path)")
                    return blob["vecs"]
            except Exception:
                pass
        print(f"  (embedding {len(texts)} tickets with {self.embedder.name} — first run, then cached)")
        vecs = self.embedder.encode(texts)
        try:
            cache.write_text(json.dumps({"key": key, "vecs": vecs}))
        except Exception:
            pass
        return vecs

    def search(self, query: str, k: int = 10):
        qv = self.embedder.encode([query])[0]
        hits = self.client.query_points(self.coll, query=qv, limit=k).points
        return [Ranked(h.payload["category"], h.payload["text"], float(h.score)) for h in hits]


# ---------- 3. Graph RAG (existing) as a ranker over categories ----------
class GraphRetriever:
    def __init__(self, tickets, cv):
        self.graph = KnowledgeGraph.build(tickets, cv)
        self.cv = cv
        self.name = "Graph RAG (symptom->cause->resolution)"
    def search(self, query: str, k: int = 10):
        try:
            res = self.graph.query(self.cv.normalize(query))
            if res and res.category:
                return [Ranked(res.category, res.resolution or "", 1.0)]
        except Exception:
            pass
        return []


# ---------- 4. Hybrid merge (Reciprocal Rank Fusion) ----------
class HybridRetriever:
    def __init__(self, retrievers, weights=None, rrf_k: int = 60):
        self.retrievers = retrievers
        self.weights = weights or {r.name: 1.0 for r in retrievers}
        self.rrf_k = rrf_k
        self.name = "Hybrid merge (RRF over BM25 + Qdrant-semantic + Graph)"
    def route(self, query: str, k: int = 10):
        cat_scores = defaultdict(float)
        per_ranker = {}
        for r in self.retrievers:
            ranked = r.search(query, k)
            per_ranker[r.name] = ranked[:3]
            w = self.weights.get(r.name, 1.0)
            for rank, item in enumerate(ranked):
                cat_scores[item.category] += w * 1.0 / (self.rrf_k + rank + 1)
        if not cat_scores:
            return {"category": None, "confidence": 0.0, "per_ranker": per_ranker}
        ordered = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        top = ordered[0]
        total = sum(cat_scores.values()) or 1.0
        return {"category": top[0], "confidence": round(top[1] / total, 3),
                "ranking": ordered, "per_ranker": per_ranker}


# ---------- demo ----------
def demo(tickets, embedder, cache_path=None):
    cv = ConceptVocabulary.from_layman_map()
    bm25 = BM25Retriever(tickets)
    sem = QdrantSemanticRetriever(tickets, embedder, cache_path=cache_path)
    graph = GraphRetriever(tickets, cv)
    hybrid = HybridRetriever([bm25, sem, graph])

    print("Hybrid retrieval — components live:")
    for r in (bm25, sem, graph):
        print(f"  • {r.name}")
    print(f"  ⇒ {hybrid.name}\n")

    probes = [
        "deadlock on the primary database replica overnight",
        "users can't reach internal websites across the office",
        "ransomware encrypting files on a finance workstation",
    ]
    for q in probes:
        out = hybrid.route(q)
        print(f"QUERY: {q!r}")
        print(f"  hybrid → {out['category']} (fused score {out['confidence']})")
        for name, items in out["per_ranker"].items():
            tops = ", ".join(f"{i.category}" for i in items) or "—"
            print(f"     {name:<42} top: {tops}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "data" / "real_3000.csv"))
    ap.add_argument("--real-model", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="use a small fast embedding model (bge-small) — recommended for live demos")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--no-cache", action="store_true", help="disable embedding cache")
    args = ap.parse_args()
    path = Path(args.inp)
    if not path.exists():
        print(f"data not found: {path}"); sys.exit(1)
    tickets = load_tickets_csv(path)
    model = "BAAI/bge-small-en-v1.5" if args.fast else args.model
    try:
        emb = LocalEmbedder(model) if (args.real_model or args.fast) else HashEmbedder()
    except Exception as e:
        print(f"(real model unavailable: {e}\n using hash stub for wiring demo only)")
        emb = HashEmbedder()
    cache = None if args.no_cache else str(ROOT / "hybrid" / "emb_cache.json")
    demo(tickets, emb, cache_path=cache)
