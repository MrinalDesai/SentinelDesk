"""
Pluggable retrieval — the seam where a vector DB drops in.

The system retrieves similar past tickets in two places (the kNN voter and,
conceptually, RAG resolution). Today that's LEXICAL (TF-IDF cosine, pure
sklearn) — interpretable, CPU-only, no infrastructure. This module makes the
backend swappable behind one interface so a semantic vector DB (Qdrant +
BGE-M3 embeddings) can replace it WITHOUT touching any caller.

  - LexicalRetriever : the working default (TF-IDF nearest-neighbour).
  - QdrantRetriever  : adapter scaffold. Imports qdrant-client lazily and is
                       only usable when a Qdrant server + an embedder are
                       provided. Ships disabled so nothing breaks; it documents
                       exactly where the semantic upgrade plugs in.

Swapping backends is a one-line change at construction; the rest of the system
is unaware which retriever it holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..corpus import LabeledTicket


@dataclass
class Retrieved:
    text: str
    category: str
    score: float          # similarity in [0,1] (1 = identical)
    resolution: str = ""


class Retriever(Protocol):
    def fit(self, tickets: list[LabeledTicket]) -> "Retriever": ...
    def search(self, query: str, k: int = 5) -> list[Retrieved]: ...


class LexicalRetriever:
    """Default backend: TF-IDF cosine similarity (no embeddings, no vector DB)."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
        self._Vec, self._NN = TfidfVectorizer, NearestNeighbors
        self._vec = None
        self._nn = None
        self._tickets: list[LabeledTicket] = []

    def fit(self, tickets: list[LabeledTicket]) -> "LexicalRetriever":
        self._tickets = tickets
        self._vec = self._Vec(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        m = self._vec.fit_transform([t.text for t in tickets])
        self._nn = self._NN(n_neighbors=min(10, len(tickets)), metric="cosine").fit(m)
        return self

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        dist, idx = self._nn.kneighbors(self._vec.transform([query]), n_neighbors=min(k, len(self._tickets)))
        out = []
        for d, i in zip(dist[0], idx[0]):
            t = self._tickets[i]
            out.append(Retrieved(t.text, t.category, 1.0 - float(d), t.resolution))
        return out


class QdrantRetriever:
    """
    Semantic backend (roadmap). Same interface as LexicalRetriever, but stores
    BGE-M3 embeddings in Qdrant and searches by vector similarity — widening
    recall to paraphrases the lexical backend misses.

    Disabled unless given a live Qdrant client and an embedder, so importing
    this module never requires qdrant-client to be installed.
    """

    def __init__(self, client=None, embedder=None, collection: str = "tickets") -> None:
        self.client = client
        self.embedder = embedder      # callable: str -> list[float]
        self.collection = collection
        if client is None or embedder is None:
            # Documents the integration point without forcing the dependency.
            raise NotImplementedError(
                "QdrantRetriever is the planned semantic backend. Provide a qdrant "
                "client and a BGE-M3 embedder to enable it; until then the system "
                "uses LexicalRetriever (TF-IDF), which needs no vector DB."
            )

    def fit(self, tickets: list[LabeledTicket]) -> "QdrantRetriever":
        from qdrant_client.models import PointStruct  # lazy import
        points = [
            PointStruct(id=i, vector=self.embedder(t.text),
                        payload={"text": t.text, "category": t.category, "resolution": t.resolution})
            for i, t in enumerate(tickets)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return self

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        hits = self.client.search(collection_name=self.collection,
                                  query_vector=self.embedder(query), limit=k)
        return [Retrieved(h.payload["text"], h.payload["category"], float(h.score),
                          h.payload.get("resolution", "")) for h in hits]
