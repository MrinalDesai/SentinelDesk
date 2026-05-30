"""
kNN voter for the edge-case resolver.

A lightweight nearest-neighbour vote over the training corpus, using the same
TF-IDF lexical space as the SVM. No embeddings, no vector DB — pure sklearn —
so it runs anywhere and needs nothing stood up. It is the third voter (with the
deterministic scorer and the SVM) in the resolver's agreement gate: it answers
"which categories do the most similar known tickets belong to?"

When real confirmed-resolved history is available, point this at that instead
of the synthetic corpus and its value rises (the synthetic neighbours largely
echo the planted vocabulary).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ..corpus import LabeledTicket


@dataclass
class KNNVote:
    category: str            # majority label among the k neighbours
    confidence: float        # share of neighbours that agree with the winner
    tally: dict[str, int]    # label -> neighbour count
    distances: list[float]   # cosine distances of the k neighbours


class KNNVoter:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._vec: TfidfVectorizer | None = None
        self._nn: NearestNeighbors | None = None
        self._labels: list[str] = []

    def fit(self, tickets: list[LabeledTicket]) -> "KNNVoter":
        texts = [t.text for t in tickets]
        self._labels = [t.category for t in tickets]
        self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        matrix = self._vec.fit_transform(texts)
        # cosine distance via brute force on sparse TF-IDF vectors
        k = min(self.k, len(tickets))
        self._nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self._nn.fit(matrix)
        return self

    def vote(self, text: str) -> KNNVote:
        if self._vec is None or self._nn is None:
            raise RuntimeError("KNNVoter not fitted")
        vec = self._vec.transform([text])
        dist, idx = self._nn.kneighbors(vec)
        neighbours = [self._labels[i] for i in idx[0]]
        tally = Counter(neighbours)
        category, count = tally.most_common(1)[0]
        return KNNVote(
            category=category,
            confidence=count / len(neighbours),
            tally=dict(tally),
            distances=[float(d) for d in dist[0]],
        )
