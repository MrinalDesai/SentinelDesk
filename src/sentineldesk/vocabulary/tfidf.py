"""
Layer 2 — TF-IDF discriminative filtering (Round 2 Algo 2).

N-gram frequency tells you what a domain talks about; TF-IDF tells you what it
talks about that *other domains do not*. We fit one vectoriser over the whole
corpus, then for each category rank terms by their mean TF-IDF within that
category's tickets. Terms common to every domain get low IDF and fall away;
domain-unique terms rise to the top.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..corpus import LabeledTicket


def build_tfidf_vocabulary(
    tickets: list[LabeledTicket],
    top_n: int = 15,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 5000,
) -> dict[str, list[str]]:
    """Return {category: [top-N discriminative terms by mean TF-IDF]}."""
    texts = [t.text for t in tickets]
    labels = [t.category for t in tickets]

    vectoriser = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # alpha tokens, len>=2
    )
    matrix = vectoriser.fit_transform(texts)
    feature_names = np.array(vectoriser.get_feature_names_out())

    result: dict[str, list[str]] = {}
    for category in sorted(set(labels)):
        rows = [i for i, lab in enumerate(labels) if lab == category]
        if not rows:
            result[category] = []
            continue
        # mean TF-IDF of each feature across this category's documents
        mean_weights = np.asarray(matrix[rows].mean(axis=0)).ravel()
        top_idx = mean_weights.argsort()[::-1]
        terms = [
            feature_names[i]
            for i in top_idx
            if mean_weights[i] > 0.0
        ][:top_n]
        result[category] = terms
    return result
