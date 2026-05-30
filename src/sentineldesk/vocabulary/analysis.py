"""
Vocabulary analysis: turn a corpus into the N-gram and TF-IDF top-N tables
(with the numbers behind them) plus a hard exclusivity metric.

  * ngram_counts_by_category  -> {cat: [(term, frequency)]}  -- "abundant"
  * tfidf_weights_by_category  -> {cat: [(term, mean_weight)]} -- "discriminative"
  * exclusivity_report         -> per-category and overall % of top-N terms
                                  that appear in NO other category's top-N.

Exclusivity is the quantitative version of "mutually exclusive": 1.0 means a
category's entire top-N is unique to it; lower means terms are leaking across
categories (which is exactly what TF-IDF should suppress relative to N-gram).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..corpus import LabeledTicket, group_by_category
from ..vocabulary.ngram import _tokenise


def ngram_counts_by_category(
    tickets: list[LabeledTicket], top_n: int = 20, ngram_max: int = 3
) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = {}
    for category, texts in group_by_category(tickets).items():
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = _tokenise(text)
            for n in range(1, ngram_max + 1):
                for i in range(len(tokens) - n + 1):
                    counter[" ".join(tokens[i : i + n])] += 1
        ranked = [
            (term, freq)
            for term, freq in counter.most_common()
            if (" " not in term) or freq > 1
        ]
        out[category] = ranked[:top_n]
    return out


def tfidf_weights_by_category(
    tickets: list[LabeledTicket],
    top_n: int = 20,
    ngram_range: tuple[int, int] = (1, 3),
    max_features: int = 5000,
) -> dict[str, list[tuple[str, float]]]:
    texts = [t.text for t in tickets]
    labels = [t.category for t in tickets]
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    matrix = vec.fit_transform(texts)
    names = np.array(vec.get_feature_names_out())
    out: dict[str, list[tuple[str, float]]] = {}
    for category in sorted(set(labels)):
        rows = [i for i, lab in enumerate(labels) if lab == category]
        mean_w = np.asarray(matrix[rows].mean(axis=0)).ravel()
        order = mean_w.argsort()[::-1]
        out[category] = [
            (str(names[i]), float(mean_w[i])) for i in order if mean_w[i] > 0.0
        ][:top_n]
    return out


def exclusivity_report(
    top_by_category: dict[str, list[tuple]],
) -> dict[str, float]:
    """Return {category: fraction_unique} plus '_overall'.

    A term is 'unique' to a category if it does not appear in any other
    category's top-N list.
    """
    term_sets = {cat: {t[0] for t in items} for cat, items in top_by_category.items()}
    report: dict[str, float] = {}
    total_unique = 0
    total_terms = 0
    for cat, terms in term_sets.items():
        others: set[str] = set()
        for other_cat, other_terms in term_sets.items():
            if other_cat != cat:
                others |= other_terms
        unique = terms - others
        report[cat] = len(unique) / len(terms) if terms else 0.0
        total_unique += len(unique)
        total_terms += len(terms)
    report["_overall"] = total_unique / total_terms if total_terms else 0.0
    return report
