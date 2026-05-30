"""
Per-category word model for term-seeded generation.

Each category carries:
  * frequent  — its top-20 terms with probability weights (these recur often),
  * unique    — its discriminative TF-IDF terms, kept separate (rarely shared).

A ticket is seeded by SAMPLING a small subset (weighted by probability), so
frequency emerges across the corpus while individual tickets stay distinct.
The default model is derived from the validated controlled corpus; authored
lists can be passed instead.
"""

from __future__ import annotations

import random
import re

from ..corpus import LabeledTicket
from ..vocabulary.analysis import (
    ngram_counts_by_category,
    tfidf_weights_by_category,
)


def _word_present(term: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


def _drop_subgrams(terms: list[str]) -> list[str]:
    """Drop a unigram if it's a word inside a longer multiword term we keep.

    e.g. ['saml assertion','saml','assertion','active','active directory']
      -> ['saml assertion','active directory']  (specific, non-ambiguous)
    """
    multi = [t for t in terms if " " in t]
    multi_words = {w for t in multi for w in t.split()}
    kept = list(multi)
    for t in terms:
        if " " not in t and t not in multi_words:
            kept.append(t)
    # preserve original order, dedup
    seen: set[str] = set()
    out = []
    for t in terms:
        if t in kept and t not in seen:
            seen.add(t)
            out.append(t)
    return out


class CategoryWordModel:
    def __init__(
        self,
        frequent: dict[str, list[tuple[str, float]]],
        unique: dict[str, list[str]],
    ) -> None:
        self.frequent = frequent          # cat -> [(term, probability)]
        self.unique = unique              # cat -> [discriminative terms]
        self.categories = sorted(frequent)
        # forbidden[c] = every OTHER category's unique terms (exclusivity gate)
        self.forbidden = {
            c: {t for oc in self.categories if oc != c for t in unique.get(oc, [])}
            for c in self.categories
        }

    @classmethod
    def from_corpus(
        cls, tickets: list[LabeledTicket], top_freq: int = 20, n_unique: int = 5
    ) -> "CategoryWordModel":
        ngram = ngram_counts_by_category(tickets, top_n=top_freq)
        tfidf = tfidf_weights_by_category(tickets, top_n=top_freq)

        # COMMON terms (appear in >=2 categories' frequent lists) are not unique
        seen: dict[str, int] = {}
        for terms in ngram.values():
            for t, _ in terms:
                seen[t] = seen.get(t, 0) + 1
        common = {t for t, c in seen.items() if c >= 2}

        frequent: dict[str, list[tuple[str, float]]] = {}
        for c, terms in ngram.items():
            total = sum(f for _, f in terms) or 1
            frequent[c] = [(t, round(f / total, 4)) for t, f in terms]

        unique: dict[str, list[str]] = {}
        for c, terms in tfidf.items():
            uniq = [t for t, _ in terms if t not in common]
            unique[c] = _drop_subgrams(uniq)[:n_unique]
        return cls(frequent, unique)

    def sample_terms(
        self,
        category: str,
        rng: random.Random,
        n_freq: int = 3,
        n_unique: int = 1,
    ) -> list[str]:
        """Pick a weighted subset of frequent terms plus a unique term or two."""
        freq = self.frequent.get(category, [])
        chosen: list[str] = []
        if freq:
            terms = [t for t, _ in freq]
            weights = [w for _, w in freq]
            k = min(n_freq, len(terms))
            # weighted sample without replacement
            pool = list(zip(terms, weights))
            for _ in range(k):
                total = sum(w for _, w in pool) or 1
                r = rng.uniform(0, total)
                acc = 0.0
                for i, (t, w) in enumerate(pool):
                    acc += w
                    if r <= acc:
                        chosen.append(t)
                        pool.pop(i)
                        break
        uniq = self.unique.get(category, [])
        if uniq and n_unique > 0:
            chosen += rng.sample(uniq, min(n_unique, len(uniq)))
        return chosen

    def passes_exclusivity(self, category: str, text: str) -> bool:
        """Gate: text must contain >=1 own term and NONE of other cats' unique terms.

        Word-boundary matching so 'san' doesn't match 'thousands' and short
        terms don't trip on coincidental substrings.
        """
        low = text.lower()
        own = [t for t, _ in self.frequent.get(category, [])] + self.unique.get(category, [])
        has_own = any(_word_present(t, low) for t in own)
        has_forbidden = any(_word_present(f, low) for f in self.forbidden.get(category, set()))
        return has_own and not has_forbidden
