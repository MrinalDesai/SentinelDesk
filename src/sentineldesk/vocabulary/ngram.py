"""
Layer 1 — N-gram frequency extraction (Round 2 Algo 1).

Per category: tokenise all tickets, drop stopwords and non-alphabetic tokens,
count 1/2/3-grams, keep the top-N most frequent. This captures the raw
vocabulary of a domain ("vpn", "active directory", "connection pool") before
TF-IDF prunes it down to the discriminative terms.
"""

from __future__ import annotations

from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

_STOPWORDS: set[str] | None = None


def _ensure_nltk() -> set[str]:
    """Lazily load NLTK data so importing this module never triggers a download."""
    global _STOPWORDS
    if _STOPWORDS is None:
        for pkg, path in (("stopwords", "corpora/stopwords"),
                          ("punkt", "tokenizers/punkt"),
                          ("punkt_tab", "tokenizers/punkt_tab")):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def _tokenise(text: str) -> list[str]:
    stop = _ensure_nltk()
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and len(t) >= 2 and t not in stop]


def build_ngram_vocabulary(
    texts_by_category: dict[str, list[str]],
    top_n: int = 20,
    ngram_max: int = 3,
) -> dict[str, list[str]]:
    """Return {category: [top-N n-gram terms by frequency]}.

    Multi-word grams are space-joined ("active directory"). A 2- or 3-gram is
    only kept if it occurs more than once, so single mentions don't crowd out
    genuinely frequent unigrams.
    """
    result: dict[str, list[str]] = {}
    for category, texts in texts_by_category.items():
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = _tokenise(text)
            for n in range(1, ngram_max + 1):
                for gram in ngrams(tokens, n):
                    counter[" ".join(gram)] += 1
        # keep frequent terms; require freq>1 for multi-word grams
        ranked = [
            term
            for term, freq in counter.most_common()
            if (" " not in term) or freq > 1
        ]
        result[category] = ranked[:top_n]
    return result
