"""
Layer 3 — Vocabulary merge (Round 2 Algo 3).

TF-IDF (discriminative) terms come first because they carry the most
classification signal; N-gram terms not already present are appended to widen
coverage. Dedup is case-insensitive but preserves the first-seen surface form.
"""

from __future__ import annotations


def _dedup_preserve(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        key = term.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(term.strip())
    return out


def merge_vocabularies(
    ngram_vocab: dict[str, list[str]],
    tfidf_vocab: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return {category: merged term list} — TF-IDF first, then new N-gram terms."""
    categories = set(ngram_vocab) | set(tfidf_vocab)
    merged: dict[str, list[str]] = {}
    for category in categories:
        combined = list(tfidf_vocab.get(category, [])) + list(
            ngram_vocab.get(category, [])
        )
        merged[category] = _dedup_preserve(combined)
    return merged
