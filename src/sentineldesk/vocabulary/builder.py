"""
Vocabulary builder — orchestrates the four layers into vocabulary.json.

build_full_vocabulary(tickets, enricher) ->
    { category: {ngram, tfidf, merged, enriched, final}, ... }

The `final` list is what the runtime VGAC prompt uses. Keeping the
intermediate layers in the output mirrors the VOCABULARY entity in the ERD
(ngram_terms / tfidf_terms / enriched_terms) and makes the ablation study in
Round 2 Section 9.2 a matter of reading different keys, not re-running.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..corpus import LabeledTicket, group_by_category
from .enricher import StubEnricher, VocabularyEnricher
from .merge import _dedup_preserve, merge_vocabularies
from .ngram import build_ngram_vocabulary
from .tfidf import build_tfidf_vocabulary


def build_full_vocabulary(
    tickets: list[LabeledTicket],
    enricher: VocabularyEnricher | None = None,
    ngram_top_n: int = 20,
    tfidf_top_n: int = 15,
    enrich_top_k: int = 8,
) -> dict[str, dict]:
    """Run layers 1-4 and return the structured per-category vocabulary."""
    enricher = enricher or StubEnricher()

    ngram_vocab = build_ngram_vocabulary(
        group_by_category(tickets), top_n=ngram_top_n
    )
    tfidf_vocab = build_tfidf_vocabulary(tickets, top_n=tfidf_top_n)
    merged = merge_vocabularies(ngram_vocab, tfidf_vocab)

    out: dict[str, dict] = {}
    for category in sorted(merged):
        merged_terms = merged[category]
        enriched_terms = enricher.enrich(category, merged_terms[:enrich_top_k])
        final = _dedup_preserve(merged_terms + enriched_terms)
        out[category] = {
            "ngram": ngram_vocab.get(category, []),
            "tfidf": tfidf_vocab.get(category, []),
            "merged": merged_terms,
            "enriched": enriched_terms,
            "final": final,
        }
    return out


def save_vocabulary(vocab: dict[str, dict], path: str | Path) -> Path:
    """Write vocabulary.json with a small metadata header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "_meta": {
            "version": 1,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "categories": sorted(vocab.keys()),
            "term_counts": {c: len(v["final"]) for c, v in vocab.items()},
        },
        "vocabulary": vocab,
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path
