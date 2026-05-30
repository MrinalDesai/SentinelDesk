"""
Vocabulary database — the single source of truth.

Built once from the corpus + layman map; read by generation (frequent/unique/
synonyms), the runtime scorer/SVM (normalized/synonyms), and the eval-set
builder. Persisted as JSON (move to SQLite/Postgres later if you want a real
query engine; the shape stays the same).

Per category:
  frequent   : [[term, probability], ...]   top-20 with scores
  unique     : [term, ...]                  TF-IDF discriminative
  synonyms   : {term: [synonym, ...]}        concept groups
  normalized : {synonym: canonical}          synonym -> canonical lookup
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..corpus import LabeledTicket
from ..data_gen.layman_map import LAYMAN_MAP
from ..data_gen.word_model import CategoryWordModel
from .concepts import ConceptVocabulary


def build_vocabulary_db(
    tickets: list[LabeledTicket],
    layman: dict[str, dict[str, list[str]]] = LAYMAN_MAP,
    top_freq: int = 20,
    n_unique: int = 5,
) -> dict:
    wm = CategoryWordModel.from_corpus(tickets, top_freq=top_freq, n_unique=n_unique)
    cv = ConceptVocabulary.from_layman_map(layman)

    categories: dict[str, dict] = {}
    for cat in wm.categories:
        freq_terms = {t for t, _ in wm.frequent[cat]}
        relevant = freq_terms | set(wm.unique[cat])
        # synonyms only for the selected terms that have them (enrich the survivors)
        syns = {
            term: list(layman.get(cat, {}).get(term, []))
            for term in relevant
            if layman.get(cat, {}).get(term)
        }
        # normalized map restricted to this category's surviving synonyms
        normalized = {
            s.lower(): term
            for term, slist in syns.items()
            for s in slist
            if cv.syn2canon.get(s.lower()) == term  # drop collisions
        }
        categories[cat] = {
            "frequent": [[t, p] for t, p in wm.frequent[cat]],
            "unique": list(wm.unique[cat]),
            "synonyms": syns,
            "normalized": normalized,
        }

    return {
        "_meta": {
            "version": 1,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "categories": wm.categories,
            "n_synonyms": sum(len(c["normalized"]) for c in categories.values()),
        },
        "categories": categories,
    }


def save_vocabulary_db(db: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(db, indent=2), encoding="utf-8")
    return path


def load_vocabulary_db(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalizer_from_db(db: dict) -> dict[str, str]:
    """Flatten the per-category normalized maps into one synonym->canonical lookup."""
    out: dict[str, str] = {}
    for cat in db["categories"].values():
        out.update(cat["normalized"])
    return out
