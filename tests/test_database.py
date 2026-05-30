"""Tests for the vocabulary database — the single source of truth."""

from sentineldesk.data_gen.controlled import generate_controlled_corpus
from sentineldesk.vocabulary.database import (
    build_vocabulary_db,
    load_vocabulary_db,
    normalizer_from_db,
    save_vocabulary_db,
)


def _db():
    return build_vocabulary_db(generate_controlled_corpus(per_category=80))


def test_db_has_four_components_per_category():
    db = _db()
    assert len(db["categories"]) == 7
    for cat in db["categories"].values():
        assert set(cat) == {"frequent", "unique", "synonyms", "normalized"}
        assert cat["frequent"] and all(len(pair) == 2 for pair in cat["frequent"])
        assert isinstance(cat["unique"], list) and cat["unique"]
        assert isinstance(cat["synonyms"], dict)
        assert isinstance(cat["normalized"], dict)


def test_frequent_has_probabilities():
    db = _db()
    for cat in db["categories"].values():
        for term, prob in cat["frequent"]:
            assert isinstance(term, str) and 0.0 <= prob <= 1.0


def test_normalizer_maps_synonym_to_canonical():
    db = _db()
    norm = normalizer_from_db(db)
    # a known layman phrasing resolves to its canonical term
    assert norm.get("internet not working") == "dns resolution"


def test_save_load_roundtrip(tmp_path):
    db = _db()
    path = save_vocabulary_db(db, tmp_path / "vocabulary_db.json")
    assert path.exists()
    reloaded = load_vocabulary_db(path)
    assert reloaded["_meta"]["categories"] == db["_meta"]["categories"]
    assert reloaded["categories"].keys() == db["categories"].keys()
