"""Tests for the self-correction loop."""

from sentineldesk.learning import CorrectionStore
from sentineldesk.vocabulary.concepts import ConceptVocabulary


def _store():
    return CorrectionStore.from_vocab(ConceptVocabulary.from_layman_map())


def test_learns_phrase_and_normalizes_to_anchor():
    cv = ConceptVocabulary.from_layman_map()
    store = _store()
    r = store.record("the box is totally frozen", "Infrastructure")
    assert r.status == "learned" and r.category == "Infrastructure"
    out = store.normalize("the box is totally frozen and won't come back", cv)
    assert r.anchor in out  # the casual phrase was rewritten to the category anchor


def test_rejects_generic_phrase():
    assert _store().record("not working", "Network").status == "rejected_generic"


def test_rejects_collision():
    store = _store()
    store.record("flibber widget", "Network")
    assert store.record("flibber widget", "Database").status == "rejected_collision"


def test_min_support_holds_until_confirmed():
    store = _store()
    r1 = store.record("zorptastic outage", "Storage", min_support=2)
    assert r1.status == "pending"
    r2 = store.record("zorptastic outage", "Storage", min_support=2)
    assert r2.status == "learned"


def test_audit_log_records_learned_entries():
    store = _store()
    store.record("the box is totally frozen", "Infrastructure")
    assert len(store.audit) == 1 and store.audit[0]["to_category"] == "Infrastructure"


def test_save_load_roundtrip(tmp_path):
    store = _store()
    store.record("the box is totally frozen", "Infrastructure")
    store.save(tmp_path / "corr.json")
    fresh = _store().load(tmp_path / "corr.json")
    assert "the box is totally frozen" in fresh.learned
