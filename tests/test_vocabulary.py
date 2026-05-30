"""
Tests for the vocabulary builder (layers 1-4).

The seed corpus is mutually exclusive by design, so we can assert that
domain-defining terms land in the right category and that TF-IDF actually
discriminates (e.g. "vpn" is a Network term, not an Application one).
"""

from pathlib import Path

import pytest

from sentineldesk.corpus import group_by_category, load_tickets_csv
from sentineldesk.vocabulary import (
    StubEnricher,
    build_enrichment_prompt,
    build_full_vocabulary,
    build_ngram_vocabulary,
    build_tfidf_vocabulary,
    merge_vocabularies,
    parse_enrichment_response,
    save_vocabulary,
)

SEED = Path(__file__).resolve().parents[1] / "data" / "seed_tickets.csv"


@pytest.fixture(scope="module")
def tickets():
    return load_tickets_csv(SEED)


def test_seed_loads_seven_categories(tickets):
    cats = {t.category for t in tickets}
    assert len(cats) == 7
    assert "Access Management" in cats


def test_ngram_excludes_stopwords(tickets):
    vocab = build_ngram_vocabulary(group_by_category(tickets))
    for terms in vocab.values():
        flat = " ".join(terms).split()
        assert "the" not in flat and "is" not in flat and "to" not in flat


def test_ngram_finds_domain_terms(tickets):
    vocab = build_ngram_vocabulary(group_by_category(tickets))
    network_flat = " ".join(vocab["Network"]).lower()
    assert "vpn" in network_flat or "network" in network_flat


def test_tfidf_discriminates(tickets):
    vocab = build_tfidf_vocabulary(tickets)
    # "vpn" should be a top Network term and NOT a top Database term
    assert any("vpn" in t for t in vocab["Network"])
    assert not any("vpn" in t for t in vocab["Database"])
    # "database" signal belongs to Database, "storage" to Storage
    assert any("database" in t for t in vocab["Database"])
    assert any("storage" in t or "backup" in t or "disk" in t
               for t in vocab["Storage"])


def test_merge_is_tfidf_first_and_deduped(tickets):
    ngram = build_ngram_vocabulary(group_by_category(tickets))
    tfidf = build_tfidf_vocabulary(tickets)
    merged = merge_vocabularies(ngram, tfidf)
    for cat, terms in merged.items():
        lowered = [t.lower() for t in terms]
        assert len(lowered) == len(set(lowered)), f"dup terms in {cat}"
        # tfidf terms appear before any ngram-only term
        if tfidf[cat]:
            assert terms[0].lower() == tfidf[cat][0].lower()


def test_full_build_with_stub(tickets):
    vocab = build_full_vocabulary(tickets, enricher=StubEnricher())
    assert len(vocab) == 7
    for cat, layers in vocab.items():
        assert set(layers) == {"ngram", "tfidf", "merged", "enriched", "final"}
        # stub adds nothing, so final == merged
        assert layers["enriched"] == []
        assert layers["final"] == layers["merged"]
        assert len(layers["final"]) > 0


def test_save_vocabulary(tickets, tmp_path):
    vocab = build_full_vocabulary(tickets, enricher=StubEnricher())
    out = save_vocabulary(vocab, tmp_path / "vocabulary.json")
    assert out.exists()
    import json
    doc = json.loads(out.read_text())
    assert "_meta" in doc and "vocabulary" in doc
    assert len(doc["_meta"]["categories"]) == 7


# --- Layer 4 pure-function tests (no live Ollama) --------------------------

def test_enrichment_prompt_contains_terms():
    prompt = build_enrichment_prompt("Network", ["vpn", "dns", "firewall"])
    assert "Network" in prompt and "vpn" in prompt
    assert "JSON" in prompt


@pytest.mark.parametrize("raw,expected", [
    ('["wifi", "wireless", "cannot connect"]', ["wifi", "wireless", "cannot connect"]),
    ('```json\n["a", "b"]\n```', ["a", "b"]),
    ('Here are the terms: ["x", "y"] hope that helps', ["x", "y"]),
    ('not json at all', []),
    ('', []),
    ('{"not": "a list"}', []),
])
def test_parse_enrichment_response(raw, expected):
    assert parse_enrichment_response(raw) == expected
