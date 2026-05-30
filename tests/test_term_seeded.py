"""Tests for term-seeded (vocabulary-first) generation."""

import re

from sentineldesk.data_gen import CategoryWordModel, TermSeededGenerator
from sentineldesk.data_gen.controlled import generate_controlled_corpus
from sentineldesk.llm import StubLLMClient


def make_model():
    return CategoryWordModel.from_corpus(generate_controlled_corpus(per_category=120))


def seeded_handler(prompt: str) -> str:
    # extract the seeded terms and the category; write a ticket that uses them
    cat = re.search(r"for the (.+?) domain", prompt)
    category = cat.group(1) if cat else "X"
    terms_m = re.search(r"these terms[^:]*:\s*(.+?)\.\n", prompt)
    terms = terms_m.group(1) if terms_m else ""
    import json
    return json.dumps({
        "title": f"Problem report involving {terms.split(',')[0].strip()}",
        "description": f"User reports a problem: {terms} are involved and failing.",
        "resolution": "investigate and remediate",
    })


def test_word_model_has_frequent_and_unique():
    m = make_model()
    assert len(m.categories) == 7
    for c in m.categories:
        assert m.frequent[c]                      # weighted frequent terms
        assert all(0 <= w <= 1 for _, w in m.frequent[c])
        assert isinstance(m.unique[c], list)


def test_sampler_returns_subset_not_all():
    m = make_model()
    import random
    terms = m.sample_terms("Network", random.Random(1), n_freq=3, n_unique=1)
    assert 1 <= len(terms) <= 4         # a subset, never the whole list of 20


def test_exclusivity_gate():
    m = make_model()
    # a Database ticket containing only Database terms passes
    assert m.passes_exclusivity("Database", "the deadlock and replication lag are bad")
    # one containing another category's unique term fails
    forbidden = next(iter(m.forbidden["Database"]))
    assert not m.passes_exclusivity("Database", f"deadlock plus {forbidden} present")


def test_generate_produces_labeled_balanced_tickets():
    m = make_model()
    gen = TermSeededGenerator(StubLLMClient(seeded_handler), m, seed=7)
    tickets = gen.generate(total=70)   # 10 per category
    assert len(tickets) > 0
    cats = {t["category"] for t in tickets}
    assert cats == set(m.categories)
    for t in tickets:
        assert t["title"] and t["description"] and t["category"]
        assert "seed_terms" in t


def test_resume_skips_complete_and_tops_up_partial():
    """Resuming from a partial corpus must skip full categories and only fill gaps."""
    from collections import Counter

    from sentineldesk.data_gen import CategoryWordModel, TermSeededGenerator
    from sentineldesk.data_gen.controlled import generate_controlled_corpus
    from sentineldesk.llm import StubLLMClient

    model = CategoryWordModel.from_corpus(generate_controlled_corpus(per_category=60))
    gen = TermSeededGenerator(StubLLMClient(lambda p: ""), model)
    gen._one = lambda category, **kw: {  # type: ignore[method-assign]
        "title": "t", "description": "d", "category": category, "resolution": "r",
    }
    per_cat = 14 // len(model.categories)
    existing = (
        [{"category": "Network", "title": "e", "description": "e", "resolution": ""}] * per_cat
        + [{"category": "Database", "title": "e", "description": "e", "resolution": ""}] * 1
    )
    out = gen.generate(total=14, existing=existing)
    counts = Counter(t["category"] for t in out)
    assert all(counts[c] == per_cat for c in model.categories)
    assert len(out) == per_cat * len(model.categories)
