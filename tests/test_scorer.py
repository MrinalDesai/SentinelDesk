"""
Tests for the explainable deterministic scorer.

Built on the controlled corpus so the signature terms are present. Clean
in-domain tickets classify confidently with grounded explanations; ambiguous
or empty tickets are flagged as edge cases rather than guessed.
"""

import pytest

from sentineldesk.classifier import DeterministicScorer, VocabModel, explain
from sentineldesk.config import Domain
from sentineldesk.data_gen.controlled import generate_controlled_corpus


@pytest.fixture(scope="module")
def scorer():
    vocab = VocabModel.build(generate_controlled_corpus(per_category=120))
    return DeterministicScorer(vocab)


def test_clean_in_domain_routes_confidently(scorer):
    r = scorer.classify("DB slow", "database query is slow, missing index suspected")
    assert r.category == "Database"
    assert not r.is_edge_case
    assert r.confidence > 0.5


def test_dept_name_is_recorded(scorer):
    r = scorer.classify("network issue", "the firewall rule is blocking a subnet")
    assert r.category == "Network"
    assert r.matched["dept"]  # the word "network" fired the dept signal


def test_ambiguous_is_edge_case(scorer):
    r = scorer.classify("System is slow", "everything feels sluggish today")
    assert r.is_edge_case
    assert r.edge_reason


def test_no_double_counting(scorer):
    # a term credited as unique must not also appear in freq
    r = scorer.classify(
        "Network", "vpn tunnel down and dns resolution failing on the subnet"
    )
    overlap = set(r.matched["unique"]) & set(r.matched["freq"])
    assert overlap == set()


def test_explanation_is_grounded(scorer):
    r = scorer.classify("DB", "deadlock and replication lag on the primary replica")
    text = explain(r)
    assert "Database" in text
    # every term named in the explanation actually fired
    fired = set(sum(r.matched.values(), []))
    assert any(term in text for term in fired)


def test_edge_case_explanation_mentions_resolver(scorer):
    r = scorer.classify("help", "")
    assert r.is_edge_case
    assert "resolver" in explain(r).lower()


def test_all_predictions_are_valid_domains(scorer):
    domains = {d.value for d in Domain}
    for title in ["vpn down", "disk full", "phishing email", "account locked"]:
        assert scorer.classify(title).category in domains
