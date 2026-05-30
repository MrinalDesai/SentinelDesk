"""Tests for the unified explainability chain."""

from sentineldesk.classifier import (
    DeterministicScorer, EdgeCaseResolver, KNNVoter, SVMClassifier, VocabModel, train_svm,
)
from sentineldesk.classifier.explainability import explain_decision
from sentineldesk.corpus import LabeledTicket
from sentineldesk.data_gen.controlled import generate_controlled_corpus
from sentineldesk.llm import StubLLMClient
from sentineldesk.rag import KnowledgeGraph
from sentineldesk.vocabulary.concepts import ConceptVocabulary

_RES = {c: f"Check the {c} system, apply the standard remediation, and verify recovery." for c in
        ["Network", "Database", "Storage", "Application", "Infrastructure", "Security", "Access Management"]}


def _setup():
    base = generate_controlled_corpus(per_category=80)
    train = [LabeledTicket(text=t.text, category=t.category, title=t.title,
                           description=t.description, resolution=_RES[t.category]) for t in base]
    scorer = DeterministicScorer(VocabModel.build(train))
    svm = SVMClassifier(train_svm(train))
    knn = KNNVoter(k=5).fit(train)
    resolver = EdgeCaseResolver(scorer, svm, knn, StubLLMClient(lambda p: "Database"))
    graph = KnowledgeGraph.build(train, ConceptVocabulary.from_layman_map())
    return scorer, resolver, graph


def test_chain_has_route_reasoning_and_resolution():
    scorer, resolver, graph = _setup()
    chain = explain_decision("primary replica deadlock",
                             "schema migration stalled, connection pool exhausted",
                             scorer, resolver, graph)
    assert chain.category == "Database"
    assert any("lexical signal" in s for s in chain.steps)
    assert any("routing" in s for s in chain.steps)
    assert chain.resolution is not None


def test_render_is_human_readable():
    scorer, resolver, graph = _setup()
    text = explain_decision("primary replica deadlock", "connection pool exhausted",
                            scorer, resolver, graph).render()
    assert "ROUTE ->" in text and "RESOLUTION:" in text


def test_works_without_graph():
    scorer, resolver, _ = _setup()
    chain = explain_decision("primary replica deadlock", "connection pool exhausted",
                             scorer, resolver, graph=None)
    assert chain.category == "Database" and chain.resolution is None
