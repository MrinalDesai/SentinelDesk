"""Tests for the end-to-end multi-agent pipeline and the retrieval seam."""

import pytest

from sentineldesk.classifier import (
    DeterministicScorer, EdgeCaseResolver, KNNVoter, SVMClassifier, VocabModel, train_svm,
)
from sentineldesk.corpus import LabeledTicket
from sentineldesk.data_gen.controlled import generate_controlled_corpus
from sentineldesk.llm import StubLLMClient
from sentineldesk.pipeline import Pipeline
from sentineldesk.rag import KnowledgeGraph
from sentineldesk.retrieval import LexicalRetriever, QdrantRetriever
from sentineldesk.vocabulary.concepts import ConceptVocabulary

_RES = {c: f"Check the {c} system, apply the standard remediation, and verify recovery." for c in
        ["Network", "Database", "Storage", "Application", "Infrastructure", "Security", "Access Management"]}


def _pipe():
    base = generate_controlled_corpus(per_category=80)
    train = [LabeledTicket(text=t.text, category=t.category, title=t.title,
                           description=t.description, resolution=_RES[t.category]) for t in base]
    scorer = DeterministicScorer(VocabModel.build(train))
    svm = SVMClassifier(train_svm(train))
    knn = KNNVoter(k=5).fit(train)
    resolver = EdgeCaseResolver(scorer, svm, knn, StubLLMClient(lambda p: "Database"))
    graph = KnowledgeGraph.build(train, ConceptVocabulary.from_layman_map())
    return Pipeline(scorer, resolver, graph), train


def test_clear_ticket_auto_resolves():
    pipe, _ = _pipe()
    s = pipe.run("primary replica deadlock", "connection pool exhausted")
    assert s.category == "Database"
    assert s.outcome == "auto_resolved"
    assert s.resolution and s.reasoning


def test_safety_gate_escalates_and_bypasses():
    pipe, _ = _pipe()
    s = pipe.run("data breach in progress", "attacker exfiltrating customer records")
    assert s.outcome == "safety_escalated"
    assert s.safe is False
    assert len(s.reasoning) == 1  # bypassed every downstream stage


def test_lexical_retriever_finds_relevant():
    _, train = _pipe()
    hits = LexicalRetriever().fit(train).search("primary replica deadlock", k=1)
    assert hits and hits[0].category == "Database" and 0.0 <= hits[0].score <= 1.0


def test_qdrant_retriever_is_disabled_until_configured():
    with pytest.raises(NotImplementedError):
        QdrantRetriever()  # no client/embedder -> documents the integration point


def test_consistency_guard_escalates_on_route_resolution_mismatch():
    from sentineldesk.classifier.explainability import ReasoningChain
    from sentineldesk.pipeline import Pipeline
    pipe, _ = _pipe()
    # force a mismatch via a fake chain: routed Network, fix says Database
    import sentineldesk.pipeline.orchestrator as orch
    orig = orch.explain_decision
    orch.explain_decision = lambda *a, **k: ReasoningChain(
        category="Network", method="agreement", confidence=0.95,
        steps=["forced"], resolution="some fix", resolution_category="Database")
    try:
        s = pipe.run("x", "y")
        assert s.outcome == "escalated" and s.resolution is None
    finally:
        orch.explain_decision = orig
