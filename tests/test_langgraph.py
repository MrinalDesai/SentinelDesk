"""Tests for the LangGraph multi-agent graph + the consistency guard."""

import pytest

from sentineldesk.classifier import (
    DeterministicScorer, EdgeCaseResolver, KNNVoter, SVMClassifier, VocabModel, train_svm,
)
from sentineldesk.corpus import LabeledTicket
from sentineldesk.data_gen.controlled import generate_controlled_corpus
from sentineldesk.llm import StubLLMClient
from sentineldesk.rag import KnowledgeGraph
from sentineldesk.vocabulary.concepts import ConceptVocabulary

pytest.importorskip("langgraph")  # skip cleanly if langgraph absent
from sentineldesk.pipeline.langgraph_app import build_graph  # noqa: E402

_RES = {c: f"Check the {c} system, apply the standard remediation, and verify recovery." for c in
        ["Network", "Database", "Storage", "Application", "Infrastructure", "Security", "Access Management"]}


def _graph():
    base = generate_controlled_corpus(per_category=80)
    train = [LabeledTicket(text=t.text, category=t.category, title=t.title,
                           description=t.description, resolution=_RES[t.category]) for t in base]
    scorer = DeterministicScorer(VocabModel.build(train))
    svm = SVMClassifier(train_svm(train))
    knn = KNNVoter(k=5).fit(train)
    resolver = EdgeCaseResolver(scorer, svm, knn, StubLLMClient(lambda p: "Database"))
    kg = KnowledgeGraph.build(train, ConceptVocabulary.from_layman_map())
    return build_graph(scorer, resolver, kg)


def test_graph_compiles_with_expected_nodes():
    nodes = list(_graph().get_graph().nodes)
    assert "intake_safety" in nodes and "classify_route_resolve" in nodes


def test_clear_ticket_auto_resolves():
    out = _graph().invoke({"title": "primary replica deadlock",
                           "description": "connection pool exhausted", "reasoning": []})
    assert out["category"] == "Database" and out["outcome"] == "auto_resolved"


def test_safety_gate_bypasses():
    out = _graph().invoke({"title": "data breach in progress",
                           "description": "attacker exfiltrating records", "reasoning": []})
    assert out["outcome"] == "safety_escalated"
