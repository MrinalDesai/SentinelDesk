"""Tests for the minimal Graph RAG (symptom -> root cause -> resolution)."""

from sentineldesk.corpus import LabeledTicket
from sentineldesk.data_gen.controlled import generate_controlled_corpus
from sentineldesk.rag import KnowledgeGraph
from sentineldesk.vocabulary.concepts import ConceptVocabulary

_RES = {
    "Network": "Checked DNS and routing; restored the path.",
    "Database": "Added the missing index; cleared the deadlock.",
    "Storage": "Expanded the volume; remounted and verified.",
    "Application": "Patched the defect and redeployed.",
    "Infrastructure": "Restarted the host and confirmed stability.",
    "Security": "Contained the threat and applied the patch.",
    "Access Management": "Reset credentials and corrected role membership.",
}


def _graph():
    base = generate_controlled_corpus(per_category=80)
    tickets = [
        LabeledTicket(text=t.text, category=t.category, title=t.title,
                      description=t.description, resolution=_RES[t.category])
        for t in base
    ]
    return KnowledgeGraph.build(tickets, ConceptVocabulary.from_layman_map())


def test_graph_has_typed_nodes_and_edges():
    g = _graph()
    s = g.stats
    assert s["symptom_nodes"] > 0
    assert s["root_cause_nodes"] > 0
    assert s["resolution_nodes"] > 0


def test_canonical_term_traverses_to_cause_and_resolution():
    g = _graph()
    r = g.query("primary replica deadlock in the connection pool")
    assert r.root_cause == "deadlock"
    assert r.category == "Database"
    assert r.resolution and "deadlock" in r.resolution.lower()
    assert len(r.path) == 3  # symptoms -> cause -> resolution


def test_layman_symptom_maps_to_canonical_cause():
    g = _graph()
    r = g.query("the websites won't load for anyone")
    assert r.root_cause == "dns resolution"
    assert r.category == "Network"


def test_unmatched_query_does_not_guess():
    g = _graph()
    r = g.query("xyzzy nothing relevant here")
    assert r.root_cause is None
    assert r.resolution is None
    assert "no symptom matched" in r.path[0]


def test_resolution_quality_rejects_stubs_and_empties():
    from sentineldesk.rag.graph_rag import resolution_quality
    assert resolution_quality("Please follow these steps to resolve the issue:") == 0
    assert resolution_quality(None) == 0
    assert resolution_quality("ok") == 0
    assert resolution_quality(
        "Check for conflicting transactions and roll back to clear the deadlock.") >= 2


def test_best_resolution_prefers_quality_over_frequency():
    from collections import Counter
    from sentineldesk.rag.graph_rag import best_resolution
    counter = Counter({
        "Please follow these steps to resolve the issue:": 9,
        "Restart the service, verify the config, and check the logs for errors.": 2,
    })
    assert best_resolution(counter).startswith("Restart")


def test_best_resolution_returns_none_when_all_stubs():
    from collections import Counter
    from sentineldesk.rag.graph_rag import best_resolution
    assert best_resolution(Counter({"see below": 4, "tbd": 2})) is None
