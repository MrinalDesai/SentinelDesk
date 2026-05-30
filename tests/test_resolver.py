"""Tests for the EdgeCaseResolver — all four decision paths."""

from sentineldesk.classifier.knn import KNNVote
from sentineldesk.classifier.resolver import EdgeCaseResolver
from sentineldesk.classifier.scorer import ScoreResult
from sentineldesk.llm import StubLLMClient


class _FakeScorer:
    def __init__(self, category):
        self.category = category

    def classify(self, title, description=""):
        return ScoreResult(
            category=self.category, confidence=0.5, scores={}, margin=2.0,
            is_edge_case=False, edge_reason=None, matched={"unique": ["term"]},
            runner_up=None,
        )


class _FakeSVM:
    def __init__(self, label, conf):
        self.label, self.conf = label, conf

    def predict(self, text):
        return self.label, self.conf


class _FakeKNN:
    def __init__(self, category):
        self.category = category

    def vote(self, text):
        return KNNVote(category=self.category, confidence=1.0,
                       tally={self.category: 5}, distances=[0.1] * 5)


def _resolver(svm_label, svm_conf, scorer_cat, knn_cat, llm=None, gate=0.80):
    return EdgeCaseResolver(
        _FakeScorer(scorer_cat), _FakeSVM(svm_label, svm_conf), _FakeKNN(knn_cat),
        llm=llm, confidence_gate=gate,
    )


def test_high_confidence_routes_via_svm_gate():
    r = _resolver("Database", 0.95, "Network", "Network")  # voters would disagree
    d = r.resolve("t", "d")
    assert d.category == "Database" and d.method == "confident_svm"  # gate short-circuits


def test_majority_agreement_routes():
    # low SVM conf -> ladder; scorer+knn agree on Network, svm says Database -> 2/3 Network
    r = _resolver("Database", 0.50, "Network", "Network")
    d = r.resolve("t", "d")
    assert d.category == "Network" and d.method == "agreement"


def test_three_way_split_goes_to_llm_tiebreak():
    # svm/scorer/knn all differ (Database/Application/Storage) -> LLM picks a candidate
    llm = StubLLMClient(lambda p: "Application")
    r = _resolver("Database", 0.50, "Application", "Storage", llm=llm)
    d = r.resolve("t", "d")
    assert d.method == "llm_tiebreak" and d.category == "Application"


def test_escalates_when_no_llm_and_no_majority():
    r = _resolver("Database", 0.50, "Application", "Storage", llm=None)
    d = r.resolve("t", "d")
    assert d.method == "escalated" and d.escalated is True


def test_escalates_when_llm_abstains():
    llm = StubLLMClient(lambda p: "Nonsense")  # not a candidate
    r = _resolver("Database", 0.50, "Application", "Storage", llm=llm)
    d = r.resolve("t", "d")
    assert d.escalated is True


def test_trace_is_populated():
    r = _resolver("Database", 0.50, "Network", "Network")
    d = r.resolve("t", "d")
    assert len(d.trace) >= 2 and any("ladder" in s for s in d.trace)
