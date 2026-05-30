#!/usr/bin/env python3
"""
Full-system smoke test — exercises every component SentinelDesk has, end to end,
and prints a pass/fail line per check plus a summary. This is a functional check
(does every piece run and produce sane output), complementary to the unit suite
(`pytest -q`, which checks correctness in detail).

    python scripts/check_everything.py
    python scripts/check_everything.py --ollama   # also exercise the live LLM tiebreak

Exit code 0 if all checks pass, 1 otherwise.
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RESULTS: list[tuple[str, bool, str]] = []
_CHECKS: list = []


def check(name):
    def deco(fn):
        def wrapped(*a, **k):
            try:
                detail = fn(*a, **k)
                RESULTS.append((name, True, detail or ""))
            except Exception as exc:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                if "-v" in sys.argv:
                    traceback.print_exc()
        _CHECKS.append(wrapped)
        return wrapped
    return deco


def build_corpus():
    from sentineldesk.corpus import LabeledTicket
    from sentineldesk.data_gen.controlled import generate_controlled_corpus
    res = {c: f"Standard {c} remediation steps." for c in
           ["Network", "Database", "Storage", "Application", "Infrastructure", "Security", "Access Management"]}
    base = generate_controlled_corpus(per_category=100)
    return [LabeledTicket(text=t.text, category=t.category, title=t.title,
                          description=t.description, resolution=res[t.category]) for t in base]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", action="store_true")
    args = ap.parse_args()

    train = build_corpus()

    @check("1. data generation (controlled corpus)")
    def _():
        assert len(train) == 700
        return f"{len(train)} tickets, 7 categories"

    @check("2. vocabulary database (freq+unique+synonyms+normalized)")
    def _():
        from sentineldesk.vocabulary.database import build_vocabulary_db, normalizer_from_db
        db = build_vocabulary_db(train, top_freq=5, n_unique=5)
        n = len(normalizer_from_db(db))
        assert db["categories"] and n > 0
        return f"{len(db['categories'])} categories, {n} synonyms in normalizer"

    @check("3. concept vocabulary + synonym normalization")
    def _():
        from sentineldesk.vocabulary.concepts import ConceptVocabulary
        cv = ConceptVocabulary.from_layman_map()
        out = cv.normalize("the websites won't load")
        assert "dns resolution" in out
        return f"{cv.n_synonyms} synonyms; 'websites won't load' -> 'dns resolution'"

    @check("4. safety layer (Stage 0)")
    def _():
        from sentineldesk.safety.safety_layer import safety_check
        clear = safety_check("printer jam", "the printer is stuck")
        breach = safety_check("data breach in progress", "attacker exfiltrating records")
        assert clear.bypass_llm is False and breach.bypass_llm is True
        return f"clear ticket passes; breach escalates to {breach.department}"

    @check("5. SVM classifier + load trained model")
    def _():
        from sentineldesk.classifier import SVMClassifier, load_model, train_svm
        model_path = Path("data/svm_model.pkl")
        clf = SVMClassifier(load_model(model_path)) if model_path.exists() \
            else SVMClassifier(train_svm(train))
        label, conf = clf.predict("primary replica deadlock in the connection pool")
        assert label == "Database"
        return f"'deadlock' -> {label} ({conf:.0%})" + ("" if model_path.exists() else " [trained fresh]")

    @check("6. deterministic scorer + explanation")
    def _():
        from sentineldesk.classifier import DeterministicScorer, VocabModel, explain
        scorer = DeterministicScorer(VocabModel.build(train))
        r = scorer.classify("primary replica deadlock", "connection pool exhausted")
        text = explain(r)
        assert r.category == "Database" and len(text) > 10
        return f"-> {r.category}; explanation grounded in matched terms"

    @check("7. kNN voter")
    def _():
        from sentineldesk.classifier import KNNVoter
        v = KNNVoter(k=5).fit(train).vote("primary replica deadlock")
        assert v.category == "Database"
        return f"5-NN vote -> {v.category} ({v.confidence:.0%} agree)"

    @check("8. edge-case resolver (agent loop)")
    def _():
        from sentineldesk.classifier import (DeterministicScorer, EdgeCaseResolver,
                                             KNNVoter, SVMClassifier, VocabModel, train_svm)
        from sentineldesk.llm import StubLLMClient
        scorer = DeterministicScorer(VocabModel.build(train))
        svm = SVMClassifier(train_svm(train))
        knn = KNNVoter(k=5).fit(train)
        r = EdgeCaseResolver(scorer, svm, knn, StubLLMClient(lambda p: "Database"))
        d = r.resolve("primary replica deadlock", "connection pool exhausted")
        assert d.category == "Database" and d.trace
        return f"-> {d.category} via {d.method} ({len(d.trace)} trace steps)"

    @check("9. Graph RAG (symptom->cause->resolution)")
    def _():
        from sentineldesk.rag import KnowledgeGraph
        from sentineldesk.vocabulary.concepts import ConceptVocabulary
        g = KnowledgeGraph.build(train, ConceptVocabulary.from_layman_map())
        gr = g.query("primary replica deadlock")
        assert gr.root_cause == "deadlock" and gr.resolution
        return f"'deadlock' -> {gr.category}; {g.stats['root_cause_nodes']} cause nodes"

    @check("10. self-correction loop")
    def _():
        from sentineldesk.learning import CorrectionStore
        from sentineldesk.vocabulary.concepts import ConceptVocabulary
        cv = ConceptVocabulary.from_layman_map()
        store = CorrectionStore.from_vocab(cv)
        res = store.record("the box is totally frozen", "Infrastructure")
        assert res.status == "learned"
        assert store.record("not working", "Network").status == "rejected_generic"
        return "learns phrase; rejects generic (guardrail works)"

    @check("11. explainability (unified reasoning chain)")
    def _():
        from sentineldesk.classifier import (DeterministicScorer, EdgeCaseResolver,
                                             KNNVoter, SVMClassifier, VocabModel, train_svm)
        from sentineldesk.classifier.explainability import explain_decision
        from sentineldesk.llm import StubLLMClient
        from sentineldesk.rag import KnowledgeGraph
        from sentineldesk.vocabulary.concepts import ConceptVocabulary
        scorer = DeterministicScorer(VocabModel.build(train))
        svm = SVMClassifier(train_svm(train))
        knn = KNNVoter(k=5).fit(train)
        resolver = EdgeCaseResolver(scorer, svm, knn, StubLLMClient(lambda p: "Database"))
        kg = KnowledgeGraph.build(train, ConceptVocabulary.from_layman_map())
        chain = explain_decision("primary replica deadlock", "connection pool exhausted",
                                 scorer, resolver, kg)
        assert "ROUTE ->" in chain.render() and chain.resolution
        return f"chain renders with {len(chain.steps)} steps + resolution"

    @check("12. retrieval seam (lexical default + Qdrant adapter)")
    def _():
        from sentineldesk.retrieval import LexicalRetriever, QdrantRetriever
        hit = LexicalRetriever().fit(train).search("primary replica deadlock", k=1)[0]
        assert hit.category == "Database"
        try:
            QdrantRetriever()
            disabled = False
        except NotImplementedError:
            disabled = True
        assert disabled
        return "lexical works; Qdrant adapter present + cleanly disabled"

    @check("13. end-to-end pipeline (dependency-free)")
    def _():
        from sentineldesk.classifier import (DeterministicScorer, EdgeCaseResolver,
                                             KNNVoter, SVMClassifier, VocabModel, train_svm)
        from sentineldesk.llm import StubLLMClient
        from sentineldesk.pipeline import Pipeline
        from sentineldesk.rag import KnowledgeGraph
        from sentineldesk.vocabulary.concepts import ConceptVocabulary
        scorer = DeterministicScorer(VocabModel.build(train))
        svm = SVMClassifier(train_svm(train))
        knn = KNNVoter(k=5).fit(train)
        resolver = EdgeCaseResolver(scorer, svm, knn, StubLLMClient(lambda p: "Database"))
        kg = KnowledgeGraph.build(train, ConceptVocabulary.from_layman_map())
        pipe = Pipeline(scorer, resolver, kg)
        ok = pipe.run("primary replica deadlock", "connection pool exhausted")
        breach = pipe.run("data breach in progress", "attacker exfiltrating records")
        assert ok.outcome == "auto_resolved" and breach.outcome == "safety_escalated"
        return f"clear->{ok.outcome}; breach->{breach.outcome}"

    @check("14. LangGraph multi-agent graph")
    def _():
        try:
            import langgraph  # noqa: F401
        except ImportError:
            return "SKIPPED (langgraph not installed: pip install langgraph)"
        from sentineldesk.classifier import (DeterministicScorer, EdgeCaseResolver,
                                             KNNVoter, SVMClassifier, VocabModel, train_svm)
        from sentineldesk.llm import StubLLMClient
        from sentineldesk.pipeline.langgraph_app import build_graph
        from sentineldesk.rag import KnowledgeGraph
        from sentineldesk.vocabulary.concepts import ConceptVocabulary
        scorer = DeterministicScorer(VocabModel.build(train))
        svm = SVMClassifier(train_svm(train))
        knn = KNNVoter(k=5).fit(train)
        resolver = EdgeCaseResolver(scorer, svm, knn, StubLLMClient(lambda p: "Database"))
        kg = KnowledgeGraph.build(train, ConceptVocabulary.from_layman_map())
        g = build_graph(scorer, resolver, kg)
        out = g.invoke({"title": "primary replica deadlock",
                        "description": "connection pool exhausted", "reasoning": []})
        assert out["outcome"] == "auto_resolved"
        return f"compiled graph runs; clear ticket -> {out['outcome']}"

    @check("15. consistency guard (route vs resolution)")
    def _():
        from sentineldesk.classifier.explainability import ReasoningChain
        from sentineldesk.classifier import (DeterministicScorer, EdgeCaseResolver,
                                             KNNVoter, SVMClassifier, VocabModel, train_svm)
        from sentineldesk.llm import StubLLMClient
        from sentineldesk.pipeline import Pipeline
        import sentineldesk.pipeline.orchestrator as orch
        scorer = DeterministicScorer(VocabModel.build(train))
        svm = SVMClassifier(train_svm(train))
        knn = KNNVoter(k=5).fit(train)
        pipe = Pipeline(scorer, EdgeCaseResolver(scorer, svm, knn, StubLLMClient(lambda p: "Database")), None)
        orig = orch.explain_decision
        orch.explain_decision = lambda *a, **k: ReasoningChain(
            category="Network", method="agreement", confidence=0.95, steps=["forced"],
            resolution="fix", resolution_category="Database")
        try:
            s = pipe.run("x", "y")
            assert s.outcome == "escalated" and s.resolution is None
        finally:
            orch.explain_decision = orig
        return "mismatch (route=Network, fix=Database) -> escalated, no auto-resolve"

    @check("16. LLM tiebreak via Ollama")
    def _():
        if not args.ollama:
            return "SKIPPED (pass --ollama to exercise the live LLM)"
        from sentineldesk.llm import OllamaClient
        resp = OllamaClient(model="mistral:7b-instruct-q8_0").complete("Reply with one word: Database")
        assert resp.strip()
        return f"Ollama responded ({len(resp)} chars)"

    # run all checks in registration order
    for fn in _CHECKS:
        fn()

    print("\n" + "=" * 70)
    print("SentinelDesk full-system check")
    print("=" * 70)
    passed = 0
    for name, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        if "SKIPPED" in detail:
            mark = "SKIP"
        else:
            passed += ok
        print(f"  [{mark}] {name}")
        if detail:
            print(f"         {detail}")
    total = sum(1 for _, _, d in RESULTS if "SKIPPED" not in d)
    print("=" * 70)
    print(f"  {passed}/{total} checks passed"
          + (f", {sum('SKIPPED' in d for _,_,d in RESULTS)} skipped" if any('SKIPPED' in d for _,_,d in RESULTS) else ""))
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
