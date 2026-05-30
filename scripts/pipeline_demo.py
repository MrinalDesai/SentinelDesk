#!/usr/bin/env python3
"""End-to-end pipeline demo: raw ticket -> safety -> classify/route -> resolve -> judge."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentineldesk.classifier import (DeterministicScorer, EdgeCaseResolver, KNNVoter,
                                     SVMClassifier, VocabModel, load_model, train_svm)
from sentineldesk.corpus import LabeledTicket
from sentineldesk.data_gen.controlled import generate_controlled_corpus
from sentineldesk.pipeline import Pipeline
from sentineldesk.rag import KnowledgeGraph
from sentineldesk.vocabulary.concepts import ConceptVocabulary

_RES = {c: f"Standard {c} remediation steps." for c in
        ["Network","Database","Storage","Application","Infrastructure","Security","Access Management"]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="data/svm_model.pkl")
    ap.add_argument("--ollama", action="store_true"); a = ap.parse_args()
    base = generate_controlled_corpus(per_category=120)
    train = [LabeledTicket(text=t.text, category=t.category, title=t.title,
             description=t.description, resolution=_RES[t.category]) for t in base]
    scorer = DeterministicScorer(VocabModel.build(train))
    svm = SVMClassifier(load_model(a.model)) if Path(a.model).exists() else SVMClassifier(train_svm(train))
    knn = KNNVoter(k=5).fit(train)
    if a.ollama:
        from sentineldesk.llm import OllamaClient; llm = OllamaClient(model="mistral:7b-instruct-q8_0")
    else:
        from sentineldesk.llm import StubLLMClient; llm = StubLLMClient(lambda p: "Database")
    resolver = EdgeCaseResolver(scorer, svm, knn, llm=llm)
    graph = KnowledgeGraph.build(train, ConceptVocabulary.from_layman_map())
    pipe = Pipeline(scorer, resolver, graph)
    for title, desc in [("primary replica deadlock","connection pool exhausted overnight"),
                        ("websites won't load","users can't reach the site"),
                        ("data breach in progress","attacker exfiltrating customer records now")]:
        s = pipe.run(title, desc)
        print(f"\n===== {title!r} =====")
        print(f"  -> {s.category}  ({s.confidence:.0%}, {s.method})   OUTCOME: {s.outcome.upper()}")
        if s.resolution: print(f"  resolution: {s.resolution}")
        for step in s.reasoning: print(f"    - {step}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
