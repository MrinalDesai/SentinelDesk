#!/usr/bin/env python3
"""
Explainability demo — print the unified reasoning chain for sample tickets.

Runs the full pipeline (scorer + resolver + Graph RAG) and shows, for each
ticket, ONE human-readable chain: the lexical signal that fired, the routing
decision and why, the voters, and the symptom->cause->resolution traversal.
Every line is grounded in a real matched term, voter, or graph edge.

    python scripts/explain_demo.py
    python scripts/explain_demo.py --query "the websites won't load for everyone"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import (                                       # noqa: E402
    DeterministicScorer, EdgeCaseResolver, KNNVoter, SVMClassifier, VocabModel, load_model, train_svm,
)
from sentineldesk.classifier.explainability import explain_decision         # noqa: E402
from sentineldesk.corpus import LabeledTicket                               # noqa: E402
from sentineldesk.data_gen.controlled import generate_controlled_corpus     # noqa: E402
from sentineldesk.rag import KnowledgeGraph                                 # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary              # noqa: E402

_RES = {
    "Network": "Checked DNS, firewall and routing; restored the path.",
    "Database": "Cleared the deadlock and added the missing index.",
    "Storage": "Expanded the volume; remounted and verified.",
    "Application": "Patched the defect and redeployed the service.",
    "Infrastructure": "Restarted the affected host and confirmed stability.",
    "Security": "Contained the threat and applied the patch.",
    "Access Management": "Reset credentials and corrected role membership.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/svm_model.pkl")
    ap.add_argument("--per-category", type=int, default=120)
    ap.add_argument("--ollama", action="store_true", help="use Ollama for the LLM tiebreak")
    ap.add_argument("--query", default=None)
    args = ap.parse_args()

    base = generate_controlled_corpus(per_category=args.per_category)
    train = [LabeledTicket(text=t.text, category=t.category, title=t.title,
                           description=t.description, resolution=_RES[t.category]) for t in base]
    scorer = DeterministicScorer(VocabModel.build(train))
    svm = SVMClassifier(load_model(args.model)) if Path(args.model).exists() \
        else SVMClassifier(train_svm(train))
    knn = KNNVoter(k=5).fit(train)
    llm = None
    if args.ollama:
        from sentineldesk.llm import OllamaClient
        llm = OllamaClient(model="mistral:7b-instruct-q8_0")
    else:
        from sentineldesk.llm import StubLLMClient
        llm = StubLLMClient(lambda p: "Database")
    resolver = EdgeCaseResolver(scorer, svm, knn, llm=llm, confidence_gate=0.80)
    graph = KnowledgeGraph.build(train, ConceptVocabulary.from_layman_map())

    if args.query:
        tickets = [("", args.query)]
    else:
        tickets = [
            ("primary replica deadlock", "schema migration stalled, connection pool exhausted"),
            ("websites won't load", "users can't reach the site across the office"),
            ("ransomware detected", "files encrypted on a finance workstation"),
        ]

    for title, desc in tickets:
        chain = explain_decision(title, desc, scorer, resolver, graph)
        label = title or desc
        print(f"\n========== ticket: {label!r} ==========")
        print(chain.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
