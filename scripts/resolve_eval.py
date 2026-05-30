#!/usr/bin/env python3
"""
Run the edge-case resolver over a corpus and report how the two-tier agent behaves.

    python scripts/resolve_eval.py --in data/real_3000.csv
    python scripts/resolve_eval.py --in data/real_3000.csv --ollama   # enable LLM tiebreak
    python scripts/resolve_eval.py --demo                             # a few traced examples

Reports the routing-tier distribution (how many tickets the SVM handled confidently
vs. how many needed the ladder), and accuracy WITHIN each tier — which is the story:
the cheap path carries the load at high accuracy, the ladder catches the hard tail.
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import (                                    # noqa: E402
    DeterministicScorer, EdgeCaseResolver, KNNVoter, SVMClassifier, VocabModel, load_model, train_svm,
)
from sentineldesk.corpus import load_tickets_csv                          # noqa: E402
from sentineldesk.data_gen.controlled import generate_controlled_corpus   # noqa: E402


def build_resolver(train, model_path, use_ollama, gate):
    scorer = DeterministicScorer(VocabModel.build(train))
    svm = SVMClassifier(load_model(model_path)) if Path(model_path).exists() \
        else SVMClassifier(train_svm(train))
    knn = KNNVoter(k=5).fit(train)
    llm = None
    if use_ollama:
        from sentineldesk.llm import OllamaClient
        llm = OllamaClient(model="mistral:7b-instruct-q8_0")
    return EdgeCaseResolver(scorer, svm, knn, llm=llm, confidence_gate=gate)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=None)
    ap.add_argument("--model", default="data/svm_model.pkl")
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--gate", type=float, default=0.80)
    ap.add_argument("--ollama", action="store_true", help="enable LLM tiebreak (needs Ollama)")
    ap.add_argument("--demo", action="store_true", help="print traced example decisions")
    args = ap.parse_args()

    train = generate_controlled_corpus(per_category=args.per_category)
    resolver = build_resolver(train, args.model, args.ollama, args.gate)

    if args.demo or not args.infile:
        examples = [
            ("primary replica deadlock", "schema migration stalled, connection pool exhausted"),
            ("user account access", "cannot get in"),
            ("application build pipeline failing", "users cannot reach the service endpoint"),
        ]
        print("=== traced example decisions ===")
        for title, desc in examples:
            d = resolver.resolve(title, desc)
            print(f"\n[{title}] -> {d.category}  ({d.method}, conf {d.confidence:.2f}"
                  f"{', ESCALATED' if d.escalated else ''})")
            for step in d.trace:
                print(f"    {step}")
        return 0

    tickets = load_tickets_csv(args.infile)
    print(f"corpus: {len(tickets)} tickets from {args.infile}  (gate={args.gate}, "
          f"llm_tiebreak={'on' if args.ollama else 'off'})\n")

    tier = collections.Counter()
    tier_ok = collections.Counter()
    overall_ok = 0
    for t in tickets:
        d = resolver.resolve(t.title or t.text, t.description or "")
        tier[d.method] += 1
        if d.category == t.category:
            overall_ok += 1
            tier_ok[d.method] += 1

    print("routing tier distribution (where each ticket was decided):")
    for m in ["confident_svm", "agreement", "llm_tiebreak", "escalated"]:
        n = tier.get(m, 0)
        if n:
            acc = tier_ok[m] / n
            pct = 100 * n / len(tickets)
            print(f"  {m:<15} {n:>5} ({pct:4.1f}%)   accuracy {acc:.2f}")
    print(f"\noverall accuracy = {overall_ok / len(tickets):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
