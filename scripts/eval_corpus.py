#!/usr/bin/env python3
"""
One-shot health check for any ticket corpus (templated OR real Mistral output).

Reports: category balance, length/duplicate stats, a sample per category,
exclusivity (N-gram + TF-IDF), and an SVM cross-eval (classifies the corpus
with the trained model, after synonym normalization).

    python scripts/eval_corpus.py --in data/synthetic_tickets.csv

When --in is your REAL Mistral corpus and --model is the templated-trained SVM,
the cross-eval is a genuine generalization test: does a model trained on clean
templated data still classify naturally-written tickets correctly?
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.corpus import load_tickets_csv                                   # noqa: E402
from sentineldesk.vocabulary.analysis import (                                     # noqa: E402
    exclusivity_report,
    ngram_counts_by_category,
    tfidf_weights_by_category,
)
from sentineldesk.vocabulary.concepts import ConceptVocabulary                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/synthetic_tickets.csv")
    ap.add_argument("--model", default="data/svm_model.pkl")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip synonym normalization before the SVM (for the ablation)")
    args = ap.parse_args()

    tickets = load_tickets_csv(args.inp)
    print(f"corpus: {len(tickets)} tickets from {args.inp}\n")

    by_cat = collections.Counter(t.category for t in tickets)
    lengths = [len(t.description.split()) for t in tickets]
    texts = [t.text.strip().lower() for t in tickets]
    dupes = len(texts) - len(set(texts))

    print("category balance:")
    for c, n in sorted(by_cat.items()):
        print(f"  {c:<20} {n}")
    print(f"\ndescription length (words): min {min(lengths)}  "
          f"avg {sum(lengths) / len(lengths):.0f}  max {max(lengths)}")
    print(f"exact duplicate texts: {dupes}  "
          f"({'clean' if dupes == 0 else 'check for repetition'})\n")

    print("sample ticket per category:")
    seen: set[str] = set()
    for t in tickets:
        if t.category not in seen:
            seen.add(t.category)
            desc = " ".join(t.description.split())[:90]
            print(f"  [{t.category}] {t.title[:48]} :: {desc}...")
    print()

    ng = exclusivity_report(ngram_counts_by_category(tickets, top_n=args.top_n))
    tf = exclusivity_report(tfidf_weights_by_category(tickets, top_n=args.top_n))
    print(f"exclusivity (top-{args.top_n}):  N-gram {ng['_overall']:.2f}   "
          f"TF-IDF {tf['_overall']:.2f}\n")

    if Path(args.model).exists():
        from sentineldesk.classifier import SVMClassifier, load_model
        clf = SVMClassifier(load_model(args.model))
        cv = ConceptVocabulary.from_layman_map()
        per = collections.Counter()
        per_ok = collections.Counter()
        confusion = collections.defaultdict(collections.Counter)  # true -> {wrong_pred: n}
        ok = 0
        for t in tickets:
            text = t.text if args.no_normalize else cv.normalize(t.text)
            pred, _ = clf.predict(text)
            per[t.category] += 1
            if pred == t.category:
                ok += 1
                per_ok[t.category] += 1
            else:
                confusion[t.category][pred] += 1
        norm_state = "OFF (ablation)" if args.no_normalize else "ON"
        print(f"SVM cross-eval (model: {args.model}, normalization: {norm_state}):")
        print(f"  overall accuracy = {ok / len(tickets):.2f}")
        for c in sorted(per):
            print(f"  {c:<20} {per_ok[c]}/{per[c]}")
        if confusion:
            print("\nwhere the misses went (true -> predicted):")
            for c in sorted(confusion):
                dests = ", ".join(f"{p} ({n})" for p, n in confusion[c].most_common())
                print(f"  {c:<20} -> {dests}")
    else:
        print(f"(no model at {args.model}; run build_pipeline.py first to enable cross-eval)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
