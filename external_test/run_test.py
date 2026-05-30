#!/usr/bin/env python3
"""
external_test/run_test.py — score the model on the curated Zenodo benchmark.

Loads external_test/zenodo_clean.csv (built by build_dataset.py from real Zenodo
tickets), runs each through the trained classifier (with synonym normalization,
exactly as in production), and reports per-domain accuracy, confusion, and
confidence. Per-domain results are only trustworthy where n is large enough;
domains with n < MIN_N are reported but flagged.

    python external_test/run_test.py
    python external_test/run_test.py --min-n 20 --examples
"""

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import SVMClassifier, load_model           # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="external_test/zenodo_clean.csv")
    ap.add_argument("--model", default="data/svm_model.pkl")
    ap.add_argument("--min-n", type=int, default=20, help="min samples for a trustworthy per-domain number")
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()

    if not Path(args.data).exists():
        print(f"benchmark not found: {args.data}  (run external_test/build_dataset.py first)")
        return 1
    if not Path(args.model).exists():
        print(f"model not found: {args.model}  (run scripts/build_pipeline.py first)")
        return 1

    rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
    clf = SVMClassifier(load_model(args.model))
    cv = ConceptVocabulary.from_layman_map()

    per = Counter()
    per_ok = Counter()
    confusion = defaultdict(Counter)
    confs = []
    samples = defaultdict(list)
    for r in rows:
        text, label = r["text"], r["label"]
        pred, conf = clf.predict(cv.normalize(text))
        confs.append(conf)
        per[label] += 1
        if pred == label:
            per_ok[label] += 1
        else:
            confusion[label][pred] += 1
        if len(samples[label]) < 3:
            samples[label].append((text, pred, conf, pred == label))

    # overall, but computed only over domains with enough samples (honest)
    big = [d for d in per if per[d] >= args.min_n]
    big_n = sum(per[d] for d in big)
    big_ok = sum(per_ok[d] for d in big)

    print("=" * 72)
    print("EXTERNAL BENCHMARK — curated real Zenodo tickets (readable, single-domain)")
    print("=" * 72)
    print(f"total curated tickets: {len(rows)}")
    print(f"\nper-domain accuracy:")
    for d in sorted(per, key=lambda x: -per[x]):
        flag = "" if per[d] >= args.min_n else "   (n too small — indicative only)"
        print(f"  {d:20} {per_ok[d]:4}/{per[d]:<4}  {per_ok[d]/per[d]:5.0%}{flag}")

    print(f"\naccuracy on domains with n >= {args.min_n} (the trustworthy number):")
    print(f"  {big_ok}/{big_n} = {big_ok/big_n:.1%}   (domains: {', '.join(sorted(big))})")

    print("\nwhere the misses went:")
    for d in sorted(confusion):
        if per[d] >= args.min_n:
            misses = ", ".join(f"{k} ({v})" for k, v in confusion[d].most_common())
            print(f"  {d:20} -> {misses}")

    hi = sum(1 for c in confs if c >= 0.80)
    print(f"\nconfidence on real text: mean {statistics.mean(confs):.2f}, "
          f">=0.80: {hi}/{len(confs)} = {hi/len(confs):.0%}")

    if args.examples:
        print("\nsample predictions:")
        for d in sorted(samples):
            for text, pred, conf, ok in samples[d]:
                print(f"  [{'OK' if ok else 'X '}] ({d}) {text[:60]}")
                print(f"        -> {pred} ({conf:.0%})")

    print("\n" + "=" * 72)
    print("Honest scope: curated to real, readable, clearly-in-domain English tickets")
    print("(labels keyword-derived, weak supervision). This is an optimistic slice —")
    print("real tickets with clear signal — not whole-dataset accuracy.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
