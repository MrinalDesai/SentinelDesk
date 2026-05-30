#!/usr/bin/env python3
"""
External validation against a REAL third-party ticket dataset (SEPARATE code/data).

The catch this script is built around: public ITSM datasets use business/functional
taxonomies (Hardware, HR Support, Access, ...), NOT our technical-domain taxonomy
(Network, Database, Access Management, ...). A raw accuracy number would be
meaningless. So we:

  [1] score accuracy ONLY on cleanly-mappable categories (a defensible number),
  [2] show where UNMAPPABLE categories route (informative, explicitly NOT scored),
  [3] print sample real tickets -> predicted domain + confidence for an eyeball check,
  [4] report the model's confidence distribution on real text.

This validates ROUTING BEHAVIOUR on real, third-party, human-written text — a
genuine step beyond all-synthetic evaluation — without pretending the label sets
match.

    python scripts/validate_external.py --in data/all_tickets_processed_improved_v3.csv
    python scripts/validate_external.py --in <csv> --text-col Document --label-col Topic_group --examples
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import SVMClassifier, load_model           # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary          # noqa: E402

# Only categories with an unambiguous mapping to our taxonomy are SCORED.
CLEAN_MAP = {
    "Access": "Access Management",
    "Administrative rights": "Access Management",
}
# Defensible-but-noisy additions (mailbox/file-share straddle Storage + Access Mgmt).
NOISY_MAP = {
    "Storage": "Storage",
}


def load_rows(path, text_col, label_col):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t, c = r.get(text_col, ""), r.get(label_col, "")
            if t and c:
                rows.append((t, c))
    return rows


def score_subset(rows, mapping, clf, cv, sample=None):
    """Return (n, correct, confusion, confidences) over rows whose label is in mapping."""
    confusion = defaultdict(Counter)
    confs = []
    n = correct = 0
    per_cat = Counter()
    for text, src_cat in rows:
        if src_cat not in mapping:
            continue
        if sample and per_cat[src_cat] >= sample:
            continue
        per_cat[src_cat] += 1
        expected = mapping[src_cat]
        pred, conf = clf.predict(cv.normalize(text))
        confs.append(conf)
        n += 1
        if pred == expected:
            correct += 1
        else:
            confusion[expected][pred] += 1
    return n, correct, confusion, confs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", default="data/svm_model.pkl")
    ap.add_argument("--text-col", default="Document")
    ap.add_argument("--label-col", default="Topic_group")
    ap.add_argument("--sample", type=int, default=None, help="cap per source category (speed)")
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()

    if not Path(args.inp).exists():
        print(f"dataset not found: {args.inp}\n(extract archive.zip into data/ first)")
        return 1
    if not Path(args.model).exists():
        print(f"model not found: {args.model}  (run scripts/build_pipeline.py first)")
        return 1

    rows = load_rows(args.inp, args.text_col, args.label_col)
    clf = SVMClassifier(load_model(args.model))
    cv = ConceptVocabulary.from_layman_map()

    src_dist = Counter(c for _, c in rows)
    print("=" * 72)
    print(f"EXTERNAL VALIDATION — {Path(args.inp).name}")
    print("=" * 72)
    print(f"total tickets: {len(rows)}")
    print("source taxonomy (theirs, NOT ours):")
    for c, n in src_dist.most_common():
        tag = "-> Access Management" if c in CLEAN_MAP else \
              ("-> Storage (noisy)" if c in NOISY_MAP else "(unmappable, not scored)")
        print(f"  {c:22} {n:6}  {tag}")

    # [1] clean mapped accuracy
    n, ok, conf, confs = score_subset(rows, CLEAN_MAP, clf, cv, args.sample)
    print("\n" + "-" * 72)
    print("[1] SCORED — clean mapping only (Access + Administrative rights -> Access Management)")
    print("-" * 72)
    if n:
        print(f"  n = {n}")
        print(f"  routed correctly to Access Management: {ok}/{n} = {ok/n:.1%}")
        if conf:
            print("  where the misses went:")
            for exp, c in conf.items():
                misses = ", ".join(f"{k} ({v})" for k, v in c.most_common())
                print(f"    {exp} -> {misses}")

    # [2] + noisy Storage
    n2, ok2, conf2, _ = score_subset(rows, {**CLEAN_MAP, **NOISY_MAP}, clf, cv, args.sample)
    print("\n[2] SCORED — incl. noisy Storage->Storage (caveat: their 'Storage' = mailbox/file-share)")
    if n2:
        print(f"  n = {n2}   accuracy = {ok2/n2:.1%}")

    # [3] confidence on real text
    if confs:
        import statistics
        hi = sum(1 for c in confs if c >= 0.80)
        print("\n[3] MODEL CONFIDENCE on real third-party text (mapped subset):")
        print(f"  mean confidence = {statistics.mean(confs):.2f}")
        print(f"  >= 0.80 (would route confidently): {hi}/{len(confs)} = {hi/len(confs):.1%}")
        print("  (low-confidence tickets would enter the edge-case ladder / escalate, not force a label)")

    # [4] routing distribution for UNMAPPABLE categories (informative only)
    print("\n" + "-" * 72)
    print("[4] INFORMATIVE (not scored) — where UNMAPPABLE categories route in our taxonomy")
    print("-" * 72)
    unmapped = defaultdict(Counter)
    seen = Counter()
    for text, src_cat in rows:
        if src_cat in CLEAN_MAP or src_cat in NOISY_MAP:
            continue
        if args.sample and seen[src_cat] >= args.sample:
            continue
        seen[src_cat] += 1
        pred, _ = clf.predict(cv.normalize(text))
        unmapped[src_cat][pred] += 1
    for src_cat, dist in unmapped.items():
        top = ", ".join(f"{k} {v}" for k, v in dist.most_common(3))
        print(f"  {src_cat:22} -> {top}")

    # [5] eyeball samples
    if args.examples:
        print("\n" + "-" * 72)
        print("[5] sample real tickets -> our predicted domain (eyeball check)")
        print("-" * 72)
        shown = Counter()
        for text, src_cat in rows:
            if shown[src_cat] >= 2:
                continue
            shown[src_cat] += 1
            pred, c = clf.predict(cv.normalize(text))
            print(f"  [{src_cat}] {text[:70].strip()}")
            print(f"      -> {pred} ({c:.0%})")

    print("\n" + "=" * 72)
    print("HONEST NOTE: this dataset's taxonomy is functional, not technical; only the")
    print("mappable slice is scored. Text is heavily pre-processed (stop-words stripped,")
    print("tokens duplicated), which is out-of-distribution vs our natural-prose training.")
    print("This validates routing behaviour on real text, not headline accuracy.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
