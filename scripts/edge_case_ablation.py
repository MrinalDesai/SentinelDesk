#!/usr/bin/env python3
"""
Edge-case ablation (SEPARATE data + code).

Generates a small corpus of tickets phrased ONLY in casual/synonym language —
the register your synthetic corpus lacks (real users say "the internet's dead",
not "dns resolution failing"). Then maps each ticket through the synonym layer
(synonym -> canonical) and classifies it with the EXISTING trained SVM, showing
accuracy BEFORE normalization vs AFTER. This is where the synonym layer earns
its keep — on input the SVM was never trained to recognise.

    python scripts/edge_case_ablation.py
    python scripts/edge_case_ablation.py --per-category 20 --examples

Writes data/edge_cases.csv (separate from your main corpora).

HONEST CAVEAT: the casual phrasings are drawn from the synonym dictionary, so
"after" normalization is a best case (the layer catches everything by
construction). It proves the mechanism and direction, not a real-world ceiling —
genuine user phrasings include synonyms not in the dictionary, which the layer
would miss. Present it as a controlled demonstration, not a field benchmark.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import SVMClassifier, load_model     # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary    # noqa: E402

_OPENERS = ["users report", "we're seeing", "tickets coming in:", "people complaining", "heads up,"]
_SCOPES = ["across the office", "for the whole team", "since this morning", "for a bunch of people", "again"]


def make_casual_corpus(cv: ConceptVocabulary, per_category: int, seed: int) -> list[dict]:
    """Build tickets containing ONLY synonym forms (no canonical terms)."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for category, groups in cv.groups.items():
        # collect non-colliding synonyms for this category (canonical EXCLUDED)
        syns = []
        for group in groups:
            canonical = group[0]
            for s in group[1:]:
                if cv.syn2canon.get(s.lower()) == canonical:
                    syns.append(s)
        if len(syns) < 2:
            continue
        for _ in range(per_category):
            a, b = rng.sample(syns, 2)
            desc = f"{rng.choice(_OPENERS)} {a}; {b} {rng.choice(_SCOPES)}"
            rows.append({"title": a, "description": desc, "category": category,
                         "resolution": "", "priority": "Medium", "request_type": "Incident"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="data/svm_model.pkl")
    ap.add_argument("--out", default="data/edge_cases.csv")
    ap.add_argument("--examples", action="store_true", help="show per-ticket before/after flips")
    args = ap.parse_args()

    cv = ConceptVocabulary.from_layman_map()
    rows = make_casual_corpus(cv, args.per_category, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"generated {len(rows)} casual (synonyms-only) tickets -> {args.out}\n")

    clf = SVMClassifier(load_model(args.model))

    def score(normalize: bool):
        ok = 0
        flips = []
        for r in rows:
            text = f"{r['title']} {r['description']}"
            pred, conf = clf.predict(cv.normalize(text) if normalize else text)
            if pred == r["category"]:
                ok += 1
            flips.append((r, pred))
        return ok / len(rows), flips

    before, flips_before = score(normalize=False)
    after, _ = score(normalize=True)

    print("ablation on casual (synonyms-only) input:")
    print(f"  BEFORE normalization (raw casual text) : {before:.2f}")
    print(f"  AFTER  normalization (synonym->canonical): {after:.2f}")
    print(f"  --> synonym layer recovers {(after - before) * 100:.0f} points on casual input\n")

    if args.examples:
        print("example before/after flips (casual ticket -> SVM pick):")
        shown = 0
        for r, pred_before in flips_before:
            if pred_before != r["category"] and shown < 8:
                text = f"{r['title']} {r['description']}"
                pred_after, _ = clf.predict(cv.normalize(text))
                if pred_after == r["category"]:
                    print(f"  [{r['category']}] {r['description'][:60]}")
                    print(f"      before: {pred_before:<18} -> after: {pred_after}  ✓")
                    shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
