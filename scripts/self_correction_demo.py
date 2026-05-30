#!/usr/bin/env python3
"""
Self-correction demo — show the loop learning your users' language.

For each casual ticket the dictionary doesn't cover:
  1) classify it BEFORE any correction (often misrouted),
  2) apply a correction (an L2 reroute teaches the phrase),
  3) classify the SAME ticket AFTER — now routed correctly, no retraining.

Then exercise the guardrails (generic + collision) and print the audit log.

    python scripts/self_correction_demo.py
    python scripts/self_correction_demo.py --save data/corrections.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import SVMClassifier, load_model        # noqa: E402
from sentineldesk.learning import CorrectionStore                    # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary       # noqa: E402

# (casual ticket, phrase the L2 highlights when rerouting, correct category)
CASES = [
    ("the box is totally frozen and won't come back", "the box is totally frozen", "Infrastructure"),
    ("our shared drive vanished overnight", "shared drive vanished", "Storage"),
    ("someone phished one of our staff", "someone phished", "Security"),
    ("the nightly job that copies the table bombed out", "nightly job that copies the table", "Database"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/svm_model.pkl")
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--save", default=None, help="write the learned corrections to this JSON")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"no model at {args.model} — run scripts/build_pipeline.py first")
        return 1

    clf = SVMClassifier(load_model(args.model))
    cv = ConceptVocabulary.from_layman_map()
    store = CorrectionStore.from_vocab(cv)

    print("=== BEFORE corrections (casual phrasings the dictionary doesn't know) ===")
    before = {}
    for text, _phrase, correct in CASES:
        pred, conf = clf.predict(store.normalize(text, cv))
        before[text] = pred
        flag = "ok" if pred == correct else "WRONG"
        print(f"  [{correct:<15}] {text!r}\n        -> {pred} ({conf:.2f})  {flag}")

    print("\n=== apply corrections (each L2 reroute teaches one phrase) ===")
    for _text, phrase, correct in CASES:
        r = store.record(phrase, correct, min_support=args.min_support)
        print(f"  learn {phrase!r:45} -> {correct:<15} [{r.status}]"
              + (f" via anchor '{r.anchor}'" if r.anchor else ""))

    print("\n=== AFTER corrections (same tickets, no retraining) ===")
    fixed = 0
    for text, _phrase, correct in CASES:
        pred, conf = clf.predict(store.normalize(text, cv))
        if pred == correct:
            fixed += 1
        change = "FIXED" if (before[text] != correct and pred == correct) else \
                 ("ok" if pred == correct else "still wrong")
        print(f"  [{correct:<15}] {text!r}\n        -> {pred} ({conf:.2f})  {change}")
    print(f"\n  corrected {fixed}/{len(CASES)} after learning")

    print("\n=== guardrails ===")
    print(f"  generic   'not working' -> Network : {store.record('not working', 'Network').status}")
    print(f"  collision (relearn first phrase elsewhere): "
          f"{store.record(CASES[0][1], 'Database').status}")

    print("\n=== audit log (every learned vocabulary change is inspectable) ===")
    for e in store.audit:
        print(f"  + {e['phrase']!r} -> {e['to_category']} (anchor '{e['anchor']}', support {e['support']})")

    if args.save:
        store.save(args.save)
        print(f"\nsaved learned corrections -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
