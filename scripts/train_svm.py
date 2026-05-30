#!/usr/bin/env python3
"""
Train the SVM ensemble classifier and report cross-validated metrics.

CPU-only — runs anywhere, no Ollama needed. Reports stratified k-fold
accuracy and macro-F1 (mean +/- std), then fits on the full corpus and writes
svm_model.pkl.

    python scripts/train_svm.py --in data/synthetic_tickets.csv --out data/svm_model.pkl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import (          # noqa: E402
    cross_validate_svm,
    save_model,
    train_svm,
)
from sentineldesk.corpus import load_tickets_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train SVM classifier")
    ap.add_argument("--in", dest="infile", default="data/seed_tickets.csv")
    ap.add_argument("--out", dest="outfile", default="data/svm_model.pkl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=5000)
    args = ap.parse_args()

    tickets = load_tickets_csv(args.infile)
    n_cat = len({t.category for t in tickets})
    print(f"loaded {len(tickets)} tickets across {n_cat} categories\n")

    print(f"cross-validating ({args.folds}-fold stratified)...")
    report = cross_validate_svm(tickets, folds=args.folds,
                                max_features=args.max_features)
    print(f"  folds used      : {report.folds}")
    print(f"  accuracy        : {report.accuracy_mean:.3f} +/- {report.accuracy_std:.3f}")
    print(f"  macro-F1        : {report.f1_macro_mean:.3f} +/- {report.f1_macro_std:.3f}")

    print("\ntraining final model on full corpus...")
    pipeline = train_svm(tickets, max_features=args.max_features)
    out = save_model(pipeline, args.outfile)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
