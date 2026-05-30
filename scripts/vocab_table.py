#!/usr/bin/env python3
"""
Build the N-gram and TF-IDF top-N tables per category and report exclusivity.

Runs on any ticket CSV; with no --in it uses the controlled reference corpus
(no LLM). Exports two wide CSVs (categories as columns) plus a long-format CSV,
and prints the exclusivity metric for both layers.

    # reference corpus (no Ollama):
    python scripts/vocab_table.py --out-dir data/vocab_tables
    # your real generated corpus:
    python scripts/vocab_table.py --in data/synthetic_tickets.csv --out-dir data/vocab_tables
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.corpus import load_tickets_csv                       # noqa: E402
from sentineldesk.data_gen.controlled import generate_controlled_corpus  # noqa: E402
from sentineldesk.vocabulary.analysis import (                          # noqa: E402
    exclusivity_report,
    ngram_counts_by_category,
    tfidf_weights_by_category,
)


def _write_wide(path: Path, top_by_cat: dict, value_fmt) -> None:
    cats = sorted(top_by_cat)
    depth = max(len(v) for v in top_by_cat.values())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank"] + cats)
        for i in range(depth):
            row = [i + 1]
            for c in cats:
                if i < len(top_by_cat[c]):
                    term, val = top_by_cat[c][i]
                    row.append(f"{term} ({value_fmt(val)})")
                else:
                    row.append("")
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=None)
    ap.add_argument("--out-dir", default="data/vocab_tables")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--per-category", type=int, default=120,
                    help="controlled corpus size when --in is omitted")
    args = ap.parse_args()

    if args.infile:
        corpus = load_tickets_csv(args.infile)
        print(f"corpus: {len(corpus)} tickets from {args.infile}")
    else:
        corpus = generate_controlled_corpus(per_category=args.per_category)
        print(f"corpus: {len(corpus)} tickets (controlled reference)")

    ng = ngram_counts_by_category(corpus, top_n=args.top_n)
    tf = tfidf_weights_by_category(corpus, top_n=args.top_n)
    ex_ng, ex_tf = exclusivity_report(ng), exclusivity_report(tf)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_wide(out / "ngram_top.csv", ng, lambda v: str(v))
    _write_wide(out / "tfidf_top.csv", tf, lambda v: f"{v:.3f}")

    print("\nexclusivity (fraction of top-N unique to the category):")
    print(f"  {'category':20s} {'N-gram':>8s} {'TF-IDF':>8s}")
    for cat in sorted(k for k in ex_ng if not k.startswith('_')):
        print(f"  {cat:20s} {ex_ng[cat]:8.2f} {ex_tf[cat]:8.2f}")
    print(f"  {'OVERALL':20s} {ex_ng['_overall']:8.2f} {ex_tf['_overall']:8.2f}")
    print(f"\nwrote {out/'ngram_top.csv'} and {out/'tfidf_top.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
