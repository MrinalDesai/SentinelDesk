#!/usr/bin/env python3
"""
Term-seeded (vocabulary-first) dataset generation.

Builds a per-category word model (frequent + unique terms), samples a weighted
subset per ticket, and has Mistral write the ticket + resolution around them.
Validation is the programmatic exclusivity gate (no LLM-judge needed at scale).

    # smoke (70 tickets, ~10/category):
    python scripts/generate_seeded.py --total 70 --model mistral:7b-instruct-q8_0
    # full run (overnight on a 5070):
    python scripts/generate_seeded.py --total 10000 --model mistral:7b-instruct-q8_0
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.corpus import load_tickets_csv                       # noqa: E402
from sentineldesk.data_gen import CategoryWordModel, TermSeededGenerator  # noqa: E402
from sentineldesk.data_gen.controlled import generate_controlled_corpus   # noqa: E402
from sentineldesk.data_gen.generator import write_tickets_csv          # noqa: E402
from sentineldesk.llm import OllamaClient                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Term-seeded ticket generation")
    ap.add_argument("--total", type=int, default=70)
    ap.add_argument("--model", default="mistral:7b-instruct-q8_0")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default="data/synthetic_tickets.csv")
    ap.add_argument("--resume", action="store_true",
                    help="resume from an existing --out file: skip complete categories, top up partial")
    ap.add_argument("--log-every", type=int, default=25,
                    help="print a progress line every N tickets (default 25)")
    ap.add_argument("--vocab-from", default=None,
                    help="CSV to derive the word model from (default: controlled corpus)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.vocab_from:
        base = load_tickets_csv(args.vocab_from)
    else:
        base = generate_controlled_corpus(per_category=120)
    model = CategoryWordModel.from_corpus(base)
    print(f"word model: {len(model.categories)} categories")
    for c in model.categories:
        print(f"  {c:18s} unique: {model.unique[c]}")

    llm = OllamaClient(model=args.model, host=args.host)
    gen = TermSeededGenerator(llm, model)
    print(f"\ngenerating ~{args.total} tickets via {args.model} "
          f"(~{args.total} LLM calls; be patient)\n")

    start = time.time()
    out_path = args.out

    existing = None
    if args.resume and Path(out_path).exists():
        import csv
        with open(out_path, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
        print(f"resume: loaded {len(existing)} existing tickets from {out_path}")

    def _checkpoint(ts: list[dict]) -> None:
        write_tickets_csv(ts, out_path)  # rewrite after each category; always a valid file

    tickets = gen.generate(
        total=args.total, log_every=args.log_every, on_checkpoint=_checkpoint, existing=existing
    )
    out = write_tickets_csv(tickets, out_path)
    mins = (time.time() - start) / 60.0

    print(f"\nwrote {len(tickets)} tickets to {out} in {mins:.1f} min")
    if not tickets:
        print("nothing generated — check Ollama is running and the model tag.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
