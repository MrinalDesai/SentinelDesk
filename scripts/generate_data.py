#!/usr/bin/env python3
"""
Generate synthetic_tickets.csv via local Mistral (Ollama).

Pipeline: generate per category -> LLM-judge validate (drop score < threshold)
-> augment (variants) -> dedup -> CSV.

RUNTIME WARNING: this makes one LLM call per generated ticket (validation) plus
one per kept ticket (augmentation). The full Round 2 target (150/category) is
~1000 generate + ~1000 validate + ~1000 augment calls. On an RTX 5070 that is
plausibly 1-3 hours. ALWAYS do a smoke run first:

    python scripts/generate_data.py --per-category 5 --model <your-q8-tag>

Then scale up once the output looks right:

    python scripts/generate_data.py --per-category 150 --variations 3 \
        --model <your-q8-tag> --out data/synthetic_tickets.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.config import ALL_DOMAINS                 # noqa: E402
from sentineldesk.data_gen import (                          # noqa: E402
    SyntheticDataGenerator,
    write_tickets_csv,
)
from sentineldesk.llm import OllamaClient                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic ITSM tickets")
    ap.add_argument("--per-category", type=int, default=5,
                    help="tickets per domain before augmentation (default 5 = smoke)")
    ap.add_argument("--threshold", type=int, default=4,
                    help="LLM-judge minimum score to keep a ticket (1-5)")
    ap.add_argument("--variations", type=int, default=3,
                    help="augmentation variants per kept ticket")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--model", default="mistral:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default="data/synthetic_tickets.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    categories = [d.value for d in ALL_DOMAINS]
    llm = OllamaClient(model=args.model, host=args.host)
    gen = SyntheticDataGenerator(llm, categories)

    est = args.per_category * len(categories)
    print(f"Generating ~{est} base tickets across {len(categories)} categories "
          f"via {args.model}. This calls the LLM many times; be patient.\n")

    start = time.time()
    tickets = gen.run(
        per_category=args.per_category,
        threshold=args.threshold,
        variations=args.variations,
        batch_size=args.batch_size,
    )
    out = write_tickets_csv(tickets, args.out)
    mins = (time.time() - start) / 60.0

    print(f"\nwrote {len(tickets)} tickets to {out} in {mins:.1f} min")
    by_cat: dict[str, int] = {}
    for t in tickets:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
    for cat in sorted(by_cat):
        print(f"  {cat:20s} {by_cat[cat]}")

    if len(tickets) == 0:
        print("\nNo tickets produced. Check that Ollama is running and the "
              "model tag is correct (ollama list).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
