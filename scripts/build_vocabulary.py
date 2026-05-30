#!/usr/bin/env python3
"""
Build vocabulary.json from a ticket CSV.

Runs the full 4-layer VGAC vocabulary build. Layers 1-3 are pure CPU. Layer 4
(LLM enrichment) requires a local Ollama server — use --enrich to turn it on.

Examples
--------
    # CPU-only build (no Ollama), good for a quick check:
    python scripts/build_vocabulary.py --in data/seed_tickets.csv --out data/vocabulary.json

    # Full build with local Mistral enrichment (your machine):
    ollama pull mistral:7b
    python scripts/build_vocabulary.py --in data/synthetic_tickets.csv \
        --out data/vocabulary.json --enrich --model mistral:7b
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.corpus import load_tickets_csv          # noqa: E402
from sentineldesk.vocabulary import (                      # noqa: E402
    OllamaEnricher,
    StubEnricher,
    build_full_vocabulary,
    save_vocabulary,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build VGAC vocabulary.json")
    ap.add_argument("--in", dest="infile", default="data/seed_tickets.csv")
    ap.add_argument("--out", dest="outfile", default="data/vocabulary.json")
    ap.add_argument("--enrich", action="store_true",
                    help="enable Layer 4 LLM enrichment via Ollama")
    ap.add_argument("--model", default="mistral:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    tickets = load_tickets_csv(args.infile)
    print(f"loaded {len(tickets)} tickets, "
          f"{len({t.category for t in tickets})} categories")

    if args.enrich:
        print(f"enrichment ON  -> Ollama {args.model} @ {args.host}")
        enricher = OllamaEnricher(model=args.model, host=args.host)
    else:
        print("enrichment OFF -> CPU-only (layers 1-3)")
        enricher = StubEnricher()

    vocab = build_full_vocabulary(tickets, enricher=enricher)
    out = save_vocabulary(vocab, args.outfile)

    print(f"\nwrote {out}")
    for cat in sorted(vocab):
        layers = vocab[cat]
        added = len(layers["enriched"])
        print(f"  {cat:20s} {len(layers['final']):3d} terms "
              f"(+{added} from enrichment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
