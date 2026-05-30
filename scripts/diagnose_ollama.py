#!/usr/bin/env python3
"""
Probe a local Ollama server: send one generation prompt and show the RAW
response plus whether the parser accepts it. Use this whenever the generator
produces empty batches — it reveals exactly what the model returned.

    python scripts/diagnose_ollama.py --model mistral:7b-instruct-q8_0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.data_gen import (                 # noqa: E402
    build_generation_prompt,
    parse_generated_tickets,
)
from sentineldesk.data_gen.prompts import GENERATION_SYSTEM  # noqa: E402
from sentineldesk.llm import OllamaClient            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral:7b-instruct-q8_0")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--category", default="Network")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--no-json-mode", action="store_true",
                    help="disable format:json to compare behaviour")
    args = ap.parse_args()

    llm = OllamaClient(model=args.model, host=args.host)
    prompt = build_generation_prompt(args.category, args.n)

    print(f"== probing {args.model} @ {args.host} ==")
    print(f"category={args.category}  n={args.n}  json_mode={not args.no_json_mode}\n")

    raw = llm.complete(prompt, system=GENERATION_SYSTEM, temperature=0.8,
                       json_mode=not args.no_json_mode)

    print("---- RAW RESPONSE ----")
    print(raw if raw else "(empty — Ollama not reachable or model not found)")
    print("---- END RAW ----\n")

    parsed = parse_generated_tickets(raw, args.category)
    print(f"parser extracted {len(parsed)} ticket(s)")
    for t in parsed[:3]:
        print(f"  - [{t['priority']}] {t['title']}")

    if not raw:
        print("\nFIX: check `ollama list` for the exact tag, and that "
              "`ollama serve` is running on the host above.")
        return 1
    if not parsed:
        print("\nThe model responded but the parser found no tickets. Paste "
              "the RAW RESPONSE above to Claude to adjust the parser.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
