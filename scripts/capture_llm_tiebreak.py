"""
Capture REAL LLM-tiebreak runs into data/llm_tiebreak_cache.json.

Run this ONCE on your machine with Ollama running:

    ollama serve                         # in another terminal
    ollama pull mistral:7b-instruct-q8_0 # if not already pulled
    python scripts/capture_llm_tiebreak.py

It sends each demo "3-way split" ticket through the resolver with the LIVE
local Mistral, and records the exact prompt + the model's raw response + the
chosen team. After this, the Explain tab / Console will replay the REAL model
decision even if Ollama is not running during the demo (source = "ollama").
"""
import json
import pathlib
import sys
import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "llm_tiebreak_cache.json"

# the demo 3-way-split tickets (extend this list as you add demo cases)
TICKETS = [
    "User accounts are not syncing properly with the centralized authentication server, resulting in intermittent access to applications.",
    "Users are experiencing intermittent issues with accessing resources on the site. Delayed and degraded performance affects everyone.",
]


def ollama_up(host="http://localhost:11434", timeout=3.0):
    import urllib.request
    try:
        urllib.request.urlopen(host + "/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def main():
    if not ollama_up():
        print("Ollama is not reachable at http://localhost:11434.")
        print("Start it (`ollama serve`) and pull the model, then re-run.")
        sys.exit(1)

    from web import server
    server._boot()
    from web.server import STATE
    from sentineldesk.llm import OllamaClient

    resolver = STATE["resolver"]
    scorer = STATE["scorer"]
    # force a real Ollama client for the capture
    resolver.llm = OllamaClient(model="mistral:7b-instruct-q8_0")

    cache = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    for tk in TICKETS:
        d = resolver.resolve("", tk)
        if d.method != "llm_tiebreak":
            print(f"[skip] not a live tiebreak ({d.method}): {tk[:60]}")
            continue
        # re-run the raw model call to capture the exact response text
        sr = scorer.classify("", tk)
        votes = d.votes
        candidates = sorted(set(votes.values()))
        evidence = ", ".join(f"{s}={t}" for s, t in sr.matched.items() if t) or "none"
        prompt = (
            "You are routing an IT support ticket to exactly one team. "
            f"The candidate teams are: {', '.join(candidates)}.\n"
            f"Ticket: {tk}\n"
            f"Lexical evidence found: {evidence}\n"
            "Reply with ONLY the exact team name from the candidate list that best "
            "matches the ROOT cause (not an incidental mention)."
        )
        raw = resolver.llm.complete(prompt, temperature=0.0).strip()
        cache[tk] = {
            "candidates": candidates,
            "votes": votes,
            "prompt": prompt,
            "raw_response": raw,
            "choice": d.category,
            "source": "ollama",
            "model": "mistral:7b-instruct-q8_0",
            "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        print(f"[captured] {d.category}  <-  {tk[:55]}  (raw: {raw[:40]!r})")

    CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"\nWrote {CACHE} with {len(cache)} entries (source=ollama).")


if __name__ == "__main__":
    main()
