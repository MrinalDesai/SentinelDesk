#!/usr/bin/env python3
"""
Build the Graph RAG knowledge graph from a corpus and show symptom->cause->resolution traversals.

    python scripts/graph_rag_demo.py --in data/real_3000.csv
    python scripts/graph_rag_demo.py --in data/real_3000.csv --query "the site keeps dropping"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.corpus import load_tickets_csv                       # noqa: E402
from sentineldesk.rag import KnowledgeGraph                            # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="data/real_3000.csv")
    ap.add_argument("--query", default=None)
    args = ap.parse_args()

    tickets = load_tickets_csv(args.infile)
    have_res = sum(1 for t in tickets if t.resolution)
    g = KnowledgeGraph.build(tickets, ConceptVocabulary.from_layman_map())
    print(f"built knowledge graph from {len(tickets)} tickets ({have_res} with resolutions)")
    print(f"graph: {g.stats}\n")

    queries = [args.query] if args.query else [
        "the websites won't load and users can't reach the site",
        "primary replica deadlock in the connection pool",
        "ransomware detected on an endpoint",
        "vpn tunnel keeps dropping for remote staff",
    ]
    for q in queries:
        r = g.query(q)
        print(f"query: {q!r}")
        if r.root_cause:
            print(f"  symptoms fired : {r.symptom_hits}")
            print(f"  root cause     : {r.root_cause}  (routes to: {r.category})")
            print(f"  resolution     : {r.resolution}")
            print(f"  traversal      : {' -> '.join(r.path)}")
        else:
            print("  no symptom matched -> hand off (escalate / semantic retrieval roadmap)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
