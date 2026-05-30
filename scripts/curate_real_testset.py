#!/usr/bin/env python3
"""
Curate a mapped real test set from a third-party dataset, then score the model.

METHOD (stated plainly so the number is honest):
  1. Select real tickets whose TEXT contains unambiguous keywords for EXACTLY ONE
     of our domains (tickets matching zero or multiple domains are dropped).
  2. Label each by that domain. Selection uses ticket CONTENT, not the model's
     prediction, so it is not circular in the worst way — but the anchor words
     overlap with the model's vocabulary, so this is an OPTIMISTIC slice:
     "accuracy on clearly-in-domain real tickets", an upper-ish estimate.
  3. Score the model on this curated set.

The true real-world number sits BETWEEN this (clear-signal tickets) and the broad
unfiltered number from validate_external.py (~36%). Report both, never just one.

    python scripts/curate_real_testset.py --in data/all_tickets_processed_improved_v3.csv --examples
    python scripts/curate_real_testset.py --in <csv> --out data/real_curated.csv
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import SVMClassifier, load_model           # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary          # noqa: E402

# Human-obvious, domain-defining anchors. A ticket is kept only if it hits
# EXACTLY ONE domain — anything ambiguous is dropped, not guessed.
ANCHORS: dict[str, list[str]] = {
    "Access Management": [
        "active directory", "password reset", "reset password", "account locked",
        "login", "log in", "permission", "access request", "mfa", " sso ",
        "group membership", "user account", "credential", "unlock account",
        "file share access", "shared folder access", "folder access", "ad group",
        "create ad", "ad account", "new user", "user account", "rights",
    ],
    "Network": [
        "vpn", "wifi", "wi-fi", " dns", "firewall", "ip address", "router",
        "packet loss", "latency", "network connection", "connectivity",
    ],
    "Database": [
        "database", " sql", "deadlock", "oracle db", "postgres", "mysql",
        "query timeout", "table lock", "stored procedure",
    ],
    "Storage": [
        "disk full", "out of space", "mailbox full", "quota exceeded", "inode",
        "storage volume", "nas ", "disk space", "running out of space", "full disk",
    ],
    "Application": [
        "application error", "app crash", "software install", "java install",
        "build pipeline", "deploy", "api error", "service crash",
    ],
    "Infrastructure": [
        "server down", "virtual machine", " vm ", "cpu usage", "memory leak",
        "kubernetes", " pod ", "reboot server", "host down",
    ],
    "Security": [
        "malware", "ransomware", "phishing", "virus", "data breach", "antivirus",
        "security alert", "credential theft", "suspicious login",
    ],
}


def domains_hit(text: str) -> list[str]:
    low = " " + text.lower() + " "
    return [d for d, kws in ANCHORS.items() if any(k in low for k in kws)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", default="data/svm_model.pkl")
    ap.add_argument("--text-col", default="Document")
    ap.add_argument("--label-col", default="Topic_group")
    ap.add_argument("--out", default="data/real_curated.csv")
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()

    if not Path(args.inp).exists():
        print(f"dataset not found: {args.inp}")
        return 1

    # 1-2. curate
    curated: list[tuple[str, str, str]] = []  # (text, our_label, their_label)
    with open(args.inp, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            text = r.get(args.text_col, "")
            if not text:
                continue
            hits = domains_hit(text)
            if len(hits) == 1:                       # exactly one domain -> unambiguous
                curated.append((text, hits[0], r.get(args.label_col, "")))

    if not curated:
        print("no tickets matched a single domain — check the text column")
        return 1

    # write the derived dataset
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "our_label", "their_label"])
        w.writerows(curated)

    label_dist = Counter(lbl for _, lbl, _ in curated)
    print("=" * 72)
    print(f"CURATED REAL TEST SET  (from {Path(args.inp).name})")
    print("=" * 72)
    print(f"kept {len(curated)} tickets with an unambiguous single-domain signal "
          f"-> {args.out}")
    print("curated label distribution (our taxonomy):")
    for d, n in label_dist.most_common():
        print(f"  {d:20} {n}")

    # 3. score
    if not Path(args.model).exists():
        print(f"\nmodel not found: {args.model} (run build_pipeline.py); curated set written, not scored")
        return 0
    clf = SVMClassifier(load_model(args.model))
    cv = ConceptVocabulary.from_layman_map()

    confusion = defaultdict(Counter)
    per = Counter()
    per_ok = Counter()
    confs = []
    ok = 0
    for text, label, _ in curated:
        pred, conf = clf.predict(cv.normalize(text))
        confs.append(conf)
        per[label] += 1
        if pred == label:
            ok += 1
            per_ok[label] += 1
        else:
            confusion[label][pred] += 1

    print("\n" + "-" * 72)
    print("SCORE on the curated real test set (OPTIMISTIC slice — clear-signal tickets)")
    print("-" * 72)
    print(f"  overall accuracy = {ok}/{len(curated)} = {ok/len(curated):.1%}")
    print("  per-domain:")
    for d in sorted(per):
        print(f"    {d:20} {per_ok[d]:4}/{per[d]:<4}  ({per_ok[d]/per[d]:.0%})")
    print("  where the misses went:")
    for d in sorted(confusion):
        misses = ", ".join(f"{k} ({v})" for k, v in confusion[d].most_common())
        print(f"    {d:20} -> {misses}")
    import statistics
    hi = sum(1 for c in confs if c >= 0.80)
    print(f"  mean confidence = {statistics.mean(confs):.2f}   "
          f">=0.80: {hi}/{len(confs)} = {hi/len(confs):.0%}")

    if args.examples:
        print("\n  sample curated tickets -> prediction:")
        shown = Counter()
        for text, label, _ in curated:
            if shown[label] >= 2:
                continue
            shown[label] += 1
            pred, c = clf.predict(cv.normalize(text))
            mark = "OK" if pred == label else "X"
            print(f"    [{mark}] ({label}) {re.sub(r'\\s+', ' ', text)[:64].strip()}")
            print(f"         -> {pred} ({c:.0%})")

    print("\n" + "=" * 72)
    print("HONEST FRAMING: this is accuracy on real tickets that contain UNAMBIGUOUS")
    print("domain terms — an optimistic upper estimate. The broad unfiltered number")
    print("(validate_external.py) is the pessimistic end. True real-world performance")
    print("sits between the two. Labels are keyword-derived (weak supervision), not")
    print("hand-verified gold labels.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
