#!/usr/bin/env python3
"""
external_test/build_dataset.py — curate a clean benchmark from the Zenodo dataset.

Source: "Classification of IT Support Tickets" (Zenodo record 7384758), real
tickets manually labeled by IT professionals, CC BY-SA. Place the source files
(X_train.csv / y_train.csv and optionally X_test.csv / y_test.csv) in
external_test/source/ and run this.

Selection rules (applied to CONTENT, never to the model's prediction — so the
resulting accuracy is honest, not cherry-picked):
  1. English only (drop tickets with obvious PT/ES/DE cues).
  2. Readable: after stripping redaction tags ([NAME], [TICKET ID], ...) and
     boilerplate ("a support ticket was forwarded..."), require >= MIN_WORDS real
     words. This removes contentless templated stubs like
     "File Share Access - [TICKET ID] - [NAME] ([COMPANY])".
  3. Unambiguous: the text must hit the anchors of EXACTLY ONE of our domains.

The output is a curated benchmark of real, readable, clearly-in-domain tickets —
an honest (if optimistic) test of routing on real text.

    python external_test/build_dataset.py
    python external_test/build_dataset.py --source external_test/source --out external_test/zenodo_clean.csv
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

MIN_WORDS = 5

ANCHORS = {
    "Access Management": [
        "active directory", "ad group", "ad account", "create ad", "new ou",
        "organizational unit", "password reset", "reset password", "reset my password",
        "account locked", "unlock account", "permission", "access rights",
        "group membership", "create user", "user creation", "new user account",
        "mfa", " sso ", "login", "log in", "credentials",
    ],
    "Network": [
        "vpn", "wifi", "wi-fi", " dns ", "firewall", "ip address", "router",
        "packet loss", "latency", "connectivity", "proxy", "subnet", "tunnel",
        "network connection",
    ],
    "Database": [
        "database", " sql ", "deadlock", "oracle", "postgres", "mysql",
        "stored procedure", "db server", "query timeout",
    ],
    "Storage": [
        "disk full", "disk space", "out of space", "mailbox full", "quota",
        "inode", "storage volume", "nas ", "running out of space",
    ],
    "Application": [
        "application error", "app crash", "software install", "java install",
        "build pipeline", "deploy", "api error", "web application", "service crash",
    ],
    "Infrastructure": [
        "virtual machine", " vm ", "cpu usage", "memory leak", "kubernetes",
        " pod ", "reboot server", "host down", "hypervisor", "server crash",
    ],
    "Security": [
        "malware", "ransomware", "phishing", " virus", "antivirus", "data breach",
        "credential theft", "security alert", "compromised account", "brute force",
    ],
}

_TAG = re.compile(r"\[[A-Z ]+\]")
_BOILER = re.compile(
    r"(a support ticket was forwarded to your role|new support ticket received|service request)",
    re.IGNORECASE,
)
_FOREIGN = ("não", "nao", " está", " esta ", "correo", "acesso", "impressora",
            "buenas", "estimado", " der ", " die ", " und ", " für ", " mit ",
            "problema", "solicit", "löschen", "minha", "máquina", "maquina")


def clean(text: str) -> str:
    t = _TAG.sub(" ", text)
    t = _BOILER.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_english(text: str) -> bool:
    low = " " + text.lower() + " "
    return not any(cue in low for cue in _FOREIGN)


def domains_hit(text: str) -> list[str]:
    low = " " + text.lower() + " "
    return [d for d, kws in ANCHORS.items() if any(k in low for k in kws)]


def load_source(source: Path) -> list[tuple[str, str]]:
    rows = []
    for xf, yf in [("X_train.csv", "y_train.csv"), ("X_test.csv", "y_test.csv")]:
        xp, yp = source / xf, source / yf
        if not xp.exists():
            continue
        X = {r["id"]: r["text"] for r in csv.DictReader(open(xp, encoding="utf-8"))}
        Y = ({r["id"]: r["category_truth"] for r in csv.DictReader(open(yp, encoding="utf-8"))}
             if yp.exists() else {})
        for i, txt in X.items():
            rows.append((txt, Y.get(i, "")))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="external_test/source")
    ap.add_argument("--out", default="external_test/zenodo_clean.csv")
    args = ap.parse_args()

    source = Path(args.source)
    raw = load_source(source)
    if not raw:
        print(f"no source files found in {source}/ "
              f"(expected X_train.csv + y_train.csv from Zenodo record 7384758)")
        return 1

    kept = []
    drop_lang = drop_short = drop_ambig = 0
    for text, src_label in raw:
        if not is_english(text):
            drop_lang += 1
            continue
        body = clean(text)
        if len(body.split()) < MIN_WORDS:
            drop_short += 1
            continue
        hits = domains_hit(body)
        if len(hits) != 1:
            drop_ambig += 1
            continue
        kept.append((body, hits[0], src_label))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label", "zenodo_label"])
        w.writerows(kept)

    print(f"source tickets: {len(raw)}")
    print(f"  dropped non-English: {drop_lang}")
    print(f"  dropped too-short/redacted-stub: {drop_short}")
    print(f"  dropped ambiguous (0 or >1 domains): {drop_ambig}")
    print(f"kept (clean, readable, single-domain): {len(kept)} -> {args.out}\n")
    print("curated label distribution (our taxonomy):")
    for d, n in Counter(l for _, l, _ in kept).most_common():
        print(f"  {d:20} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
