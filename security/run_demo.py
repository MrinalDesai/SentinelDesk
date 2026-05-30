#!/usr/bin/env python3
"""
Security controls demo — run the real controls on the security dataset.

Demonstrates, on security/data/*:
  1. PII redaction      (IMPLEMENTED) — redact synthetic PII from tickets
  2. Encryption at rest (IMPLEMENTED) — Fernet round-trip; at-rest blob is ciphertext
  3. RBAC               (IMPLEMENTED) — enforced allow/deny on access scenarios
  4. Secrets provider   (SEAM)        — local backend works; Vault adapter documented
  5. mTLS               (DEPLOYMENT)  — generate demo certs; transport is config, not code

    python security/generate_dataset.py   # once, to create the data
    python security/run_demo.py

Leaves the main pipeline untouched — these controls are standalone here, with
the documented integration point being an ingestion stage before classification.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from security.encryption import EncryptedStore, SecretsProvider, VaultSecretsProvider  # noqa: E402
from security.mtls import DEPLOYMENT_NOTE, generate_demo_certs                          # noqa: E402
from security.pii_redaction import PIIRedactor                                          # noqa: E402
from security.rbac import AccessControl                                                 # noqa: E402

KNOWN_NAMES = ["Smith", "Johnson", "Patel", "Garcia", "Lee"]


def main() -> int:
    pii_path = Path("security/data/pii_tickets.csv")
    acc_path = Path("security/data/access_requests.csv")
    if not pii_path.exists() or not acc_path.exists():
        print("dataset missing — run: python security/generate_dataset.py")
        return 1

    print("=" * 70)
    print("SENTINELDESK SECURITY CONTROLS — demo on security/data/*")
    print("=" * 70)

    # 1. PII redaction
    print("\n[1] PII REDACTION  (IMPLEMENTED)")
    redactor = PIIRedactor(known_names=KNOWN_NAMES)
    tickets = list(csv.DictReader(open(pii_path, encoding="utf-8")))
    totals: Counter = Counter()
    clean_records = []
    for t in tickets:
        r = redactor.redact(t["text"])
        totals.update(r.counts)
        clean_records.append({"id": t["id"], "text": r.text})
    print(f"  redacted {len(tickets)} tickets; entities removed: "
          + ", ".join(f"{k}={v}" for k, v in totals.most_common()))
    print("  example:")
    print(f"    before: {tickets[2]['text']}")
    print(f"    after : {redactor.redact(tickets[2]['text']).text}")

    # 2. Encryption at rest (on the REDACTED records)
    print("\n[2] ENCRYPTION AT REST  (IMPLEMENTED — Fernet/AES)")
    store = EncryptedStore(SecretsProvider())
    blob = store.encrypt(clean_records[2])
    roundtrip = store.decrypt(blob)
    print(f"  plaintext record: {clean_records[2]}")
    print(f"  at-rest ciphertext (truncated): {blob[:48]}...")
    print(f"  ciphertext leaks no plaintext: {store.is_ciphertext(blob, clean_records[2])}")
    print(f"  decrypt round-trip matches: {roundtrip == clean_records[2]}")

    # 3. RBAC
    print("\n[3] RBAC  (IMPLEMENTED)")
    ac = AccessControl()
    for r in csv.DictReader(open(acc_path, encoding="utf-8")):
        d = ac.check(r["role"], r["action"], domain=r["domain"],
                     reads_pii=r["reads_pii"].lower() == "true")
        mark = "ALLOW" if d.allowed else "DENY "
        print(f"  [{mark}] {r['user']:6} {r['role']:11} {r['action']:9} "
              f"{('('+r['domain']+')') if r['domain'] else '':12} -> {d.reason}")

    # 4. Secrets provider
    print("\n[4] SECRETS PROVIDER  (local backend IMPLEMENTED; Vault = SEAM)")
    key = SecretsProvider().get_encryption_key()
    print(f"  local provider returned a key ({len(key)} bytes), no secret hard-coded")
    try:
        VaultSecretsProvider()
    except NotImplementedError:
        print("  Vault adapter present but cleanly disabled (production seam)")

    # 5. mTLS (deployment)
    print("\n[5] mTLS  (DEPLOYMENT-layer — cert material demo, transport is config)")
    certs = generate_demo_certs()
    print(f"  generated demo CA + server cert: {', '.join(Path(p).name for p in certs.values())}")
    print("  " + DEPLOYMENT_NOTE.strip().replace("\n", "\n  "))

    print("\n" + "=" * 70)
    print("HONEST LABELS: PII redaction / encryption-at-rest / RBAC = IMPLEMENTED &")
    print("demonstrated above. Vault = seam. mTLS = deployment config (certs shown,")
    print("transport not implemented in code). Presidio NER name-redaction = roadmap.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
