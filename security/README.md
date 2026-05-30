# security — standalone security controls (demonstrable)

Real, runnable security controls for SentinelDesk, kept **separate** from the
classification pipeline (own code, own dataset, no changes to `src/`). Each
control is honestly labeled by maturity so nothing here is overclaimed.

## Maturity labels

| Control | Status | What's real |
|---|---|---|
| PII redaction | **IMPLEMENTED** | Regex redaction of email/phone/IP/SSN/card + title-prefixed & known names. Runs on `data/pii_tickets.csv`. |
| Encryption at rest | **IMPLEMENTED** | Fernet (AES-128 + HMAC) encrypt/decrypt of ticket records; verified no plaintext leak. |
| RBAC | **IMPLEMENTED** | Roles, permissions, elevation rules (Security tickets + PII reads need elevated roles); enforced on `data/access_requests.csv`. |
| Secrets provider | **SEAM** | Local env/key-file backend works; HashiCorp Vault adapter present but disabled until a client is supplied. |
| Presidio NER redaction | **ROADMAP** | Adapter present; full ML name/entity detection needs presidio-analyzer. |
| mTLS / encryption in transit | **DEPLOYMENT** | Demo CA + server cert generated to show the pattern; the transport terminates at the ingress (nginx/Envoy/Istio), **not** in this code. |

## Run
```
python security/generate_dataset.py   # creates synthetic PII tickets + access scenarios (all PII is FAKE)
python security/run_demo.py           # runs PII redaction, encryption, RBAC, secrets, mTLS cert demo
python -m pytest security/tests -q    # 6 unit tests
```

## Honest framing for the deck
- **Say:** "PII-redaction-first ingestion, encryption at rest, and RBAC are
  implemented and demonstrated on a dedicated security dataset; Vault-backed
  secrets and mTLS are deployment-layer integrations (seam + config shown)."
- **Do not say:** "100% PII Safe", "Vault implemented", or "mTLS implemented".
  Regex redaction is high-precision on structured PII but name coverage is
  partial without Presidio (NER) — state that.

## Integration point (not wired, per scope)
The intended production placement is an **ingestion stage between the safety gate
and the classifier**: redact PII → encrypt the stored record → enforce RBAC on
who may view/route/resolve/read-PII. The live pipeline is intentionally left
untouched here.
