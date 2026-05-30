# external_test — validation on real third-party tickets

A self-contained, honest validation of SentinelDesk's routing against a **real,
human-labeled** IT-support ticket dataset — separate from the synthetic training
pipeline (separate code, separate data, no imports into the main package beyond
the trained model + vocabulary).

## Why this exists
All of SentinelDesk's training/eval data is synthetic. This folder tests the
trained classifier on real, third-party, human-written tickets to check that it
generalizes beyond synthetic prose.

## Source
"Classification of IT Support Tickets" — Zenodo record **7384758**
(https://zenodo.org/records/7384758). Real tickets manually classified by IT
professionals; PII redacted. Licensed **CC BY-SA**. Place the source files in
`external_test/source/`:
```
external_test/source/X_train.csv  y_train.csv   (and optionally X_test.csv y_test.csv)
```

## Method (so the number is honest, not cherry-picked)
`build_dataset.py` curates a benchmark using rules applied to ticket **content**,
never to the model's prediction:
1. **English only** (drop PT/ES/DE tickets).
2. **Readable** — strip redaction tags (`[NAME]`, `[TICKET ID]`, ...) and ticket
   boilerplate, then require ≥ 5 real words. Removes contentless stubs.
3. **Unambiguous** — keep only tickets that hit exactly ONE of our 7 domains'
   keyword anchors.

This yields real tickets that clearly belong to one domain — an **optimistic**
slice (clear-signal tickets), not whole-dataset accuracy. Labels are
keyword-derived (weak supervision), not hand-verified gold.

## Usage
```
python external_test/build_dataset.py            # -> external_test/zenodo_clean.csv
python external_test/run_test.py --examples      # score the trained model on it
```
`run_test.py` reports per-domain accuracy, only treating domains with n ≥ 20 as
trustworthy (others flagged indicative-only), plus a confusion breakdown and the
model's confidence distribution on real text.

## Result (this build)
On the domains with real coverage in this corpus:
- **Network: 117/118 = 99%**
- **Access Management: 64/85 = 75%**
- Combined (n ≥ 20): **89.2%**

Confidence on real redacted text drops to mean ~0.71 (vs ~0.93 on synthetic
in-domain text) — i.e. the confidence gate correctly flags real tickets as harder
rather than guessing. Other domains (Database/Application/Security/Infrastructure)
have too few clean single-domain tickets in this corpus to score (n < 5) and are
reported as indicative-only.

## Honest framing for the deck
> "Validated on real third-party tickets (Zenodo IT-support corpus): Network 99%,
> Access Management 75% on the readable, clearly-in-domain subset. The model's
> confidence correctly drops on redacted/templated real text, so it escalates
> rather than guessing. Broader real-world coverage and robustness to redacted
> text are roadmap items."

Do **not** cite a single whole-dataset accuracy from public ITSM corpora: their
taxonomies are functional (Hardware/HR/Access), not technical, so most tickets
don't map to our domains, and a raw number would be misleading.
