#!/usr/bin/env python3
"""
End-to-end: generate data -> build vocabulary database -> train SVM -> report.

Deterministic stand-in for the Mistral run (templated text), so it runs without
Ollama. On your machine, swap generation for scripts/generate_seeded.py to get
real LLM writeups; the database, SVM, and metrics are identical in shape.

    python scripts/build_pipeline.py --per-category 200
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import SVMClassifier, cross_validate_svm, save_model, train_svm  # noqa: E402
from sentineldesk.corpus import LabeledTicket                                  # noqa: E402
from sentineldesk.data_gen.controlled import generate_controlled_corpus        # noqa: E402
from sentineldesk.data_gen.generator import write_tickets_csv                  # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary                 # noqa: E402
from sentineldesk.vocabulary.database import (                                 # noqa: E402
    build_vocabulary_db,
    normalizer_from_db,
    save_vocabulary_db,
)

RESOLUTIONS = {
    "Network": "Checked DNS, firewall and routing config; restored the affected network path.",
    "Application": "Reviewed logs and stack trace; patched the defect and redeployed the service.",
    "Database": "Analyzed query plan and locks; added the missing index and validated replication.",
    "Storage": "Reclaimed/expanded capacity; remounted the volume and verified the filesystem.",
    "Infrastructure": "Inspected resource usage; restarted the affected host/pod and confirmed stability.",
    "Access Management": "Reset credentials / unlocked the account; corrected role membership and confirmed sign-in.",
    "Security": "Contained the threat; removed the malicious artifact and applied the relevant patch.",
}


def to_rows(tickets: list[LabeledTicket]) -> list[dict]:
    return [
        {
            "title": t.title,
            "description": t.description,
            "category": t.category,
            "resolution": RESOLUTIONS.get(t.category, ""),
            "priority": "Medium",
            "request_type": "Incident",
        }
        for t in tickets
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--out-csv", default="data/synthetic_tickets.csv")
    ap.add_argument("--out-db", default="data/vocabulary_db.json")
    ap.add_argument("--out-model", default="data/svm_model.pkl")
    args = ap.parse_args()

    print("1) generating data (canonical train + casual eval)")
    train = generate_controlled_corpus(per_category=args.per_category, register="canonical")
    casual = generate_controlled_corpus(per_category=max(20, args.per_category // 5),
                                        seed=99, register="casual")
    write_tickets_csv(to_rows(train), args.out_csv)
    print(f"   train={len(train)}  casual_eval={len(casual)}  -> {args.out_csv}")

    print("2) building vocabulary database (5 frequent + 5 unique + synonyms + normalized)")
    db = build_vocabulary_db(train, top_freq=5, n_unique=5)
    save_vocabulary_db(db, args.out_db)
    norm_map = normalizer_from_db(db)
    cv = ConceptVocabulary.from_layman_map()  # for normalize()
    print(f"   {db['_meta']['n_synonyms']} synonyms in normalizer  -> {args.out_db}")

    print("3) training SVM + cross-validation (clean benchmark)")
    rep = cross_validate_svm(train, folds=5)
    clf = SVMClassifier(train_svm(train))
    save_model(clf.pipeline, args.out_model)
    print(f"   CV accuracy = {rep.accuracy_mean:.3f} +/- {rep.accuracy_std:.3f}")
    print(f"   CV macro-F1 = {rep.f1_macro_mean:.3f} +/- {rep.f1_macro_std:.3f}  -> {args.out_model}")

    print("4) ablation: SVM on casual input, without vs with synonym normalization")
    def acc(normalize: bool) -> float:
        ok = 0
        for t in casual:
            text = cv.normalize(t.text) if normalize else t.text
            if clf.predict(text)[0] == t.category:
                ok += 1
        return ok / len(casual)
    raw, norm = acc(False), acc(True)
    print(f"   casual, NO normalization   = {raw:.3f}")
    print(f"   casual, WITH normalization = {norm:.3f}")
    print(f"   --> synonym layer recovers {(norm - raw) * 100:.0f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
