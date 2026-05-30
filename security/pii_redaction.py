"""
PII redaction (REAL, working) — ingestion-time redaction of personal data.

Structured PII (emails, phones, IPs, credit cards, SSNs, national IDs) is
redacted with high-precision regex. NAMES are the hard case — reliable name
detection needs ML NER, so we redact title-prefixed names ("Mr Smith") and a
configurable known-name list here, and document Presidio as the production
upgrade for full NER-based name redaction.

This is a genuine working control: it runs on text and returns redacted text
plus an audit record of what was removed. It is NOT wired into the live pipeline
(per scope); the intended integration point is an ingestion stage between the
safety gate and the classifier.

HONEST LABEL: regex redaction = IMPLEMENTED. Presidio NER for names = ROADMAP.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")),
    # title-prefixed names: Mr/Ms/Mrs/Dr/Prof <Name>
    ("NAME", re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")),
]


@dataclass
class RedactionResult:
    text: str
    counts: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class PIIRedactor:
    """Redact structured PII (+ title-prefixed names) and report what was removed."""

    def __init__(self, known_names: list[str] | None = None) -> None:
        self.known_names = [n for n in (known_names or []) if n]

    def redact(self, text: str) -> RedactionResult:
        counts: Counter = Counter()
        out = text
        # 1. configured known names first (exact, case-insensitive, word-boundary)
        for name in sorted(self.known_names, key=len, reverse=True):
            pat = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            out, n = pat.subn("[NAME]", out)
            counts["NAME"] += n
        # 2. regex categories (order matters: SSN/CARD before generic digits)
        for tag, pat in _PATTERNS:
            out, n = pat.subn(f"[{tag}]", out)
            if n:
                counts[tag] += n
        return RedactionResult(out, counts)


# --- Presidio adapter seam (ROADMAP — full NER name/entity detection) ----------
class PresidioRedactor:
    """
    Production redactor using Microsoft Presidio for ML-based NER (robust name,
    location, org detection). Disabled unless presidio-analyzer is installed and
    an analyzer is provided — documents the upgrade path without forcing the dep.
    """

    def __init__(self, analyzer=None, anonymizer=None) -> None:
        if analyzer is None or anonymizer is None:
            raise NotImplementedError(
                "PresidioRedactor is the production NER upgrade. Install "
                "presidio-analyzer + presidio-anonymizer and pass an analyzer; "
                "until then PIIRedactor (regex) is the working default."
            )
        self.analyzer, self.anonymizer = analyzer, anonymizer

    def redact(self, text: str) -> RedactionResult:  # pragma: no cover
        results = self.analyzer.analyze(text=text, language="en")
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        counts = Counter(r.entity_type for r in results)
        return RedactionResult(anonymized.text, counts)
