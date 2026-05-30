"""
Self-correction loop (minimal, real).

When a human reroutes a misrouted ticket, the phrasing the dictionary missed is
learned as a new synonym pointing to the correct category's anchor term. It
joins the normalizer immediately — future similar tickets route correctly with
NO retraining, because routing improvement is a dictionary lookup.

Two things make this safe and credible:
  - GUARDS: a phrase is rejected if it already maps to a different category
    (collision) or is too generic, so corrections can't recreate a magnet class.
  - AUDIT: every learned entry is logged (phrase, from -> to, support) — the
    vocabulary change is inspectable, unlike opaque model retraining.

Optionally requires `min_support` consistent corrections before committing, so a
single mistaken correction doesn't pollute the vocabulary.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..vocabulary.concepts import ConceptVocabulary

_GENERIC = {
    "not working", "broken", "down", "issue", "problem", "error", "failing",
    "slow", "stuck", "help", "urgent", "the system", "it", "stuff",
}


@dataclass
class CorrectionResult:
    status: str            # "learned" | "pending" | "rejected_collision" | "rejected_generic"
    phrase: str
    category: str | None = None
    anchor: str | None = None
    detail: str = ""


@dataclass
class CorrectionStore:
    anchors: dict[str, str]                                  # category -> anchor canonical term
    learned: dict[str, str] = field(default_factory=dict)   # phrase -> anchor canonical
    learned_category: dict[str, str] = field(default_factory=dict)  # phrase -> category
    audit: list[dict] = field(default_factory=list)
    _pending: dict = field(default_factory=lambda: defaultdict(Counter))

    @classmethod
    def from_vocab(cls, cv: ConceptVocabulary) -> "CorrectionStore":
        # anchor = the canonical of each category's first concept group (a term the SVM knows)
        anchors = {cat: groups[0][0] for cat, groups in cv.groups.items() if groups}
        return cls(anchors=anchors)

    def record(self, phrase: str, correct_category: str, min_support: int = 1) -> CorrectionResult:
        phrase = " ".join(phrase.lower().split())
        if not phrase or phrase in _GENERIC or len(phrase) < 3:
            return CorrectionResult("rejected_generic", phrase, detail="too generic to be a reliable signal")
        if phrase in self.learned_category and self.learned_category[phrase] != correct_category:
            return CorrectionResult("rejected_collision", phrase,
                                    detail=f"already learned for {self.learned_category[phrase]}")
        if correct_category not in self.anchors:
            return CorrectionResult("rejected_generic", phrase, detail=f"unknown category {correct_category}")

        self._pending[phrase][correct_category] += 1
        cat, support = self._pending[phrase].most_common(1)[0]
        if support < min_support:
            return CorrectionResult("pending", phrase, category=cat,
                                    detail=f"{support}/{min_support} confirmations")
        anchor = self.anchors[cat]
        self.learned[phrase] = anchor
        self.learned_category[phrase] = cat
        self.audit.append({"phrase": phrase, "to_category": cat, "anchor": anchor, "support": support})
        return CorrectionResult("learned", phrase, category=cat, anchor=anchor)

    def normalize(self, text: str, cv: ConceptVocabulary) -> str:
        """Base normalization + the learned overlay."""
        out = cv.normalize(text)
        for phrase, anchor in sorted(self.learned.items(), key=lambda kv: len(kv[0]), reverse=True):
            out = re.sub(r"\b" + re.escape(phrase) + r"\b", anchor, out, flags=re.IGNORECASE)
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"learned": self.learned, "learned_category": self.learned_category, "audit": self.audit},
            indent=2), encoding="utf-8")
        return path

    def load(self, path: str | Path) -> "CorrectionStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.learned = data.get("learned", {})
        self.learned_category = data.get("learned_category", {})
        self.audit = data.get("audit", [])
        return self
