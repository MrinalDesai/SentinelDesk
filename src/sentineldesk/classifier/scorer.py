"""
Deterministic lexical scorer (Stage 2 classification core).

Transparent, fast, no LLM. It scores each category from four signals and
records every term that fired, so an explanation is just reading the result
back. The LLM/ensemble only gets involved for the edge cases this flags.

    score(c) = w_dept·dept_mention(c)
             + w_unique·unique_hits(c)     (discriminative TF-IDF terms)
             + w_freq·freq_hits(c)         (frequent N-gram terms, shared filler removed)
             + w_layman·layman_hits(c)     (casual phrasings from the layman map)

A ticket is an edge case (-> resolver ladder) when nothing matches, the
top-two margin is too thin, or only shared/common words fired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..corpus import LabeledTicket
from ..data_gen.layman_map import LAYMAN_MAP
from ..vocabulary.analysis import ngram_counts_by_category, tfidf_weights_by_category


# Explicit department-name words. dept_mention is a strong weight, not a
# short-circuit (the score still decides), per the design.
DEPT_WORDS: dict[str, list[str]] = {
    "Network": ["network", "networking"],
    "Application": ["application", "app"],
    "Database": ["database"],
    "Storage": ["storage"],
    "Infrastructure": ["infrastructure", "infra", "server"],
    "Security": ["security"],
    "Access Management": ["access management", "identity", "login", "account"],
}


@dataclass
class ScoreResult:
    category: str
    confidence: float
    scores: dict[str, float]
    margin: float
    is_edge_case: bool
    edge_reason: str | None
    matched: dict[str, list[str]]  # signal -> terms that fired for the winner
    runner_up: str | None
    common_hits: list[str] = field(default_factory=list)


def _present(term: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


class VocabModel:
    """Holds the per-category term sets the scorer matches against."""

    def __init__(
        self,
        unique_terms: dict[str, set[str]],
        freq_terms: dict[str, set[str]],
        common: set[str],
        dept_words: dict[str, list[str]] | None = None,
        layman: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        self.unique_terms = unique_terms
        self.freq_terms = freq_terms
        self.common = common
        self.dept_words = dept_words or DEPT_WORDS
        self.layman = layman or LAYMAN_MAP
        self.categories = sorted(unique_terms)

    @classmethod
    def build(
        cls, tickets: list[LabeledTicket], top_unique: int = 20, top_freq: int = 20
    ) -> "VocabModel":
        tfidf = tfidf_weights_by_category(tickets, top_n=top_unique)
        ngram = ngram_counts_by_category(tickets, top_n=top_freq)

        # COMMON = terms appearing in >=2 categories' frequent lists (leaked filler)
        seen: dict[str, int] = {}
        for terms in ngram.values():
            for t, _ in terms:
                seen[t] = seen.get(t, 0) + 1
        common = {t for t, c in seen.items() if c >= 2}

        unique_terms = {c: {t for t, _ in v} for c, v in tfidf.items()}
        freq_terms = {c: {t for t, _ in v} - common for c, v in ngram.items()}
        return cls(unique_terms, freq_terms, common)


class DeterministicScorer:
    def __init__(
        self,
        vocab: VocabModel,
        w_dept: float = 3.0,
        w_unique: float = 2.0,
        w_freq: float = 1.0,
        w_layman: float = 1.0,
        edge_margin: float = 2.0,
    ) -> None:
        self.v = vocab
        self.w_dept = w_dept
        self.w_unique = w_unique
        self.w_freq = w_freq
        self.w_layman = w_layman
        self.edge_margin = edge_margin

    def classify(self, title: str, description: str = "") -> ScoreResult:
        text = f"{title} {description}".lower()
        text_norm = text.replace("'", "").replace("\u2019", "")

        scores: dict[str, float] = {}
        matched_all: dict[str, dict[str, list[str]]] = {}
        for c in self.v.categories:
            dept = [w for w in self.v.dept_words.get(c, []) if _present(w, text)]
            uniq = [t for t in self.v.unique_terms.get(c, set()) if _present(t, text)]
            uniq_set = set(uniq)
            # a term counts once, at its strongest tier: drop freq terms already
            # credited as unique, so nothing is double-counted.
            freq = [
                t
                for t in self.v.freq_terms.get(c, set())
                if t not in uniq_set and _present(t, text)
            ]
            lay = []
            for term, phrasings in self.v.layman.get(c, {}).items():
                for p in phrasings:
                    if p.lower().replace("'", "") in text_norm:
                        lay.append(p)
            score = (
                self.w_dept * (1 if dept else 0)
                + self.w_unique * len(uniq)
                + self.w_freq * len(freq)
                + self.w_layman * len(lay)
            )
            scores[c] = score
            matched_all[c] = {"dept": dept, "unique": uniq, "freq": freq, "layman": lay}

        common_hits = sorted({t for t in self.v.common if _present(t, text)})

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        (top_cat, top_score) = ranked[0]
        (second_cat, second_score) = ranked[1] if len(ranked) > 1 else (None, 0.0)
        margin = top_score - second_score
        confidence = top_score / (top_score + second_score) if top_score > 0 else 0.0

        edge_reason = None
        if top_score == 0:
            edge_reason = "no category vocabulary matched"
        elif margin < self.edge_margin:
            edge_reason = f"top two are too close ({top_score:g} vs {second_score:g})"
        else:
            win = matched_all[top_cat]
            if not win["dept"] and not win["unique"] and common_hits:
                edge_reason = "only shared/common words matched"
        is_edge = edge_reason is not None

        return ScoreResult(
            category=top_cat,
            confidence=round(confidence, 3),
            scores=scores,
            margin=margin,
            is_edge_case=is_edge,
            edge_reason=edge_reason,
            matched=matched_all[top_cat],
            runner_up=second_cat,
            common_hits=common_hits,
        )
