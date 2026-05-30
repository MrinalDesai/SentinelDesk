"""
Concept-grouped vocabulary: each concept is a list of surface forms
(canonical first, then synonyms). Built from the curated layman map (the LLM
tops it up on a real run). Provides synonym -> canonical normalization, with
cross-category collisions demoted to ambiguous (dropped from the map).
"""

from __future__ import annotations

import re

from ..data_gen.layman_map import LAYMAN_MAP


class ConceptVocabulary:
    def __init__(self, groups: dict[str, list[list[str]]], syn2canon: dict[str, str]) -> None:
        self.groups = groups            # category -> [[canonical, syn, syn, ...], ...]
        self.syn2canon = syn2canon      # synonym(lower) -> canonical term
        # replace longest phrases first so "websites won't load" beats "load"
        self._ordered = sorted(syn2canon.keys(), key=len, reverse=True)

    @classmethod
    def from_layman_map(cls, layman: dict[str, dict[str, list[str]]] = LAYMAN_MAP) -> "ConceptVocabulary":
        groups: dict[str, list[list[str]]] = {}
        syn_to_canons: dict[str, set[str]] = {}
        for category, terms in layman.items():
            glist: list[list[str]] = []
            for term, syns in terms.items():
                glist.append([term, *syns])
                for s in syns:
                    syn_to_canons.setdefault(s.lower(), set()).add(term)
            groups[category] = glist
        # collision demotion: a synonym that maps to >1 canonical is ambiguous
        syn2canon = {s: next(iter(c)) for s, c in syn_to_canons.items() if len(c) == 1}
        return cls(groups, syn2canon)

    def normalize(self, text: str) -> str:
        """Rewrite known synonyms to their canonical term (word-boundary, longest-first)."""
        out = text
        for syn in self._ordered:
            out = re.sub(r"\b" + re.escape(syn) + r"\b", self.syn2canon[syn], out, flags=re.IGNORECASE)
        return out

    @property
    def n_synonyms(self) -> int:
        return len(self.syn2canon)
