"""
Term-seeded generation (vocabulary-first).

For each ticket: sample a weighted subset of the category's words, hand them to
the LLM with a rotated scenario, and let it write the ticket text + resolution.
Validation is the cheap programmatic exclusivity gate (own terms present, no
forbidden cross-category terms) — at 10k tickets, LLM-judging each is
impractical, so the gate is deterministic and the label is free.
"""

from __future__ import annotations

import logging
import random
from collections import Counter

from ..llm import LLMClient
from .parsers import parse_single_ticket
from .prompts import SCENARIO_HINTS, SEEDED_SYSTEM, build_seeded_prompt
from .word_model import CategoryWordModel

logger = logging.getLogger("sentineldesk.data_gen.seeded")


class TermSeededGenerator:
    def __init__(self, llm: LLMClient, model: CategoryWordModel, seed: int = 42) -> None:
        self.llm = llm
        self.model = model
        self.rng = random.Random(seed)

    def _one(self, category: str, max_retries: int = 2) -> dict | None:
        for _ in range(max_retries + 1):
            terms = self.model.sample_terms(category, self.rng)
            scenario = self.rng.choice(SCENARIO_HINTS)
            prompt = build_seeded_prompt(category, terms, scenario)
            raw = self.llm.complete(
                prompt, system=SEEDED_SYSTEM, temperature=0.9, json_mode=True
            )
            ticket = parse_single_ticket(raw)
            if not ticket:
                continue
            blob = f"{ticket['title']} {ticket['description']}"
            if self.model.passes_exclusivity(category, blob):
                ticket["category"] = category
                ticket["seed_terms"] = terms
                return ticket
        return None  # exhausted retries; caller counts the miss

    def generate(
        self,
        total: int,
        log_every: int = 500,
        on_checkpoint=None,
        existing: list[dict] | None = None,
    ) -> list[dict]:
        """Generate `total` tickets, balanced across categories.

        on_checkpoint(tickets) runs after each category so a long run persists progress.
        existing: already-generated tickets (from a prior partial run) — categories
        already at target are skipped, partial ones are topped up. Enables resume.
        """
        cats = self.model.categories
        per_cat = total // len(cats)
        tickets: list[dict] = list(existing) if existing else []
        have = Counter(t.get("category") for t in tickets)
        if existing:
            logger.info("resuming from %d existing tickets: %s", len(tickets), dict(have))
        misses = 0
        for category in cats:
            made = have.get(category, 0)
            if made >= per_cat:
                logger.info("%s: %d tickets (already complete, skipping)", category, made)
                continue
            attempts = 0
            while made < per_cat and attempts < per_cat * 4:
                attempts += 1
                try:
                    t = self._one(category)
                except Exception as exc:  # transient LLM/network error — don't kill an overnight run
                    logger.warning("generation error (%s): %s", category, exc)
                    t = None
                if t:
                    tickets.append(t)
                    made += 1
                    if len(tickets) % log_every == 0:
                        logger.info("generated %d/%d", len(tickets), per_cat * len(cats))
                else:
                    misses += 1
            logger.info("%s: %d tickets (%d attempts)", category, made, attempts)
            if on_checkpoint is not None:
                on_checkpoint(tickets)  # persist progress; a later crash can't lose it
        if misses:
            logger.info("rejected %d tickets that failed the exclusivity gate", misses)
        return tickets
