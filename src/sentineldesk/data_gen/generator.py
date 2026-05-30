"""
Synthetic ticket generation pipeline (Round 2 Section 4.2 steps 1-6).

generate -> validate (LLM-judge, drop score < threshold) -> augment -> dedup.

Every LLM interaction goes through the injected LLMClient, so the whole
pipeline runs against StubLLMClient in tests and OllamaClient in production.
The orchestration, filtering, augmentation, and dedup logic is fully
deterministic and tested here; only the model's actual outputs differ.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from ..llm import LLMClient
from .parsers import (
    parse_augmentations,
    parse_generated_tickets,
    parse_validation_score,
)
from .prompts import (
    GENERATION_SYSTEM,
    build_augmentation_prompt,
    build_generation_prompt,
    build_validation_prompt,
)

logger = logging.getLogger("sentineldesk.data_gen")

CSV_COLUMNS = ["title", "description", "category", "resolution", "priority", "request_type"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class SyntheticDataGenerator:
    def __init__(self, llm: LLMClient, categories: list[str]) -> None:
        self.llm = llm
        self.categories = categories

    # --- step 1: generation ------------------------------------------------
    def generate(self, per_category: int, batch_size: int = 10) -> list[dict]:
        tickets: list[dict] = []
        for category in self.categories:
            collected: list[dict] = []
            while len(collected) < per_category:
                want = min(batch_size, per_category - len(collected))
                prompt = build_generation_prompt(category, want)
                raw = self.llm.complete(
                    prompt, system=GENERATION_SYSTEM, temperature=0.8, json_mode=True
                )
                batch = parse_generated_tickets(raw, category)
                if not batch:
                    logger.warning(
                        "empty/invalid batch for %s; stopping early. "
                        "raw response head: %r",
                        category, (raw or "")[:300],
                    )
                    break
                collected.extend(batch)
            logger.info("generated %d tickets for %s", len(collected), category)
            tickets.extend(collected[:per_category])
        return tickets

    # --- step 2: validation (LLM-as-judge) ---------------------------------
    def validate(self, tickets: list[dict], threshold: int = 4) -> list[dict]:
        kept: list[dict] = []
        for t in tickets:
            prompt = build_validation_prompt(t["title"], t["description"], t["category"])
            raw = self.llm.complete(prompt, temperature=0.0, json_mode=True)
            score = parse_validation_score(raw)
            if score >= threshold:
                kept.append(t)
        logger.info("validation: kept %d/%d (threshold>=%d)", len(kept), len(tickets), threshold)
        return kept

    # --- step 3: augmentation ----------------------------------------------
    def augment(self, tickets: list[dict], variations: int = 3) -> list[dict]:
        out: list[dict] = list(tickets)  # keep originals
        for t in tickets:
            prompt = build_augmentation_prompt(t["title"], t["description"], variations)
            raw = self.llm.complete(prompt, temperature=0.9, json_mode=True)
            for variant in parse_augmentations(raw):
                out.append(
                    {
                        "title": variant["title"],
                        "description": variant["description"],
                        "category": t["category"],
                        "resolution": t["resolution"],
                        "priority": t["priority"],
                        "request_type": t["request_type"],
                    }
                )
        logger.info("augmentation: %d -> %d tickets", len(tickets), len(out))
        return out

    # --- step 4: dedup -----------------------------------------------------
    @staticmethod
    def dedup(tickets: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for t in tickets:
            key = _norm(f"{t['title']} {t['description']}")
            if key and key not in seen:
                seen.add(key)
                out.append(t)
        if len(out) != len(tickets):
            logger.info("dedup: removed %d duplicates", len(tickets) - len(out))
        return out

    # --- full pipeline -----------------------------------------------------
    def run(
        self,
        per_category: int,
        threshold: int = 4,
        variations: int = 3,
        batch_size: int = 10,
    ) -> list[dict]:
        generated = self.generate(per_category, batch_size=batch_size)
        validated = self.validate(generated, threshold=threshold)
        augmented = self.augment(validated, variations=variations)
        return self.dedup(augmented)


def write_tickets_csv(tickets: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for t in tickets:
            writer.writerow({c: t.get(c, "") for c in CSV_COLUMNS})
    return path
