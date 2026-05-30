"""
Corpus loading for vocabulary building and SVM training.

Kept dependency-free (stdlib csv, no pandas) because the vocabulary layers
only need a list of (text, category) pairs. pandas enters later for the SVM
feature pipeline where it actually earns its weight.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabeledTicket:
    text: str          # title + description, the surface we model
    category: str
    title: str = ""
    description: str = ""
    priority: str = "Medium"
    resolution: str = ""   # the fix text, used to build the Graph-RAG resolution edges


def load_tickets_csv(path: str | Path) -> list[LabeledTicket]:
    """Load tickets from a CSV with columns: title, description, category, priority.

    Matches the Round 2 synthetic_tickets.csv schema. `text` is the
    concatenation the classifier sees.
    """
    path = Path(path)
    tickets: list[LabeledTicket] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            title = (row.get("title") or "").strip()
            desc = (row.get("description") or "").strip()
            cat = (row.get("category") or "").strip()
            if not cat:
                continue
            tickets.append(
                LabeledTicket(
                    text=f"{title} {desc}".strip(),
                    category=cat,
                    title=title,
                    description=desc,
                    priority=(row.get("priority") or "Medium").strip(),
                    resolution=(row.get("resolution") or "").strip(),
                )
            )
    return tickets


def group_by_category(tickets: list[LabeledTicket]) -> dict[str, list[str]]:
    """Return {category: [text, ...]} for the per-category n-gram pass."""
    grouped: dict[str, list[str]] = {}
    for t in tickets:
        grouped.setdefault(t.category, []).append(t.text)
    return grouped
