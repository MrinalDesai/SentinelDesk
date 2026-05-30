"""
Parsers for the LLM responses in the generation pipeline.

All pure functions, all tolerant of the usual LLM mess (markdown fences,
leading prose, trailing junk). They return clean Python structures or empty
results — never raise — so a single malformed batch can't abort a long run.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _extract_json(raw: str) -> Any:
    """Pull the first JSON array or object out of a possibly-messy string."""
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    # try the whole thing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # else grab the first [...] or {...} block
    match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


# Keys an LLM commonly uses to wrap an array when forced into object output.
_LIST_KEYS = ("tickets", "variants", "items", "data", "results", "list")


def _coerce_list(data: Any) -> list:
    """Return a list whether the model emitted a bare array or wrapped it in
    an object. Ollama's format:"json" tends to produce a top-level object, so
    we look for the first list-valued key (preferring the common ones)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in _LIST_KEYS:
            if isinstance(data.get(key), list):
                return data[key]
        # otherwise take the first list value present
        for value in data.values():
            if isinstance(value, list):
                return value
    return []


_VALID_PRIORITIES = {"Critical", "High", "Medium", "Low"}
_VALID_REQUEST_TYPES = {"Incident", "Service Request", "Problem", "Change"}


def parse_generated_tickets(raw: str, category: str) -> list[dict]:
    """Return a list of normalised ticket dicts. Drops malformed entries."""
    data = _coerce_list(_extract_json(raw))
    tickets: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not title or not desc:
            continue
        priority = str(item.get("priority", "Medium")).strip().title()
        if priority not in _VALID_PRIORITIES:
            priority = "Medium"
        rtype = str(item.get("request_type", "Incident")).strip()
        if rtype not in _VALID_REQUEST_TYPES:
            rtype = "Incident"
        tickets.append(
            {
                "title": title,
                "description": desc,
                "resolution": str(item.get("resolution", "")).strip(),
                "category": category,
                "priority": priority,
                "request_type": rtype,
            }
        )
    return tickets


def parse_validation_score(raw: str) -> int:
    """Return an integer score 1-5, or 0 if unparseable (treated as reject)."""
    data = _extract_json(raw)
    if isinstance(data, dict) and "score" in data:
        try:
            score = int(round(float(data["score"])))
            return max(0, min(5, score))
        except (ValueError, TypeError):
            return 0
    # fallback: first bare digit 1-5 in the text
    m = re.search(r"\b([1-5])\b", raw or "")
    return int(m.group(1)) if m else 0


def parse_augmentations(raw: str) -> list[dict]:
    """Return a list of {title, description} variant dicts."""
    data = _coerce_list(_extract_json(raw))
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        desc = str(item.get("description", "")).strip()
        if title and desc:
            out.append({"title": title, "description": desc})
    return out


def parse_single_ticket(raw: str) -> dict | None:
    """Parse one {title, description, resolution} object. None if unusable."""
    data = _extract_json(raw)
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else None
    if not isinstance(data, dict):
        return None
    title = str(data.get("title", "")).strip()
    desc = str(data.get("description", "")).strip()
    if not title or not desc:
        return None
    return {
        "title": title,
        "description": desc,
        "resolution": str(data.get("resolution", "")).strip(),
    }
