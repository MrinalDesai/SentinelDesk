"""
Stage 0 — deterministic safety gate.

Runs BEFORE any LLM call. Scans the ticket's title + description against the
compiled high-stakes patterns. A match means: escalate immediately, bypass
every downstream LLM stage. No match means the ticket proceeds to PII
redaction. The whole call is pure regex over short text, so it is sub-5ms.
"""

from __future__ import annotations

import time

from ..models import SafetyResult
from .patterns import HIGH_STAKES


def safety_check(title: str, description: str) -> SafetyResult:
    """Return a SafetyResult. `bypass_llm=True` => escalate now.

    First-match wins: HIGH_STAKES is ordered most-unambiguous first, so the
    earliest match is the most defensible escalation reason. We scan title and
    description as one blob because a high-stakes signal anywhere is enough.
    """
    start = time.perf_counter()
    text = f"{title}\n{description}"

    for category in HIGH_STAKES:
        for pattern in category.compiled:
            m = pattern.search(text)
            if m:
                latency_ms = (time.perf_counter() - start) * 1000.0
                return SafetyResult(
                    bypass_llm=True,
                    matched_category=category.name,
                    trigger=m.group(0),
                    department=category.department,
                    severity=category.severity,
                    latency_ms=latency_ms,
                )

    latency_ms = (time.perf_counter() - start) * 1000.0
    return SafetyResult(bypass_llm=False, latency_ms=latency_ms)
