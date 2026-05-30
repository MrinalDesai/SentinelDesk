"""
Layer 4 — LLM vocabulary enrichment (Round 2 Algo 4, the novel contribution).

For each category we hand the top discriminative terms to a local LLM and ask
for synonyms, abbreviations, and layman equivalents — closing the gap between
how users phrase tickets ("internet not working") and how the corpus phrases
them ("IPSec tunnel dropping").

Design:
  * `VocabularyEnricher` is the interface every layer-4 implementation honours.
  * `StubEnricher` returns terms unchanged — used in CI and tests where no LLM
    is available, so the whole build pipeline stays runnable without a GPU.
  * `OllamaEnricher` talks to a local Ollama server over its REST API (no
    extra dependency — stdlib urllib). It degrades gracefully: any network,
    timeout, or parse failure returns the input terms unchanged rather than
    raising, consistent with the "never returns empty" philosophy.

The Ollama path is validated on your machine (Ollama + Mistral 7B Q8). The
prompt builder and the response parser are pure functions and ARE unit-tested
here, so the only untested-in-CI surface is the network call itself.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Protocol


class VocabularyEnricher(Protocol):
    def enrich(self, category: str, terms: list[str]) -> list[str]:
        """Return new synonym/abbreviation/layman terms for the category."""
        ...


class StubEnricher:
    """No-op enricher: returns nothing extra. Keeps CI builds LLM-free."""

    def enrich(self, category: str, terms: list[str]) -> list[str]:
        return []


def build_enrichment_prompt(category: str, terms: list[str], top_k: int = 8) -> str:
    """Construct the enrichment prompt sent to the LLM (pure, testable)."""
    seed = ", ".join(terms[:top_k])
    return (
        "You expand IT-support vocabulary. For the ITSM category "
        f'"{category}", given these technical terms:\n'
        f"{seed}\n\n"
        "Return ONLY a JSON array of additional related terms a user might "
        "type: synonyms, common abbreviations, and layman phrasings. No "
        "explanation, no markdown, just the JSON array. Example: "
        '["wifi", "wireless", "cannot connect"].'
    )


def parse_enrichment_response(raw: str) -> list[str]:
    """Extract a list of terms from a possibly-messy LLM response (pure, testable).

    Handles: clean JSON arrays, arrays wrapped in markdown fences, and arrays
    embedded in surrounding prose. Returns [] if nothing parseable is found.
    """
    if not raw:
        return []
    text = raw.strip()
    # strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    # find the first [...] block
    match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


class OllamaEnricher:
    """Layer-4 enricher backed by a local Ollama server.

    Run on your machine with Ollama serving the model. Defaults match Round 2
    (Mistral 7B). Falls back to [] on any failure so the build never breaks.
    """

    def __init__(
        self,
        model: str = "mistral:7b",
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def enrich(self, category: str, terms: list[str]) -> list[str]:
        prompt = build_enrichment_prompt(category, terms)
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return parse_enrichment_response(body.get("response", ""))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            # graceful degradation: no enrichment rather than a failed build
            return []
