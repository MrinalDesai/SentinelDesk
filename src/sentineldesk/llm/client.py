"""
Shared LLM client used by every stage that talks to Mistral (data generation,
LLM-as-judge, VGAC classification, resolution, escalation scoring).

  * `LLMClient` is the interface every implementation honours.
  * `OllamaClient` talks to a local Ollama server via /api/chat (stdlib urllib,
    no extra dependency). Supports JSON-forced output via Ollama's
    `format: "json"`. Returns "" on any failure so callers can decide how to
    degrade rather than crashing.
  * `StubLLMClient` drives a user-supplied handler(prompt) -> str, so the full
    generate/validate/augment pipeline is testable without a GPU.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Optional, Protocol


class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        """Return the model's text response, or "" on failure."""
        ...


class StubLLMClient:
    """Deterministic test double. `handler` maps a prompt to a canned response."""

    def __init__(self, handler: Callable[[str], str]) -> None:
        self._handler = handler
        self.calls: list[str] = []  # prompts seen, for assertions

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(prompt)
        return self._handler(prompt)


class CacheLLMClient:
    """Replays tiebreak responses recorded from a REAL Ollama run.

    Used as the fallback when a live Ollama server isn't reachable, so a demo
    can still show the rare LLM-tiebreak path resolving instead of silently
    escalating. The cache is a JSON map: ticket-text -> {raw_response, choice,
    candidates, source, ...}. On a cache miss it returns "" — exactly how a
    missing model degrades — so behaviour stays honest for unseen tickets.
    """

    def __init__(self, cache_path) -> None:
        import json
        import pathlib
        import re

        self.path = pathlib.Path(cache_path)
        try:
            self.cache = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.cache = {}
        self._re = re.compile(r"Ticket:\s*(.+)")
        self.calls: list[str] = []

    def _lookup(self, ticket: str):
        e = self.cache.get(ticket)
        if e:
            return e
        for k, v in self.cache.items():          # tolerate minor whitespace drift
            if k.strip()[:80] == ticket.strip()[:80]:
                return v
        return None

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(prompt)
        m = self._re.search(prompt)
        if not m:
            return ""
        entry = self._lookup(m.group(1).strip())
        if not entry:
            return ""
        return entry.get("raw_response") or entry.get("choice") or ""


class OllamaClient:
    """LLM client backed by a local Ollama server (your machine)."""

    def __init__(
        self,
        model: str = "mistral:7b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            body["format"] = "json"

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("message", {}).get("content", "") or ""
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return ""
