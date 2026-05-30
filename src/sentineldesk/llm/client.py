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
