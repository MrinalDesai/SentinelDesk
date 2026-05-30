"""Ticket data contracts: the raw inbound ticket and the redacted form."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from ..config import Priority


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PIIEntity(BaseModel):
    """A single piece of PII located by Presidio, kept for the audit log.

    We store the entity *type* and offsets, never the original text, so the
    audit record itself does not become a PII leak.
    """

    entity_type: str  # e.g. EMAIL, IP_ADDRESS, PHONE_NUMBER, PERSON, URL, DATE
    start: int
    end: int
    score: float = Field(ge=0.0, le=1.0)
    field: str  # "title" or "description" — PII is redacted per-field


class RawTicket(BaseModel):
    """Inbound ticket exactly as submitted, before any processing."""

    title: str
    description: str
    priority: Priority = Priority.MEDIUM
    submitted_by: Optional[str] = None  # user id / email; redacted downstream
    submitted_at: datetime = Field(default_factory=_utcnow)

    def combined_text(self) -> str:
        """Title + description, the surface most stages classify against."""
        return f"{self.title}\n{self.description}".strip()


class CleanTicket(BaseModel):
    """Ticket after Stage 1 PII redaction. Carries the audit handle forward."""

    clean_title: str
    clean_description: str
    priority: Priority
    audit_id: str
    pii_entities: list[PIIEntity] = Field(default_factory=list)

    def combined_text(self) -> str:
        return f"{self.clean_title}\n{self.clean_description}".strip()
