"""The state object LangGraph passes from agent to agent, and the Stage 0 result."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..config import Priority, TicketState
from .classification import (
    Classification,
    EscalationDecision,
    Resolution,
    Routing,
)
from .ticket import CleanTicket, RawTicket


class SafetyResult(BaseModel):
    """Stage 0 output. `bypass_llm` true means escalate immediately."""

    bypass_llm: bool
    matched_category: Optional[str] = None
    trigger: Optional[str] = None       # the substring/pattern that fired
    department: Optional[str] = None     # where to escalate
    severity: Optional[str] = None       # e.g. "Critical"
    latency_ms: float = 0.0


class PipelineState(BaseModel):
    """Mutable state threaded through agents 1->5.

    Each stage reads what it needs and writes its own result field. Keeping a
    single object (rather than passing growing tuples) is what lets every
    agent write one audit_log entry from a consistent view of the world.
    """

    ticket_id: str
    raw: RawTicket
    state: TicketState = TicketState.SUBMITTED

    # filled in stage by stage
    safety: Optional[SafetyResult] = None
    clean: Optional[CleanTicket] = None
    classification: Optional[Classification] = None
    routing: Optional[Routing] = None
    resolution: Optional[Resolution] = None
    decision: Optional[EscalationDecision] = None

    # running audit trail of (agent_name, note) entries
    audit_trail: list[str] = Field(default_factory=list)

    @property
    def priority(self) -> Priority:
        return self.raw.priority

    def note(self, agent: str, message: str) -> None:
        """Append an audit breadcrumb. The real audit log is the DB table;
        this is the in-memory mirror used for tests and reconstruction."""
        self.audit_trail.append(f"{agent}: {message}")
