"""Tests for the data contracts: enums, priority ranking, pipeline state."""

from sentineldesk.config import (
    CONFIDENCE_THRESHOLD,
    Action,
    Domain,
    Priority,
    TicketState,
)
from sentineldesk.models import (
    Classification,
    PipelineState,
    RawTicket,
)


def test_seven_domains():
    assert len(list(Domain)) == 7
    assert Domain.ACCESS_MANAGEMENT == "Access Management"


def test_priority_ordering():
    assert Priority.CRITICAL.level > Priority.HIGH.level
    assert Priority.HIGH.level > Priority.MEDIUM.level
    assert Priority.MEDIUM.level > Priority.LOW.level


def test_rawticket_combined_text():
    t = RawTicket(title="VPN down", description="cannot connect")
    assert "VPN down" in t.combined_text()
    assert "cannot connect" in t.combined_text()


def test_pipeline_state_defaults_and_notes():
    state = PipelineState(
        ticket_id="T-1",
        raw=RawTicket(title="x", description="y", priority=Priority.HIGH),
    )
    assert state.state == TicketState.SUBMITTED
    assert state.priority == Priority.HIGH
    state.note("IntakeAgent", "redacted 2 entities")
    assert state.audit_trail == ["IntakeAgent: redacted 2 entities"]


def test_classification_confidence_bounds():
    c = Classification(category=Domain.NETWORK, confidence=0.83)
    assert 0.0 <= c.confidence <= 1.0
    assert CONFIDENCE_THRESHOLD == 0.75


def test_models_roundtrip_json():
    state = PipelineState(
        ticket_id="T-2",
        raw=RawTicket(title="t", description="d"),
    )
    dumped = state.model_dump_json()
    restored = PipelineState.model_validate_json(dumped)
    assert restored.ticket_id == "T-2"


def test_action_enum():
    assert Action.AUTO_RESOLVE.value == "AUTO_RESOLVE"
    assert Action.ESCALATE.value == "ESCALATE"
