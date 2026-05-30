"""
SentinelDesk central configuration.

Single source of truth for the 7 domain categories, priority levels, and the
decision thresholds that appear throughout the Round 2 design (confidence
0.75 / 0.90, quality 3/5). Importing constants from here keeps the agents,
the SVM trainer, and the tests from drifting out of sync.
"""

from __future__ import annotations

from enum import Enum


class Domain(str, Enum):
    """The 7 ITSM domains the classifier routes into.

    Defined as a str-Enum so values serialize cleanly to JSON and compare
    equal to their string form (Domain.NETWORK == "Network").
    """

    INFRASTRUCTURE = "Infrastructure"
    APPLICATION = "Application"
    SECURITY = "Security"
    DATABASE = "Database"
    STORAGE = "Storage"
    NETWORK = "Network"
    ACCESS_MANAGEMENT = "Access Management"


class Priority(str, Enum):
    """Ticket priority as stated by the submitter (may be overridden by impact)."""

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

    @property
    def level(self) -> int:
        """Numeric rank used by the RBAC routing query (higher = more urgent).

        The router selects rules WHERE priority_level <= ticket level, so we
        need a stable integer ordering, not the enum's declaration order.
        """
        return {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}[self.value]


class TicketState(str, Enum):
    """Lifecycle states from the Round 2 state machine (Section 8)."""

    SUBMITTED = "SUBMITTED"
    SAFETY_CHECK = "SAFETY_CHECK"
    CRITICAL_ESCALATION = "CRITICAL_ESCALATION"
    PII_REDACTION = "PII_REDACTION"
    CLASSIFYING = "CLASSIFYING"
    ROUTING = "ROUTING"
    RESOLVING = "RESOLVING"
    EVALUATING = "EVALUATING"
    RESOLVED = "RESOLVED"
    ESCALATING = "ESCALATING"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"
    CORRECTION_LOOP = "CORRECTION_LOOP"


class Action(str, Enum):
    """Terminal action chosen by the escalation agent."""

    AUTO_RESOLVE = "AUTO_RESOLVE"
    ESCALATE = "ESCALATE"


# --- Decision thresholds (Round 2 §3.5, §14.2) -----------------------------

# Below this ensemble confidence, the ticket is escalated rather than resolved.
CONFIDENCE_THRESHOLD: float = 0.75

# At or above this, both Mistral and SVM are treated as strongly agreeing.
HIGH_CONFIDENCE_THRESHOLD: float = 0.90

# LLM-as-judge resolution quality (1-5). Below this, escalate.
QUALITY_THRESHOLD: int = 3

# Penalty multiplier applied when Mistral and SVM disagree and we fall back
# to the higher-confidence single prediction (Round 2 R7).
ENSEMBLE_DISAGREEMENT_PENALTY: float = 0.9

# Repeated-issue automation trigger: N occurrences within the window.
REPEAT_ISSUE_COUNT: int = 3
REPEAT_ISSUE_WINDOW_DAYS: int = 7


ALL_DOMAINS: tuple[Domain, ...] = tuple(Domain)
