"""
Tests for the Stage 0 safety layer.

Two things matter here and both are pinned:
  - TRUE POSITIVES: real high-stakes tickets must escalate to the right team.
  - BENIGN NEAR-MISSES: tickets that merely *mention* a scary word in a
    routine context must NOT escalate. These guard against the over-escalation
    that bare-keyword matching would cause. If a future retune breaks one of
    these, the test makes it visible instead of silently flooding the SOC.
"""

import pytest

from sentineldesk.safety import (
    DOCUMENTED_COUNT,
    HIGH_STAKES,
    PROPOSED_COUNT,
    safety_check,
)


# (title, description, expected_category)
TRUE_POSITIVES = [
    ("Ransomware alert", "ransomware encrypting files on the share", "Ransomware/Malware"),
    ("AV console", "malware detected on 12 endpoints", "Ransomware/Malware"),
    ("Urgent", "we have a data breach, customer records exposed", "Data Breach"),
    ("Help", "unauthorized access to the finance system", "Data Breach"),
    ("MAJOR", "all systems down across the org", "Complete Outage"),
    ("Outage", "complete outage of the platform", "Complete Outage"),
    ("DB", "production database is down", "Production DB Down"),
    ("prod", "prod db crashed", "Production DB Down"),
    ("Sec", "zero day being exploited", "Security Incident"),
    ("vuln", "remote code execution on the gateway", "Security Incident"),
    ("Compliance", "possible GDPR breach reported", "DPDP/Compliance"),
    ("Legal", "DPDP violation in the export job", "DPDP/Compliance"),
    ("Facilities", "fire in the datacenter", "Physical Emergency"),
    ("Power", "power failure at the primary site", "Physical Emergency"),
    # proposed categories
    ("SOC", "lateral movement observed on the domain controller", "Active Intrusion"),
    ("Leak", "employee credentials leaked on pastebin", "Mass Credential Leak"),
    ("Net", "ongoing DDoS against the edge", "DDoS Attack"),
    ("DB", "database corruption detected after the patch", "Data Loss/Corruption"),
]

# Tickets that mention a sensitive word but should be routed normally, NOT
# escalated by Stage 0.
BENIGN_NEAR_MISSES = [
    ("License renewal", "please renew our anti-malware software license"),
    ("Training", "scheduling the annual ransomware-awareness training session"),
    ("Access request", "user forgot password and is locked out of email"),
    ("Report", "monthly security report shows no incidents this period"),
    ("Question", "what is our data protection policy for new hires"),
    ("Provisioning", "need access to the production database read replica"),
    ("Docs", "update the disaster recovery runbook for power outages"),
    ("Survey", "exploring options to reduce data storage costs"),
]


@pytest.mark.parametrize("title,desc,expected", TRUE_POSITIVES)
def test_true_positives_escalate(title, desc, expected):
    result = safety_check(title, desc)
    assert result.bypass_llm is True, f"expected escalation for: {desc!r}"
    assert result.matched_category == expected
    assert result.department  # a routing target was set
    assert result.severity == "Critical"


@pytest.mark.parametrize("title,desc", BENIGN_NEAR_MISSES)
def test_benign_near_misses_pass_through(title, desc):
    result = safety_check(title, desc)
    assert result.bypass_llm is False, (
        f"false escalation (over-trigger) for benign ticket: {desc!r} "
        f"matched {result.matched_category!r} on {result.trigger!r}"
    )


def test_latency_under_budget():
    # Stage 0 must be well under the 5ms budget even on a longer ticket.
    long_desc = ("user reports intermittent slowness " * 40) + "all systems down"
    result = safety_check("Perf", long_desc)
    assert result.bypass_llm is True
    assert result.latency_ms < 5.0, f"latency {result.latency_ms:.3f}ms exceeds 5ms"


def test_trigger_is_reported():
    result = safety_check("x", "ransomware on the file server")
    assert result.trigger is not None
    assert "ransomware" in result.trigger.lower()


def test_case_insensitive():
    assert safety_check("X", "RANSOMWARE OUTBREAK").bypass_llm is True
    assert safety_check("X", "Zero-Day Exploit Detected").bypass_llm is True


def test_clean_ticket_passes():
    result = safety_check("VPN", "vpn not connecting since this morning")
    assert result.bypass_llm is False
    assert result.matched_category is None


def test_pattern_count_matches_prose_claim():
    # The Round 2 prose claims 11 high-stakes patterns: 7 documented + 4 proposed.
    assert DOCUMENTED_COUNT == 7
    assert PROPOSED_COUNT == 4
    assert len(HIGH_STAKES) == 11
