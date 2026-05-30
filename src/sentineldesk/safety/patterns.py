"""
High-stakes pattern definitions for the Stage 0 safety layer.

DESIGN NOTES (read before tuning):

1. The Round 2 document claims "11 high-stakes patterns" but the table in
   Section 2.4 only enumerates 7. The 7 documented categories are marked
   documented=True below; 4 additional ones are marked documented=False
   (PROPOSED) so the count reaches the 11 the prose claims. Confirm or
   replace the proposed four before this is treated as final.

2. Stage 0's whole job is to escalate before any LLM runs, on a deliberate
   "better to escalate than be wrong" bias. But a *bare* keyword like
   "malware" would also fire on benign tickets ("renew anti-malware
   license"). So where a bare token is ambiguous we require an action/context
   word (malware *detected*, *infection*). Where a token is never benign in an
   enterprise ticket ("ransomware", "zero day") we match it bare. Every
   pattern is a conscious point on that precision/recall tradeoff, and the
   test suite pins both true positives and benign near-misses.

3. Patterns are case-insensitive, compiled once at import (the Round 2
   "compiled regex cached at startup" optimisation). Word boundaries (\\b)
   prevent substring false matches like "exploration" -> "exploit".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Pattern


@dataclass(frozen=True)
class HighStakesCategory:
    name: str
    department: str
    severity: str
    patterns: tuple[str, ...]
    documented: bool = True            # False => PROPOSED, not in Round 2 doc
    compiled: tuple[Pattern[str], ...] = field(default=(), compare=False, repr=False)

    def compile(self) -> "HighStakesCategory":
        """Return a copy with patterns compiled (frozen dataclass workaround)."""
        object.__setattr__(
            self, "compiled",
            tuple(re.compile(p, re.IGNORECASE) for p in self.patterns),
        )
        return self


# Ordered most-unambiguous first; scanning stops at the first match, so order
# also encodes a rough triage priority.
_RAW: tuple[HighStakesCategory, ...] = (
    # --- DOCUMENTED (Round 2 section 2.4) ---------------------------------
    HighStakesCategory(
        name="Ransomware/Malware",
        department="SOC Team",
        severity="Critical",
        patterns=(
            # Bare "ransomware" is high-signal, but suppress obvious benign
            # contexts (awareness training, policy docs, tabletop drills). The
            # negative lookahead is the single point where we trade a tiny bit
            # of recall for not paging the SOC about a training calendar invite.
            # DECISION POINT: delete the lookahead to escalate on ANY mention.
            r"\bransomware\b(?![- ](?:awareness|training|policy|drill|simulation|webinar|exercise|tabletop))",
            r"\bcrypto[- ]?locker\b",
            r"\bmalware\s+(?:detected|found|infection|outbreak|alert)\b",
            r"\b(?:detected|found)\s+malware\b",
            r"\bfiles?\s+(?:are\s+)?(?:being\s+)?encrypted\b.*\bransom\b",
        ),
    ),
    HighStakesCategory(
        name="Data Breach",
        department="SOC Team",
        severity="Critical",
        patterns=(
            r"\bdata\s+breach\b",
            r"\bdata\s+(?:leak|exfiltration|exfiltrated)\b",
            r"\bunauthor(?:i[sz]ed)\s+access\b",
            r"\bcustomer\s+data\s+(?:exposed|leaked)\b",
        ),
    ),
    HighStakesCategory(
        name="Complete Outage",
        department="Bridge Call - P1",
        severity="Critical",
        patterns=(
            r"\ball\s+systems?\s+down\b",
            r"\bcomplete\s+outage\b",
            r"\btotal\s+outage\b",
            r"\beverything\s+is\s+down\b",
            r"\bentire\s+\w+\s+(?:is\s+)?down\b",
        ),
    ),
    HighStakesCategory(
        name="Production DB Down",
        department="DBA Team",
        severity="Critical",
        patterns=(
            r"\bprod(?:uction)?\s+(?:database|db)\s+(?:is\s+)?(?:down|crashed|offline|unreachable)\b",
            r"\b(?:database|db)\s+(?:is\s+)?down\s+in\s+prod(?:uction)?\b",
        ),
    ),
    HighStakesCategory(
        name="Security Incident",
        department="SOC Team",
        severity="Critical",
        patterns=(
            r"\bzero[- ]?day\b",
            r"\bexploit\s+(?:detected|active|in\s+the\s+wild)\b",
            r"\bactive\s+exploit\b",
            r"\bremote\s+code\s+execution\b",
            r"\brce\s+vuln(?:erability)?\b",
        ),
    ),
    HighStakesCategory(
        name="DPDP/Compliance",
        department="Legal + CISO",
        severity="Critical",
        patterns=(
            r"\bdpdp\s+(?:violation|breach)\b",
            r"\bgdpr\s+(?:violation|breach)\b",
            r"\bhipaa\s+(?:violation|breach)\b",
            r"\bcompliance\s+breach\b",
            r"\bdata\s+protection\s+violation\b",
        ),
    ),
    HighStakesCategory(
        name="Physical Emergency",
        department="Facilities + Infra",
        severity="Critical",
        patterns=(
            r"\b(?:datacenter|data\s+center|server\s+room)\s+fire\b",
            r"\bfire\s+in\s+(?:the\s+)?(?:datacenter|data\s+center|server\s+room)\b",
            r"\b(?:datacenter|data\s+center|server\s+room)\s+(?:flood|flooding|flooded)\b",
            r"\bpower\s+failure\b",
            r"\bups\s+failure\b",
            r"\bsmoke\s+detected\b",
        ),
    ),
    # --- PROPOSED (to reach the 11 the prose claims; confirm or replace) ---
    HighStakesCategory(
        name="Active Intrusion",
        department="SOC Team",
        severity="Critical",
        documented=False,
        patterns=(
            r"\bintrusion\s+detected\b",
            r"\blateral\s+movement\b",
            r"\battacker\s+(?:inside|in\s+(?:the\s+)?network)\b",
            r"\bcompromised\s+(?:host|server|account)\b.*\bactive\b",
        ),
    ),
    HighStakesCategory(
        name="Mass Credential Leak",
        department="SOC + IAM",
        severity="Critical",
        documented=False,
        patterns=(
            r"\bcredential(?:s)?\s+(?:leak|dump|leaked|dumped)\b",
            r"\bpassword\s+dump\b",
            r"\bmass\s+account\s+compromise\b",
        ),
    ),
    HighStakesCategory(
        name="DDoS Attack",
        department="SOC + Network",
        severity="Critical",
        documented=False,
        patterns=(
            r"\bddos\b",
            r"\bdistributed\s+denial[- ]of[- ]service\b",
            r"\bdenial[- ]of[- ]service\s+attack\b",
        ),
    ),
    HighStakesCategory(
        name="Data Loss/Corruption",
        department="DBA + Infra",
        severity="Critical",
        documented=False,
        patterns=(
            r"\bdata\s+(?:loss|corruption)\b",
            r"\b(?:database|db)\s+corrupt(?:ed|ion)?\b",
            r"\bbackups?\s+(?:are\s+)?(?:missing|corrupt(?:ed)?|gone)\b",
        ),
    ),
)

# Public, compiled view.
HIGH_STAKES: tuple[HighStakesCategory, ...] = tuple(c.compile() for c in _RAW)

DOCUMENTED_COUNT = sum(1 for c in HIGH_STAKES if c.documented)
PROPOSED_COUNT = sum(1 for c in HIGH_STAKES if not c.documented)
