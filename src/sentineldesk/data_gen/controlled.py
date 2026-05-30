"""
Controlled reference corpus (no LLM).

Purpose: validate that the N-gram and TF-IDF layers recover an *abundant* and
*mutually exclusive* top-N per category BEFORE spending hours on Mistral
generation. Each category draws only on its own signature lexicon and never on
another category's; neutral filler is rotated from a large pool so no filler
word accumulates enough frequency to crowd the signature terms out of the
top-N. This is a reference target and a test harness, not the final corpus —
the same analysis re-runs on the real Mistral output to check it holds.
"""

from __future__ import annotations

import random

from ..corpus import LabeledTicket

# ~14 signature terms per category (mix of unigrams and recurring phrases).
# These are the terms generation must reuse and that extraction should recover.
SIGNATURE_LEXICON: dict[str, list[str]] = {
    "Network": [
        "vpn tunnel", "dns resolution", "firewall rule", "packet loss",
        "subnet", "vlan", "switch port", "routing table", "gateway",
        "bandwidth", "dhcp lease", "wifi", "latency", "mtu",
    ],
    "Application": [
        "null pointer exception", "stack trace", "api endpoint", "deployment",
        "microservice", "build pipeline", "code regression", "frontend",
        "backend", "exception thrown", "rest call", "payload", "rollback",
    ],
    "Database": [
        "query plan", "missing index", "deadlock", "replication lag",
        "connection pool", "table lock", "transaction rollback", "slow query",
        "schema migration", "primary replica", "vacuum", "stored procedure",
    ],
    "Storage": [
        "disk full", "volume capacity", "nas mount", "san", "snapshot",
        "raid array", "filesystem", "inode", "block storage", "quota exceeded",
        "archive tier", "lun", "mount point",
    ],
    "Infrastructure": [
        "cpu utilization", "memory exhaustion", "oom killer", "host reboot",
        "hypervisor", "virtual machine", "kubernetes node", "pod restart",
        "load average", "kernel panic", "cgroup", "node taint",
    ],
    "Access Management": [
        "account locked", "password reset", "single sign", "saml assertion",
        "mfa otp", "ldap sync", "active directory", "group membership",
        "permission denied", "user provisioning", "okta", "role assignment",
    ],
    "Security": [
        "phishing", "malware", "ransomware", "brute force", "vulnerability",
        "cve", "suspicious login", "unauthorized access", "intrusion",
        "credential theft", "threat actor", "exploit attempt",
    ],
}

# Neutral, cross-domain filler. Rotated widely so none reaches the top-N.
_TITLE_NOUNS = ["issue", "alert", "ticket", "report", "fault", "case", "request"]
_VERBS = [
    "failing", "degraded", "unstable", "intermittent", "unresponsive",
    "stalling", "flapping", "erratic", "delayed", "recurring",
]
_OPENERS = [
    "", "users report", "monitoring flagged", "we are seeing",
    "ongoing", "noticed that", "tracking", "escalated because",
]
_SCOPES = [
    "across several users", "for the team", "in one department",
    "at multiple sites", "in a branch office", "in production",
    "during the night shift", "after the upgrade",
]


def generate_controlled_corpus(
    per_category: int = 120,
    seed: int = 42,
    register: str = "canonical",
) -> list[LabeledTicket]:
    """register='canonical' uses signature terms; 'casual' swaps in layman forms
    (so the eval set tests robustness to how real users actually phrase things)."""
    from .layman_map import LAYMAN_MAP

    rng = random.Random(seed)

    def surface(category: str, term: str) -> str:
        if register == "casual":
            forms = LAYMAN_MAP.get(category, {}).get(term)
            if forms:
                return rng.choice(forms)
        return term

    tickets: list[LabeledTicket] = []
    for category, lexicon in SIGNATURE_LEXICON.items():
        for _ in range(per_category):
            s1, s2 = rng.sample(lexicon, 2)
            d1, d2 = surface(category, s1), surface(category, s2)
            noun = rng.choice(_TITLE_NOUNS)
            v1, v2, v3 = (rng.choice(_VERBS) for _ in range(3))
            opener = rng.choice(_OPENERS)
            scope = rng.choice(_SCOPES)
            title = f"{d1} {noun}"
            desc = f"{opener} {d1} {v1}; {d2} {v2} {scope}; {d1} {v3}".strip()
            tickets.append(
                LabeledTicket(
                    text=f"{title} {desc}",
                    category=category,
                    title=title,
                    description=desc,
                )
            )
    return tickets
