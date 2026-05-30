"""
Prompt builders for synthetic ticket generation (Round 2 Section 4.2).

All builders are pure functions returning prompt strings, so they're unit
tested without any LLM. The generation prompt anchors each category on
domain-exclusive signals (and explicit exclusions) to enforce the mutual
exclusivity the VGAC classifier relies on.
"""

from __future__ import annotations

# Per-domain anchor signals and the language each domain must NOT contain.
# Access Management / Security split mirrors Round 2 Section 11.3.
SEED_SIGNALS: dict[str, dict[str, list[str]]] = {
    "Network": {
        "include": ["vpn", "dns", "firewall", "latency", "switch", "wifi", "subnet", "routing"],
        "exclude": ["database", "disk", "login", "malware"],
    },
    "Application": {
        "include": ["null pointer", "api", "500 error", "frontend", "deployment", "crash", "endpoint"],
        "exclude": ["disk", "firewall", "password reset", "ransomware"],
    },
    "Database": {
        "include": ["query", "index", "deadlock", "replication", "connection pool", "table lock", "restore"],
        "exclude": ["vpn", "wifi", "login", "phishing"],
    },
    "Storage": {
        "include": ["disk full", "volume", "nas", "san", "snapshot", "backup space", "capacity"],
        "exclude": ["vpn", "deadlock", "password", "malware"],
    },
    "Infrastructure": {
        "include": ["cpu", "memory", "oom", "host", "vm", "kubernetes node", "patch", "reboot"],
        "exclude": ["password reset", "phishing", "query", "vpn"],
    },
    "Access Management": {
        "include": ["account locked", "sso", "saml", "mfa", "password reset", "ldap", "permission denied", "active directory"],
        "exclude": ["ransomware", "breach", "suspicious", "malware", "brute force"],
    },
    "Security": {
        "include": ["phishing", "brute force", "malware", "vulnerability", "suspicious login", "unauthorized access", "cve"],
        "exclude": ["forgot password", "account locked", "sso issue", "mfa setup"],
    },
}

GENERATION_SYSTEM = (
    "You generate realistic enterprise IT support tickets for a specific "
    "category. Tickets must use that category's terminology and must NOT use "
    "language belonging to other categories. Output strictly valid JSON."
)


def build_generation_prompt(category: str, n: int) -> str:
    # Anchor the include list on the validated signature lexicon (the terms we
    # proved are abundant + mutually exclusive); keep the curated exclusions.
    from .controlled import SIGNATURE_LEXICON

    include_terms = SIGNATURE_LEXICON.get(category) or SEED_SIGNALS.get(
        category, {}
    ).get("include", [])
    include = ", ".join(include_terms)
    exclude = ", ".join(SEED_SIGNALS.get(category, {}).get("exclude", []))
    return (
        f"Generate {n} distinct IT support tickets for the category "
        f'"{category}".\n'
        f"Each ticket MUST draw on signals like: {include}.\n"
        f"Each ticket MUST NOT contain language like: {exclude}.\n\n"
        "Return ONLY a JSON object with a single key \"tickets\" whose value "
        f"is an array of {n} objects. Each object has keys:\n"
        '  "title" (short), "description" (1-3 sentences), '
        '"resolution" (concrete fix steps), '
        '"priority" (one of Critical, High, Medium, Low), '
        '"request_type" (one of Incident, Service Request, Problem, Change).\n'
        'No markdown, no commentary. Example shape: {"tickets": [ {...}, {...} ]}.'
    )


def build_validation_prompt(title: str, description: str, category: str) -> str:
    return (
        "Score how well this ticket fits its assigned category and how "
        "realistic it is, from 1 (poor) to 5 (excellent).\n"
        f"Category: {category}\nTitle: {title}\nDescription: {description}\n\n"
        'Return ONLY JSON: {"score": <1-5>, "reason": "<short>"}.'
    )


def build_augmentation_prompt(title: str, description: str, n: int = 3) -> str:
    return (
        f"Rewrite this IT ticket {n} different ways. Keep the same meaning and "
        "the same underlying issue, but vary the wording, tone, and phrasing "
        "(some terse, some verbose, some layman).\n"
        f"Title: {title}\nDescription: {description}\n\n"
        'Return ONLY a JSON object with key "variants" whose value is an '
        'array of objects, each with keys "title" and "description".'
    )


def build_boundary_prompt(category_a: str, category_b: str, n: int) -> str:
    return (
        f"Generate {n} deliberately AMBIGUOUS IT tickets that could plausibly "
        f'belong to either "{category_a}" or "{category_b}". They should be '
        "genuinely hard to classify, to stress-test a router.\n\n"
        "Return ONLY a JSON array of objects with keys "
        '"title", "description", and "plausible_categories" (array of strings).'
    )


# Scenario hints rotated into seeded generation so 10k tickets vary in shape,
# not just in which terms were sampled.
SCENARIO_HINTS = [
    "a sudden complete failure",
    "intermittent, comes and goes",
    "gradual degradation over days",
    "started right after a change or update",
    "affecting a single user",
    "affecting an entire team or site",
    "includes a log snippet or error code",
    "a vague, frustrated end-user report",
    "a monitoring alert that fired overnight",
    "a recurring problem that keeps coming back",
]

SEEDED_SYSTEM = (
    "You write a single realistic enterprise IT support ticket. Use the given "
    "terms naturally; do not list them. Output strictly valid JSON only."
)


def build_seeded_prompt(category: str, terms: list[str], scenario: str) -> str:
    term_str = ", ".join(terms)
    return (
        f"Write one realistic IT support ticket for the {category} domain.\n"
        f"Naturally incorporate these terms (don't just list them): {term_str}.\n"
        f"Shape of the incident: {scenario}.\n\n"
        'Return ONLY a JSON object: {"title": "...", "description": "...", '
        '"resolution": "..."}. The resolution is concrete fix steps. '
        "No markdown, no commentary."
    )
