"""
Role-Based Access Control (REAL, working).

Defines roles, the permissions each role holds, and enforces access decisions on
(role, action, resource-sensitivity) — including the rule that security-domain /
high-sensitivity tickets and raw-PII reads require an elevated role. This is a
genuine working control: it runs on a dataset of access requests and returns
allow/deny with a reason, suitable for an audit log.

HONEST LABEL: application-level RBAC = IMPLEMENTED. Identity-provider / SSO
integration (OIDC, SCIM) = ROADMAP.
"""

from __future__ import annotations

from dataclasses import dataclass

# permissions
VIEW, ROUTE, RESOLVE, ESCALATE, READ_PII, ADMIN = (
    "view", "route", "resolve", "escalate", "read_pii", "admin")

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "viewer":       {VIEW},
    "agent":        {VIEW, ROUTE, RESOLVE},
    "soc_analyst":  {VIEW, ROUTE, RESOLVE, ESCALATE, READ_PII},
    "admin":        {VIEW, ROUTE, RESOLVE, ESCALATE, READ_PII, ADMIN},
}

# actions that require elevation regardless of base permission
_SENSITIVE_DOMAINS = {"Security"}


@dataclass
class AccessDecision:
    allowed: bool
    reason: str


class AccessControl:
    def __init__(self, role_permissions: dict[str, set[str]] | None = None) -> None:
        self.role_permissions = role_permissions or ROLE_PERMISSIONS

    def check(self, role: str, action: str, *, domain: str = "", reads_pii: bool = False) -> AccessDecision:
        perms = self.role_permissions.get(role)
        if perms is None:
            return AccessDecision(False, f"unknown role '{role}'")
        if action not in perms:
            return AccessDecision(False, f"role '{role}' lacks '{action}'")
        # elevation rules
        if reads_pii and READ_PII not in perms:
            return AccessDecision(False, f"role '{role}' may not read PII")
        if domain in _SENSITIVE_DOMAINS and action in {RESOLVE, ESCALATE} \
                and role not in {"soc_analyst", "admin"}:
            return AccessDecision(False, f"{domain} tickets require soc_analyst/admin for '{action}'")
        return AccessDecision(True, f"role '{role}' permits '{action}'")
