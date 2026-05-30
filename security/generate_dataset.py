#!/usr/bin/env python3
"""
Generate the security demo dataset (SEPARATE data — all PII below is FAKE).

Produces:
  security/data/pii_tickets.csv      tickets with embedded synthetic PII
  security/data/access_requests.csv  (user, role, action, domain, reads_pii) scenarios

    python security/generate_dataset.py
"""

import csv
import random
from pathlib import Path

random.seed(7)

# fully synthetic — no real personal data
NAMES = ["Mr Smith", "Ms Johnson", "Dr Patel", "Mrs Garcia", "Prof Lee"]
TEMPLATES = [
    "User {name} cannot log in. Email {email}, phone {phone}. Reset from IP {ip}.",
    "Ticket from {name}: VPN drops. Contact {email} or {phone}.",
    "Payment failed for card {card}. Account holder {name}, SSN {ssn}.",
    "{name} reports mailbox full. Forward to {email}; server IP {ip}.",
    "Access request: {name} ({email}) needs AD group membership. Phone {phone}.",
    "Database deadlock reported by {name}. Callback {phone}, escalate via {email}.",
]


def fake_pii():
    return {
        "name": random.choice(NAMES),
        "email": f"user{random.randint(100,999)}@example.com",
        "phone": f"555-{random.randint(100,999)}-{random.randint(1000,9999)}",
        "ip": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "card": f"4{random.randint(100,999)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}",
        "ssn": f"{random.randint(100,899)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
    }


def main():
    Path("security/data").mkdir(parents=True, exist_ok=True)

    with open("security/data/pii_tickets.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text"])
        for i in range(40):
            p = fake_pii()
            w.writerow([i, random.choice(TEMPLATES).format(**p)])
    print("wrote security/data/pii_tickets.csv (40 tickets with synthetic PII)")

    scenarios = [
        ("alice", "viewer", "view", "", False),
        ("alice", "viewer", "resolve", "", False),          # deny: no resolve
        ("bob", "agent", "route", "Network", False),
        ("bob", "agent", "resolve", "Security", False),     # deny: Security needs soc/admin
        ("bob", "agent", "read_pii", "", True),             # deny: no PII read
        ("carol", "soc_analyst", "resolve", "Security", False),
        ("carol", "soc_analyst", "read_pii", "Security", True),
        ("dave", "admin", "admin", "", False),
        ("eve", "guest", "view", "", False),                # deny: unknown role
    ]
    with open("security/data/access_requests.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user", "role", "action", "domain", "reads_pii"])
        w.writerows(scenarios)
    print("wrote security/data/access_requests.csv (9 access scenarios)")


if __name__ == "__main__":
    main()
