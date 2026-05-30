"""Tests for the standalone security controls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from security.encryption import EncryptedStore, SecretsProvider, VaultSecretsProvider
from security.pii_redaction import PIIRedactor, PresidioRedactor
from security.rbac import AccessControl
import pytest


def test_pii_redacts_structured_entities():
    r = PIIRedactor(known_names=["Smith"]).redact(
        "Mr Smith email a@b.com phone 555-123-4567 ip 10.0.0.1 ssn 123-45-6789")
    assert "[EMAIL]" in r.text and "[PHONE]" in r.text and "[IP]" in r.text and "[SSN]" in r.text
    assert "[NAME]" in r.text and r.total >= 5


def test_pii_leaves_clean_text_alone():
    r = PIIRedactor().redact("the database had a deadlock in the connection pool")
    assert r.total == 0


def test_encryption_roundtrip_and_no_plaintext_leak():
    store = EncryptedStore(SecretsProvider(key_file="security/data/.test_key"))
    rec = {"id": "1", "text": "secret ticket body"}
    blob = store.encrypt(rec)
    assert store.decrypt(blob) == rec
    assert store.is_ciphertext(blob, rec)
    Path("security/data/.test_key").unlink(missing_ok=True)


def test_vault_seam_disabled_until_configured():
    with pytest.raises(NotImplementedError):
        VaultSecretsProvider()


def test_presidio_seam_disabled_until_configured():
    with pytest.raises(NotImplementedError):
        PresidioRedactor()


def test_rbac_enforces_permissions_and_elevation():
    ac = AccessControl()
    assert ac.check("viewer", "view").allowed
    assert not ac.check("viewer", "resolve").allowed
    assert not ac.check("agent", "resolve", domain="Security").allowed   # needs soc/admin
    assert ac.check("soc_analyst", "resolve", domain="Security").allowed
    assert not ac.check("agent", "view", reads_pii=True).allowed
    assert not ac.check("guest", "view").allowed                          # unknown role
