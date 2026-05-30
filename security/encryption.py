"""
Encryption at rest (REAL) + secrets provider (local backend + Vault seam).

EncryptedStore uses Fernet (AES-128-CBC + HMAC, from the `cryptography` library)
to encrypt ticket records at rest. This is a genuine working control: records go
in as plaintext dicts, are stored as ciphertext, and round-trip back. The
encryption key comes from the SecretsProvider, never hard-coded.

SecretsProvider reads secrets from the environment or a local key file today;
the Vault adapter documents the production path (HashiCorp Vault via hvac) and
is disabled until configured — the same honest seam pattern as the Qdrant
retriever.

HONEST LABEL: Fernet encryption at rest = IMPLEMENTED. HashiCorp Vault =
ROADMAP/SEAM. Encryption in transit (TLS/mTLS) = DEPLOYMENT-layer (see mtls.py),
not application code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cryptography.fernet import Fernet


# --- secrets provider ----------------------------------------------------------
class SecretsProvider:
    """Fetch secrets from env or a local key file. No secret is hard-coded."""

    def __init__(self, key_file: str | Path = "security/data/.fernet_key") -> None:
        self.key_file = Path(key_file)

    def get_encryption_key(self) -> bytes:
        env = os.environ.get("SENTINEL_FERNET_KEY")
        if env:
            return env.encode()
        if self.key_file.exists():
            return self.key_file.read_bytes()
        key = Fernet.generate_key()                     # first run: create + persist
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self.key_file.write_bytes(key)
        try:
            os.chmod(self.key_file, 0o600)              # owner-only (best-effort on Windows)
        except OSError:
            pass
        return key


class VaultSecretsProvider:
    """
    Production secrets via HashiCorp Vault (hvac). Disabled unless a Vault client
    + mount are supplied — documents the integration point without the dependency.
    """

    def __init__(self, client=None, mount: str = "secret", path: str = "sentineldesk") -> None:
        if client is None:
            raise NotImplementedError(
                "VaultSecretsProvider is the production secrets backend. Provide an "
                "authenticated hvac Vault client; until then SecretsProvider (env/"
                "local key file) is the working default."
            )
        self.client, self.mount, self.path = client, mount, path

    def get_encryption_key(self) -> bytes:  # pragma: no cover
        resp = self.client.secrets.kv.v2.read_secret_version(mount_point=self.mount, path=self.path)
        return resp["data"]["data"]["fernet_key"].encode()


# --- encryption at rest --------------------------------------------------------
class EncryptedStore:
    """Encrypt/decrypt ticket records at rest with Fernet."""

    def __init__(self, secrets: SecretsProvider | None = None) -> None:
        self._fernet = Fernet((secrets or SecretsProvider()).get_encryption_key())

    def encrypt(self, record: dict) -> bytes:
        return self._fernet.encrypt(json.dumps(record).encode())

    def decrypt(self, blob: bytes) -> dict:
        return json.loads(self._fernet.decrypt(blob).decode())

    def is_ciphertext(self, blob: bytes, record: dict) -> bool:
        """True if the at-rest blob does not leak the plaintext (sanity check)."""
        sample = json.dumps(record)[:40]
        return sample.encode() not in blob
