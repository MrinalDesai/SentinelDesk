"""
mTLS — DEPLOYMENT-LAYER, not application code.

Mutual TLS (encryption + mutual authentication in transit) is a deployment
concern: it lives in the reverse proxy / service mesh / API gateway in front of
the services, not in this Python code. It cannot "run on a dataset" — there is
nothing to classify, only a transport to configure. Claiming mTLS as an
implemented application feature would be an overclaim.

What this module honestly provides:
  - a helper to GENERATE a self-signed CA + server/client certs, to DEMONSTRATE
    the mTLS material and pattern (real certs, via `cryptography`);
  - documentation of where mTLS terminates in the production topology.

HONEST LABEL: mTLS = DEPLOYMENT (config), demonstrated via cert generation;
NOT an implemented runtime control in this codebase.
"""

from __future__ import annotations

import datetime
from pathlib import Path


def generate_demo_certs(out_dir: str | Path = "security/data/certs") -> dict:
    """Generate a self-signed CA + a server cert signed by it (demo material)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _key():
        return rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def _name(cn):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    now = datetime.datetime.utcnow()
    ca_key = _key()
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("SentinelDesk-Demo-CA"))
        .issuer_name(_name("SentinelDesk-Demo-CA"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    srv_key = _key()
    srv_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("sentineldesk.local"))
        .issuer_name(ca_cert.subject)
        .public_key(srv_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=90))
        .sign(ca_key, hashes.SHA256())
    )
    paths = {}
    for name, obj, is_key in [
        ("ca.crt", ca_cert, False), ("server.crt", srv_cert, False),
        ("server.key", srv_key, True),
    ]:
        p = out / name
        if is_key:
            p.write_bytes(obj.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption()))
        else:
            p.write_bytes(obj.public_bytes(serialization.Encoding.PEM))
        paths[name] = str(p)
    return paths


DEPLOYMENT_NOTE = """\
mTLS terminates at the ingress (reverse proxy / service mesh), NOT in this code:
  client --mTLS--> [ingress: nginx/Envoy/Istio] --localhost--> FastAPI service
The ingress presents server.crt, requires a client cert signed by ca.crt, and
forwards verified requests internally. In Kubernetes this is an Istio
PeerAuthentication policy (mode: STRICT). This module only generates demo cert
material to show the pattern; it does not implement the transport.
"""
