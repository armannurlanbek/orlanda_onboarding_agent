"""Symmetric encryption for secrets stored at rest.

Currently used for per-user monday.com OAuth access tokens. Uses Fernet (AES-128-CBC +
HMAC-SHA256) from the already-required ``cryptography`` dependency.

The key comes from ``RAG_MONDAY_TOKEN_ENC_KEY`` when set (a urlsafe-base64 32-byte key as
produced by ``Fernet.generate_key()``), otherwise it is derived deterministically from
``RAG_AGENT_SECRET_KEY``. ``SECRET_KEY`` is already required to be stable and identical
across both cluster servers, so derived keys stay valid across a failover (the same
property the auth token-hashing already relies on).
"""
from __future__ import annotations

import base64
import hashlib
from functools import lru_cache

from cryptography.fernet import Fernet

from rag_agent.config import RAG_MONDAY_TOKEN_ENC_KEY, SECRET_KEY


def _fernet_key() -> bytes:
    """Return a Fernet-compatible urlsafe-base64 32-byte key."""
    if RAG_MONDAY_TOKEN_ENC_KEY:
        # Trust an explicitly provided key verbatim; Fernet() validates it on use.
        return RAG_MONDAY_TOKEN_ENC_KEY.encode("utf-8")
    # Derive a stable 32-byte key from SECRET_KEY.
    digest = hashlib.sha256(SECRET_KEY.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    return Fernet(_fernet_key())


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret string into a urlsafe token (str) safe for DB storage."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(token: str) -> str:
    """Decrypt a token produced by ``encrypt_secret``.

    Raises ``cryptography.fernet.InvalidToken`` if the ciphertext was tampered with or the
    encryption key changed.
    """
    return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
