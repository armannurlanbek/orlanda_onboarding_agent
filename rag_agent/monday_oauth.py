"""
Monday OAuth helpers.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from rag_agent.config import (
    MONDAY_CLIENT_ID,
    MONDAY_CLIENT_SECRET,
    MONDAY_OAUTH_AUTHORIZE_URL,
    MONDAY_OAUTH_REDIRECT_URI,
    MONDAY_OAUTH_SCOPES,
    MONDAY_OAUTH_TOKEN_URL,
)


def monday_oauth_enabled() -> bool:
    return bool(MONDAY_CLIENT_ID and MONDAY_CLIENT_SECRET and MONDAY_OAUTH_REDIRECT_URI)


def build_authorize_url(state: str) -> str:
    params = {
        "client_id": MONDAY_CLIENT_ID,
        "redirect_uri": MONDAY_OAUTH_REDIRECT_URI,
        "response_type": "code",
        "state": state,
    }
    if MONDAY_OAUTH_SCOPES:
        params["scope"] = MONDAY_OAUTH_SCOPES
    return f"{MONDAY_OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_token(code: str) -> dict[str, Any]:
    payload = {
        "grant_type": "authorization_code",
        "client_id": MONDAY_CLIENT_ID,
        "client_secret": MONDAY_CLIENT_SECRET,
        "redirect_uri": MONDAY_OAUTH_REDIRECT_URI,
        "code": code,
    }
    body = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(
        MONDAY_OAUTH_TOKEN_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or not data.get("access_token"):
        raise RuntimeError("Monday token exchange returned invalid payload.")
    return data
