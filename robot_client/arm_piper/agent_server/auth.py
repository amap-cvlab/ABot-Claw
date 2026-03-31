"""API key authentication — two-tier (client / admin), localhost bypass."""

from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Optional, Tuple

from fastapi import HTTPException, Request, WebSocket
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that are always public (no key required)
PUBLIC_PATHS = frozenset({
    "/health",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
})

_LOCALHOST_ADDRS = frozenset({"127.0.0.1", "::1"})


class KeyStore:
    """Loads and validates API keys from a JSON file + ROBOT_API_KEY env var."""

    def __init__(self, path: str | None = None) -> None:
        self._keys: dict[str, Tuple[str, str]] = {}  # key -> (name, role)
        self._load_file(path)
        self._load_env()

    def _load_file(self, path: str | None) -> None:
        if path is None or not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for entry in data.get("keys", []):
                self._keys[entry["key"]] = (entry["name"], entry["role"])
            logger.info("Loaded %d API keys from %s", len(self._keys), path)
        except Exception as e:
            logger.error("Failed to load API keys from %s: %s", path, e)

    def _load_env(self) -> None:
        env_key = os.getenv("ROBOT_API_KEY")
        if env_key and env_key not in self._keys:
            self._keys[env_key] = ("env-key", "admin")
            logger.info("Loaded API key from ROBOT_API_KEY env var")

    @property
    def enabled(self) -> bool:
        return len(self._keys) > 0

    def lookup(self, key: str) -> Optional[Tuple[str, str]]:
        """Return (name, role) if key is valid, else None. Timing-safe."""
        for stored_key, info in self._keys.items():
            if hmac.compare_digest(stored_key, key):
                return info
        return None


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces API key auth on non-localhost requests."""

    def __init__(self, app, key_store: KeyStore) -> None:
        super().__init__(app)
        self._ks = key_store

    async def dispatch(self, request: Request, call_next):
        # 1. Auth disabled → pass through as admin
        if not self._ks.enabled:
            request.state.auth_user = "local"
            request.state.auth_role = "admin"
            return await call_next(request)

        path = request.url.path.rstrip("/") or "/"

        # 2. Public whitelist
        if path in PUBLIC_PATHS:
            request.state.auth_user = "public"
            request.state.auth_role = "public"
            return await call_next(request)

        # 3. Localhost bypass → admin
        client_ip = request.client.host if request.client else None
        if client_ip in _LOCALHOST_ADDRS:
            request.state.auth_user = "localhost"
            request.state.auth_role = "admin"
            return await call_next(request)

        # 4. Extract key from header or query param
        key = request.headers.get("x-api-key") or request.query_params.get("api_key")

        if not key:
            return JSONResponse(
                status_code=401,
                content={"detail": "API key required. Use X-API-Key header or ?api_key= query param."},
            )

        result = self._ks.lookup(key)
        if result is None:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )

        name, role = result
        request.state.auth_user = name
        request.state.auth_role = role
        return await call_next(request)


async def require_admin(request: Request) -> None:
    """FastAPI dependency that enforces admin role."""
    role = getattr(request.state, "auth_role", None)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")


def check_ws_auth(
    ws: WebSocket,
    key_store: KeyStore,
) -> Tuple[bool, str, str]:
    """Check WebSocket auth. Returns (authorized, name, role).

    Checks localhost first, then ?api_key= query param, then x-api-key header.
    """
    if not key_store.enabled:
        return True, "local", "admin"

    # Localhost bypass
    client_ip = ws.client.host if ws.client else None
    if client_ip in _LOCALHOST_ADDRS:
        return True, "localhost", "admin"

    # Query param
    key = ws.query_params.get("api_key")
    # Header
    if not key:
        key = ws.headers.get("x-api-key")

    if not key:
        return False, "", ""

    result = key_store.lookup(key)
    if result is None:
        return False, "", ""

    return True, result[0], result[1]
