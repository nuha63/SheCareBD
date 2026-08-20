"""
SheCare BD — Daily Chat Limit Service
=======================================
Tracks and enforces a per-anonymous-user daily AI chat limit.

Since the app currently has no authentication or persistent user accounts,
we use the client IP address + calendar date as the anonymous identifier.

Design decisions:
- In-memory store (dict). Resets automatically on server restart and
  naturally cleans up old dates on each write. This is intentional for
  a privacy-first anonymous app — no chat counts are persisted to disk.
- If the request has no identifiable IP (e.g. in tests), a fallback
  key is used that does NOT trigger limits (allows all test requests).
- Daily limit is configurable via DAILY_CHAT_LIMIT constant.

Limitations (documented, not hidden):
- The limit resets on server restart (acceptable for anonymous wellness).
- If multiple server workers are used, each process has its own counter.
  For production multi-worker deployment, replace _store with Redis.
- IP-based limiting is bypassable with VPN. This is acceptable because the
  limit purpose is cost/quota management, not strict security enforcement.
"""
import logging
from datetime import date
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Configurable daily limit
DAILY_CHAT_LIMIT: int = 5

# In-memory store: { (ip_key, date_str): count }
_store: Dict[Tuple[str, str], int] = {}


def _today() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return date.today().isoformat()


def _clean_old_entries() -> None:
    """Remove entries from previous days to prevent unbounded memory growth."""
    today = _today()
    old_keys = [k for k in _store if k[1] != today]
    for k in old_keys:
        del _store[k]


def get_usage_count(ip_key: str) -> int:
    """Return how many chat requests this IP has made today."""
    return _store.get((ip_key, _today()), 0)


def is_limit_reached(ip_key: str) -> bool:
    """
    Return True if the daily limit has been reached for this IP.
    Always returns False for the anonymous test key ("__test__").
    """
    if ip_key == "__test__":
        return False
    return get_usage_count(ip_key) >= DAILY_CHAT_LIMIT


def increment_usage(ip_key: str) -> int:
    """
    Increment the usage count for this IP today.
    Returns the NEW count after incrementing.
    Call this ONLY after a successful (non-blocked) Gemini request.
    """
    if ip_key == "__test__":
        return 0

    _clean_old_entries()
    today = _today()
    key = (ip_key, today)
    _store[key] = _store.get(key, 0) + 1
    new_count = _store[key]
    logger.debug("Daily usage for %s: %d/%d", ip_key, new_count, DAILY_CHAT_LIMIT)
    return new_count


def reset_usage(ip_key: str) -> None:
    """
    Reset the usage count for an IP (used in tests).
    """
    today = _today()
    _store.pop((ip_key, today), None)
