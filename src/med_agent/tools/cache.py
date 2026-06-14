"""In-process TTL cache for external API responses.

Usage:
    from med_agent.tools.cache import get_cached

    result = get_cached(
        key=f"PubMedSearch:{query}",
        fetcher=lambda: do_expensive_api_call(query),
        ttl=3600,
    )
"""

import time
import threading
from typing import Any, Callable

_cache: dict[str, tuple[float, Any]] = {}
_lock = threading.Lock()


def get_cached(key: str, fetcher: Callable[[], Any], ttl: int = 3600) -> Any:
    now = time.monotonic()
    with _lock:
        entry = _cache.get(key)
        if entry is not None:
            ts, value = entry
            if now - ts < ttl:
                return value
    result = fetcher()
    with _lock:
        _cache[key] = (time.monotonic(), result)
    return result


def invalidate(key: str) -> None:
    with _lock:
        _cache.pop(key, None)


def clear() -> None:
    with _lock:
        _cache.clear()
