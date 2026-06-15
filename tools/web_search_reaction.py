"""Gateway↔tool bridge: react to the triggering message during a web search.

Mirrors ``tools.approval`` / ``tools.clarify_gateway``: the gateway registers a
per-session callback keyed by the agent's ``task_id`` (which, for gateway runs,
equals the ``session_id`` passed to ``run_conversation``).  A ``pre_tool_call``
hook invokes that callback the first time a web-search tool runs in the turn,
so the platform adapter (currently Discord) can add a 🌐 reaction to the user's
message — a lightweight "searching the web" acknowledgement that is a reaction,
never a reply.

The hook is observer-only: it always returns ``None`` so it can never block or
veto the tool call, and any failure in the callback is swallowed.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Tool names whose execution counts as "doing a web search".
WEB_SEARCH_TOOLS = frozenset({"web_search"})

_lock = threading.Lock()
_notify_cbs: Dict[str, Callable[[], None]] = {}


def register_notify(task_id: str, cb: Callable[[], None]) -> None:
    """Register a per-session web-search reaction callback."""
    if not task_id:
        return
    with _lock:
        _notify_cbs[task_id] = cb


def unregister_notify(task_id: str) -> None:
    """Drop the per-session callback (called when the run ends)."""
    if not task_id:
        return
    with _lock:
        _notify_cbs.pop(task_id, None)


def get_notify(task_id: str) -> Optional[Callable[[], None]]:
    with _lock:
        return _notify_cbs.get(task_id)


def pre_tool_call(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    task_id: str = "",
    **kwargs: Any,
) -> None:
    """``pre_tool_call`` hook: fire the reaction callback for web searches.

    Observer-only — always returns ``None`` so it never blocks the tool.
    """
    if tool_name not in WEB_SEARCH_TOOLS:
        return None
    cb = get_notify(task_id)
    if cb is None:
        return None
    try:
        cb()
    except Exception as exc:  # a reaction failure must never break the tool
        logger.debug("web-search reaction callback failed: %s", exc)
    return None
