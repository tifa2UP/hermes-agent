"""Reminder action buttons (Mark Done / Snooze) for Discord cron reminders.

When a one-shot cron reminder is delivered to Discord, the adapter attaches
native buttons that let the user resolve it without typing:

    [✅ Done] [💤 15m] [💤 1h] [💤 6h] [💤 24h]

This module holds the platform-agnostic pieces so they can be unit-tested
without a live Discord client:

* ``ReminderActionStore`` — a small disk-backed map from an opaque ``token``
  to the payload needed to act on the reminder later. It is disk-backed (not
  just in-memory) because a snooze button may be clicked hours later and/or
  after a gateway restart — the discord.py ``DynamicItem`` that routes the
  click is stateless and reconstructs everything from the ``token``.
* Snooze duration/label tables and ``custom_id`` build/parse helpers.
* ``recreate_reminder`` — snooze action. A one-shot reminder is *deleted* the
  instant it fires (``cron.jobs.mark_job_run`` pops it), so "snooze" cannot
  reschedule the original job id; it re-creates an equivalent one-shot from the
  stored payload, scheduled ``N`` minutes out.
* ``mark_reminder_done`` — done action. Removes the underlying job if it still
  exists (recurring jobs do; one-shots are already gone) and is a no-op
  otherwise.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# custom_id layout: ``hermes:rem:<token>:<action>`` (well under Discord's
# 100-char limit). ``token`` is hex, ``action`` is one of the keys below.
CUSTOM_ID_PREFIX = "hermes:rem"
CUSTOM_ID_TEMPLATE = re.compile(r"hermes:rem:(?P<token>[0-9a-f]+):(?P<action>[a-z0-9]+)")

ACTION_DONE = "done"

# action key -> snooze minutes
SNOOZE_MINUTES: Dict[str, int] = {
    "snooze15": 15,
    "snooze60": 60,
    "snooze360": 360,
    "snooze1440": 1440,
}

# Buttons rendered, in order. (action_key, label, emoji)
BUTTON_SPECS: List[Tuple[str, str, str]] = [
    (ACTION_DONE, "Done", "✅"),
    ("snooze15", "15m", "💤"),
    ("snooze60", "1h", "💤"),
    ("snooze360", "6h", "💤"),
    ("snooze1440", "24h", "💤"),
]

# Max stored reminder-action entries; oldest (by created_at) are evicted.
_MAX_ENTRIES = 500


def new_token() -> str:
    """Return a fresh opaque token for a reminder-action set."""
    return uuid.uuid4().hex[:12]


def build_custom_id(token: str, action: str) -> str:
    """Build the Discord component ``custom_id`` for *token*/*action*."""
    return f"{CUSTOM_ID_PREFIX}:{token}:{action}"


def parse_custom_id(custom_id: str) -> Optional[Tuple[str, str]]:
    """Parse a ``custom_id`` into ``(token, action)``, or None if it isn't ours."""
    if not custom_id:
        return None
    match = CUSTOM_ID_TEMPLATE.fullmatch(custom_id)
    if not match:
        return None
    return match.group("token"), match.group("action")


def snooze_label(minutes: int) -> str:
    """Human label for a snooze duration (e.g. 15 -> '15 minutes', 1440 -> '24 hours')."""
    if minutes % 1440 == 0:
        days = minutes // 1440
        return f"{days} day" if days == 1 else f"{days} days"
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"


def _now_iso() -> str:
    from hermes_time import now as _hermes_now

    return _hermes_now().isoformat()


class ReminderActionStore:
    """Disk-backed token → reminder-action payload map.

    Mirrors ``gateway.platforms.helpers.ThreadParticipationTracker``: state
    lives under ``get_hermes_home()`` and every mutation is persisted with an
    atomic write. Reads go to disk on each call so that a click handled after a
    restart (or by a freshly-reconnected adapter) still resolves the token.
    """

    def __init__(self, max_entries: int = _MAX_ENTRIES):
        self._max_entries = max_entries

    def _state_path(self):
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "discord_reminder_actions.json"

    def _load(self) -> Dict[str, Any]:
        path = self._state_path()
        if path.exists():
            try:
                import json

                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.debug("Failed to read reminder-action store at %s", path, exc_info=True)
        return {}

    def _save(self, data: Dict[str, Any]) -> None:
        from utils import atomic_json_write

        # Evict oldest entries beyond the cap (by created_at, missing sorts last).
        if len(data) > self._max_entries:
            ordered = sorted(
                data.items(),
                key=lambda kv: kv[1].get("created_at", "") if isinstance(kv[1], dict) else "",
            )
            for token, _ in ordered[: len(data) - self._max_entries]:
                data.pop(token, None)
        atomic_json_write(self._state_path(), data, indent=None)

    def put(self, token: str, entry: Dict[str, Any]) -> None:
        """Persist *entry* under *token* (stamps ``created_at`` if absent)."""
        data = self._load()
        entry = dict(entry)
        entry.setdefault("created_at", _now_iso())
        data[token] = entry
        self._save(data)

    def get(self, token: str) -> Optional[Dict[str, Any]]:
        """Return the entry for *token*, or None."""
        return self._load().get(token)

    def remove(self, token: str) -> bool:
        """Drop *token*; return True if it existed."""
        data = self._load()
        if token in data:
            data.pop(token, None)
            self._save(data)
            return True
        return False


def recreate_reminder(payload: Dict[str, Any], minutes: int) -> Dict[str, Any]:
    """Snooze: re-create the reminder as a fresh one-shot ``minutes`` out.

    *payload* is the dict produced by ``cron.jobs.reminder_payload_from_job``.
    Returns the newly created job dict.
    """
    from cron.jobs import create_job

    kwargs = dict(payload or {})
    # Defensive: never let a stale schedule/repeat sneak through — we always
    # want a fresh one-shot at now + minutes.
    kwargs.pop("schedule", None)
    kwargs.pop("repeat", None)
    return create_job(schedule=f"{minutes}m", **kwargs)


def mark_reminder_done(job_id: Optional[str]) -> bool:
    """Done: remove the underlying job if it still exists.

    One-shot reminders are already gone (auto-deleted on fire), so this is
    typically a no-op returning False; recurring jobs are removed and return
    True. Either way the click is acknowledged by the caller.
    """
    if not job_id:
        return False
    try:
        from cron.jobs import remove_job

        return bool(remove_job(job_id))
    except Exception:
        logger.debug("mark_reminder_done: remove_job(%s) failed", job_id, exc_info=True)
        return False
