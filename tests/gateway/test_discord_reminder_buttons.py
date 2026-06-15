"""Tests for Discord reminder action buttons (Mark Done / Snooze).

When a one-shot cron reminder is delivered to Discord the adapter attaches a
persistent button row:

    [✅ Done] [💤 15m] [💤 1h] [💤 6h] [💤 24h]

Because a one-shot reminder is deleted the instant it fires, "snooze"
re-creates an equivalent one-shot N minutes out rather than rescheduling a
now-deleted job id. Clicks are routed by a discord.py ``DynamicItem`` whose
``custom_id`` carries an opaque token; the token maps to a disk-backed payload
so a click works even after a gateway restart.

Coverage:
  · custom_id build/parse + snooze duration/label tables
  · cron.jobs.reminder_payload_from_job (recreate payload subset)
  · scheduler gating (_is_reminder_job) + metadata injection
  · ReminderActionStore put/get/remove + persistence + eviction
  · recreate_reminder (snooze) / mark_reminder_done (done)
  · build_reminder_action_view + DiscordAdapter._make_reminder_view
  · _handle_reminder_click: done / snooze / stale-token / unauthorized
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Repo root importable
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# Triggers the shared discord mock from tests/gateway/conftest.py before
# importing the production module (no-op when real discord.py is installed).
from plugins.platforms.discord.adapter import (  # noqa: E402
    DiscordAdapter,
    build_reminder_action_view,
    _handle_reminder_click,
)
from plugins.platforms.discord import reminder_actions as ra  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
import cron.jobs as jobs_mod  # noqa: E402
import cron.scheduler as scheduler  # noqa: E402


# Tokens must be hex — the DynamicItem template (real discord.py) validates
# custom_ids against ``[0-9a-f]+``; new_token() produces uuid hex.
_HEX_TOKEN = "a1b2c3d4e5f6"
_HAS_DYNAMIC_ITEMS = build_reminder_action_view(_HEX_TOKEN) is not None


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolate cron job storage to a tempdir (JOBS_FILE et al. are module
    constants resolved at import, so they must be patched per-test)."""
    home = tmp_path / ".hermes"
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", home / "cron" / "output")
    (home / "cron").mkdir(parents=True, exist_ok=True)
    return home


def _make_adapter(*, allowed_users=None, allowed_roles=None, reminder_buttons=True):
    config = PlatformConfig(
        enabled=True, token="test-token", extra={"reminder_buttons": reminder_buttons}
    )
    adapter = DiscordAdapter(config)
    adapter._client = MagicMock()
    adapter._allowed_user_ids = set(allowed_users or [])
    adapter._allowed_role_ids = set(allowed_roles or [])
    return adapter


def _make_interaction(adapter, *, user_id="42", content="Reminder: call mom"):
    user = SimpleNamespace(id=user_id, display_name="Tester", name="tester", roles=[])
    interaction = SimpleNamespace()
    interaction.user = user
    interaction.client = SimpleNamespace(_hermes_discord_adapter=adapter)
    interaction.message = SimpleNamespace(content=content, embeds=[])
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    return interaction


def _one_shot_job(**overrides):
    job = {
        "id": "job123",
        "name": "call mom",
        "prompt": "Remind me to call mom",
        "schedule": {"kind": "once", "run_at": "2026-06-02T10:00:00+00:00"},
        "deliver": "origin",
        "origin": {"platform": "discord", "chat_id": "555", "thread_id": None},
        "skills": [],
        "model": None,
        "provider": None,
        "base_url": None,
        "script": None,
        "context_from": None,
        "enabled_toolsets": None,
        "workdir": None,
        "profile": None,
        "no_agent": False,
    }
    job.update(overrides)
    return job


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestCustomIdAndTables:
    def test_custom_id_round_trip(self):
        cid = ra.build_custom_id("abc123", "snooze60")
        assert cid == "hermes:rem:abc123:snooze60"
        assert ra.parse_custom_id(cid) == ("abc123", "snooze60")

    def test_parse_rejects_foreign_custom_ids(self):
        assert ra.parse_custom_id("clarify:xyz:0") is None
        assert ra.parse_custom_id("") is None
        assert ra.parse_custom_id("hermes:rem:") is None

    def test_snooze_tables_cover_all_buttons(self):
        # Every snooze button maps to a positive duration; Done is not a snooze.
        snooze_actions = [a for a, _, _ in ra.BUTTON_SPECS if a != ra.ACTION_DONE]
        assert set(snooze_actions) == set(ra.SNOOZE_MINUTES)
        assert ra.SNOOZE_MINUTES == {
            "snooze15": 15,
            "snooze60": 60,
            "snooze360": 360,
            "snooze1440": 1440,
        }

    def test_snooze_labels(self):
        assert ra.snooze_label(15) == "15 minutes"
        assert ra.snooze_label(60) == "1 hour"
        assert ra.snooze_label(360) == "6 hours"
        assert ra.snooze_label(1440) == "1 day"


# ---------------------------------------------------------------------------
# cron.jobs.reminder_payload_from_job
# ---------------------------------------------------------------------------

class TestReminderPayload:
    def test_payload_subset_and_exclusions(self):
        job = _one_shot_job()
        payload = jobs_mod.reminder_payload_from_job(job)
        # Carries the defining fields...
        assert payload["prompt"] == "Remind me to call mom"
        assert payload["origin"] == {"platform": "discord", "chat_id": "555", "thread_id": None}
        assert payload["deliver"] == "origin"
        assert payload["no_agent"] is False
        # ...but never the schedule/repeat (caller supplies a fresh one).
        assert "schedule" not in payload
        assert "repeat" not in payload
        assert "id" not in payload  # not a create_job kwarg

    def test_payload_is_create_job_compatible(self, cron_env):
        job = _one_shot_job()
        payload = jobs_mod.reminder_payload_from_job(job)
        created = jobs_mod.create_job(schedule="15m", **payload)
        assert created["prompt"] == job["prompt"]
        assert created["origin"] == job["origin"]


# ---------------------------------------------------------------------------
# scheduler gating + metadata injection
# ---------------------------------------------------------------------------

class TestSchedulerGating:
    def test_is_reminder_job(self):
        assert scheduler._is_reminder_job(_one_shot_job()) is True
        assert scheduler._is_reminder_job({"schedule": {"kind": "interval", "minutes": 10}}) is False
        assert scheduler._is_reminder_job({"schedule": {"kind": "cron", "expr": "* * * * *"}}) is False
        assert scheduler._is_reminder_job({}) is False

    def test_reminder_send_metadata_adds_descriptor_and_preserves_base(self):
        job = _one_shot_job()
        meta = scheduler._reminder_send_metadata(job, {"thread_id": "t99"})
        assert meta["thread_id"] == "t99"  # base preserved
        descriptor = meta["reminder"]
        assert descriptor["job_id"] == "job123"
        assert descriptor["name"] == "call mom"
        assert descriptor["payload"]["prompt"] == "Remind me to call mom"

    def test_reminder_send_metadata_with_no_base(self):
        meta = scheduler._reminder_send_metadata(_one_shot_job(), None)
        assert "reminder" in meta
        assert "thread_id" not in meta


# ---------------------------------------------------------------------------
# ReminderActionStore (uses per-test HERMES_HOME from the autouse fixture)
# ---------------------------------------------------------------------------

class TestReminderActionStore:
    def test_put_get_remove(self):
        store = ra.ReminderActionStore()
        store.put("tok1", {"job_id": "j1", "payload": {"prompt": "x"}})
        got = store.get("tok1")
        assert got["job_id"] == "j1"
        assert "created_at" in got  # stamped on put
        assert store.remove("tok1") is True
        assert store.get("tok1") is None
        assert store.remove("tok1") is False

    def test_persists_across_instances(self):
        ra.ReminderActionStore().put("tok2", {"job_id": "j2", "payload": {}})
        # A fresh instance (simulating a gateway restart) still resolves it.
        assert ra.ReminderActionStore().get("tok2")["job_id"] == "j2"

    def test_eviction_keeps_newest(self):
        store = ra.ReminderActionStore(max_entries=3)
        for i in range(5):
            store.put(f"t{i}", {"job_id": str(i), "created_at": f"2026-06-02T00:00:0{i}+00:00"})
        remaining = {k for k in ("t0", "t1", "t2", "t3", "t4") if store.get(k)}
        assert remaining == {"t2", "t3", "t4"}


# ---------------------------------------------------------------------------
# snooze (recreate) + done (remove)
# ---------------------------------------------------------------------------

class TestReminderActions:
    def test_recreate_reminder_creates_fresh_one_shot(self, cron_env):
        payload = jobs_mod.reminder_payload_from_job(_one_shot_job())
        before = len(jobs_mod.load_jobs())
        new_job = ra.recreate_reminder(payload, 60)
        after = len(jobs_mod.load_jobs())
        assert after == before + 1
        assert new_job["id"] != "job123"  # genuinely new
        assert new_job["schedule"]["kind"] == "once"
        assert new_job["origin"] == {"platform": "discord", "chat_id": "555", "thread_id": None}
        assert new_job["prompt"] == "Remind me to call mom"

    def test_recreate_ignores_stale_schedule_and_repeat(self, cron_env):
        payload = jobs_mod.reminder_payload_from_job(_one_shot_job())
        payload["schedule"] = {"kind": "interval", "minutes": 999}
        payload["repeat"] = {"times": 7, "completed": 3}
        new_job = ra.recreate_reminder(payload, 15)
        assert new_job["schedule"]["kind"] == "once"
        assert new_job["repeat"]["times"] == 1  # one-shot

    def test_mark_done_removes_existing_job(self, cron_env):
        job = jobs_mod.create_job(prompt="do thing", schedule="30m", name="thing")
        assert jobs_mod.get_job(job["id"]) is not None
        assert ra.mark_reminder_done(job["id"]) is True
        assert jobs_mod.get_job(job["id"]) is None

    def test_mark_done_noop_on_missing_job(self, cron_env):
        assert ra.mark_reminder_done("does-not-exist") is False
        assert ra.mark_reminder_done(None) is False


# ---------------------------------------------------------------------------
# View construction + adapter wiring
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_DYNAMIC_ITEMS, reason="discord.ui.DynamicItem unavailable")
class TestViewAndAdapter:
    def test_build_view_buttons(self):
        view = build_reminder_action_view(_HEX_TOKEN)
        assert view.timeout is None
        cids = [getattr(c, "item", c).custom_id for c in view.children]
        assert cids == [
            f"hermes:rem:{_HEX_TOKEN}:done",
            f"hermes:rem:{_HEX_TOKEN}:snooze15",
            f"hermes:rem:{_HEX_TOKEN}:snooze60",
            f"hermes:rem:{_HEX_TOKEN}:snooze360",
            f"hermes:rem:{_HEX_TOKEN}:snooze1440",
        ]

    def test_make_reminder_view_returns_tuple_and_defers_persist(self):
        adapter = _make_adapter()
        descriptor = {"job_id": "j1", "name": "n", "payload": {"prompt": "p"}}
        view, token, entry = adapter._make_reminder_view(descriptor, chat_id="555", thread_id=None)
        assert view is not None and token and entry
        # Token is the one embedded in the buttons.
        assert getattr(view.children[0], "item", view.children[0]).custom_id == f"hermes:rem:{token}:done"
        assert entry["job_id"] == "j1"
        assert entry["payload"] == {"prompt": "p"}
        assert entry["channel_id"] == "555"
        # Crucially NOT persisted yet — send() persists only after a successful
        # send so a failed send leaves no orphan store entry.
        assert adapter._get_reminder_store().get(token) is None

    def test_make_reminder_view_disabled_by_flag(self):
        adapter = _make_adapter(reminder_buttons=False)
        assert adapter._reminder_buttons is False
        assert adapter._make_reminder_view({"job_id": "x"}, "555", None) == (None, None, None)

    def test_make_reminder_view_none_descriptor(self):
        adapter = _make_adapter()
        assert adapter._make_reminder_view(None, "555", None) == (None, None, None)

    @pytest.mark.asyncio
    async def test_send_persists_token_after_success(self):
        adapter = _make_adapter()
        sent = []

        async def fake_send(**kwargs):
            sent.append(kwargs)
            return SimpleNamespace(id=98765)

        adapter._client.get_channel = MagicMock(return_value=SimpleNamespace(send=fake_send))
        adapter._is_forum_parent = lambda c: False

        descriptor = {"job_id": "j9", "name": "n", "payload": {"prompt": "p"}}
        res = await adapter.send("555", "hi", metadata={"reminder": descriptor})

        assert res.success
        view = next(k["view"] for k in sent if "view" in k)  # buttons on the (only) chunk
        token = getattr(view.children[0], "item", view.children[0]).custom_id.split(":")[2]
        assert adapter._get_reminder_store().get(token) is not None  # persisted after send

    @pytest.mark.asyncio
    async def test_send_failure_leaves_no_orphan_token(self):
        adapter = _make_adapter()
        captured = {}

        async def fail_send(**kwargs):
            v = kwargs.get("view")
            if v is not None:
                captured["token"] = getattr(
                    v.children[0], "item", v.children[0]
                ).custom_id.split(":")[2]
            raise RuntimeError("simulated discord send failure")

        adapter._client.get_channel = MagicMock(return_value=SimpleNamespace(send=fail_send))
        adapter._is_forum_parent = lambda c: False

        descriptor = {"job_id": "j9", "name": "n", "payload": {"prompt": "p"}}
        res = await adapter.send("555", "hi", metadata={"reminder": descriptor})

        assert not res.success
        assert "token" in captured  # the view (with token) was attempted
        assert adapter._get_reminder_store().get(captured["token"]) is None  # no orphan


# ---------------------------------------------------------------------------
# Click handling
# ---------------------------------------------------------------------------

class TestHandleReminderClick:
    @pytest.mark.asyncio
    async def test_done_acknowledges_and_clears_token(self, cron_env):
        adapter = _make_adapter()
        job = jobs_mod.create_job(prompt="call mom", schedule="30m", name="call mom")
        store = adapter._get_reminder_store()
        store.put("tokD", {"job_id": job["id"], "name": "call mom", "payload": {}})
        interaction = _make_interaction(adapter)

        await _handle_reminder_click(interaction, "tokD", "done")

        # Underlying job removed, token cleared, message edited (buttons gone).
        assert jobs_mod.get_job(job["id"]) is None
        assert store.get("tokD") is None
        interaction.response.edit_message.assert_awaited_once()
        kwargs = interaction.response.edit_message.await_args.kwargs
        assert kwargs.get("view") is None
        assert "Marked done" in kwargs.get("content", "")

    @pytest.mark.asyncio
    async def test_snooze_recreates_reminder(self, cron_env):
        adapter = _make_adapter()
        payload = jobs_mod.reminder_payload_from_job(_one_shot_job())
        store = adapter._get_reminder_store()
        store.put("tokS", {"job_id": "job123", "name": "call mom", "payload": payload})
        interaction = _make_interaction(adapter)

        before = len(jobs_mod.load_jobs())
        await _handle_reminder_click(interaction, "tokS", "snooze60")
        after = len(jobs_mod.load_jobs())

        assert after == before + 1  # a fresh one-shot was created
        assert store.get("tokS") is None
        interaction.response.edit_message.assert_awaited_once()
        kwargs = interaction.response.edit_message.await_args.kwargs
        assert kwargs.get("view") is None
        assert "Snoozed" in kwargs.get("content", "")

    @pytest.mark.asyncio
    async def test_stale_token_disables_buttons(self, cron_env):
        adapter = _make_adapter()
        interaction = _make_interaction(adapter)
        await _handle_reminder_click(interaction, "nope", "done")
        # No crash; buttons removed so the dead message stops inviting clicks.
        interaction.response.edit_message.assert_awaited_once_with(view=None)

    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected(self, cron_env):
        adapter = _make_adapter(allowed_users=["999"])  # interaction user is "42"
        store = adapter._get_reminder_store()
        store.put("tokU", {"job_id": "j", "payload": {}})
        interaction = _make_interaction(adapter, user_id="42")

        await _handle_reminder_click(interaction, "tokU", "done")

        # Ephemeral rejection; token untouched; no message edit.
        interaction.response.send_message.assert_awaited_once()
        assert interaction.response.send_message.await_args.kwargs.get("ephemeral") is True
        assert store.get("tokU") is not None
        interaction.response.edit_message.assert_not_called()
