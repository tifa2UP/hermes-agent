"""Tests for Discord message reactions tied to processing lifecycle hooks."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome, SendResult
from gateway.session import SessionSource, build_session_key


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn

        return decorator


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    return adapter


def _make_event(message_id: str, raw_message) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="123",
            chat_type="dm",
            user_id="42",
            user_name="Jezza",
        ),
        raw_message=raw_message,
        message_id=message_id,
    )


@pytest.mark.asyncio
async def test_process_message_background_adds_and_swaps_reactions(adapter):
    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("1", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    assert raw_message.add_reaction.await_args_list[0].args == ("👀",)
    assert raw_message.remove_reaction.await_args_list[0].args == ("👀", adapter._client.user)
    assert raw_message.add_reaction.await_args_list[1].args == ("✅",)


@pytest.mark.asyncio
async def test_reaction_only_empty_turn_is_success_without_text_send(adapter):
    """An empty (reaction-only) agent turn must be delivered cleanly: the
    gateway sends NO text message and reports SUCCESS, so the lifecycle
    swaps 👀→✅ (never ❌). This is the intentional 'remind me to X →
    just react' outcome."""
    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return ""  # agent acknowledged via a reaction tool; no text reply

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("8", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    # No text message is sent for an empty, intentional turn.
    adapter.send.assert_not_awaited()
    # 👀 on start → removed on complete → ✅ (SUCCESS), never ❌ (FAILURE).
    assert raw_message.add_reaction.await_args_list[0].args == ("👀",)
    assert raw_message.remove_reaction.await_args_list[0].args == ("👀", adapter._client.user)
    assert raw_message.add_reaction.await_args_list[1].args == ("✅",)
    added = [c.args for c in raw_message.add_reaction.await_args_list]
    assert ("❌",) not in added


@pytest.mark.asyncio
async def test_interaction_backed_events_do_not_attempt_reactions(adapter):
    interaction = SimpleNamespace(guild_id=123456789)

    async def handler(_event):
        await asyncio.sleep(0)
        return None

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter._add_reaction = AsyncMock()
    adapter._remove_reaction = AsyncMock()
    adapter._keep_typing = hold_typing

    event = MessageEvent(
        text="/status",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="123",
            chat_type="dm",
            user_id="42",
            user_name="Jezza",
        ),
        raw_message=interaction,
        message_id="2",
    )

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter._add_reaction.assert_not_awaited()
    adapter._remove_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_reaction_helper_failures_do_not_break_message_flow(adapter):
    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(side_effect=[RuntimeError("no perms"), RuntimeError("no perms")]),
        remove_reaction=AsyncMock(side_effect=RuntimeError("no perms")),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("3", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_reactions_disabled_via_env(adapter, monkeypatch):
    """When DISCORD_REACTIONS=false, no reactions should be added."""
    monkeypatch.setenv("DISCORD_REACTIONS", "false")

    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("4", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    raw_message.add_reaction.assert_not_awaited()
    raw_message.remove_reaction.assert_not_awaited()
    # Response should still be sent
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_reactions_disabled_via_env_zero(adapter, monkeypatch):
    """DISCORD_REACTIONS=0 should also disable reactions."""
    monkeypatch.setenv("DISCORD_REACTIONS", "0")

    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    event = _make_event("5", raw_message)
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    raw_message.add_reaction.assert_not_awaited()
    raw_message.remove_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_reactions_enabled_by_default(adapter, monkeypatch):
    """When DISCORD_REACTIONS is unset, reactions should still work (default: true)."""
    monkeypatch.delenv("DISCORD_REACTIONS", raising=False)

    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    event = _make_event("6", raw_message)
    await adapter.on_processing_start(event)

    raw_message.add_reaction.assert_awaited_once_with("👀")


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_removes_eyes_without_terminal_reaction(adapter):
    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    event = _make_event("7", raw_message)
    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    raw_message.remove_reaction.assert_awaited_once_with("👀", adapter._client.user)
    raw_message.add_reaction.assert_not_awaited()


# ── react_to_message (web-search reaction bridge) ────────────────────


@pytest.mark.asyncio
async def test_react_to_message_adds_reaction(adapter):
    target = SimpleNamespace(add_reaction=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=target))
    adapter._client.get_channel = lambda _id: channel

    ok = await adapter.react_to_message(chat_id="123", message_id="55", emoji="🌐")

    assert ok is True
    channel.fetch_message.assert_awaited_once_with(55)
    target.add_reaction.assert_awaited_once_with("🌐")


@pytest.mark.asyncio
async def test_react_to_message_prefers_thread_metadata(adapter):
    target = SimpleNamespace(add_reaction=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=target))
    seen_ids = []

    def _get_channel(_id):
        seen_ids.append(_id)
        return channel

    adapter._client.get_channel = _get_channel

    ok = await adapter.react_to_message(
        chat_id="123", message_id="55", emoji="🌐", metadata={"thread_id": "777"}
    )

    assert ok is True
    assert seen_ids == [777]  # thread_id wins over chat_id


@pytest.mark.asyncio
async def test_react_to_message_disabled_via_env(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REACTIONS", "false")
    channel = SimpleNamespace(fetch_message=AsyncMock())
    adapter._client.get_channel = lambda _id: channel

    ok = await adapter.react_to_message(chat_id="123", message_id="55", emoji="🌐")

    assert ok is False
    channel.fetch_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_react_to_message_swallows_errors(adapter):
    channel = SimpleNamespace(fetch_message=AsyncMock(side_effect=RuntimeError("boom")))
    adapter._client.get_channel = lambda _id: channel

    ok = await adapter.react_to_message(chat_id="123", message_id="55", emoji="🌐")

    assert ok is False
