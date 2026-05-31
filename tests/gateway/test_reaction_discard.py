"""Gateway-level tests for reaction-only acknowledgement suppression.

When the agent emits a successful platform reaction this turn (e.g. discord
``react_to_message``), the reaction IS the entire user-facing response.  The
agent blanks ``final_response`` and reports ``turn_exit_reason ==
"terminal_ack_tool"``.  The gateway must:

  1. Propagate ``turn_exit_reason`` out of ``_run_agent`` (the bug: it was
     dropped before reaching ``_normalize_empty_agent_response``).
  2. With streaming on, delete any text message that was already streamed
     into the chat before the turn ended (via ``discard_sent()``), so the
     user sees ONLY the reaction.

A normal turn (no reaction) must be unaffected.
"""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class StreamCaptureAdapter(BasePlatformAdapter):
    """Adapter recording sends / edits / deletes for inspection."""

    _next_mid = 500

    def __init__(self, platform=Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.deleted = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    def _mint_id(self) -> str:
        StreamCaptureAdapter._next_mid += 1
        return str(StreamCaptureAdapter._next_mid)

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        mid = self._mint_id()
        self.sent.append({"chat_id": chat_id, "content": content, "message_id": mid})
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content, **kwargs) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.append({"chat_id": chat_id, "message_id": str(message_id)})
        return True

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class ReactionAgent:
    """Streams a redundant text reply, then signals a reaction-only turn.

    Mirrors the production agent: a reaction succeeded this turn (sets
    ``_reaction_emitted_this_turn``), the model ALSO wrote text (streamed via
    the delta callback), and ``run_conversation`` returns the blanked
    terminal-ack result that ``agent/conversation_loop.py`` produces.
    """

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.stream_delta_callback = None
        self.tools = []
        self._reaction_emitted_this_turn = False

    def run_conversation(self, message, conversation_history=None, task_id=None):
        # The stream-delta callback is wired by the gateway AFTER construction
        # (agent.stream_delta_callback = ...).  Push redundant text through it
        # so the consumer sends a real message before the turn ends.
        cb = self.stream_delta_callback
        if cb is not None:
            # Give the consumer task time to start polling, then stream text.
            time.sleep(0.15)
            cb("Done — reminder set for 5 minutes.")
            # Let the consumer flush the delta into a real platform message.
            time.sleep(0.25)
        # A reaction succeeded this turn → the agent blanks the text response.
        self._reaction_emitted_this_turn = True
        return {
            "final_response": "",
            "messages": [],
            "api_calls": 2,
            "completed": True,
            "failed": False,
            "turn_exit_reason": "terminal_ack_tool",
        }


class NormalAgent:
    """Streams text and returns it normally — no reaction this turn."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.stream_delta_callback = None
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.stream_delta_callback
        if cb is not None:
            time.sleep(0.15)
            cb("Here is the answer.")
            time.sleep(0.25)
        return {
            "final_response": "Here is the answer.",
            "messages": [],
            "api_calls": 1,
            "completed": True,
            "turn_exit_reason": "text_response(finish_reason=stop)",
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    # Streaming ON with a tight edit interval so the consumer flushes fast.
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=StreamingConfig(
            enabled=True, transport="edit", edit_interval=0.01, buffer_threshold=1,
        ),
    )
    return runner


def _install_fakes(monkeypatch, agent_cls):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    return gateway_run


@pytest.mark.asyncio
async def test_reaction_turn_propagates_exit_reason_and_discards_streamed_text(
    monkeypatch, tmp_path
):
    """A reaction-only turn that ALSO streamed text: the gateway returns
    final_response="" with turn_exit_reason terminal_ack_tool, and deletes
    the already-streamed message via discard_sent()."""
    adapter = StreamCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, ReactionAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.DISCORD, chat_id="chan-1", chat_type="dm")
    session_key = "agent:main:discord:dm:chan-1"

    result = await runner._run_agent(
        message="remind me in 5 minutes",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    # (1) turn_exit_reason propagated out of _run_agent (the dropped-field bug).
    assert result["turn_exit_reason"] == "terminal_ack_tool"
    # (2) No text is delivered — the reaction is the whole response.
    assert result["final_response"] == ""
    # (3) The streamed text message was sent, then discarded (deleted).
    assert len(adapter.sent) >= 1, f"expected a streamed message; sent={adapter.sent}"
    streamed_ids = {m["message_id"] for m in adapter.sent}
    deleted_ids = {d["message_id"] for d in adapter.deleted}
    assert streamed_ids & deleted_ids, (
        f"streamed message not deleted; sent={adapter.sent} deleted={adapter.deleted}"
    )

    # _normalize_empty_agent_response keeps the empty response silent because
    # turn_exit_reason survived the trip out of _run_agent.
    normalized = gateway_run._normalize_empty_agent_response(result, result["final_response"])
    assert normalized == ""


@pytest.mark.asyncio
async def test_normal_turn_is_unaffected_no_discard(monkeypatch, tmp_path):
    """Regression: a normal turn (no reaction) delivers its text and never
    deletes the streamed message."""
    adapter = StreamCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = _install_fakes(monkeypatch, NormalAgent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(platform=Platform.DISCORD, chat_id="chan-2", chat_type="dm")
    session_key = "agent:main:discord:dm:chan-2"

    result = await runner._run_agent(
        message="look something up",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-2",
        session_key=session_key,
    )

    assert result["final_response"] == "Here is the answer."
    assert result["turn_exit_reason"] != "terminal_ack_tool"
    # The streamed message must NOT be deleted on a normal turn.
    assert adapter.deleted == [], f"normal turn should not delete; deleted={adapter.deleted}"
