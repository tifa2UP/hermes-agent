"""Tests for the web-search reaction gateway↔tool bridge."""

import pytest

from tools import web_search_reaction as wsr


@pytest.fixture(autouse=True)
def _clear_registry():
    # Ensure a clean registry around every test.
    wsr._notify_cbs.clear()
    yield
    wsr._notify_cbs.clear()


def test_register_and_get_notify():
    calls = []
    wsr.register_notify("sess-1", lambda: calls.append(1))
    assert wsr.get_notify("sess-1") is not None
    wsr.unregister_notify("sess-1")
    assert wsr.get_notify("sess-1") is None


def test_register_ignores_empty_task_id():
    wsr.register_notify("", lambda: None)
    assert wsr._notify_cbs == {}


def test_pre_tool_call_fires_for_web_search():
    calls = []
    wsr.register_notify("sess-1", lambda: calls.append("hit"))

    result = wsr.pre_tool_call("web_search", {"query": "hi"}, task_id="sess-1")

    assert result is None  # observer-only, never blocks
    assert calls == ["hit"]


def test_pre_tool_call_ignores_other_tools():
    calls = []
    wsr.register_notify("sess-1", lambda: calls.append("hit"))

    wsr.pre_tool_call("read_file", {"path": "x"}, task_id="sess-1")

    assert calls == []


def test_pre_tool_call_noop_without_registered_session():
    # No callback registered for this task_id → must not raise.
    assert wsr.pre_tool_call("web_search", {}, task_id="unknown") is None


def test_pre_tool_call_swallows_callback_errors():
    def _boom():
        raise RuntimeError("nope")

    wsr.register_notify("sess-1", _boom)

    # Must not propagate — a reaction failure can never break the tool call.
    assert wsr.pre_tool_call("web_search", {}, task_id="sess-1") is None
