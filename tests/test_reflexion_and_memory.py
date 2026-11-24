"""Tests for the Reflexion retry loop and auto-memory file names. No model server needed:
the model is a scripted fake client.

    uv run --with pytest pytest tests
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # harness.py lives at the repo root

import harness  # noqa: E402
from agentic_harness_build import auto_memory, memory  # noqa: E402
from agentic_harness_build.reflection import SUBMITTED_MARKER, ReflectionEngine  # noqa: E402


# --- fakes ------------------------------------------------------------------------


def _text(content):
    return SimpleNamespace(content=content, tool_calls=None)


def _tool_call(name, args, call_id="c1"):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(content="", tool_calls=[call])


class FakeClient:
    """Answers chat requests from a script; reflection prompts get a fixed reflection."""

    def __init__(self, script, reflection="Try the venue's own site."):
        self.script = list(script)
        self.reflection = reflection
        self.requests = []  # (kind, messages)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        messages = [dict(m) for m in kwargs["messages"]]
        is_reflection = len(messages) == 1 and messages[0]["content"].rstrip().endswith("Reflection:")
        kind = "reflection" if is_reflection else "agent"
        self.requests.append((kind, messages))
        message = self.script.pop(0) if kind == "agent" else _text(self.reflection)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def agent_requests(self):
        return [m for kind, m in self.requests if kind == "agent"]

    def reflection_count(self):
        return sum(kind == "reflection" for kind, _ in self.requests)


class FakeAgent:
    log_level = "warn"
    reply = {}

    def __init__(self, task_type, max_retries=2, reflect_after="failure_only", max_steps=3, tool_results=None):
        self.model = {"name": "fake", "max_steps": max_steps, "context_window": 16384}
        self.reflection = {
            "enabled": True, "strategy": "reflexion", "task_type": task_type,
            "reflect_after": reflect_after, "max_retries": max_retries, "max_reflections": 5,
        }
        self.tool_results = tool_results or {}
        self.calls = []

    def openai_tools(self):
        return []

    def call_tool(self, name, arguments):
        self.calls.append(name)
        return self.tool_results.get(name, "ok")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # ReflectionStorage writes to ./data/reflections
    monkeypatch.setattr(harness.ui, "say", lambda *a, **k: None)
    memory_dir = tmp_path / "data" / "memory"
    monkeypatch.setattr(auto_memory, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(memory, "MEMORY_DIR", memory_dir)  # ensure_memory_dir reads this one
    return tmp_path


def _start(question="table for 2 at Meatbar Thursday 15:00?"):
    return [{"role": "system", "content": "You are a booking assistant."}, {"role": "user", "content": question}]


def _stored_reflections(tmp_path, task_type):
    path = tmp_path / "data" / "reflections" / f"{task_type}.jsonl"
    return path.read_text().splitlines() if path.exists() else []


# --- 1. auto-memory file names cannot leave the memory folder --------------------


@pytest.mark.parametrize("name", ["../../README.md", "../x", "/etc/passwd.md", "a/b.md", "MEMORY.md", "", None, 42])
def test_unsafe_memory_filenames_are_rejected(name):
    assert auto_memory._safe_filename(name) is None


def test_safe_memory_filename_gets_md():
    assert auto_memory._safe_filename("user_prefs") == "user_prefs.md"


def test_save_learnings_never_writes_outside_memory_dir(isolated):
    outside = isolated / "README.md"
    outside.write_text("original\n")
    auto_memory.save_learnings([
        {"type": "project", "content": "planted", "file": "../../README.md"},
        {"type": "../../evil", "content": "planted too"},
        {"type": "user", "content": "likes sushi", "file": "food.md"},
    ])
    assert outside.read_text() == "original\n"
    written = sorted(p.name for p in (isolated / "data" / "memory").iterdir())
    assert written == ["MEMORY.md", "food.md", "project_context.md"]
    assert not list(isolated.glob("**/evil*"))


# --- 2. booking only retries on real failures -----------------------------------


def test_booking_availability_answer_is_not_retried(isolated):
    client = FakeClient([_text("Yes, 15:00 Thursday is free for 2. Book here: https://ontopo.com/x")])
    agent = FakeAgent("booking", max_retries=3)
    answer = harness.run_agent(agent, client, _start(), "table for 2?")
    assert "15:00" in answer
    assert len(client.agent_requests()) == 1
    assert client.reflection_count() == 0


def test_submitted_booking_is_never_retried(isolated):
    # the user approved the final click, then the model ran out of steps: still no retry
    client = FakeClient([_tool_call("browser_click", {"ref": "e1"})] * 3 + [_text("")])
    agent = FakeAgent("booking", max_retries=3, tool_results={"browser_click": f"{SUBMITTED_MARKER} Clicked [e1]"})
    answer = harness.run_agent(agent, client, _start(), "book it")
    assert answer.startswith("(stopped after")
    assert agent.calls.count("browser_click") == 3  # one attempt only


def test_booking_retries_when_out_of_steps(isolated):
    client = FakeClient(
        [_tool_call("web_search", {"query": "meatbar"})] * 2 + [_text("not sure")]  # attempt 1: out of steps
        + [_text("Free at 15:00, book at https://ontopo.com/x")]  # attempt 2
    )
    agent = FakeAgent("booking", max_retries=3, max_steps=2)
    answer = harness.run_agent(agent, client, _start(), "table?")
    assert answer.startswith("Free at 15:00")


def test_attempt_failed_rules():
    booking, research = ReflectionEngine(agent_type="booking"), ReflectionEngine(agent_type="research")
    assert not booking.attempt_failed("Free at 19:00, link: https://x", {})
    assert booking.attempt_failed("(stopped after 40 steps without a final answer)", {})
    assert not booking.attempt_failed("(stopped after 40 steps)", {"submitted": True})
    assert booking.attempt_failed("anything", {"out_of_steps": True})
    assert research.attempt_failed("no", {})
    assert not research.attempt_failed("Apple EPS $1.98, source: https://x", {})


# --- 3. a retry starts from a clean conversation ---------------------------------


def test_retry_starts_from_the_original_messages_plus_reflection(isolated):
    client = FakeClient(
        [_tool_call("web_search", {"query": "a"})] * 2 + [_text("")]  # attempt 1: out of steps, empty
        + [_text("Answer with source https://example.com")]  # attempt 2
    )
    agent = FakeAgent("research", max_retries=2, max_steps=2)
    messages = _start("Apple EPS?")
    answer = harness.run_agent(agent, client, messages, "Apple EPS?")
    assert answer.startswith("Answer with source")

    retry_request = client.agent_requests()[3]  # first model call of attempt 2
    assert [m["role"] for m in retry_request] == ["system", "user"]  # nothing left over from attempt 1
    assert "Try the venue's own site." in retry_request[0]["content"]  # but the reflection is there
    assert retry_request[1]["content"] == "Apple EPS?"
    # the caller's list now holds the kept attempt, and its system prompt was not edited in place
    assert messages[-1] == {"role": "assistant", "content": answer}


# --- 4. each attempt is reflected on at most once -------------------------------


def test_one_reflection_per_failed_attempt(isolated):
    client = FakeClient(
        [_tool_call("web_search", {"query": "a"})] * 2 + [_text("")]  # attempt 1 fails
        + [_tool_call("web_search", {"query": "b"})] * 2 + [_text("")]  # attempt 2 fails, no retries left
    )
    agent = FakeAgent("research", max_retries=1, max_steps=2, reflect_after="failure_only")
    harness.run_agent(agent, client, _start(), "q")
    assert client.reflection_count() == 2
    assert len(_stored_reflections(isolated, "research")) == 2


def test_success_with_failure_only_writes_no_reflection(isolated):
    client = FakeClient([_text("EPS $1.98, source https://x")])
    agent = FakeAgent("research", reflect_after="failure_only")
    harness.run_agent(agent, client, _start(), "q")
    assert client.reflection_count() == 0
    assert _stored_reflections(isolated, "research") == []


def test_success_with_every_turn_writes_exactly_one(isolated):
    client = FakeClient([_text("EPS $1.98, source https://x")])
    agent = FakeAgent("research", reflect_after="every_turn")
    harness.run_agent(agent, client, _start(), "q")
    assert client.reflection_count() == 1
    assert len(_stored_reflections(isolated, "research")) == 1
