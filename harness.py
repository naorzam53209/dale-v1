import json
import os
import re
import sys
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

from openai import APIError, OpenAI

from agentic_harness_build import ui
from agentic_harness_build.manifest import ManifestError, load_assistant
from agentic_harness_build.reflection import (
    SUBMITTED_MARKER, USER_ANSWER_PREFIXES, USER_TOOLS, ReflectionEngine, ReflexionStrategy,
)
from agentic_harness_build.memory import load_memory_index, ensure_memory_dir
from agentic_harness_build.auto_memory import extract_learnings, save_learnings

# --- which agent to run: everything about it lives in its manifest -----------

DEFAULT_MANIFEST = "agents/assistant.agents.json"  # research + booking behind one router
API_KEY = "ollama"  # Ollama ignores it, but the client requires one
ROUTER_HISTORY_MESSAGES = 4  # the router sees the last two turns
ROUTER_MAX_TOKENS = 10  # it only answers with an agent name
_clients: dict = {}
RESULT_PREVIEW_CHARS = 300  # how much of each tool result to show at log_level "debug"
DEFAULT_CONTEXT_WINDOW = 4096  # Ollama's default; set x-model.context_window to what the server really uses
CONTEXT_FILL = 0.7  # fill at most this share with the prompt; the rest is for thinking and the answer
CHARS_PER_TOKEN = 3.5  # rough estimate, good enough for budgeting
TRIMMED_RESULT_CHARS = 400  # what is left of an old tool result after trimming
EMPTY_ANSWER_NUDGE = "Your last reply was empty. Answer my question now using what you found so far."
PHONE = re.compile(r"(?<![\d\w])(?:\+\d{1,3}[\s-]?)?\(?0?\d{1,3}\)?[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)")
UNVERIFIED_PHONE_REQUEST = (
    "These phone numbers do not appear in any page you read: {numbers}. Never make up phone numbers. "
    "Rewrite your answer using only numbers you actually saw, or give the venue's website link instead."
)
MARKDOWN = re.compile(r"\*\*|^\s{0,3}(#{1,6}\s|[-*•]\s)", re.M)
SHORTEN_REQUEST = (
    "Rewrite your last answer in at most {limit} characters, as plain text: only the key facts "
    "(or the one question you need answered), then at most one source link. No markdown, no lists."
)
OUT_OF_STEPS_NUDGE = (
    "You have used all your tool steps. Answer my question now with what you found so far, "
    "say what you could not verify, and list your sources."
)


def _assistant_entry(message) -> dict:
    """Keep only what the model needs next turn: the answer and tool calls, not its thinking."""
    entry = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        entry["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in message.tool_calls
        ]
    return entry


def _fit_context(messages: list, tools: list, window: int):
    """Shorten old tool results, oldest first, until the conversation fits the model's context window.

    Ollama silently drops the *start* of a prompt that is too long, which would lose the
    instructions and the user's question; shrinking old page dumps is much less harmful."""
    budget = int(window * CONTEXT_FILL * CHARS_PER_TOKEN) - len(json.dumps(tools))
    size = lambda: sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in messages)
    tool_results = [m for m in messages if m["role"] == "tool"]
    for m in tool_results:
        overflow = size() - budget
        if overflow <= 0:
            return
        keep = max(TRIMMED_RESULT_CHARS, len(m["content"]) - overflow) if m is tool_results[-1] else TRIMMED_RESULT_CHARS
        if len(m["content"]) > keep:
            m["content"] = m["content"][:keep] + "\n(rest removed to fit the context window)"


def _ask(agent, client, messages: list, **extra):
    model = agent.model
    _fit_context(messages, agent.openai_tools(), model.get("context_window", DEFAULT_CONTEXT_WINDOW))
    response = client.chat.completions.create(
        model=model["name"], temperature=model.get("temperature", 0.7), messages=messages, **extra
    )
    return response.choices[0].message


def _plain(text: str) -> str:
    """Drop markdown the terminal can't show: **bold**, # headings, bullets."""
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m[1] or m[2], text)
    text = re.sub(r"^\s{0,3}(#{1,6}\s+|[-*•]\s+)", "", text, flags=re.M)
    return re.sub(r"\n{2,}", "\n", text).strip()


def _digits(text: str) -> str:
    return re.sub(r"(?<=\d)[\s\-().]+(?=\d)", "", text)


def _unverified_phones(answer: str, sources: list[str]) -> list[str]:
    """Phone numbers in the answer that appear in none of the tool results the model read this turn."""
    seen = _digits("\n".join(sources))
    seen += "\n" + re.sub(r"\+?972-?", "0", seen)  # +972-3-... on a page is 03-... in an answer
    missing = []
    for number in PHONE.findall(answer):
        digits = re.sub(r"\D", "", number)
        local = "0" + digits[3:] if digits.startswith("972") else digits  # and the other way round
        if digits not in seen and local not in seen:
            missing.append(number.strip())
    return missing


def _ground_phones(agent, client, messages: list, answer: str, sources: list[str]) -> str:
    """Small models invent phone numbers. Ask for one correction, then remove what is still made up."""
    if not agent.reply.get("verify_phone_numbers"):
        return answer
    missing = _unverified_phones(answer, sources)
    if missing:
        request = {"role": "user", "content": UNVERIFIED_PHONE_REQUEST.format(numbers=", ".join(missing))}
        fixed = (_ask(agent, client, messages + [request]).content or "").strip()
        if fixed:
            messages[-1] = {"role": "assistant", "content": fixed}
            answer = fixed
    for number in _unverified_phones(answer, sources):
        answer = answer.replace(number, "(number not found on their site)")
    return answer


def _finish(agent, client, messages: list, answer: str, sources: list[str]) -> str:
    answer = _ground_phones(agent, client, messages, answer, sources)
    answer = _shorten(agent, client, messages, answer)
    return _ground_phones(agent, client, messages, answer, sources) if agent.reply.get("verify_phone_numbers") else answer


def _shorten(agent, client, messages: list, answer: str) -> str:
    """Small models ignore length rules, so enforce x-reply.max_chars with one rewrite pass."""
    limit = agent.reply.get("max_chars")
    if not limit or (len(answer) <= limit and not MARKDOWN.search(answer)):
        return _plain(answer)
    request = {"role": "user", "content": SHORTEN_REQUEST.format(limit=limit)}
    rewrite = (_ask(agent, client, messages + [request]).content or "").strip()
    if rewrite and len(rewrite) < len(answer):
        messages[-1] = {"role": "assistant", "content": rewrite}  # history keeps the short version
        answer = rewrite
    return _plain(answer)


def _reflection_engine(agent):
    """The agent's ReflectionEngine with its stored reflections loaded, or None when reflection is off."""
    if not agent.reflection.get("enabled", False):
        return None
    task_type = agent.reflection.get("task_type", "default")
    engine = ReflectionEngine(
        strategy=ReflexionStrategy(agent.reflection.get("strategy", "reflexion")),
        task_type=task_type,
        agent_type=task_type,  # task_type doubles as the agent type (booking, research...) for its prompts
    )
    engine.load_stored_reflections(limit=agent.reflection.get("max_reflections", 5))
    return engine


def _with_reflections(base: list, engine) -> list:
    """A fresh copy of the turn's starting messages, with the current reflections added to the system prompt."""
    attempt = [dict(m) for m in base]
    context = engine.get_reflections_context() if engine else ""
    if context and attempt and attempt[0].get("role") == "system":
        attempt[0]["content"] += "\n\n" + context
    return attempt


def run_agent(agent, client, messages: list, user_input: str = "") -> str:
    """Answer one user message, retrying with a self-reflection when an attempt fails (Reflexion).

    Every attempt starts from the same messages the turn started with, plus the reflections so far:
    a retry never sees the failed attempt's tool calls or answer, only the lesson drawn from it.
    Each attempt is reflected on at most once. On return, `messages` holds the attempt that was kept.
    """
    engine = _reflection_engine(agent)
    max_retries = agent.reflection.get("max_retries", 0) if engine else 0
    base = [dict(m) for m in messages]

    for attempt in range(max_retries + 1):
        if attempt > 0:
            ui.say(f"[reflexion] Retry attempt {attempt}/{max_retries}")
        attempt_messages = _with_reflections(base, engine)
        answer, outcome = _run_agent_once(agent, client, attempt_messages)
        failed = engine.attempt_failed(answer, outcome) if engine else False
        if not (engine and failed and engine.should_retry(answer, max_retries, attempt, outcome)):
            break
        # reflect on the failed attempt (once), then run the turn again from the start
        engine.update_reflections(user_input, outcome["tool_log"], answer, client, agent.model, reflect_after="every_turn")
        _reset_tool_state(agent)

    messages[:] = attempt_messages
    # the kept attempt has not been reflected on yet: do it now if the manifest asks for it
    if engine and (agent.reflection.get("reflect_after", "every_turn") == "every_turn" or failed):
        engine.update_reflections(user_input, outcome["tool_log"], answer, client, agent.model, reflect_after="every_turn")
    return answer


def _reset_tool_state(agent):
    """Undo what a failed attempt left behind that the retry cannot see in its fresh conversation.

    The browser would still show the old page with stale refs and keep an approved card usable,
    so close it (which also wipes the card). Lists like the failed-URL cache are kept on purpose:
    their errors explain themselves."""
    tool = agent.tools.get("browser_close") if hasattr(agent, "tools") else None
    if tool is not None:
        try:
            tool.function()
        except Exception as e:  # noqa: BLE001  (a browser that won't close must not end the turn)
            ui.say(f"(could not reset the browser before retrying: {e})")


def _run_agent_once(agent, client, messages: list) -> tuple[str, dict]:
    """One attempt: call the model, run the tools it asks for, repeat until it answers in text.

    Returns the answer and what happened: submitted (the user approved a final submit),
    asked_user (the user was asked or confirmed something), out_of_steps, empty, and
    tool_log (the tool results, shortened, for reflection)."""
    max_steps = agent.model.get("max_steps", 10)
    nudged = False
    sources = []  # full tool results of this turn (the history copies may get trimmed)
    outcome = {"submitted": False, "asked_user": False, "out_of_steps": False, "empty": False, "tool_log": ""}
    tool_log = []

    for _ in range(max_steps):
        message = _ask(agent, client, messages, tools=agent.openai_tools())

        if not message.tool_calls:
            if (message.content or "").strip():
                messages.append(_assistant_entry(message))
                outcome["tool_log"] = "\n".join(tool_log)
                return _finish(agent, client, messages, message.content, sources), outcome
            if nudged:
                outcome.update(empty=True, tool_log="\n".join(tool_log))
                return "(the model gave an empty answer twice; try rephrasing the question)", outcome
            messages.append({"role": "user", "content": EMPTY_ANSWER_NUDGE})
            nudged = True
            continue

        messages.append(_assistant_entry(message))
        for call in message.tool_calls:
            result = agent.call_tool(call.function.name, call.function.arguments)
            failed = result.startswith("error:")
            if call.function.name == "browser_click" and result.startswith(SUBMITTED_MARKER):
                outcome["submitted"] = True
            if call.function.name in USER_TOOLS or (
                call.function.name == "browser_click" and result.startswith(USER_ANSWER_PREFIXES)
            ):
                outcome["asked_user"] = True
            tool_log.append(f"Tool: {call.function.name}, Args: {call.function.arguments}, Result: {result[:200]}")

            if agent.log_level == "debug":
                ui.end_status()
                print(f"  [tool] {call.function.name}({call.function.arguments})")
                print(f"         -> {result[:RESULT_PREVIEW_CHARS]}")
            elif agent.log_level == "info":
                ui.status(call.function.name, failed)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            sources.append(result)

    # out of steps: make it answer with what it has
    outcome.update(out_of_steps=True, tool_log="\n".join(tool_log))
    messages.append({"role": "user", "content": OUT_OF_STEPS_NUDGE})
    message = _ask(agent, client, messages)
    messages.append(_assistant_entry(message))
    if not message.content:
        return f"(stopped after {max_steps} steps without a final answer)", outcome
    return _finish(agent, client, messages, message.content, sources), outcome


def _client(model: dict) -> OpenAI:
    """One API client per server + timeout (each agent's x-model may differ)."""
    base_url = os.getenv(model.get("base_url_env", ""), "") or model["base_url"]
    key = (base_url, model.get("timeout_seconds", 120))
    if key not in _clients:
        _clients[key] = OpenAI(base_url=base_url, api_key=API_KEY, timeout=key[1], max_retries=0)
    return _clients[key]


def _route(assistant, history: list, user_input: str):
    """Ask the router model which agent should handle this message (thinking off: ~0.2s)."""
    if assistant.router is None:
        return next(iter(assistant.routes.values()))
    router = assistant.router
    options = "\n".join(f"- {r.id}: {r.description}" for r in assistant.routes.values())
    messages = [{"role": "system", "content": f"{router.instructions}\n\nAssistants:\n{options}"}]
    messages += history[-ROUTER_HISTORY_MESSAGES:]  # recent turns, so follow-ups go to the right place
    messages.append({"role": "user", "content": user_input})
    response = _client(router.model).chat.completions.create(
        model=router.model["name"], temperature=router.model.get("temperature", 0),
        messages=messages, reasoning_effort="none", max_tokens=ROUTER_MAX_TOKENS,
    )
    answer = (response.choices[0].message.content or "").lower()
    intent = next((name for name in assistant.routes if name in answer), "")
    return assistant.pick(intent, user_input)


def main():
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("AGENT_MANIFEST", DEFAULT_MANIFEST)
    try:
        assistant = load_assistant(manifest_path)
    except ManifestError as e:
        sys.exit(f"error: {e}")

    today = date.today().isoformat()
    memory_index = load_memory_index()
    systems = {r.id: f"{r.agent.instructions}\n\nToday's date: {today}\n\nMemory Index:\n{memory_index}" for r in assistant.routes.values()}
    first = next(iter(assistant.routes.values())).agent
    print(f"{assistant.name} ({', '.join(assistant.routes) if assistant.router else first.name}) "
          f"| {first.model['name']} @ {_client(first.model).base_url}  (exit to quit)")

    history = []  # shared by all agents: user messages and final answers only, no tool traffic
    while True:
        try:
            user_input = input("\nuser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input in ("", "exit", "quit"):
            break
        try:
            route = _route(assistant, history, user_input)
            ui.new_turn(route.id if assistant.router else None)
            messages = [{"role": "system", "content": systems[route.id]}, *history, {"role": "user", "content": user_input}]
            answer = run_agent(route.agent, _client(route.agent.model), messages, user_input).strip()
        except APIError as e:
            ui.say(f"(error talking to the model: {e})")
            continue  # nothing is added to the history, so the conversation stays valid
        ui.say(answer)
        history += [{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}]

        # Auto memory extraction (if enabled for this agent)
        if route.agent.memory.get("auto_extract", False):
            try:
                client = _client(route.agent.model)
                actions_summary = "\n".join([m.get("content", "")[:200] for m in messages if m.get("role") == "tool"])
                learnings = extract_learnings(user_input, actions_summary, answer, client, route.agent.model)
                if learnings:
                    save_learnings(learnings, route.agent.memory.get("max_learnings_per_turn", 3))
            except Exception as e:
                print(f"[auto_memory] Error: {e}")


if __name__ == "__main__":
    main()
