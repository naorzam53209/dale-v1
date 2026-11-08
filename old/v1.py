



import json
import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama")
MODEL = "qwen3.5:4b"

WORKSPACE = Path(__file__).parent / "workspace"


# --- the actual tools: plain Python functions ---------------------------

def list_files() -> str:
    """List the files in the assistant's workspace folder."""
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    """Read a file from the workspace folder."""
    path = (WORKSPACE / filename).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        return f"error: {filename} is outside the workspace"
    if not path.is_file():
        return f"error: no file named {filename}"
    return path.read_text(encoding="utf-8")


TOOLS = {"list_files": list_files, "read_file": read_file}

# --- what the model sees: a JSON description of each tool -------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the user's workspace folder.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
]


MAX_STEPS = 8  # stop a confused model from looping forever


def run_tool(name: str, raw_args: str) -> str:
    """Run one tool call. Any mistake goes back to the model as text, not a crash."""
    if name not in TOOLS:
        return f"error: unknown tool {name!r}. Available: {', '.join(TOOLS)}"
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        return f"error: arguments were not valid JSON ({e})"
    try:
        return TOOLS[name](**args)
    except TypeError as e:
        return f"error: bad arguments for {name}: {e}"


def chat(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful personal assistant."},
        {"role": "user", "content": user_message},
    ]

    # The agent loop: keep offering tools until the model answers in plain text.
    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        messages.append(message)
        for call in message.tool_calls:
            print(f"  [tool] {call.function.name}({call.function.arguments})")
            result = run_tool(call.function.name, call.function.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    return f"(stopped after {MAX_STEPS} tool steps without a final answer)"


if __name__ == "__main__":
    print(f"v1 assistant ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant:", chat(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")