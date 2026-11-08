"""Terminal output and prompts: the compact tool status line, questions, confirmations, secrets.

A turn looks like:
    user: book me a table at Golden Dragon tonight
    agent: sure [web_search, fetch_page, browser_open, auto_fill]
    agent: Booked for 19:30, 2 people. Confirmation #GD-7731.
"""

from getpass import getpass

_status_open = False  # is an "agent: sure [..." line currently being written?
_status_count = 0  # tool names on the open line
_said_sure = False  # "sure" only on the first tool line of a turn
_label = None  # which agent is working this turn, shown as "agent: sure [booking: ..."


def new_turn(label: str | None = None):
    global _said_sure, _label
    _said_sure, _label = False, label


def status(tool_name: str, failed: bool = False):
    """Add one tool to the status line, opening it if needed."""
    global _status_open, _status_count, _said_sure
    if not _status_open:
        prefix = "agent: sure [" if not _said_sure else "agent: ["
        print(prefix + (f"{_label}: " if _label else ""), end="", flush=True)
        _status_open, _status_count, _said_sure = True, 0, True
    print(("" if _status_count == 0 else ", ") + tool_name + (" ✗" if failed else ""), end="", flush=True)
    _status_count += 1


def end_status():
    """Close the status line (before printing anything else)."""
    global _status_open
    if _status_open:
        print("]", flush=True)
        _status_open = False


def say(text: str):
    """A line from the agent."""
    end_status()
    print(f"agent: {text}")


def show(text: str):
    """A question or notice from a tool, shown before the user is asked something."""
    say(text.replace("\n", "\n       "))


def ask(prompt: str, default: str | None = None) -> str:
    """Ask for a value. With a default, pressing Enter keeps it."""
    end_status()
    suffix = f" [{default}]" if default else ""
    answer = input(f"  {prompt}{suffix}: ").strip()
    return answer or (default or "")


def ask_secret(prompt: str) -> str:
    """Ask without echoing what is typed."""
    end_status()
    return getpass(f"  {prompt} (hidden): ").strip()


def confirm(prompt: str, strict: bool = False) -> bool:
    """Yes/no question. With strict=True only the full word 'yes' counts."""
    end_status()
    answer = input(f"  {prompt} [{'type yes' if strict else 'y/N'}]: ").strip().lower()
    return answer == "yes" if strict else answer in ("y", "yes")
