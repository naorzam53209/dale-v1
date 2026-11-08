"""The user's saved details (name, phone, address...) and the ask_user tool.

Answers to known fields are remembered in data/profile.json, so next time they are
pre-filled and auto_fill can put them into web forms without asking again.
"""

import json
import re
from pathlib import Path

from . import ui

PROFILE_PATH = Path(__file__).resolve().parents[2] / "data" / "profile.json"

# fields worth remembering between sessions, with the label shown to the user
PROFILE_FIELDS = {
    "name": "Full name",
    "first_name": "First name",
    "last_name": "Last name",
    "phone": "Phone",
    "email": "Email",
    "address": "Street address",
    "city": "City",
    "postal_code": "Postal code",
}
# a booking can only be submitted once the user has given or confirmed these in this session
REQUIRED_FOR_BOOKING = ("name", "phone", "address", "city")
_confirmed: dict[str, str] = {}  # details the user typed or confirmed (Enter on a saved value) this session

CARD_WORDS = re.compile(r"card|credit|cvc|cvv|expir|payment|iban|bank", re.I)


def load_profile() -> dict:
    if PROFILE_PATH.is_file():
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return {}


def save_profile(profile: dict):
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


def _key(field: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")


def ask_user(question: str, fields: list | None = None) -> str:
    if fields and any(CARD_WORDS.search(str(f)) for f in fields):
        return "error: never collect card or bank details with ask_user; use request_payment instead"

    ui.show(f"The agent asks: {question}")
    if not fields:
        answer = ui.ask("Your answer (Enter when done)")
        return f"user answered: {answer or '(pressed Enter)'}"

    profile = load_profile()
    answers = {}
    for field in fields:
        key = _key(str(field))
        label = PROFILE_FIELDS.get(key, str(field))
        answers[key] = ui.ask(label, default=profile.get(key))

    remembered = {k: v for k, v in answers.items() if k in PROFILE_FIELDS and v}
    _confirmed.update(remembered)
    if remembered and remembered != {k: profile.get(k) for k in remembered}:
        profile.update(remembered)
        save_profile(profile)
    return "user answered:\n" + "\n".join(f"{k}: {v or '(left empty)'}" for k, v in answers.items())


def confirmed_details() -> dict:
    """The user's details as given or confirmed in this session; saved values alone don't count."""
    details = dict(_confirmed)
    if details.get("name"):  # many forms split the name
        first, _, last = details["name"].partition(" ")
        details.setdefault("first_name", first)
        if last:
            details.setdefault("last_name", last)
    return details


def missing_for_booking() -> list[str]:
    return [k for k in REQUIRED_FOR_BOOKING if not _confirmed.get(k)]
