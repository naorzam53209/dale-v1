"""Card details for bookings, kept away from the model.

The user types the card into hidden prompts (or reuses one saved in the macOS Keychain).
The model only ever sees placeholders like {{card.number}}; the browser tools swap in the
real values at the moment they fill a form, and only on the site the user approved.
The CVC is never saved, and the whole card is wiped from memory after a submission.
"""

import json
import re
from urllib.parse import urlparse

import keyring

from . import ui

KEYCHAIN_SERVICE = "dale-agent"
KEYCHAIN_ACCOUNT = "card"
CARD_KEYS = ("number", "expiry", "exp_month", "exp_year", "cvc", "name")
PLACEHOLDER = re.compile(r"\{\{card\.(\w+)\}\}")

_card: dict | None = None  # the card for the current booking, in memory only
_domain: str | None = None  # the only site it may be used on
_masked: dict = {}  # card values to hide from the model; outlives the card until the browser closes,
#                     because confirmation pages often echo the card after it was used


def _base_domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return 12 <= len(number) <= 19 and total % 10 == 0


def _brand(number: str) -> str:
    if number.startswith("4"):
        return "Visa"
    if number[:2] in ("34", "37"):
        return "Amex"
    if 51 <= int(number[:2]) <= 55 or 2221 <= int(number[:4]) <= 2720:
        return "Mastercard"
    return "Card"


def describe(card: dict) -> str:
    return f"{_brand(card['number'])} ending {card['number'][-4:]}"


# --- keychain --------------------------------------------------------------------


def _load_saved() -> dict | None:
    try:
        saved = keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
    except keyring.errors.KeyringError:
        return None
    return json.loads(saved) if saved else None


def _save(card: dict):
    without_cvc = {k: card[k] for k in ("number", "exp_month", "exp_year", "name")}
    keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT, json.dumps(without_cvc))


def forget_saved_card() -> str:
    try:
        keyring.delete_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
        return "saved card removed from the Keychain"
    except keyring.errors.PasswordDeleteError:
        return "no saved card"


# --- entering a card ---------------------------------------------------------------


def _enter_card() -> dict | None:
    for _ in range(3):
        number = re.sub(r"[\s-]", "", ui.ask_secret("Card number"))
        if number.isdigit() and _luhn_ok(number):
            break
        print("  │  That card number doesn't look right, try again.")
    else:
        return None
    for _ in range(3):
        expiry = ui.ask("Expiry (MM/YY)")
        match = re.fullmatch(r"(0[1-9]|1[0-2])\s*/\s*(\d{2})", expiry)
        if match:
            break
        print("  │  Use the format MM/YY, e.g. 07/28.")
    else:
        return None
    name = ui.ask("Name on card")
    return {"number": number, "exp_month": match[1], "exp_year": match[2], "name": name}


def request_payment(reason: str) -> str:
    global _card, _domain, _masked
    from .browser import current_url  # imported here: browser.py imports this module

    url = current_url()
    if not url:
        return "error: open the booking page with browser_open first, so the card can be tied to that site"
    domain = _base_domain(url)

    ui.show(f"The agent needs a payment card.\nReason: {reason}\nSite:   {domain}\n(The agent never sees the card number.)")
    if not ui.confirm(f"Allow a card to be used on {domain}?"):
        return "user declined to provide card details; do not ask again unless the user brings it up"

    saved = _load_saved()
    card = saved if saved and ui.confirm(f"Use your saved {describe(saved)}?") else None
    if card is None:
        card = _enter_card()
        if card is None:
            return "error: the user could not enter a valid card; tell them and stop the payment step"
        if ui.confirm("Save this card in your macOS Keychain for next time? (the CVC is never saved)"):
            _save(card)

    cvc = ""
    for _ in range(3):
        cvc = ui.ask_secret("CVC")
        if re.fullmatch(r"\d{3,4}", cvc):
            break
    else:
        return "error: no valid CVC entered; tell the user and stop the payment step"

    _card = {**card, "cvc": cvc, "expiry": f"{card['exp_month']}/{card['exp_year']}"}
    _domain = domain
    _masked = {k: _card[k] for k in ("number", "cvc", "expiry")}
    placeholders = ", ".join(f"{{{{card.{k}}}}}" for k in CARD_KEYS)
    return (
        f"card ready: {describe(_card)}, usable on {domain} only. "
        f"auto_fill fills card fields automatically; for fields it misses use these placeholders as values: {placeholders}. "
        "You will never see the real numbers."
    )


# --- used by the browser tools -------------------------------------------------------


def has_card() -> bool:
    return _card is not None


def allowed_on(url: str) -> bool:
    domain = _base_domain(url)
    return _domain is not None and (domain == _domain or domain.endswith("." + _domain))


def get(key: str) -> str | None:
    return _card.get(key) if _card else None


def fill_placeholders(value: str, page_url: str) -> str:
    """Swap {{card.x}} for the real value. Raises if the card can't be used here."""
    if not PLACEHOLDER.search(value):
        return value
    if not has_card():
        raise ValueError("no card yet; call request_payment first")
    if not allowed_on(page_url):
        raise ValueError(f"the card is approved for {_domain} only, not {_base_domain(page_url)}")

    def swap(match):
        if match[1] not in CARD_KEYS:
            raise ValueError(f"unknown placeholder {match[0]}")
        return _card[match[1]]

    return PLACEHOLDER.sub(swap, value)


def mask(text: str, include_short: bool = False) -> str:
    """Replace any card value in text the model will see with its placeholder."""
    if not _masked:
        return text
    spaced = re.compile(r"[\s-]?".join(re.escape(d) for d in _masked["number"]))  # also "4242 4242 ..."
    text = spaced.sub("{{card.number}}", text)
    if include_short:  # CVC and expiry are short, so only mask them inside form values, not page text
        text = re.sub(rf"(?<!\d){re.escape(_masked['cvc'])}(?!\d)", "{{card.cvc}}", text)
        text = text.replace(_masked["expiry"], "{{card.expiry}}")
    return text


def wipe(forget_masks: bool = False):
    """Make the card unusable. With forget_masks, also drop the values kept for masking."""
    global _card, _domain, _masked
    _card, _domain = None, None
    if forget_masks:
        _masked = {}
