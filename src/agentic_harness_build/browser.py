"""A real browser the agent can drive to book things: open, auto-fill, click.

It runs hidden. When the user has to step in (captcha, login), browser_show reopens the
current page in a visible window; cookies are kept, so the session carries over. Every page is
returned to the model as text plus a numbered list of form fields and buttons ("refs").
Clicking anything that looks like a final submit (book, pay, confirm...) asks the user first.
"""

import os
import re
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from . import payment, ui
from .profile import REQUIRED_FOR_BOOKING, confirmed_details, missing_for_booking

BROWSER_DATA = Path(__file__).resolve().parents[2] / "data" / "browser"  # cookies/logins persist here
HEADLESS = os.getenv("BROWSER_HEADLESS", "1") != "0"  # hidden unless BROWSER_HEADLESS=0
PAGE_TEXT_CHARS = 3000
MAX_ELEMENTS = 80
MAX_LINKS = 25
MAX_CLICKABLES = 50
SETTLE_INTERVAL_MS = 400
AFTER_CLICK_MS = 1500
SETTLE_CHECKS = 10  # at most 4 seconds  # JS-clickable widgets: time slots, pickers, calendar days
ACTION_TIMEOUT_MS = 8000

BLOCKED_STATUSES = {401, 403, 429}
BLOCK_PAGE = re.compile(r"request could not be satisfied|access denied|attention required|just a moment|are you a robot", re.I)

# button text that means "this sends the booking or the payment" (English + Hebrew)
SUBMIT_WORDS = re.compile(
    r"\b(book|reserve|confirm|complete|pay|purchase|order|checkout|place|submit|finish)\b|הזמ|אישור|אשר|לתשלום|שלם|סיום",
    re.I,
)

TERMS = re.compile(r"agree|terms|consent|מסכים|מסכימה|תנאי|תקנון|הסכמה", re.I)
TIME_SLOT = re.compile(r"\d{1,2}:\d{2}\b")
NOT_SUBMIT_PHRASES = re.compile(r"אישור מיידי|instant confirmation|confirmed instantly", re.I)

# field hints -> value key, checked in order (card first, so "name on card" isn't the profile name)
FIELD_RULES = [
    (r"cc-number|card.?num|cardnumber|מספר.?כרטיס", "card.number"),
    (r"cc-exp-month|exp.?month|expiry.?month", "card.exp_month"),
    (r"cc-exp-year|exp.?year|expiry.?year", "card.exp_year"),
    (r"cc-exp|expir|mm.?/.?yy|valid.?thru|תוקף", "card.expiry"),
    (r"cc-csc|cvc|cvv|security.?code|csc|קוד.?אבטחה", "card.cvc"),
    (r"cc-name|name.?on.?card|card.?holder|שם.?בעל", "card.name"),
    (r"given-name|first.?name|fname|שם.?פרטי", "profile.first_name"),
    (r"family-name|last.?name|lname|surname|שם.?משפחה", "profile.last_name"),
    (r"e-?mail|דוא\"?ל|מייל", "profile.email"),
    (r"(?<![a-z])tel(?![a-z])|phone|mobile|cell|טלפון|נייד", "profile.phone"),
    (r"postal|zip|מיקוד", "profile.postal_code"),
    (r"address-level2|(?<![a-z])city(?![a-z])|town|עיר|ישוב", "profile.city"),
    (r"street|address|כתובת|רחוב", "profile.address"),
    (r"(?<![a-z])name(?![a-z])|full.?name|שם", "profile.name"),
]
FILLABLE_TYPES = {"", "text", "email", "tel", "number", "search", "url"}  # never passwords

SNAPSHOT_JS = """
([maxElements, maxLinks, maxClickables]) => {
  const visible = el => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return el.type !== 'hidden' && s.display !== 'none' && s.visibility !== 'hidden' && (r.width > 0 || r.height > 0);
  };
  const labelOf = el => {
    let l = el.id && document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
    l = l || el.closest('label');
    return ((l && l.innerText) || el.getAttribute('aria-label') || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
  };
  const controls = [...document.querySelectorAll('input, select, textarea, button, [role=button], input[type=submit]')];
  const allLinks = [...document.querySelectorAll('a[href]')];
  const links = allLinks.filter(visible).slice(0, maxLinks);
  // booking links often sit in closed menus: list them even when hidden (open them with browser_open)
  const bookingRe = /reserv|\/book|booking|book.?a.?table|book.?now|tabit|ontopo|opentable|resy\.com|thefork|sevenrooms|tock|quandoo|eatapp|הזמנ|שולחן|מקום/i;
  const hiddenBooking = allLinks.filter(a => !visible(a) && !/facebook/i.test(a.href) && bookingRe.test((a.innerText || a.textContent || '') + ' ' + a.href)).slice(0, 15);
  // widgets (time slots, pickers, calendars) are often plain divs made clickable with JS:
  // take the outermost element with a pointer cursor (cursor is inherited, so its parent has none)
  const pointer = el => getComputedStyle(el).cursor === 'pointer';
  const clickRoles = ['option', 'menuitem', 'radio', 'tab', 'gridcell', 'checkbox', 'switch'];
  const clickables = [...document.querySelectorAll('div, span, li, td, label, p, img')].filter(el => {
    if (!visible(el) || el.closest('button, a, [role=button], select, label input')) return false;
    const text = (el.innerText || el.getAttribute('aria-label') || '').trim();
    if (!text || text.length > 60) return false;
    const role = el.getAttribute('role');
    if (clickRoles.includes(role) || el.hasAttribute('onclick')) return true;
    return pointer(el) && !(el.parentElement && pointer(el.parentElement));
  }).slice(0, maxClickables);
  const out = [];
  for (const el of [...controls.filter(visible), ...clickables, ...links, ...hiddenBooking]) {
    if (out.length >= maxElements) break;
    if (!el.dataset.agentRef) el.dataset.agentRef = 'e' + (window.__agentRefs = (window.__agentRefs || 0) + 1);
    const tag = el.tagName.toLowerCase();
    out.push({
      ref: el.dataset.agentRef, tag, type: (el.type || '').toLowerCase(),
      name: el.name || '', id: el.id || '', label: labelOf(el),
      placeholder: el.placeholder || '', autocomplete: el.getAttribute('autocomplete') || '',
      clickable: !['input', 'select', 'textarea', 'button', 'a'].includes(tag) && el.getAttribute('role') !== 'button',
      text: (tag === 'input' ? '' : (el.innerText || el.getAttribute('aria-label') || el.textContent || '')).trim().replace(/\\s+/g, ' ').slice(0, 80),
      value: tag === 'select' ? (el.selectedOptions[0]?.text || '') : (el.value || ''),
      checked: !!el.checked, href: tag === 'a' ? el.href.slice(0, 200) : '', hidden: !visible(el),
      options: tag === 'select' ? [...el.options].slice(0, 20).map(o => o.text.trim()) : [],
      required: !!el.required || el.getAttribute('aria-required') === 'true', in_form: !!el.closest('form'),
    });
  }
  return out;
}
"""


class _Session:
    def __init__(self):
        self.playwright = None
        self.context = None
        self.page = None
        self.elements = {}  # public ref -> element info (with its frame)
        self.filled = {}  # public ref -> what was filled, for the confirmation summary
        self.headless = HEADLESS

    def start(self):
        if self.page is None or self.page.is_closed():
            BROWSER_DATA.mkdir(parents=True, exist_ok=True)
            self.playwright = self.playwright or sync_playwright().start()
            self.context = self.playwright.chromium.launch_persistent_context(
                str(BROWSER_DATA), headless=self.headless, viewport={"width": 1280, "height": 900}
            )
            self.context.set_default_timeout(ACTION_TIMEOUT_MS)
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        return self.page

    def close(self):
        if self.context:
            self.context.close()
        if self.playwright:
            self.playwright.stop()
        self.__init__()


_session = _Session()


def current_url() -> str | None:
    page = _session.page
    return page.url if page and not page.is_closed() else None


# --- reading the page ----------------------------------------------------------


def _describe(info: dict) -> str:
    kind = info["tag"] if info["tag"] != "input" else f"input {info['type'] or 'text'}"
    if info.get("clickable"):
        kind = "clickable"  # a time slot, picker option or calendar day: use browser_click
    title = info["label"] or info["text"] or info["placeholder"] or info["name"] or info["id"]
    details = [f"{k}={info[k]}" for k in ("name", "autocomplete") if info[k] and info[k] != title]
    line = f"[{info['ref']}] {kind} \"{title}\""
    if details:
        line += f" ({', '.join(details)})"
    if info["tag"] in ("input", "textarea", "select") and info["type"] not in ("submit", "button"):
        if info["type"] in ("checkbox", "radio"):
            line += " checked" if info["checked"] else " unchecked"
        else:
            line += f' value="{payment.mask(info["value"], include_short=True)}"'
    if info.get("required"):
        line += " REQUIRED"
    if info["options"]:
        line += " options: " + ", ".join(info["options"])
    if info["href"]:
        line += f" -> {info['href']}" + (" (inside a closed menu: open it with browser_open)" if info.get("hidden") else "")
    return line


def _settle(page):
    """Wait until the page stops changing: widgets draw slots and pickers a moment after a click."""
    last = -1
    for _ in range(SETTLE_CHECKS):
        try:
            size = page.evaluate("() => document.body ? document.body.innerText.length : 0")
        except PlaywrightError:
            return
        if size == last:
            return
        last = size
        page.wait_for_timeout(SETTLE_INTERVAL_MS)


def _snapshot(note: str = "") -> str:
    page = _session.page
    try:
        page.wait_for_load_state("domcontentloaded", timeout=ACTION_TIMEOUT_MS)
    except PlaywrightError:
        pass
    _settle(page)
    _session.elements = {}
    lines = []
    for index, frame in enumerate(page.frames):  # booking widgets often live in iframes
        try:
            found = frame.evaluate(SNAPSHOT_JS, [MAX_ELEMENTS, MAX_LINKS if index == 0 else 0, MAX_CLICKABLES])
        except PlaywrightError:
            continue  # frame detached or cross-origin and not ready
        for info in found:
            public_ref = info["ref"] if index == 0 else f"f{index}{info['ref']}"
            info = {**info, "ref": public_ref, "dom_ref": info["ref"], "frame": frame}
            _session.elements[public_ref] = info
            lines.append(_describe(info))
        if len(lines) >= MAX_ELEMENTS:
            break

    try:
        text = page.inner_text("body", timeout=ACTION_TIMEOUT_MS)
    except PlaywrightError:
        text = ""
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) > PAGE_TEXT_CHARS:
        text = text[:PAGE_TEXT_CHARS] + "\n(page text truncated)"
    parts = [note] if note else []
    parts += [
        f"Page: {page.title()}\nURL: {page.url}",
        f"Text:\n{text}",
        "Elements (use the [ref] with auto_fill or browser_click):\n" + ("\n".join(lines) or "(none found)"),
    ]
    return payment.mask("\n\n".join(parts))  # title, URL and text can all echo the card


def _element(ref: str) -> tuple[dict, object]:
    info = _session.elements.get(ref)
    if info is None:
        raise ValueError(f"no element [{ref}] on the current page; call browser_view to see the current refs")
    return info, info["frame"].locator(f'[data-agent-ref="{info["dom_ref"]}"]').first


# --- tools -----------------------------------------------------------------------


def browser_open(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "error: url must start with https://"
    page = _session.start()
    response = page.goto(url, wait_until="domcontentloaded")
    _session.filled = {}
    if (response and response.status in BLOCKED_STATUSES) or BLOCK_PAGE.search(page.title()):
        return (
            f"error: this booking site refuses automated browsers (HTTP {response.status if response else '?'}). "
            "Online booking DOES exist, the user just has to open it themselves. Stop using tools now and reply: "
            f"'<venue> takes online bookings here: {url} (it blocks automated booking). Or call <phone from a page you read>.'"
        )
    return _snapshot()


def browser_view() -> str:
    if not current_url():
        return "error: no page open; use browser_open first"
    return _snapshot()


def _field_key(info: dict) -> str | None:
    """What a form field asks for, e.g. 'profile.phone' or 'card.number', from its attributes and label."""
    hints = " ".join(info[k] for k in ("autocomplete", "name", "id", "label", "placeholder")).lower()
    return next((k for pattern, k in FIELD_RULES if re.search(pattern, hints, re.I)), None)


def _value_for(key: str, details: dict) -> str | None:
    source, _, field = key.partition(".")
    if source == "card":
        return payment.get(field) if payment.allowed_on(current_url() or "") else None
    return details.get(field)  # only what the user gave or confirmed this session


def _fill(info: dict, locator, value: str):
    if info["tag"] == "select":
        try:
            locator.select_option(label=value)
        except PlaywrightError:
            locator.select_option(value=value)
    elif info["type"] in ("checkbox", "radio"):
        locator.check() if value.lower() in ("true", "yes", "on", "1", "x") else locator.uncheck()
    else:
        locator.fill(value)


def auto_fill(fields: list | None = None) -> str:
    if not current_url():
        return "error: no page open; use browser_open first"
    details = confirmed_details()
    page_url = current_url()
    explicit = {f["ref"]: str(f["value"]) for f in (fields or []) if isinstance(f, dict) and "ref" in f and "value" in f}
    report, missing, rejected = [], [], []

    for ref, value in explicit.items():  # values the model chose, e.g. date, time, party size
        info, locator = _element(ref)
        shown = value
        real = payment.fill_placeholders(value, page_url)
        real = re.sub(r"\{\{profile\.(\w+)\}\}", lambda m: details.get(m[1], ""), real)
        key = _field_key(info)
        if key and key.startswith("profile.") and " ".join(real.split()) != details.get(key.split(".")[1]):
            # contact details must be the user's own, never typed in by the model
            rejected.append(f"[{ref}] needs the user's {key.split('.')[1]}: ask the user with ask_user, then auto_fill fills it")
            continue
        _fill(info, locator, real)
        _session.filled[ref] = (info, shown)
        report.append(f"[{ref}] {shown}")

    for ref, info in _session.elements.items():  # everything else: match the field to saved data
        if ref in explicit or info["tag"] not in ("input", "textarea", "select"):
            continue
        if (info["tag"] == "input" and info["type"] not in FILLABLE_TYPES) or info["value"]:
            continue
        key = _field_key(info)
        if key is None:
            continue
        value = _value_for(key, details)
        if not value:
            if key.startswith("card."):
                why = "the card is approved for another site" if payment.has_card() else "needs the card (request_payment)"
            else:
                why = f"needs the user's {key.split('.')[1]} (ask_user)"
            missing.append(f"[{ref}] \"{info['label'] or info['name'] or info['placeholder']}\" {why}")
            continue
        _, locator = _element(ref)
        try:
            _fill(info, locator, value)
        except PlaywrightError as e:
            missing.append(f"[{ref}] could not be filled ({str(e).splitlines()[0]})")
            continue
        shown = f"{{{{{key}}}}}" if key.startswith("card.") else f"{value} ({{{{{key}}}}})"
        _session.filled[ref] = (info, shown)
        report.append(f"[{ref}] {shown}")

    note = "Filled:\n" + ("\n".join(report) or "(nothing)")
    if rejected:
        note += "\n\nRejected (contact details must come from the user):\n" + "\n".join(rejected)
    if missing:
        note += "\n\nNot filled yet:\n" + "\n".join(missing)
    return _snapshot(note)


def _not_ready_to_submit(final: bool) -> str | None:
    """Refuse to send a form while the user's details or the site's required fields are missing."""
    _snapshot()  # fresh values after any filling
    fields = [i for i in _session.elements.values() if i["tag"] in ("input", "textarea", "select")]
    has_contact_fields = any((_field_key(i) or "").startswith("profile.") for i in fields)
    if final or has_contact_fields:
        missing = missing_for_booking()
        if missing:
            return (
                f"error: cannot submit yet: the user has not given or confirmed their {', '.join(missing)} in this session. "
                f"Call ask_user with fields {list(REQUIRED_FOR_BOOKING)} plus anything else the form needs, then auto_fill."
            )
    empty = []
    for i in fields:
        if not i.get("required") or i["type"] in ("submit", "button", "hidden"):
            continue
        if (i["type"] in ("checkbox", "radio") and not i["checked"]) or (i["type"] not in ("checkbox", "radio") and not i["value"].strip()):
            empty.append(f"[{i['ref']}] \"{i['label'] or i['name'] or i['placeholder']}\"")
    if empty:
        return (
            "error: cannot submit yet: required fields are empty: " + ", ".join(empty)
            + ". Ask the user for these with ask_user (or request_payment for card fields), then auto_fill."
        )
    return None


def browser_click(ref: str) -> str:
    if not current_url():
        return "error: no page open; use browser_open first"
    info, locator = _element(ref)
    title = info["text"] or info["label"] or info["value"] or info["name"]
    # links only navigate ("Book a table" opens the form); buttons are what send data
    details_entered = any("{{profile." in shown for _, shown in _session.filled.values())
    # picking a time slot isn't submitting, even when the tile says "instant confirmation" (אישור מיידי)
    is_slot = bool(TIME_SLOT.match(title.strip()))
    label = NOT_SUBMIT_PHRASES.sub("", f"{title} {info['name']} {info['id']}")
    is_submit = info["tag"] != "a" and not is_slot and (
        bool(SUBMIT_WORDS.search(label))
        or details_entered  # sites label the final button "Continue" too; once details are in, always ask
        or (payment.has_card() and info["type"] == "submit")
    )

    sends_form = info["tag"] in ("button", "input") and info["type"] in ("", "submit") and info.get("in_form")
    if is_submit or sends_form:
        problem = _not_ready_to_submit(final=is_submit)
        if problem:
            return problem
        info, locator = _element(ref)  # the check re-read the page

    if is_submit:
        filled = "\n".join(
            f"{i['label'] or i['name'] or i['placeholder'] or ref}: {payment.mask(shown, include_short=True)}"
            for ref, (i, shown) in _session.filled.items()
        )
        ui.show(
            f"The agent wants to click \"{title}\" on {current_url()}\n"
            + ("Filled in:\n" + filled if filled else "No fields were filled by the agent on this page.")
            + ("\nYour card will be used." if payment.has_card() else "")
        )
        if not ui.confirm("Submit?", strict=True):
            return "the user declined this submission; ask them what to change, or stop"

    if TERMS.search(title):  # accepting terms is the user's legal consent, not the agent's
        ui.show(f"The site asks you to accept: \"{title}\" ({current_url()})")
        if not ui.confirm("Accept?"):
            return "the user did not accept the terms; stop and tell them the booking needs it"

    old_url = current_url()
    locator.click()
    page = _session.page
    page.wait_for_timeout(AFTER_CLICK_MS)  # many widgets navigate (e.g. to checkout) a moment after the click
    try:
        page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
    except PlaywrightError:
        pass  # pages that keep polling never go idle; the snapshot below is still useful
    if is_submit and payment.has_card():
        payment.wipe()  # card used once; a new payment needs a new request_payment
    if current_url() != old_url:
        _session.filled = {}
    return _snapshot(f"Clicked [{ref}] \"{title}\".")


def browser_show() -> str:
    if not _session.headless:
        return "the browser window is already visible to the user"
    url = current_url()
    if _session.context:
        _session.context.close()  # same profile folder, so cookies and logins carry over
    _session.context, _session.page, _session.headless = None, None, False
    page = _session.start()
    if url:
        page.goto(url, wait_until="domcontentloaded")
    _session.filled = {}
    return _snapshot("The browser window is now visible to the user. Fields filled before were reset; fill them again if needed.")


def browser_close() -> str:
    _session.close()
    payment.wipe(forget_masks=True)
    return "browser closed and card details wiped from memory"
