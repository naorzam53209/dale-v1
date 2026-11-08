import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urljoin

from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from openpyxl import Workbook
from openpyxl.styles import Font

# --- sandbox: every tool works only inside this folder ------------------------

WORKSPACE = (Path(__file__).resolve().parents[2] / "workspace").resolve()

SEARCH_TIMEOUT = 20
SNIPPET_CHARS = 600  # per search result; Tavily extracts can be long
MAX_PAGE_CHARS = 8000  # cap fetched pages so they don't flood a small model's context
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DaleResearchAgent/1.0; personal research assistant)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
_failed_urls: dict[str, str] = {}  # url -> why it failed, for this session
# reservation platforms and words that mean "book" (English + Hebrew); facebook is not a booking site
BOOKING_LINK = re.compile(
    r"(?!.*facebook)(reserv|/book|booking|book.?a.?table|book.?now|tabit|ontopo|opentable|resy\.com|thefork|"
    r"sevenrooms|tock|quandoo|eatapp|הזמנ|שולחן|מקום)",
    re.I,
)
# food orders, delivery and gift cards look like booking links but are not table reservations
ORDER_LINK = re.compile(r"tabit-order|/order|delivery|takeaway|pickup|wolt\.com|10bis|gift|משלוח|איסוף|גיפט", re.I)
MAX_BOOKING_LINKS = 15
MAX_OTHER_LINKS = 15
MAX_LINK_CHARS = 200
URL_SAFE_CHARS = ":/?#[]@!$&'()*+,;=%~-._"
BLOCK_TAGS = ["p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "table", "dt", "dd", "pre", "blockquote"]


def _safe_path(path: str) -> Path:
    """Resolve a path relative to the workspace and refuse anything outside it."""
    full = (WORKSPACE / path).resolve()
    if not full.is_relative_to(WORKSPACE):
        raise ValueError(f"path '{path}' is outside the workspace")
    return full


# --- tool functions: plain Python, each returns a string for the model --------
# Their names, descriptions and JSON schemas live in the agent manifest (agents/*.agents.json).


def list_files(subdir: str = "") -> str:
    folder = _safe_path(subdir)
    if not folder.is_dir():
        return f"error: '{subdir}' is not a folder"
    names = sorted(
        str(p.relative_to(WORKSPACE)) + ("/" if p.is_dir() else "")
        for p in folder.iterdir()
        if not p.name.startswith(".")
    )
    return "\n".join(names) or "(empty)"


def read_file(path: str) -> str:
    full = _safe_path(path)
    if not full.is_file():
        return f"error: file '{path}' not found"
    return full.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    full = _safe_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {path}"


def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient  # imported here so the key is read at call time

    response = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(
        query, max_results=max_results, search_depth="basic", timeout=SEARCH_TIMEOUT
    )
    return [{"title": r["title"], "url": r["url"], "text": r.get("content", "")} for r in response.get("results", [])]


def _ddg_search(query: str, max_results: int) -> list[dict]:
    try:
        results = DDGS().text(query, max_results=max_results)
    except DDGSException:  # raised for "No results found." and rate limits alike
        return []
    return [{"title": r["title"], "url": r["href"], "text": r["body"]} for r in results]


def web_search(query: str, max_results: int = 5) -> str:
    max_results = max(1, min(int(max_results), 10))
    results = []
    if os.getenv("TAVILY_API_KEY"):  # Tavily: built for agents, returns relevant page extracts
        try:
            results = _tavily_search(query, max_results)
        except Exception:
            results = []  # bad key, quota or network: fall back to DuckDuckGo
    if not results:
        results = _ddg_search(query, max_results)
    if not results:
        return "no results; try different words"
    return "\n\n".join(
        f"{i}. {r['title']}\n{r['url']}\n{r['text'][:SNIPPET_CHARS]}" for i, r in enumerate(results, 1)
    )


def fetch_page(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "error: url must start with http:// or https://"
    if url in _failed_urls:  # small models like to retry the same broken link
        return f"error: this URL already failed ({_failed_urls[url]}); do not retry it, use a different source"

    url = quote(url, safe=URL_SAFE_CHARS)  # Hebrew or spaces in a URL must be %-encoded for urllib
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if "html" not in response.headers.get("Content-Type", ""):
                return f"error: not a web page ({response.headers.get('Content-Type')})"
            html = response.read().decode(response.headers.get_content_charset() or "utf-8", "replace")
    except urllib.error.HTTPError as e:
        reason = f"HTTP {e.code}" + (": the site does not allow automated access" if e.code in (401, 403, 429) else "")
        _failed_urls[url] = reason
        return f"error: {reason}; use a different source"
    except Exception as e:
        _failed_urls[url] = str(e).splitlines()[0][:120]
        raise

    soup = BeautifulSoup(html, "html.parser")
    booking_links, order_links, other_links = _links(soup, url)  # before nav/header are removed: menus hold the links
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg"]):
        tag.decompose()
    # keep table rows on one line as "a | b | c" so the model can see the columns
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
        row.replace_with(" | ".join(cells) + "\n")
    # line breaks only after block elements, so links and bold text stay inside their sentence
    for block in soup.find_all(BLOCK_TAGS):
        block.append("\n")

    lines = (" ".join(line.split()) for line in soup.get_text().splitlines())
    text = "\n".join(line for line in lines if line)
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + f"\n\n(truncated: page has {len(text)} characters)"
    title = soup.title.get_text(strip=True) if soup.title else url
    page = f"{title}\n{url}\n\n{text}"
    if booking_links:
        page += "\n\nTable reservation links:\n" + "\n".join(booking_links)
    if order_links:
        page += "\n\nFood order / delivery / gift card links (NOT table reservations):\n" + "\n".join(order_links)
    if other_links:
        page += "\n\nOther links:\n" + "\n".join(other_links)
    return page


def _links(soup, base_url: str) -> tuple[list[str], list[str], list[str]]:
    """Links on the page as 'text -> url': table reservations, food orders, and the rest."""
    seen, booking, orders, other = set(), [], [], []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0]
        if not href.startswith(("http://", "https://")) or href in seen or href.rstrip("/") == base_url.rstrip("/"):
            continue
        seen.add(href)
        text = " ".join(a.get_text(" ", strip=True).split())[:60] or "(no text)"
        line = f"{text} -> {href[:MAX_LINK_CHARS]}"
        if ORDER_LINK.search(f"{text} {href}"):
            orders.append(line)
        elif BOOKING_LINK.search(f"{text} {href}"):
            booking.append(line)
        else:
            other.append(line)
    return booking[:MAX_BOOKING_LINKS], orders[:MAX_BOOKING_LINKS], other[:MAX_OTHER_LINKS]


def _cell(value):
    """Turn numeric strings like '42' or '3.5' into numbers so the spreadsheet can do math on them."""
    if isinstance(value, str):
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value
    return value


def create_spreadsheet(path: str, headers: list, rows: list) -> str:
    if not path.endswith(".xlsx"):
        path += ".xlsx"
    full = _safe_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.append([str(h) for h in headers])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in rows:
        sheet.append([_cell(v) for v in row])

    # widen each column to fit its longest value
    for column in sheet.columns:
        longest = max(len(str(c.value)) for c in column if c.value is not None)
        sheet.column_dimensions[column[0].column_letter].width = min(longest + 2, 60)

    sheet.freeze_panes = "A2"
    workbook.save(full)
    return f"created {path} with {len(headers)} columns and {len(rows)} rows"
