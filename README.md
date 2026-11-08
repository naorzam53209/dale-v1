# Agentic Harness Build

A small agent harness: a local model (via Ollama's OpenAI-compatible API) that can use tools to work with files in `workspace/`.

## Setup

Runs entirely on your own computer. Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) installed and running (the model is about 3.3 GB). The agents use `qwen3.5:4b-16k`, which is `qwen3.5:4b` with a 16k context (Ollama's 4k default is too small for web pages):

```sh
ollama pull qwen3.5:4b
curl http://localhost:11434/api/create -d '{"model":"qwen3.5:4b-16k","from":"qwen3.5:4b","parameters":{"num_ctx":16384}}'

uv sync
uv run playwright install chromium   # browser for the Booking Assistant
```

Web search uses [Tavily](https://tavily.com) when `TAVILY_API_KEY` is set (add `export TAVILY_API_KEY=tvly-...` to `~/.zshrc`), and falls back to DuckDuckGo when the key is missing or Tavily fails.

## Run

```sh
uv run python harness.py                                      # the Assistant: research + booking in one
uv run python harness.py agents/booking-assistant.agents.json  # or run a single agent
```

### Docker

```sh
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t dale-agent .
docker run -it --rm \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_URL=http://host.docker.internal:11434/v1 \
  -e TAVILY_API_KEY=your_key \
  -v $(pwd)/workspace:/app/workspace \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  dale-agent
```

**Note**: Inside the container, `localhost` is the container itself, so Docker reaches the Ollama on your computer through `host.docker.internal`. On Linux, Ollama must also listen beyond 127.0.0.1: start it with `OLLAMA_HOST=0.0.0.0`.

Ollama is expected at `http://localhost:11434/v1`. To use a server on another machine, set `OLLAMA_URL` (e.g. `OLLAMA_URL=http://<server-ip>:11434/v1`).

Type a request after `user:`, and `exit` (or an empty line) to quit. Output is kept short:

```
user: what do analysts expect from Apple's earnings today?
agent: sure [research: web_search, fetch_page]
agent: Consensus EPS $1.98, revenue $113.3B, reported after the close today. Source: https://...

user: does Meatbar Herzliya have a table for 3 on Thursday at 15:00?
agent: sure [booking: web_search, fetch_page, browser_open ✗]
agent: Meatbar Herzliya books via Tabit, which blocks automated booking: https://tabitisrael.co.il/... or call 09-9546000.
```

The `[...]` line shows which agent took the message and the tools it used (`✗` marks a failed call). If it needs something from you, it asks right there. Set `observability.log_level` to `debug` in the manifest to see every tool's arguments and results instead.

Examples:

```
you> find the 5 largest cities in Japan and make a spreadsheet with name, prefecture and population   # Research Analyst
you> does Golden Dragon in Tel Aviv have a table for 2 tonight? if so, book it                        # Booking Assistant
```

## Agents

| Agent | Manifest | What it does |
|---|---|---|
| Assistant (default) | [assistant.agents.json](agents/assistant.agents.json) | Routes each message to research or booking; one shared conversation |
| Research Analyst | [research-analyst.agents.json](agents/research-analyst.agents.json) | Web research with verified, cited numbers; delivers spreadsheets and Markdown reports |
| Booking Assistant | [booking-assistant.agents.json](agents/booking-assistant.agents.json) | Finds a restaurant, checks availability, and books it in a visible browser |

### How the Assistant routes

[assistant.agents.json](agents/assistant.agents.json) uses the spec's `graph` profile: a `router` node plus one node per agent (`ref` points to that agent's manifest, `metadata.description` tells the router what it handles), and edges with conditions like `message.intent == 'booking'` (an edge without a condition is the fallback).

For each message the router asks the model, with thinking turned off (under a second), which agent fits, using the last two turns so follow-ups like "book the first one" land right. The chosen agent runs with its own instructions, tools and policies. The conversation (your messages and the final answers) is shared by all agents; tool output stays with the agent that produced it. To add an agent: write its manifest, then add a node and an edge.

### Booking: the Restaurant Reservation Execution Scheme

The Booking Assistant follows a standard operating procedure kept in its manifest (`x-sop`), so the stages can be edited without touching code:

**Triage Request → Fetch & Match → Check Availability → Verify & Confirm → Handoff**

1. **Triage**: cuisine/vibe, party size (2 if not given), date and time, location, urgency. A named venue skips discovery.
2. **Fetch & Match**: search, drop closed / walk-in-only / off-profile venues, keep at most 3, labeled best food / best ambience / most convenient.
3. **Check Availability**: open each candidate's own reservation link and read live slots; try ±1 hour if the exact time is taken; check opening hours.
4. **Verify & Confirm**: table fits the party, location makes sense, surface cancellation fees, deposits, dress codes, time limits.
5. **Handoff**: one line per option with confirmed slots (or "slots unconfirmed"), the direct booking link and the phone number. If you asked it to book, it completes the booking (rules below).

Adapted to a small local model: at most 3 candidates instead of 5-20, and no maps/transit tool (location is checked from addresses).

### How the Booking Assistant books

1. Finds the venue's official website and uses the booking link it publishes (Tabit, Ontopo, OpenTable...), and asks (once) for any missing date, time or party size. If that booking site refuses automated browsers, it gives you the direct link and the phone number instead.
2. Opens the page in a hidden browser and checks availability. For "is there space?" questions it stops here and reports.
3. Asks for your details with `ask_user`: always name, phone, address and city, plus anything the form marks required. They are saved in `data/profile.json` and pre-filled next time, but you confirm them (Enter) in every session.
4. `auto_fill` fills the form: contact fields are matched automatically (by `autocomplete`, name, label, placeholder; English and Hebrew), the rest (date, time, guests) are passed by the model.

Enforced in code ([browser.py](src/agentic_harness_build/browser.py), [profile.py](src/agentic_harness_build/profile.py)), not just asked of the model:
- No form with contact fields, and no final booking button, can be submitted until you have given or confirmed your name, phone, address and city in this session (`REQUIRED_FOR_BOOKING`).
- Contact fields are only ever filled with what you gave; a name, phone, email or address the model types itself is rejected.
- Submitting is blocked while any field the site marks required is empty (including required checkboxes such as terms).
5. Only if the site requires a card, `request_payment` asks you for it (see below), then `auto_fill` fills the card fields.
6. Clicking a final button (book, confirm, pay, הזמנה, אישור...) shows you everything that was filled and needs you to type `yes`.

The browser runs hidden. For a captcha or login, the agent calls `browser_show`, which reopens the page in a visible window for you to complete, then continues. Cookies persist in `data/browser/`, so a site you logged into stays logged in. Set `BROWSER_HEADLESS=0` to watch the browser the whole time.

### Card safety

- You type the card into hidden prompts in the terminal. The model never sees the numbers: it only gets placeholders like `{{card.number}}`, which the browser swaps for the real value at the moment of filling.
- The card is tied to the site that was open when you approved it; it won't fill on any other domain.
- Anything the model reads back (page text, URL, field values) is masked, including confirmation pages that echo the card.
- The card is wiped from memory after the submission; a new payment needs a new approval.
- Optional: save the card in the macOS Keychain (service `dale-agent`). The CVC is never saved; you type it each time. To remove the saved card: `uv run python -c "from agentic_harness_build.payment import forget_saved_card; print(forget_saved_card())"`.
- The policy `card-placeholders-stay-in-forms` blocks card placeholders in any tool except `auto_fill`, and the audit log for this agent stores no tool results (`redact_sensitive: true`).

## Agents are JSON manifests

Each agent is a single [JSON Agents](https://github.com/JSON-Agents/Standard) manifest in `agents/*.agents.json` (spec v1.0, profiles `core`, `exec`, `gov`). It is the single source of truth for the agent:

| Manifest field | Used for |
|---|---|
| `agent`, `capabilities`, `modalities` | Identity and description |
| `tools[]` | Tools offered to the model: `name`, `description`, `input_schema`; `endpoint` (`module:function`) is the Python function that runs |
| `policies[]` | Rules checked before every tool call (see below) |
| `observability` | `log_level` (`debug` / `info` / `warn` / `error`) and `log_sink` for the audit log |
| `x-model` | Model name, server URL, temperature, timeout, `max_steps`, `context_window` |
| `x-instructions` | The system prompt (a string, or a list of lines) |
| `x-sop` | A standard operating procedure: `name` and `stages` (each with `id`, `name`, `goal`, `steps`). Rendered into the system prompt where `x-instructions` has a `{{sop}}` line (or at the end) |
| `x-reply` | `max_chars`: longer final answers are rewritten shorter, and markdown is stripped. `verify_phone_numbers`: every phone number in an answer must appear in a page read that turn, otherwise it is corrected or removed |
| `x-reflection` | Enables verbal reinforcement learning: `enabled` (bool), `strategy` (none/last_attempt/reflexion/last_attempt_and_reflexion), `task_type` (for organizing reflections), `reflect_after` (every_turn/failure_only), `max_reflections` (how many to keep in context) |

The harness validates the manifest on startup against the official schema ([schemas/json-agents.schema.json](schemas/json-agents.schema.json)) and refuses to run if it is invalid. To check manifests without running:

```sh
uv run python -m agentic_harness_build.manifest            # all agents/*.agents.json
```

### Reflection (Verbal Reinforcement Learning)

Agents use [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) to learn from failures and retry with improved strategies. When enabled, the agent:

1. Attempts the task
2. Checks for success using agent-specific detection
3. If failed, generates a self-reflection analyzing what went wrong
4. Retries the task with the reflection injected into context
5. Repeats until success or max retries reached
6. Stores reflections in `data/reflections/` for future learning

Configuration (in the manifest's `x-reflection` field):
- `enabled`: Set to `true` to activate reflection
- `strategy`: How to reflect:
  - `none`: No reflection (baseline)
  - `last_attempt`: Include the full previous attempt in context
  - `reflexion`: Generate and use self-reflections (recommended)
  - `last_attempt_and_reflexion`: Both previous attempt and self-reflections
- `task_type`: Organizes reflections by domain (e.g., "research", "booking"). Also determines the agent type for specialized prompts
- `reflect_after`: When to generate reflections:
  - `every_turn`: Reflect after every interaction (for continuous improvement)
  - `failure_only`: Only reflect when the task fails (for success/failure detection)
- `max_reflections`: Maximum number of reflections to keep in context (default: 5)
- `max_retries`: Maximum number of retry attempts for failed tasks (default: 0, recommended: 2-3 for 4B models)

**Agent-Specific Behavior:**
- **Booking Assistant**: Uses `failure_only` reflection with `max_retries: 3`. Only retries when booking fails (detected by absence of confirmation patterns like "Booked:" or "Confirmation #"). The reflection focuses on overcoming booking obstacles (site blocking, missing info, timing issues, captchas).
- **Research Analyst**: Uses `every_turn` reflection with `max_retries: 2`. Retries when research fails (detected by lack of sources or substantive answer). The reflection focuses on improving research quality: better sources, verification, completeness.

**Optimized for Qwen 3.5 4B:**
- Shorter reflection prompts (~370 chars vs ~800 in original)
- Reduced max_tokens for reflection generation (200 vs 500)
- Simplified success detection logic
- Configurable retry limits to balance quality vs speed

Reflections are stored per-task in `data/reflections/<task_type>.jsonl` and persist across sessions.

### Policies

Policies use the spec's expression language (`==`, `!=`, `<`, `>`, `~` regex, `in`, `contains`, `starts_with`, `ends_with`, `&&`, `||`, `not`). They are evaluated against `tool` (the tool's manifest entry), `args` (the call's arguments), `args_text` (all arguments as one JSON string) and `agent`:

| Effect | What happens when `where` matches |
|---|---|
| `deny` | The call is blocked; the model gets the policy's description as the error |
| `audit` | The call is recorded in the audit log (`logs/audit.jsonl`) |
| `notify` | A line is printed to the console |
| `allow` | Nothing (calls are allowed unless denied) |

The Research Analyst ships with `https-only`, `no-private-network`, `no-executable-files` (deny) and `audit-file-writes` (audit). The Booking Assistant has `https-only`, `no-private-network`, `no-food-order-links`, `card-placeholders-stay-in-forms` (deny) and `audit-booking-actions` (audit). Blocked calls are always audited.

## Tools

Plain Python functions; their schemas are in each agent's manifest. File paths are relative to `workspace/` and can't escape it.

| Tool | Arguments | What it does |
|---|---|---|
| `list_files` | `subdir` (optional) | List files in the workspace |
| `read_file` | `path` | Read a text file |
| `write_file` | `path`, `content` | Create or overwrite a text file |
| `web_search` | `query`, `max_results` (optional, 1-10) | Search the web with Tavily (or DuckDuckGo without a key) and return titles, links and extracts |
| `fetch_page` | `url` | Read a web page as plain text (tables as `a \| b \| c` rows, capped at 8,000 characters), followed by its links: table reservations, food orders, and the rest |
| `create_spreadsheet` | `path`, `headers`, `rows` | Create an `.xlsx` file with a bold header row; numeric strings become numbers |
| `ask_user` | `question`, `fields` (optional) | Ask the user; saved profile fields are pre-filled. Refuses card fields |
| `browser_open` | `url` | Open a page; returns its text and fields/buttons with `[ref]`s (including inside iframes) |
| `browser_view` | | Show the current page again |
| `browser_show` | | Reopen the current page in a visible window (captcha, login) |
| `auto_fill` | `fields` (optional `[{ref, value}]`) | Fill the form: saved details and the approved card automatically, plus the given fields |
| `browser_click` | `ref` | Click; final submit buttons need the user to type `yes` |
| `request_payment` | `reason` | Ask the user for a card (hidden input or Keychain); the model only gets placeholders |
| `browser_close` | | Close the browser and wipe the card |

To add a tool: write the function, then add an entry to the manifest's `tools` array with its `endpoint` (`module:function`) and `input_schema`.

## Layout

| Path | Contents |
|---|---|
| `harness.py` | Loads an agent manifest, runs the tool-call loop and the chat prompt |
| `agents/` | Agent manifests (`*.agents.json`) |
| `schemas/` | Official JSON Agents schema (Apache-2.0, from JSON-Agents/Standard) |
| `src/agentic_harness_build/tools.py` | File, web and spreadsheet tools |
| `src/agentic_harness_build/browser.py` | Browser tools: open, view, show, auto_fill, click, close |
| `src/agentic_harness_build/profile.py` | Saved user details and `ask_user` |
| `src/agentic_harness_build/payment.py` | Card handling: hidden input, Keychain, placeholders, masking |
| `src/agentic_harness_build/ui.py` | Terminal output (compact tool line) and prompts used by the tools |
| `src/agentic_harness_build/manifest.py` | Manifest loading, validation, policy-checked tool calls, graph (router) assistants |
| `src/agentic_harness_build/policy.py` | Policy expression parser/evaluator |
| `logs/` | Audit log (git-ignored) |
| `data/` | Saved profile and browser cookies (git-ignored) |
| `workspace/` | The agent's sandbox folder |
| `old/` | Earlier versions: `v0` plain chat, `v1`–`v2` file tools, `v3` MCP servers (time, fetch) |
| `Memory.md` | Project notes |
