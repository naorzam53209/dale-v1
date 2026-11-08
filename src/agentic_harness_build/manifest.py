"""Load a JSON Agents manifest (*.agents.json) and turn it into a runnable agent.

The manifest is the single source of truth: identity, tools, model settings,
instructions and policies all come from it. Validate one from the command line:

    uv run python -m agentic_harness_build.manifest agents/research-analyst.agents.json
"""

import importlib
import inspect
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

from . import ui
from .policy import PolicyError, compile_expression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "json-agents.schema.json"
SUPPORTED_ACTIONS = {"tool.call"}
AUDIT_PREVIEW_CHARS = 200  # long arguments (e.g. file contents) are shortened in the audit log

# Our own extension fields. The spec allows any x-* field; these are the ones this harness reads.
EXTENSION_SCHEMA = {
    "type": "object",
    "required": ["x-model", "x-instructions"],
    "properties": {
        "x-model": {
            "type": "object",
            "required": ["name", "base_url"],
            "additionalProperties": False,
            "properties": {
                "provider": {"type": "string"},
                "name": {"type": "string"},
                "base_url": {"type": "string"},
                "base_url_env": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
                "max_steps": {"type": "integer", "minimum": 1},
                "context_window": {"type": "integer", "minimum": 1024},
            },
        },
        "x-instructions": {
            "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
        },
        "x-sop": {
            "type": "object",
            "required": ["name", "stages"],
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "stages": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name", "goal", "steps"],
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "goal": {"type": "string"},
                            "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        },
                    },
                },
            },
        },
        "x-reply": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "max_chars": {"type": "integer", "minimum": 50, "description": "Longer final answers get rewritten shorter."},
                "verify_phone_numbers": {"type": "boolean", "description": "Phone numbers must appear in a page read this turn."},
            },
        },
        "x-reflection": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enabled": {"type": "boolean", "description": "Enable verbal reinforcement learning via reflection."},
                "strategy": {"type": "string", "enum": ["none", "last_attempt", "reflexion", "last_attempt_and_reflexion"], "description": "Reflection strategy to use."},
                "task_type": {"type": "string", "description": "Task type for organizing reflections (e.g., 'research', 'booking')."},
                "reflect_after": {"type": "string", "enum": ["every_turn", "failure_only"], "description": "When to generate reflections."},
                "max_reflections": {"type": "integer", "minimum": 1, "description": "Maximum number of reflections to keep in context."},
                "max_retries": {"type": "integer", "minimum": 0, "description": "Maximum number of retry attempts for failed tasks (Reflexion retry loop)."},
            },
        },
        "x-memory": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enabled": {"type": "boolean", "description": "Enable memory tool for persistent learnings."},
                "auto_extract": {"type": "boolean", "description": "Automatically extract learnings after each turn."},
                "max_learnings_per_turn": {"type": "integer", "minimum": 0, "description": "Max learnings to extract per turn."},
                "memory_dir": {"type": "string", "description": "Memory directory path (default: data/memory)."},
            },
        },
    },
}


class ManifestError(ValueError):
    pass


@dataclass
class Tool:
    name: str
    spec: dict  # the tool entry from the manifest
    function: object

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.spec.get("description", ""),
                "parameters": self.spec.get("input_schema", {"type": "object", "properties": {}}),
            },
        }


@dataclass
class Policy:
    id: str
    effect: str  # allow | deny | audit | notify
    description: str
    matches: object  # compiled `where` expression: context -> bool


@dataclass
class Agent:
    path: Path
    manifest: dict
    tools: dict[str, Tool]
    policies: list[Policy]
    audit_log: Path | None = None
    log_level: str = "info"
    redact_sensitive: bool = False
    model: dict = field(default_factory=dict)
    instructions: str = ""
    reply: dict = field(default_factory=dict)
    reflection: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.manifest["agent"]["name"]

    def openai_tools(self) -> list[dict]:
        return [tool.openai_schema() for tool in self.tools.values()]

    def call_tool(self, name: str, arguments: str) -> str:
        """Run one tool call from the model, after checking it against the policies.

        Every problem comes back as text so the model can read it and recover."""
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return f"error: arguments are not valid JSON ({e})"
        if not isinstance(args, dict):
            return "error: arguments must be a JSON object"

        # args_text lets a policy search all arguments at once, e.g. args_text contains '{{card.'
        args_text = json.dumps(args, ensure_ascii=False)
        context = {"tool": tool.spec, "args": args, "args_text": args_text, "agent": self.manifest["agent"]}
        matched = [p for p in self.policies if p.matches(context)]
        for policy in matched:
            if policy.effect == "notify":
                ui.say(f"[policy {policy.id}] {policy.description or 'matched'}")
        denied = next((p for p in matched if p.effect == "deny"), None)
        if denied:
            result = f"error: blocked by policy '{denied.id}': {denied.description}"
        else:
            try:
                inspect.signature(tool.function).bind(**args)
            except TypeError as e:
                result = f"error: wrong arguments for {name} ({e})"
            else:
                try:
                    result = str(tool.function(**args))
                except Exception as e:
                    result = f"error: {e}"

        if any(p.effect == "audit" for p in matched) or denied:
            self._audit(name, args, result, blocked_by=denied.id if denied else None)
        return result

    def _audit(self, tool_name: str, args: dict, result: str, blocked_by: str | None):
        if self.audit_log is None:
            return
        shorten = lambda v: v[:AUDIT_PREVIEW_CHARS] + "..." if isinstance(v, str) and len(v) > AUDIT_PREVIEW_CHARS else v
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agent": self.manifest["agent"]["id"],
            "tool": tool_name,
            "args": {k: shorten(v) for k, v in args.items()},
            "outcome": "blocked" if blocked_by else ("error" if result.startswith("error:") else "ok"),
            "blocked_by": blocked_by,
        }
        if not self.redact_sensitive:  # results can hold personal data (answers, page text)
            record["result"] = shorten(result)
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- loading -------------------------------------------------------------------


def render_sop(sop: dict) -> str:
    """Turn the x-sop stages into prompt text the model follows."""
    stages = sop["stages"]
    lines = [f"## Standard operating procedure: {sop['name']}"]
    lines.append("Work through these stages in order: " + " -> ".join(s["name"] for s in stages) + ".")
    for number, stage in enumerate(stages, 1):
        lines.append(f"\n### {number}. {stage['name']}\nGoal: {stage['goal']}")
        lines += [f"- {step}" for step in stage["steps"]]
    return "\n".join(lines)


def _schema_errors(validator: Draft202012Validator, data: dict) -> list[str]:
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}" for e in errors]


def _resolve_endpoint(endpoint: str, tool_name: str):
    """'package.module:function' -> the Python function."""
    module_name, _, function_name = endpoint.partition(":")
    if not function_name:
        raise ManifestError(f"tool '{tool_name}': endpoint must look like 'module:function', got '{endpoint}'")
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as e:
        raise ManifestError(f"tool '{tool_name}': cannot load endpoint '{endpoint}' ({e})") from e
    if not callable(function):
        raise ManifestError(f"tool '{tool_name}': endpoint '{endpoint}' is not a function")
    return function


def _audit_path(log_sink: str | None) -> Path | None:
    if not log_sink:
        return None
    if not log_sink.startswith("file://"):
        raise ManifestError(f"observability.log_sink: only file:// sinks are supported, got '{log_sink}'")
    path = Path(log_sink.removeprefix("file://"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_agent(path: str | Path) -> Agent:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"manifest not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ManifestError(f"{path.name} is not valid JSON: {e}") from None

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = _schema_errors(Draft202012Validator(schema), manifest)
    errors += _schema_errors(Draft202012Validator(EXTENSION_SCHEMA), manifest)
    if errors:
        raise ManifestError(f"{path.name} is not a valid manifest:\n  " + "\n  ".join(errors))

    tools = {}
    for spec in manifest.get("tools", []):
        name = spec["name"]
        if name in tools:
            raise ManifestError(f"duplicate tool name '{name}'")
        if "endpoint" not in spec:
            raise ManifestError(f"tool '{name}': missing endpoint")
        tools[name] = Tool(name=name, spec=spec, function=_resolve_endpoint(spec["endpoint"], name))

    policies = []
    for rule in manifest.get("policies", []):
        if rule["action"] not in SUPPORTED_ACTIONS:
            raise ManifestError(f"policy '{rule['id']}': action '{rule['action']}' is not supported (only tool.call)")
        try:
            matches = compile_expression(rule["where"]) if "where" in rule else (lambda ctx: True)
        except PolicyError as e:
            raise ManifestError(f"policy '{rule['id']}': {e}") from e
        policies.append(Policy(rule["id"], rule["effect"], rule.get("description", ""), matches))

    instructions = manifest["x-instructions"]
    instructions = "\n".join(instructions) if isinstance(instructions, list) else instructions
    if "x-sop" in manifest:  # the SOP goes where {{sop}} is, or at the end
        sop = render_sop(manifest["x-sop"])
        instructions = instructions.replace("{{sop}}", sop) if "{{sop}}" in instructions else f"{instructions}\n\n{sop}"
    observability = manifest.get("observability", {})
    return Agent(
        path=path,
        manifest=manifest,
        tools=tools,
        policies=policies,
        audit_log=_audit_path(observability.get("log_sink")),
        log_level=observability.get("log_level", "info"),
        redact_sensitive=observability.get("redact_sensitive", False),
        model=manifest["x-model"],
        instructions=instructions,
        reply=manifest.get("x-reply", {}),
        reflection=manifest.get("x-reflection", {}),
        memory=manifest.get("x-memory", {}),
    )


# --- assistants: several agents behind one router (the spec's graph profile) ------


@dataclass
class Route:
    id: str
    agent: Agent
    description: str


@dataclass
class Assistant:
    """What the harness runs: one agent, or a router that picks an agent for each message."""

    name: str
    routes: dict[str, Route]  # node id -> agent
    edges: list[tuple[str, object]] = field(default_factory=list)  # (target node, compiled condition)
    router: Agent | None = None  # None: a single agent, no routing

    def pick(self, intent: str, text: str) -> Route:
        """First edge whose condition matches the message; an edge without a condition always matches."""
        context = {"message": {"intent": intent, "text": text}}
        for target, matches in self.edges:
            if matches(context):
                return self.routes[target]
        return next(iter(self.routes.values()))


def load_assistant(path: str | Path) -> Assistant:
    """Load a manifest. With a `graph`, every non-router node is an agent manifest of its own."""
    top = load_agent(path)
    graph = top.manifest.get("graph")
    if not graph:
        return Assistant(name=top.name, routes={"agent": Route("agent", top, top.manifest["agent"].get("description", ""))})

    own_id = top.manifest["agent"]["id"]
    router_ids, routes = set(), {}
    for node in graph["nodes"]:
        if node["ref"] == own_id:
            router_ids.add(node["id"])
            continue
        if not node["ref"].endswith(".agents.json"):
            raise ManifestError(f"graph node '{node['id']}': ref must be a path to an .agents.json file, got '{node['ref']}'")
        description = node.get("metadata", {}).get("description") or ""
        try:
            agent = load_agent(node["ref"])
        except ManifestError as e:
            raise ManifestError(f"graph node '{node['id']}': {e}") from e
        routes[node["id"]] = Route(node["id"], agent, description or agent.manifest["agent"].get("description", ""))
    if not router_ids or not routes:
        raise ManifestError("graph needs a router node (ref = this manifest's agent.id) and at least one agent node")

    edges = []
    for edge in graph.get("edges", []):
        if edge["from"] not in router_ids or edge["to"] not in routes:
            raise ManifestError(f"graph edge {edge['from']} -> {edge['to']}: must go from the router to an agent node")
        try:
            matches = compile_expression(edge["condition"]) if "condition" in edge else (lambda ctx: True)
        except PolicyError as e:
            raise ManifestError(f"graph edge {edge['from']} -> {edge['to']}: {e}") from e
        edges.append((edge["to"], matches))
    return Assistant(name=top.name, routes=routes, edges=edges, router=top)


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(str(p) for p in (PROJECT_ROOT / "agents").glob("*.agents.json"))
    failed = False
    for manifest_path in paths:
        try:
            assistant = load_assistant(manifest_path)
            if assistant.router:
                print(f"ok    {manifest_path}: {assistant.name}, routes to {', '.join(assistant.routes)}")
            else:
                agent = assistant.routes["agent"].agent
                print(f"ok    {manifest_path}: {agent.name}, {len(agent.tools)} tools, {len(agent.policies)} policies")
        except ManifestError as e:
            print(f"FAIL  {manifest_path}: {e}")
            failed = True
    sys.exit(1 if failed else 0)
