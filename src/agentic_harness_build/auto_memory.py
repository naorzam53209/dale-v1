"""Auto memory - background extraction of learnings from agent interactions.

After each agent turn, extracts learnings worth remembering and writes them to
memory files. This is the "auto memory" part of Anthropic's architecture.
"""

import json
import re
from datetime import date
from pathlib import Path

from .memory import MEMORY_DIR, ensure_memory_dir

# Prompt for extracting learnings
EXTRACTION_PROMPT = """Analyze this agent interaction and extract learnings worth remembering for future sessions.

User request: {user_input}
Agent actions: {actions_summary}
Agent response: {agent_response}

Extract 0-3 learnings. Only include information that would be useful in a FUTURE conversation.
Skip obvious facts, one-time errors, or information already in the codebase.

Categories:
- user: User preferences, role, expertise, working style
- feedback: Corrections given, approaches confirmed, things to avoid
- project: Ongoing work, deadlines, decisions, context that persists
- reference: External resources, where to find information

Return ONLY a JSON array, no other text:
[{{"type": "user|feedback|project|reference", "content": "brief learning", "file": "suggested_filename.md"}}]

If nothing worth remembering, return: []"""


def _get_memory_type_filename(memory_type: str) -> str:
    """Get standard filename for a memory type."""
    type_files = {
        "user": "user_preferences.md",
        "feedback": "feedback_corrections.md",
        "project": "project_context.md",
        "reference": "external_references.md",
    }
    return type_files.get(memory_type, f"{memory_type}_notes.md")


SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.md")


def _safe_filename(name) -> str | None:
    """A plain file name inside the memory folder, or None.

    The name comes from the model, whose input includes web page text, so it must not
    be able to point anywhere else: no folders, no '..', no other extensions, and never
    the index itself."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name.endswith(".md"):
        name += ".md"
    if not SAFE_FILENAME.fullmatch(name) or name.upper() == "MEMORY.MD":
        return None
    if (MEMORY_DIR / name).resolve().parent != MEMORY_DIR.resolve():
        return None
    return name


def _update_memory_index(new_file: str, description: str):
    """Update MEMORY.md index with new memory file."""
    index_file = MEMORY_DIR / "MEMORY.md"
    if not index_file.exists():
        return

    content = index_file.read_text(encoding="utf-8")

    # Check if file already in index
    if new_file in content:
        return

    # Add to index
    lines = content.split("\n")
    # Find the end of the index section
    insert_pos = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("#") or (line.strip() and not line.startswith("-")):
            insert_pos = i
            break

    new_entry = f"- [{new_file.replace('.md', '').replace('_', ' ').title()}]({new_file}) - {description}"
    lines.insert(insert_pos, new_entry)

    index_file.write_text("\n".join(lines), encoding="utf-8")


def _append_to_memory_file(file_path: Path, learning: dict):
    """Append a learning to a memory file."""
    memory_type = learning.get("type", "project")
    content = learning.get("content", "")

    if not content:
        return

    # Create file with frontmatter if it doesn't exist
    if not file_path.exists():
        frontmatter = f"""---
type: {memory_type}
created: {date.today().isoformat()}
---

# {memory_type.title()} Memory

"""
        file_path.write_text(frontmatter, encoding="utf-8")

    # Append learning
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {content}\n")


def extract_learnings(user_input: str, actions_summary: str, agent_response: str,
                      client, model: dict) -> list[dict]:
    """Extract learnings from an agent interaction using the LLM."""
    prompt = EXTRACTION_PROMPT.format(
        user_input=user_input[:500],  # Truncate long inputs
        actions_summary=actions_summary[:1000],
        agent_response=agent_response[:1000]
    )

    try:
        response = client.chat.completions.create(
            model=model["name"],
            temperature=0.3,  # Lower temperature for consistent extraction
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        content = (response.choices[0].message.content or "").strip()

        # Parse JSON response
        # Handle cases where model wraps JSON in markdown
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        learnings = json.loads(content)
        return learnings if isinstance(learnings, list) else []

    except Exception as e:
        print(f"[auto_memory] Failed to extract learnings: {e}")
        return []


def save_learnings(learnings: list[dict], max_per_turn: int = 3):
    """Save extracted learnings to memory files."""
    if not learnings:
        return

    ensure_memory_dir()

    # Limit number of learnings per turn
    learnings = learnings[:max_per_turn]

    for learning in learnings:
        if not isinstance(learning, dict):
            continue
        memory_type = str(learning.get("type", "project"))
        filename = _safe_filename(learning.get("file")) or _safe_filename(_get_memory_type_filename(memory_type))
        if not filename:
            continue
        file_path = MEMORY_DIR / filename

        # Append learning to file
        _append_to_memory_file(file_path, learning)

        # Update index if new file
        _update_memory_index(filename, learning.get("content", "")[:50])

    print(f"[auto_memory] Saved {len(learnings)} learnings")
