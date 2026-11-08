"""Memory tool for agents - file-based persistent memory.

Implements Anthropic's memory tool spec: agents can view, create, edit, and delete
files in a memory directory that persists across sessions.

Memory lives in data/memory/ with a MEMORY.md index and topic files.
All paths are restricted to the memory directory (path traversal protection).
"""

import os
import re
from pathlib import Path

MEMORY_DIR = Path(__file__).resolve().parents[2] / "data" / "memory"
MEMORY_PREFIX = "/memories"

# Limits
MAX_FILE_LINES = 999999
MAX_VIEW_CHARS = 16000
MAX_INDEX_LINES = 200
MAX_INDEX_CHARS = 25000


def _safe_path(path: str) -> Path:
    """Resolve a memory path and ensure it's inside MEMORY_DIR."""
    # Strip /memories prefix and leading slashes
    if path.startswith(MEMORY_PREFIX):
        path = path[len(MEMORY_PREFIX):]
    path = path.lstrip("/")

    # Resolve to absolute path
    full = (MEMORY_DIR / path).resolve()

    # Ensure it's inside MEMORY_DIR
    try:
        full.relative_to(MEMORY_DIR.resolve())
    except ValueError:
        raise ValueError(f"path '{path}' is outside the memory directory")

    return full


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ["B", "K", "M", "G"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}" if unit != "B" else f"{size_bytes}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}T"


def _list_directory(path: Path, depth: int = 0, max_depth: int = 2) -> str:
    """List directory contents up to max_depth levels."""
    if not path.exists():
        return f"The path {path} does not exist. Please provide a valid path."

    if not path.is_dir():
        return f"Error: {path} is not a directory"

    lines = []
    for item in sorted(path.iterdir()):
        if item.name.startswith(".") or item.name == "node_modules":
            continue

        rel_path = item.relative_to(MEMORY_DIR)
        display_path = f"{MEMORY_PREFIX}/{rel_path}"

        if item.is_file():
            size = _format_size(item.stat().st_size)
            lines.append(f"{size}\t{display_path}")
        elif item.is_dir() and depth < max_depth:
            lines.append(f"{'4.0K'}\t{display_path}/")
            # Recurse into subdirectory
            sub_lines = _list_directory(item, depth + 1, max_depth)
            if sub_lines and not sub_lines.startswith("The path") and not sub_lines.startswith("Error"):
                for line in sub_lines.split("\n"):
                    if line and not line.startswith("Here're"):
                        lines.append(line)

    return "\n".join(lines)


def _view_file(path: Path, view_range: list | None = None) -> str:
    """View file contents with line numbers."""
    if not path.exists():
        return f"The path {MEMORY_PREFIX}/{path.relative_to(MEMORY_DIR)} does not exist. Please provide a valid path."

    if path.is_dir():
        return f"Error: {path} is a directory, not a file"

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

    lines = content.split("\n")
    if len(lines) > MAX_FILE_LINES:
        return f"File {path} exceeds maximum line limit of {MAX_FILE_LINES} lines."

    # Apply view_range if specified
    if view_range and len(view_range) == 2:
        start, end = view_range
        if start < 0 or end < -1 or start > len(lines):
            return f"Error: Invalid view_range [{start}, {end}]. File has {len(lines)} lines."
        if end == -1:
            end = len(lines)
        lines = lines[start - 1:end] if start > 0 else lines[:end]

    # Format with line numbers
    numbered = []
    for i, line in enumerate(lines, 1):
        numbered.append(f"{i:6d}\t{line}")

    result = "\n".join(numbered)
    if len(result) > MAX_VIEW_CHARS:
        result = result[:MAX_VIEW_CHARS] + "\n... (truncated)"

    return f"Here's the content of {MEMORY_PREFIX}/{path.relative_to(MEMORY_DIR)} with line numbers:\n{result}"


def memory(command: str, path: str, file_text: str | None = None, old_str: str | None = None,
           new_str: str | None = None, insert_line: int | None = None, insert_text: str | None = None,
           old_path: str | None = None, new_path: str | None = None, view_range: list | None = None) -> str:
    """Memory tool entry point - dispatches to appropriate command handler."""
    # Ensure memory directory exists
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if command == "view":
            full_path = _safe_path(path)
            if full_path == MEMORY_DIR or path in ("/memories", "memories", "."):
                # List root directory
                listing = _list_directory(MEMORY_DIR)
                header = f"Here're the files and directories up to 2 levels deep in {MEMORY_PREFIX}, excluding hidden items and node_modules:"
                return f"{header}\n{listing}"
            elif full_path.is_dir():
                listing = _list_directory(full_path)
                header = f"Here're the files and directories up to 2 levels deep in {MEMORY_PREFIX}/{full_path.relative_to(MEMORY_DIR)}, excluding hidden items and node_modules:"
                return f"{header}\n{listing}"
            else:
                return _view_file(full_path, view_range)

        elif command == "create":
            if file_text is None:
                return "Error: file_text is required for create command"
            full_path = _safe_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(file_text, encoding="utf-8")
            return f"File created successfully at: {MEMORY_PREFIX}/{full_path.relative_to(MEMORY_DIR)}"

        elif command == "str_replace":
            if old_str is None:
                return "Error: old_str is required for str_replace command"
            full_path = _safe_path(path)
            if not full_path.exists():
                return f"Error: The path {MEMORY_PREFIX}/{full_path.relative_to(MEMORY_DIR)} does not exist. Please provide a valid path."
            if full_path.is_dir():
                return f"Error: {path} is a directory, not a file"

            content = full_path.read_text(encoding="utf-8")
            count = content.count(old_str)
            if count == 0:
                return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
            elif count > 1:
                # Find line numbers
                lines = content.split("\n")
                line_nums = []
                for i, line in enumerate(lines, 1):
                    if old_str in line:
                        line_nums.append(i)
                return f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines: {', '.join(map(str, line_nums))}. Please ensure it is unique"

            new_content = content.replace(old_str, new_str or "", 1)
            full_path.write_text(new_content, encoding="utf-8")
            return f"The memory file has been edited.\n\n{_view_file(full_path)}"

        elif command == "insert":
            if insert_line is None or insert_text is None:
                return "Error: insert_line and insert_text are required for insert command"
            full_path = _safe_path(path)
            if not full_path.exists():
                return f"Error: The path {MEMORY_PREFIX}/{full_path.relative_to(MEMORY_DIR)} does not exist"
            if full_path.is_dir():
                return f"Error: {path} is a directory, not a file"

            content = full_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            if insert_line < 0 or insert_line > len(lines):
                return f"Error: Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: [0, {len(lines)}]"

            # Insert after insert_line (0 = beginning)
            lines.insert(insert_line, insert_text.rstrip("\n"))
            full_path.write_text("\n".join(lines), encoding="utf-8")
            return f"The file {MEMORY_PREFIX}/{full_path.relative_to(MEMORY_DIR)} has been edited."

        elif command == "delete":
            full_path = _safe_path(path)
            if full_path == MEMORY_DIR:
                return "Error: Cannot delete the /memories directory itself"
            if not full_path.exists():
                return f"Error: The path {MEMORY_PREFIX}/{full_path.relative_to(MEMORY_DIR)} does not exist"

            if full_path.is_dir():
                import shutil
                shutil.rmtree(full_path)
            else:
                full_path.unlink()
            return f"Successfully deleted {MEMORY_PREFIX}/{full_path.relative_to(MEMORY_DIR)}"

        elif command == "rename":
            if not old_path or not new_path:
                return "Error: old_path and new_path are required for rename command"
            old_full = _safe_path(old_path)
            new_full = _safe_path(new_path)

            if old_full == MEMORY_DIR:
                return "Error: Cannot rename the /memories directory itself"
            if not old_full.exists():
                return f"Error: The path {MEMORY_PREFIX}/{old_full.relative_to(MEMORY_DIR)} does not exist"
            if new_full.exists():
                return f"Error: The destination {MEMORY_PREFIX}/{new_full.relative_to(MEMORY_DIR)} already exists"

            new_full.parent.mkdir(parents=True, exist_ok=True)
            old_full.rename(new_full)
            return f"Successfully renamed {MEMORY_PREFIX}/{old_full.relative_to(MEMORY_DIR)} to {MEMORY_PREFIX}/{new_full.relative_to(MEMORY_DIR)}"

        else:
            return f"Error: unknown command {command}"

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def load_memory_index() -> str:
    """Load MEMORY.md index for injection into system prompt."""
    index_file = MEMORY_DIR / "MEMORY.md"
    if not index_file.exists():
        return ""
    try:
        content = index_file.read_text(encoding="utf-8")
        # Return first MAX_INDEX_LINES lines or MAX_INDEX_CHARS characters
        lines = content.split("\n")[:MAX_INDEX_LINES]
        result = "\n".join(lines)
        return result[:MAX_INDEX_CHARS]
    except Exception:
        return ""


def ensure_memory_dir():
    """Create memory directory and MEMORY.md if they don't exist."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    index_file = MEMORY_DIR / "MEMORY.md"
    if not index_file.exists():
        index_file.write_text("""---
type: index
updated: {date}
---

# Memory Index

No memories yet. Use the memory tool to store learnings.
""".format(date=__import__('datetime').date.today().isoformat()), encoding="utf-8")
