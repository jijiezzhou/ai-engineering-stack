"""
Week 3 — tool definitions for Lantern.

A `ToolSpec` is a Pydantic model with a `run()` method:

    class read_file(ToolSpec):
        '''Read the contents of a file at `path`.'''
        path: str

        def run(self, root: Path) -> str:
            return Path(self.path).read_text()

That single class declaration carries everything the model needs (name = class
name, description = docstring, args = JSON Schema from the fields) AND the
executor (`run()`). One source of truth per tool.

Lantern's default toolset is the trio a senior engineer uses to onboard:
read_file, list_dir, grep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

# Cap on what a single tool returns to the model. Keeps the context window
# from exploding on a multi-megabyte file or a grep that hits everything.
MAX_OUTPUT_CHARS = 16_000

# Names we never traverse / read / grep into. Not a substitute for .gitignore;
# just enough to keep build artefacts and secrets out of the model's view.
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".idea", ".vscode", ".claude", "dist", "build", ".next", ".uv",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
})
SKIP_SUFFIXES: frozenset[str] = frozenset({".pyc", ".lock"})


class ToolSpec(BaseModel):
    """Subclass to define a tool. The class becomes name + description + arg
    schema. Implement `run(root)` to execute it."""

    def run(self, root: Path) -> str:  # noqa: D401
        raise NotImplementedError


# ---------------------------------------------------------------- safety

def _resolve_safely(p: str | Path, root: Path) -> Path:
    """Resolve `p` (relative to `root` if not absolute) and ensure the result
    stays inside `root`. Raises ValueError otherwise.

    Why: the model picks tool args. We don't want a stray `path="../../.ssh"`
    to read your private keys.
    """
    root = root.resolve()
    candidate = Path(p) if Path(p).is_absolute() else root / p
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"path {p!r} resolves outside the repo root {root}"
        ) from e
    return candidate


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [truncated; {len(text) - limit} more chars]"


def _walk(root: Path) -> Iterable[Path]:
    """Yield files under `root`, skipping common bloat directories."""
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        if path.is_file():
            yield path


# ---------------------------------------------------------------- the tools

class read_file(ToolSpec):
    """Read the contents of a single file.

    Use this when you need to understand what a specific module, function, or
    config does. Prefer `list_dir` first if you don't yet know the path."""

    path: str = Field(
        description="File path, relative to the repository root."
    )

    def run(self, root: Path) -> str:
        try:
            full = _resolve_safely(self.path, root)
        except ValueError as e:
            return f"ERROR: {e}"
        if not full.is_file():
            return f"ERROR: not a file: {self.path}"
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"ERROR: cannot read {self.path}: {e}"
        return _truncate(content)


class list_dir(ToolSpec):
    """List immediate children of a directory.

    Use this to discover the shape of a codebase before reading any files.
    Hides build artefacts (`__pycache__`, `node_modules`, `.git`, ...)."""

    path: str = Field(
        default=".",
        description="Directory path, relative to the repository root. Default `.` is the root.",
    )

    def run(self, root: Path) -> str:
        try:
            full = _resolve_safely(self.path, root)
        except ValueError as e:
            return f"ERROR: {e}"
        if not full.is_dir():
            return f"ERROR: not a directory: {self.path}"
        entries: list[str] = []
        for child in sorted(full.iterdir()):
            if child.name in SKIP_DIRS:
                continue
            if child.suffix in SKIP_SUFFIXES:
                continue
            kind = "dir " if child.is_dir() else "file"
            try:
                rel = child.relative_to(root)
            except ValueError:
                rel = child.name
            entries.append(f"{kind}  {rel}")
        return _truncate("\n".join(entries) or "(empty directory)")


class grep(ToolSpec):
    """Find lines containing `pattern` (case-insensitive substring) anywhere
    under `path`. Returns `file:line:content` rows.

    Use this to locate where a name, string, or concept appears."""

    pattern: str = Field(description="Substring to search for. Case-insensitive.")
    path: str = Field(
        default=".",
        description="Search root, relative to the repository root.",
    )

    def run(self, root: Path) -> str:
        try:
            full = _resolve_safely(self.path, root)
        except ValueError as e:
            return f"ERROR: {e}"
        if not full.exists():
            return f"ERROR: not found: {self.path}"
        needle = self.pattern.lower()
        hits: list[str] = []
        running_chars = 0
        for file in _walk(full):
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if needle in line.lower():
                    try:
                        rel = file.relative_to(root)
                    except ValueError:
                        rel = file.name
                    snippet = line.strip()[:200]
                    hit = f"{rel}:{i}: {snippet}"
                    hits.append(hit)
                    running_chars += len(hit) + 1
                    if running_chars > MAX_OUTPUT_CHARS:
                        return _truncate("\n".join(hits))
        return "\n".join(hits) if hits else f"(no matches for {self.pattern!r})"


# ---------------------------------------------------------------- registry & wire formats

DEFAULT_TOOLS: list[type[ToolSpec]] = [read_file, list_dir, grep]


def to_ollama_tools(tools: list[type[ToolSpec]]) -> list[dict]:
    """Convert a list of ToolSpec subclasses into Ollama's OpenAI-compatible tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": cls.__name__,
                "description": (cls.__doc__ or "").strip(),
                "parameters": cls.model_json_schema(),
            },
        }
        for cls in tools
    ]


def to_anthropic_tools(tools: list[type[ToolSpec]]) -> list[dict]:
    """Convert a list of ToolSpec subclasses into Anthropic's tool format."""
    return [
        {
            "name": cls.__name__,
            "description": (cls.__doc__ or "").strip(),
            "input_schema": cls.model_json_schema(),
        }
        for cls in tools
    ]


def parse_tool_call(name: str, args: dict, tools: list[type[ToolSpec]]) -> ToolSpec:
    """Build the right ToolSpec instance from a model-emitted tool call."""
    by_name = {cls.__name__: cls for cls in tools}
    if name not in by_name:
        raise ValueError(f"unknown tool {name!r}; choices: {sorted(by_name)}")
    return by_name[name].model_validate(args)
