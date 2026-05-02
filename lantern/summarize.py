"""
Week 2 — typed file summarization.

Given a source-file path, produce a `FileSummary`: a Pydantic model with the
fields a new contributor would want before opening the file. This is the first
real use of `LLM.structured()`, and the building block week 4 retrieval and
week 6 agent will both stand on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from lantern.llm import LLM

# Cap on how many characters we send to the model. ~30k chars ≈ 7-8k tokens for
# code, well within Qwen2.5-Coder-7B's 32k context window even with overhead.
MAX_FILE_CHARS = 30_000

LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".sh": "bash",
    ".sql": "sql",
    ".md": "markdown",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}


class FileSummary(BaseModel):
    """Structured summary of a single source file. The contract every later
    week relies on."""

    path: str = Field(
        description="Echo back the file path you were given, exactly."
    )
    language: str = Field(
        description="Primary language, lowercase (python, typescript, go, ...)."
    )
    one_liner: str = Field(
        description="One sentence: what this file is for. No filler."
    )
    public_api: list[str] = Field(
        default_factory=list,
        description="Top-level functions, classes, or constants meant to be "
        "imported or called from outside this file. Names only.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="External modules or packages this file imports. "
        "Names as written in import statements.",
    )
    notable: list[str] = Field(
        default_factory=list,
        description="Things a new contributor should know: gotchas, patterns, "
        "non-obvious invariants, hacks. Empty list if nothing stands out.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Your confidence in this summary, 0.0 to 1.0. Use 0.9+ when "
        "the file is short and clear, 0.6-0.8 when partial, below 0.5 if guessing.",
    )


SUMMARIZE_SYSTEM = (
    "You are a senior engineer onboarding to a new codebase. For each file you "
    "read, you extract a tight structured summary that lets another engineer "
    "understand the file without opening it. Be concrete and use real names — "
    "names, not paraphrases of names."
)


def summarize_file(
    path: str | Path,
    *,
    llm: Optional[LLM] = None,
) -> FileSummary:
    """Read a source file and return its structured summary."""
    p = Path(path)
    code = p.read_text(encoding="utf-8", errors="replace")
    truncated = len(code) > MAX_FILE_CHARS
    if truncated:
        code = (
            code[:MAX_FILE_CHARS]
            + f"\n\n... [truncated; full file is {len(code)} chars]"
        )

    lang = LANGUAGE_BY_SUFFIX.get(p.suffix.lower(), "")
    fence_lang = lang or ""

    prompt = (
        f"Summarize the following source file.\n\n"
        f"Path: `{p}`\n\n"
        f"```{fence_lang}\n{code}\n```\n\n"
        "Fill in every field. Specifically:\n"
        f"- `path`: copy `{p}` verbatim.\n"
        "- `language`: lowercase (python, typescript, go, etc).\n"
        "- `public_api`: list every top-level def, class, and CONSTANT defined "
        "in this file (names only). If the file defines none, return [].\n"
        "- `dependencies`: list every module name that appears in an `import` "
        "or `from ... import` statement.\n"
        "- `notable`: list 1-3 concrete things a new contributor should know "
        "(invariants, gotchas, non-obvious patterns). Use [] if nothing is "
        "notable.\n"
        "- `confidence`: your honest self-rating, 0.0-1.0."
    )

    llm = llm or LLM()
    return llm.structured(
        prompt,
        FileSummary,
        system=SUMMARIZE_SYSTEM,
        temperature=0.0,
    )
