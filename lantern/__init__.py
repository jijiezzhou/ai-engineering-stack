"""Lantern — a local-first coding agent that explains unfamiliar codebases."""

from lantern.agent import ask
from lantern.llm import LLM, ToolCall
from lantern.summarize import FileSummary, summarize_file
from lantern.tools import DEFAULT_TOOLS, ToolSpec, grep, list_dir, read_file

__all__ = [
    # week 1
    "LLM",
    # week 2
    "FileSummary",
    "summarize_file",
    # week 3
    "ToolCall",
    "ToolSpec",
    "DEFAULT_TOOLS",
    "read_file",
    "list_dir",
    "grep",
    "ask",
]
