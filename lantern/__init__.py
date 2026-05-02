"""Lantern — a local-first coding agent that explains unfamiliar codebases."""

from lantern.llm import LLM
from lantern.summarize import FileSummary, summarize_file

__all__ = ["LLM", "FileSummary", "summarize_file"]
