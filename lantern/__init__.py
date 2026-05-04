"""Lantern — a local-first coding agent that explains unfamiliar codebases."""

from lantern.agent import ask
from lantern.chunk import Chunk, chunk_file
from lantern.evals import EvalReport, TestCase, evaluate, load_tests
from lantern.index import index_repo
from lantern.llm import LLM, ToolCall
from lantern.rerank import rerank
from lantern.search import Hit, bm25_search, hybrid_search, search
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
    # week 4
    "Chunk",
    "chunk_file",
    "index_repo",
    "Hit",
    "search",
    # week 5
    "bm25_search",
    "hybrid_search",
    "rerank",
    "TestCase",
    "EvalReport",
    "load_tests",
    "evaluate",
]
