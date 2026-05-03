"""
Week 4 — code-aware chunking.

Splits a source file into chunks that embed well:
  - Python: tree-sitter aware. One chunk per top-level def/class, plus a
    "header" chunk for module-level code (imports, docstring).
  - Anything else: fixed-size with overlap, broken on newlines when possible.

Why this matters: a chunk that ends mid-function embeds badly. Splitting on
natural boundaries massively improves retrieval — the difference between
"close enough" and "actually finds the right function."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Picked to fit comfortably under nomic-embed-text's ~8K-token window with
# room for the function signature + docstring + body of typical methods.
MAX_CHARS = 1500
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    path: str          # repo-root-relative or absolute, as the caller passes it
    start_line: int    # 1-indexed inclusive
    end_line: int
    kind: str          # "function" | "class" | "header" | "fixed"
    name: str          # symbol name, or "" for header/fixed
    content: str


def chunk_file(path: Path, *, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    """Read `path` and return its chunks. Empty list on read failure."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not text.strip():
        return []
    if path.suffix == ".py":
        chunks = _chunk_python(text, path)
        if not chunks:
            chunks = _chunk_fixed(text, path, max_chars=max_chars, overlap=overlap)
    else:
        chunks = _chunk_fixed(text, path, max_chars=max_chars, overlap=overlap)
    # A tree-sitter chunk for a long class can blow past the embedder's
    # context window. Split anything oversized into fixed sub-chunks while
    # keeping the parent's kind/name so retrieval still surfaces it as
    # "class:LLM" rather than "fixed".
    return [sc for c in chunks for sc in _cap(c, max_chars, overlap)]


def _cap(c: Chunk, max_chars: int, overlap: int) -> list[Chunk]:
    if len(c.content) <= max_chars:
        return [c]
    sub = _chunk_fixed(c.content, Path(c.path), max_chars=max_chars, overlap=overlap)
    offset = c.start_line - 1
    for s in sub:
        s.start_line += offset
        s.end_line += offset
        s.kind = c.kind
        s.name = c.name
    return sub


def _chunk_python(text: str, path: Path) -> list[Chunk]:
    try:
        import tree_sitter_python as ts_python
        from tree_sitter import Language, Parser
    except ImportError:
        return []
    parser = Parser(Language(ts_python.language()))
    tree = parser.parse(text.encode("utf-8"))

    top_defs = [
        c for c in tree.root_node.children
        if c.type in ("function_definition", "decorated_definition", "class_definition")
    ]

    chunks: list[Chunk] = []
    src_bytes = text.encode("utf-8")

    # Header chunk: everything before the first top-level def/class.
    header_end = top_defs[0].start_byte if top_defs else len(src_bytes)
    header = src_bytes[:header_end].decode("utf-8", errors="replace").rstrip()
    if header:
        chunks.append(Chunk(
            path=str(path),
            start_line=1,
            end_line=text[:header_end].count("\n") + 1,
            kind="header",
            name="",
            content=header,
        ))

    for node in top_defs:
        kind, name = _python_kind_and_name(node, src_bytes)
        snippet = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        chunks.append(Chunk(
            path=str(path),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            kind=kind,
            name=name,
            content=snippet,
        ))

    return chunks


def _python_kind_and_name(node, src_bytes: bytes) -> tuple[str, str]:
    target = node
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                target = child
                break
    name_node = target.child_by_field_name("name")
    name = ""
    if name_node is not None:
        name = src_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
    if target.type == "function_definition":
        return "function", name
    if target.type == "class_definition":
        return "class", name
    return "block", name


def _chunk_fixed(text: str, path: Path, *, max_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + max_chars, len(text))
        # Prefer breaking on a newline in the second half of the window.
        if end < len(text):
            last_nl = text.rfind("\n", cursor + max_chars // 2, end)
            if last_nl != -1:
                end = last_nl + 1
        snippet = text[cursor:end]
        start_line = text.count("\n", 0, cursor) + 1
        end_line = start_line + snippet.count("\n")
        chunks.append(Chunk(
            path=str(path),
            start_line=start_line,
            end_line=end_line,
            kind="fixed",
            name="",
            content=snippet,
        ))
        if end >= len(text):
            break
        # Step forward, with overlap, but never zero-progress.
        cursor = max(end - overlap, cursor + 1)
    return chunks
