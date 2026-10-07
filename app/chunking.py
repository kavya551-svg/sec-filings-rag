"""Section-aware chunking for 10-K text.

10-Ks are organized into "Items" (Item 1. Business, Item 1A. Risk Factors, Item 7. MD&A, ...).
Chunks never cross an Item boundary, and each chunk records its section so answers can
cite where in the filing a fact came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)
MAX_HEADING_CHARS = 150


@dataclass
class Chunk:
    id: int
    section: str
    text: str


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split filing text into (section name, section text) pairs using Item headings."""
    sections: list[tuple[str, list[str]]] = [("Cover page", [])]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = ITEM_RE.match(line)
        if match and len(line) <= MAX_HEADING_CHARS:
            number, title = match.group(1).upper(), match.group(2).strip(" .")
            name = f"Item {number}. {title}" if title else f"Item {number}"
            sections.append((name, []))
        else:
            sections[-1][1].append(line)
    return [(name, " ".join(lines)) for name, lines in sections if lines]


def chunk_text(text: str, max_words: int = 300, overlap: int = 50, min_words: int = 40) -> list[Chunk]:
    """Split each section into overlapping word windows of at most max_words."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    chunks: list[Chunk] = []
    step = max_words - overlap
    for section, body in split_sections(text):
        words = body.split()
        if len(words) < min_words:  # skip table-of-contents stubs and empty headings
            continue
        for start in range(0, len(words), step):
            window = words[start : start + max_words]
            if start > 0 and len(window) < min_words:
                break
            chunks.append(Chunk(id=len(chunks), section=section, text=" ".join(window)))
            if start + max_words >= len(words):
                break
    return chunks
