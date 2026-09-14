"""Section-aware chunking.

Why chunk at all? A 7B model has a finite context and -- more importantly --
attention degrades over long inputs. Feeding a 6-page academic CV as one blob
makes qwen2.5-coder lose the middle. Instead we:

1. Split on resume section headings (Experience / Education / Skills / ...).
2. Split any section that is still too long into overlapping character
   windows (overlap keeps a bullet that straddles a boundary intact).
3. Rank chunks by *deterministic* JD-keyword density and only send the top
   `--max-chunks` to the LLM (usually 3-4). This is the biggest single
   speed win: the model never reads the "References" section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import ChunkConfig
from .textutils import clean_lines, detect_section


@dataclass
class Chunk:
    index: int
    section: str
    text: str
    keyword_hits: int = 0
    rel_score: float = 0.0

    def as_llm_block(self) -> str:
        return f"[chunk {self.index} | {self.section}]\n{self.text}"


@dataclass
class ChunkedDoc:
    chunks: list[Chunk] = field(default_factory=list)
    sections_found: list[str] = field(default_factory=list)
    truncated: bool = False


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_name, section_text), ...] preserving order."""
    lines = clean_lines(text)
    sections: list[tuple[str, list[str]]] = []
    current = "header"
    buf: list[str] = []
    for ln in lines:
        sec = detect_section(ln)
        if sec and sec != "other":
            if buf:
                sections.append((current, buf))
            current, buf = sec, []
            continue
        buf.append(ln)
    if buf:
        sections.append((current, buf))
    return [(name, "\n".join(body)) for name, body in sections if "\n".join(body).strip()]


def _window(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out, step, i = [], max(1, size - overlap), 0
    while i < len(text):
        piece = text[i : i + size].strip()
        if piece:
            out.append(piece)
        if i + size >= len(text):
            break
        i += step
    return out


def chunk_document(text: str, cfg: ChunkConfig) -> ChunkedDoc:
    doc = ChunkedDoc()
    idx = 0
    for sec_name, body in split_into_sections(text):
        doc.sections_found.append(sec_name)
        for piece in _window(body, cfg.max_chars, cfg.overlap_chars):
            piece = piece.strip()
            if len(piece) < cfg.min_chunk_chars:
                continue
            doc.chunks.append(Chunk(index=idx, section=sec_name, text=piece))
            idx += 1
    if not doc.chunks:
        # Degenerate text (e.g. brutal OCR): fall back to raw windows.
        for piece in _window(text.strip(), cfg.max_chars, cfg.overlap_chars):
            doc.chunks.append(Chunk(index=idx, section="body", text=piece))
            idx += 1
    return doc


def rank_chunks(doc: ChunkedDoc, keywords: list[str], cfg: ChunkConfig) -> list[Chunk]:
    """Order chunks by JD relevance so the LLM only sees the useful ones.

    `keywords` should already be squashed (lowercase alnum only).
    """
    for ch in doc.chunks:
        sq = re.sub(r"[^a-z0-9]+", "", ch.text.lower())
        hits = sum(1 for k in keywords if k and k in sq)
        ch.keyword_hits = hits
        # Density beats raw count: a short Skills block with 6 hits is gold,
        # a 2400-char Experience wall with 6 hits is less so.
        density = hits / max(1.0, len(ch.text) / 500.0)
        boost = 1.25 if ch.section in ("skills", "header", "summary") else 1.0
        if ch.section == "projects":
            boost = 1.1
        ch.rel_score = density * boost
    ranked = sorted(doc.chunks, key=lambda c: (-c.rel_score, c.index))
    keep = ranked[: cfg.max_chunks_to_llm]
    keep.sort(key=lambda c: c.index)  # present in reading order
    doc.truncated = len(ranked) > len(keep)
    return keep
