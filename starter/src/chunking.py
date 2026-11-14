"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into deterministic character windows."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    text = page.text
    if not text:
        return []
    chunks: list[Chunk] = []
    step = size - overlap
    for index, start in enumerate(range(0, len(text), step)):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(_make_chunk(page, piece, "fixed", index))
        if start + size >= len(text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group complete paragraphs, splitting only paragraphs exceeding max_size."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    paragraphs = [part.strip() for part in page.text.split("\n\n") if part.strip()]
    chunks: list[Chunk] = []
    current = ""
    index = 0
    for paragraph in paragraphs:
        pieces = _split_oversized(paragraph, max_size)
        for piece in pieces:
            if current and len(current) + 2 + len(piece) > max_size:
                chunks.append(_make_chunk(page, current, "paragraph", index))
                index += 1
                current = ""
            current = piece if not current else f"{current}\n\n{piece}"
    if current:
        chunks.append(_make_chunk(page, current, "paragraph", index))
    return chunks


def _split_oversized(text: str, max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]
    return [text[start:start + max_size].strip()
            for start in range(0, len(text), max_size)
            if text[start:start + max_size].strip()]


def _make_chunk(page: DocumentPage, text: str, strategy: str, index: int) -> Chunk:
    chunk_id = f"{page.document_id}:{page.page}:{strategy}:{index}"
    return Chunk(chunk_id, page.document_id, page.source, page.page, text,
                 {**page.metadata, "chunk_strategy": strategy})
