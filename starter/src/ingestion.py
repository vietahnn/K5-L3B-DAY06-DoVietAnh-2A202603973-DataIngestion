"""Read and normalize source data while preserving physical-page traceability."""
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract every physical page and retain empty-page diagnostics in metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    current_section = "unclassified"
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        raw_text = pdf_page.extract_text() or ""
        text = normalize_text(raw_text)
        detected = _detect_section(text)
        if detected:
            current_section = detected
        metadata: dict[str, Any] = {
            "section": current_section,
            "document_type": "poverty_prosperity_planet_overview",
            "language": "en",
            "extraction_status": "ok" if text else "empty",
        }
        pages.append(DocumentPage(document_id, source, page_number, text, metadata))
    if not pages:
        raise ValueError(f"PDF contains no pages: {path}")
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode, line endings and whitespace without flattening paragraphs."""
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00ad", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    output: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        output.append(line)
        previous_blank = blank
    return "\n".join(output).strip()


def _detect_section(text: str) -> Optional[str]:
    """Use only heading-like text from the page; never invent a source value."""
    for line in text.splitlines()[:12]:
        candidate = line.strip(" :-")
        if not candidate or len(candidate) > 100 or len(candidate.split()) > 12:
            continue
        if candidate.isupper() and any(char.isalpha() for char in candidate):
            return candidate.lower()
        if re.match(r"^(chapter|part|section|annex|appendix)\b", candidate, re.I):
            return candidate.lower()
    return None
