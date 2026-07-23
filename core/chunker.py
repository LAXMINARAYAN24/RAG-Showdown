# core/chunker.py
import re
from pypdf import PdfReader

# Boilerplate that repeats on nearly every page and dilutes embeddings:
# copyright footers and standalone page numbers.
_BOILERPLATE = re.compile(r"©\s*Confluent\s+Inc\.\s*\d{4}|\bPage\s+\d+\b", re.IGNORECASE)


def chunk_words(words, chunk_size=200, overlap=50):
    """Split a word list into overlapping windows.

    Overlap keeps sentences that straddle a boundary fully present in at
    least one chunk, so a passage can't be lost to an unlucky cut.
    """
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks


def _is_toc_page(text):
    """Detect a table-of-contents page: a 'Contents' header plus many lines
    that are just numbers/dots (page listings)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    has_contents_header = any(ln.lower() in ("contents", "table of contents") for ln in lines)
    numberish = sum(1 for ln in lines if re.fullmatch(r"[\d\s.]+", ln))
    return has_contents_header and numberish >= len(lines) * 0.3


def load_and_chunk(pdf_path, chunk_size=200, overlap=50):
    """Read a PDF, return overlapping text chunks with boilerplate removed.

    Compared to the original (flat 500-word windows over the concatenated
    document), this skips table-of-contents pages, strips copyright footers
    and page numbers, and uses smaller overlapping windows — so prose
    passages get their own focused chunks instead of being diluted by
    title/TOC text.
    """
    reader = PdfReader(pdf_path)

    page_texts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if _is_toc_page(text):
            continue
        text = _BOILERPLATE.sub(" ", text)
        page_texts.append(text)

    words = "\n".join(page_texts).split()
    return chunk_words(words, chunk_size=chunk_size, overlap=overlap)
