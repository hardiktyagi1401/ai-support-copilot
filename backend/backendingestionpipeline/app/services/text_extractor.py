"""
Text extraction from uploaded documents.

Architecture note:
    We use PyMuPDF (imported as `fitz`) for PDF extraction. Alternatives:
    - pdfplumber: better for tables, slower, more memory
    - pypdf: pure Python, misses some formatting, slower
    - pdfminer: powerful but complex API
    - Amazon Textract / Google Document AI: accurate for scanned docs, costs money

    PyMuPDF wins for our use case: fast, handles text-based PDFs well,
    preserves page structure, actively maintained.

    What we track per page:
    - Page number: enables "this answer is from page 4" citations
    - The text blocks in reading order: top-to-bottom, left-to-right

What we DON'T handle here (production extensions):
    - Scanned PDFs (images of text): needs OCR (Tesseract or cloud API)
    - Password-protected PDFs: need to decrypt first
    - PDFs with tables: PyMuPDF misses table structure — pdfplumber is better
    - Word documents (.docx): use python-docx
    - Presentations (.pptx): use python-pptx

Failure modes:
    - Encrypted PDF: fitz raises a PermissionError — catch and return FAILED
    - Corrupt PDF: fitz raises a RuntimeError — catch and return FAILED
    - PDF with only images (scanned): returns empty text per page — we detect
      this via a minimum character count check
    - Encoding issues: PyMuPDF handles this internally, but some CJK PDFs fail

Production consideration:
    For very large PDFs (200+ pages), extraction can take 10-30 seconds.
    Consider streaming progress updates to the client, or breaking the file
    into page-range batches processed in parallel.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.core.app_logging import get_logger

logger = get_logger(__name__)

# Minimum characters per page to consider it "has text" (not a scanned image)
MIN_CHARS_PER_PAGE = 50


@dataclass
class ExtractedPage:
    """Text content from a single page, with its page number for citations."""
    page_number: int  # 1-indexed (human-readable page number)
    text: str
    char_count: int


@dataclass
class ExtractionResult:
    """Full extraction result from a document."""
    pages: list[ExtractedPage]
    total_chars: int
    scanned_page_count: int  # pages with < MIN_CHARS_PER_PAGE characters
    mime_type: str


def extract_text(file_path: Path, mime_type: str) -> ExtractionResult:
    """
    Extract text from a document, routing by MIME type.

    Returns an ExtractionResult with per-page text and metadata.
    Raises ValueError for unsupported MIME types.
    Raises RuntimeError for corrupt or unreadable files.
    """
    if mime_type == "application/pdf":
        return _extract_pdf(file_path)
    elif mime_type in ("text/plain", "text/markdown"):
        return _extract_plaintext(file_path)
    else:
        raise ValueError(f"Unsupported MIME type for extraction: {mime_type}")


def _extract_pdf(file_path: Path) -> ExtractionResult:
    """
    Extract text from a PDF, preserving page boundaries.

    We extract text in "block" order (top-to-bottom reading order) rather
    than raw character order, which respects multi-column layouts better.
    """
    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc

    pages: list[ExtractedPage] = []
    scanned_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # "blocks" sorts text into reading order within the page
        blocks = page.get_text("blocks", sort=True)
        page_text = "\n".join(
            block[4]  # index 4 is the text content in the blocks tuple
            for block in blocks
            if block[6] == 0  # block type 0 = text (type 1 = image)
        ).strip()

        char_count = len(page_text)
        if char_count < MIN_CHARS_PER_PAGE:
            scanned_count += 1
            logger.warning(
                "low_text_page",
                page_number=page_num + 1,
                char_count=char_count,
                file_path=str(file_path),
                hint="Page may be a scanned image — consider OCR",
            )

        pages.append(ExtractedPage(
            page_number=page_num + 1,  # convert to 1-indexed
            text=page_text,
            char_count=char_count,
        ))

    doc.close()

    total_chars = sum(p.char_count for p in pages)
    logger.info(
        "pdf_extracted",
        page_count=len(pages),
        total_chars=total_chars,
        scanned_pages=scanned_count,
    )

    return ExtractionResult(
        pages=pages,
        total_chars=total_chars,
        scanned_page_count=scanned_count,
        mime_type="application/pdf",
    )


def _extract_plaintext(file_path: Path) -> ExtractionResult:
    """
    Extract text from plain text files.
    Treated as a single "page" since there's no page structure.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"Failed to read text file: {exc}") from exc

    page = ExtractedPage(page_number=1, text=text.strip(), char_count=len(text))
    logger.info("plaintext_extracted", char_count=len(text))

    return ExtractionResult(
        pages=[page],
        total_chars=len(text),
        scanned_page_count=0,
        mime_type="text/plain",
    )
