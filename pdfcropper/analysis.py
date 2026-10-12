from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


class TextAvailability(str, Enum):
    """Simple classification for how useful the embedded text layer is."""

    NO_TEXT = "no_text"
    LOW_TEXT = "low_text"
    USEFUL = "useful"


@dataclass(frozen=True)
class ExtractedWord:
    """One word from the PDF text layer, expressed in page coordinates.

    Coordinate convention: every bounding box is in the same coordinate system as
    the source PDF page's current rotation state. In other words, values are the
    raw coordinates returned by PyMuPDF for that page, not converted to an image
    pixel grid. This keeps the geometry aligned with the rest of the application,
    which already uses the page's `fitz.Rect` values directly.
    """

    text: str
    rect: fitz.Rect
    page_index: int
    block_no: int = 0
    line_no: int = 0
    word_no: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rect", fitz.Rect(self.rect))


@dataclass
class PageAnalysis:
    """Per-page analysis produced from the PDF text layer."""

    page_index: int
    width: float
    height: float
    rotation: int = 0
    bounds: fitz.Rect = field(default_factory=lambda: fitz.Rect(0, 0, 0, 0))
    text: str = ""
    text_blocks: list[dict[str, Any]] = field(default_factory=list)
    words: list[ExtractedWord] = field(default_factory=list)
    availability: TextAvailability = TextAvailability.NO_TEXT
    has_usable_text: bool = False


@dataclass
class DocumentAnalysis:
    """Document-level PDF analysis result."""

    path: str | None = None
    page_count: int = 0
    pages: list[PageAnalysis] = field(default_factory=list)
    valid: bool = True
    error: str | None = None

    @property
    def has_usable_pages(self) -> bool:
        return any(page.has_usable_text for page in self.pages)


def page_indices_for_range(first_page: int, last_page: int, page_count: int) -> range:
    """Validate an inclusive 1-based page range and return zero-based indices."""

    values = (first_page, last_page, page_count)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("Page range values and page count must be integers.")
    if page_count < 1:
        raise ValueError("Page count must be at least 1.")
    if first_page < 1 or last_page < 1:
        raise ValueError("Page numbers must be at least 1.")
    if first_page > page_count or last_page > page_count:
        raise ValueError(f"Page numbers must not exceed the document page count ({page_count}).")
    if first_page > last_page:
        raise ValueError("First page must not be greater than last page.")
    return range(first_page - 1, last_page)


def _validate_page_indices(page_indices: Sequence[int], page_count: int) -> tuple[int, ...]:
    try:
        indices = tuple(page_indices)
    except TypeError as exc:
        raise ValueError("Selected page indices must be an iterable of integers.") from exc
    if not indices:
        raise ValueError("At least one page index must be selected.")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("Selected page indices must be integers.")
    if any(index < 0 or index >= page_count for index in indices):
        raise ValueError(f"Selected page indices must be between 0 and {page_count - 1}.")
    if len(set(indices)) != len(indices):
        raise ValueError("Selected page indices must not contain duplicates.")
    return tuple(sorted(indices))


def analyze_pdf_document(
    pdf_path: str | Path,
    *,
    page_indices: Sequence[int] | None = None,
) -> DocumentAnalysis:
    """Inspect all pages, or selected original zero-based pages, for embedded text.

    The routine intentionally keeps the extraction logic separate from the GUI so it
    can be used by tests and future OCR/analysis workflows without opening a
    window. ``page_count`` always reports the full source document count; when
    ``page_indices`` is supplied, ``pages`` contains only those pages in source order.
    """

    path = str(pdf_path)
    try:
        document = fitz.open(path)
    except (fitz.FileDataError, RuntimeError, ValueError, OSError) as exc:
        return DocumentAnalysis(path=path, page_count=0, pages=[], valid=False, error=f"Could not open PDF: {exc}")

    try:
        if not document.is_pdf:
            return DocumentAnalysis(path=path, page_count=0, pages=[], valid=False, error="File is not a valid PDF document.")

        source_page_count = document.page_count
        selected_indices = (
            tuple(range(source_page_count))
            if page_indices is None
            else _validate_page_indices(page_indices, source_page_count)
        )
        pages: list[PageAnalysis] = []
        for page_index in selected_indices:
            page = document[page_index]
            page_word_boxes = _extract_words(page, page_index)
            blocks = _extract_blocks(page)
            text = page.get_text("text").strip()
            availability = _classify_page_text(page_word_boxes)
            page_analysis = PageAnalysis(
                page_index=page_index,
                width=float(page.rect.width),
                height=float(page.rect.height),
                rotation=int(page.rotation),
                bounds=fitz.Rect(page.rect),
                text=text,
                text_blocks=blocks,
                words=page_word_boxes,
                availability=availability,
                has_usable_text=availability == TextAvailability.USEFUL,
            )
            pages.append(page_analysis)

        return DocumentAnalysis(path=path, page_count=source_page_count, pages=pages, valid=True)
    finally:
        document.close()


def _extract_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    """Return a normalized list of page blocks and their line content."""

    blocks: list[dict[str, Any]] = []
    raw_blocks = page.get_text("dict").get("blocks", [])
    for block_index, block in enumerate(raw_blocks):
        block_bbox = fitz.Rect(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        lines: list[dict[str, Any]] = []
        for line in block.get("lines", []):
            spans: list[dict[str, Any]] = []
            line_bbox = fitz.Rect(line.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            for span in line.get("spans", []):
                span_text = str(span.get("text", "")).strip()
                if not span_text:
                    continue
                spans.append({
                    "text": span_text,
                    "bbox": fitz.Rect(span.get("bbox", (0.0, 0.0, 0.0, 0.0))),
                    "size": span.get("size"),
                    "flags": span.get("flags"),
                })
            if not spans and not line_bbox.is_empty:
                lines.append({"bbox": line_bbox, "spans": []})
            elif spans:
                lines.append({"bbox": line_bbox, "spans": spans})
        text = "\n".join(span["text"] for line in lines for span in line.get("spans", []))
        if not text and block_bbox.is_empty:
            continue
        blocks.append({
            "index": block_index,
            "bbox": block_bbox,
            "text": text,
            "lines": lines,
        })
    return blocks


def _extract_words(page: fitz.Page, page_index: int) -> list[ExtractedWord]:
    """Build a word list and bounding boxes from PyMuPDF's structured text output."""

    words: list[ExtractedWord] = []
    raw_words = page.get_text("words")
    for word_index, item in enumerate(raw_words):
        if len(item) < 5:
            continue
        x0, y0, x1, y1, text, *rest = item
        text_value = str(text).strip()
        if not text_value:
            continue
        rect = fitz.Rect(float(x0), float(y0), float(x1), float(y1))
        if not _is_finite_rect(rect):
            continue
        block_no = int(rest[0]) if len(rest) > 0 else 0
        line_no = int(rest[1]) if len(rest) > 1 else 0
        word_no = int(rest[2]) if len(rest) > 2 else word_index
        words.append(
            ExtractedWord(
                text=text_value,
                rect=rect,
                page_index=page_index,
                block_no=block_no,
                line_no=line_no,
                word_no=word_no,
                metadata={"source": "embedded_pdf_text"},
            )
        )
    return words


def _classify_page_text(words: list[ExtractedWord]) -> TextAvailability:
    """Determine whether the embedded text is useful for automated processing.

    The heuristic intentionally uses simple, explainable checks rather than a
    single arbitrary character count: pages with no words are treated as empty,
    pages with a handful of words or very short text are treated as low-value,
    and pages with multiple words spanning meaningful text are treated as useful.
    """

    if not words:
        return TextAvailability.NO_TEXT

    total_text = " ".join(word.text for word in words)
    non_empty_words = [word for word in words if word.text.strip()]
    if len(non_empty_words) == 0:
        return TextAvailability.NO_TEXT
    if len(non_empty_words) < 3 and len(total_text.strip()) < 20:
        return TextAvailability.LOW_TEXT
    if len(non_empty_words) >= 3 or len(total_text.strip()) >= 20:
        return TextAvailability.USEFUL
    return TextAvailability.LOW_TEXT


def _is_finite_rect(rect: fitz.Rect) -> bool:
    numbers = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    return all(math.isfinite(value) for value in numbers)
