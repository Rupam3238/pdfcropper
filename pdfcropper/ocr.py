from __future__ import annotations

import csv
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

from .analysis import ExtractedWord, PageAnalysis, TextAvailability, _classify_page_text

DEFAULT_OCR_DPI = 300.0


def _validate_dpi(dpi: float) -> float:
    """Validate OCR DPI and return a positive finite value."""

    value = float(dpi)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"OCR DPI must be a positive finite number; got {dpi!r}.")
    return value


def _render_page_for_ocr(page: fitz.Page, dpi: float = DEFAULT_OCR_DPI):
    """Render a single page to an image for OCR.

    The conversion is always done from page points to rendered image pixels, using
    the page's current rotation. That keeps the OCR output in the same coordinate
    convention described in the Task 1 analysis layer: page-space coordinates, not
    an unrelated image coordinate system.
    """

    validated_dpi = _validate_dpi(dpi)
    scale = validated_dpi / 72.0
    return page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)


def _image_to_page_rect(left: float, top: float, right: float, bottom: float, scale: float) -> fitz.Rect:
    """Convert OCR image bounding boxes to the page-coordinate system used by Task 1.

    Tesseract reports regions in rendered-image pixels. PyMuPDF returns page-space
    coordinates in points. Since the rendered page is generated from the page's own
    coordinate system at a known scale factor, the conversion is a single linear
    scale across x and y. We intentionally do not re-rotate coordinates here;
    that would drift away from the page's own rotated coordinate convention.
    """

    if scale <= 0 or not math.isfinite(scale):
        raise ValueError(f"OCR scale must be positive and finite; got {scale!r}.")
    x0 = float(left) / scale
    y0 = float(top) / scale
    x1 = float(right) / scale
    y1 = float(bottom) / scale
    rect = fitz.Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if not all(math.isfinite(value) for value in (rect.x0, rect.y0, rect.x1, rect.y1)):
        raise ValueError("OCR coordinates produced a non-finite bounding box.")
    return rect


def _parse_confidence(value: Any) -> float | None:
    """Parse a confidence value from Tesseract or return None when no meaningful confidence is available."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        try:
            numeric = float(str(value).strip())
        except ValueError:
            return None
    if not math.isfinite(numeric) or numeric < 0.0:
        return None
    return numeric


def _parse_tsv_words(tsv_text: str, page_index: int, render_scale: float) -> list[ExtractedWord]:
    """Parse Tesseract TSV output into Task 1 word objects."""

    lines = [line for line in (tsv_text or "").splitlines() if line.strip()]
    if not lines:
        return []

    try:
        reader = csv.DictReader(lines, delimiter="\t")
    except csv.Error:
        return []

    words: list[ExtractedWord] = []
    for word_no, row in enumerate(reader):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        try:
            left = float(row.get("left", 0.0) or 0.0)
            top = float(row.get("top", 0.0) or 0.0)
            width = float(row.get("width", 0.0) or 0.0)
            height = float(row.get("height", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue

        confidence = _parse_confidence(row.get("conf"))
        block_no = _safe_int(row.get("block_num"), default=0)
        line_no = _safe_int(row.get("line_num"), default=0)
        word_num = _safe_int(row.get("word_num"), default=word_no)

        rect = _image_to_page_rect(left, top, left + width, top + height, render_scale)
        words.append(
            ExtractedWord(
                text=text,
                rect=rect,
                page_index=page_index,
                block_no=block_no,
                line_no=line_no,
                word_no=word_num,
                metadata={"source": "ocr", "confidence": confidence},
            )
        )
    return words


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _run_tesseract_ocr(image_path: str | Path) -> str:
    """Run Tesseract locally and return OCR TSV output."""

    exe = shutil.which("tesseract")
    if not exe:
        raise RuntimeError(
            "Tesseract is not installed or not on PATH. Install Tesseract locally and ensure the 'tesseract' executable is available before using OCR."
        )

    cmd = [exe, str(image_path), "stdout", "--psm", "6", "tsv"]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Failed to execute Tesseract because the binary was not found.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        message = stderr or exc.stdout or "Tesseract execution failed."
        raise RuntimeError(f"Tesseract OCR failed: {message}") from exc

    output = completed.stdout or ""
    if output.strip() == "":
        return ""
    return output


def ocr_page_words(
    pdf_path: str | Path | fitz.Page,
    page_index: int | None = None,
    *,
    dpi: float = DEFAULT_OCR_DPI,
) -> list[ExtractedWord]:
    """OCR a single PDF page and return word-level results in Task 1 page coordinates.

    Parameters
    ----------
    pdf_path:
        A PDF file path or an already-open PyMuPDF page. When a page object is
        supplied, `page_index` is optional and only used for metadata.
    page_index:
        The page index to OCR when a PDF path is passed.
    dpi:
        Render DPI for OCR. The value is expressed in document points-per-inch and
        is converted through the standard 72-point-per-inch PDF coordinate system.
    """

    document = None
    page = None
    target_index = 0

    if isinstance(pdf_path, fitz.Page):
        page = pdf_path
        target_index = page_index if page_index is not None else 0
    else:
        path = str(pdf_path)
        try:
            document = fitz.open(path)
        except (fitz.FileDataError, OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid or unreadable PDF: {path}") from exc
        try:
            if not document.is_pdf:
                raise ValueError(f"Invalid or unreadable PDF: {path}")
            if page_index is None:
                raise ValueError("A page index is required when OCR is run on a PDF path.")
            if page_index < 0 or page_index >= document.page_count:
                raise IndexError(f"Page index {page_index} is out of range for a {document.page_count}-page document.")
            page = document[page_index]
            target_index = page_index
        except Exception:
            if document is not None:
                document.close()
            raise

    try:
        render_scale = _validate_dpi(dpi) / 72.0
        pixmap = _render_page_for_ocr(page, dpi=dpi)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            pixmap.save(str(temp_path))
            tsv_text = _run_tesseract_ocr(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return _parse_tsv_words(tsv_text, target_index, render_scale)
    finally:
        if document is not None:
            document.close()


def ocr_page_analysis(
    pdf_path: str | Path | fitz.Page,
    page_index: int | None = None,
    *,
    dpi: float = DEFAULT_OCR_DPI,
) -> PageAnalysis:
    """Return a Task 1-compatible page analysis that includes OCR words."""

    if isinstance(pdf_path, fitz.Page):
        page = pdf_path
        resolved_index = page_index if page_index is not None else 0
        width = float(page.rect.width)
        height = float(page.rect.height)
        rotation = int(page.rotation)
        bounds = fitz.Rect(page.rect)
        words = ocr_page_words(page, resolved_index, dpi=dpi)
    else:
        value_path = str(pdf_path)
        try:
            document = fitz.open(value_path)
        except (fitz.FileDataError, OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid or unreadable PDF: {value_path}") from exc
        try:
            if not document.is_pdf:
                raise ValueError(f"Invalid or unreadable PDF: {value_path}")
            if page_index is None:
                raise ValueError("A page index is required when OCR is run on a PDF path.")
            if page_index < 0 or page_index >= document.page_count:
                raise IndexError(f"Page index {page_index} is out of range for a {document.page_count}-page document.")
            page = document[page_index]
            resolved_index = page_index
            width = float(page.rect.width)
            height = float(page.rect.height)
            rotation = int(page.rotation)
            bounds = fitz.Rect(page.rect)
        finally:
            document.close()
        words = ocr_page_words(value_path, resolved_index, dpi=dpi)

    text = " ".join(word.text for word in words)
    availability = _classify_page_text(words)
    return PageAnalysis(
        page_index=resolved_index,
        width=width,
        height=height,
        rotation=rotation,
        bounds=bounds,
        text=text,
        text_blocks=[],
        words=words,
        availability=availability,
        has_usable_text=availability == TextAvailability.USEFUL,
    )
