from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Mapping, Sequence

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from .analysis import DocumentAnalysis, PageAnalysis, TextAvailability, analyze_pdf_document
from .layout import PageLayout, analyze_page_layout
from .ocr import DEFAULT_OCR_DPI, ocr_page_analysis
from .questions import QuestionAnalysis, analyze_questions
from .rectangles import (
    DEFAULT_RECTANGLE_PADDING,
    ProposedQuestionRectangle,
    RectangleGenerationResult,
    propose_question_rectangles,
)


class PageLayoutMode(str, Enum):
    AUTOMATIC = "automatic"
    SINGLE_COLUMN = "single_column"
    TWO_COLUMNS = "two_columns"


@dataclass(frozen=True)
class PageLayoutOptions:
    """Per-source-page layout instruction for automated analysis."""

    mode: PageLayoutMode = PageLayoutMode.AUTOMATIC
    divider_normalized: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PageLayoutMode):
            try:
                object.__setattr__(self, "mode", PageLayoutMode(self.mode))
            except ValueError as exc:
                raise ValueError(f"Unsupported page layout mode: {self.mode!r}.") from exc
        if self.divider_normalized is not None:
            value = float(self.divider_normalized)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError("Normalized divider must be finite and strictly between 0 and 1.")
            object.__setattr__(self, "divider_normalized", value)


@dataclass
class PageQuestionAnalysisResult:
    """All analysis stages and diagnostics for one original source page."""

    page_index: int
    embedded_text_availability: TextAvailability
    text_source: str
    page_analysis: PageAnalysis
    layout: PageLayout | None = None
    questions: QuestionAnalysis | None = None
    rectangles: RectangleGenerationResult | None = None
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def proposals(self) -> list[ProposedQuestionRectangle]:
        return self.rectangles.proposals if self.rectangles is not None else []

    @property
    def ambiguous_lines(self):
        return self.questions.ambiguous_lines if self.questions is not None else []

    @property
    def skipped_candidate_orders(self) -> list[int]:
        return self.rectangles.skipped_candidate_orders if self.rectangles is not None else []


@dataclass
class QuestionAnalysisRun:
    """Structured result for one selected page range."""

    document_path: str
    page_count: int
    selected_page_indices: tuple[int, ...]
    pages: list[PageQuestionAnalysisResult] = field(default_factory=list)
    document_error: str | None = None

    @property
    def proposals(self) -> list[ProposedQuestionRectangle]:
        return [proposal for page in self.pages for proposal in page.proposals]

    @property
    def proposal_count(self) -> int:
        return sum(len(page.proposals) for page in self.pages)

    @property
    def warnings(self) -> list[str]:
        return [warning for page in self.pages for warning in page.warnings]

    @property
    def failures(self) -> list[str]:
        return [failure for page in self.pages for failure in page.failures]


def analyze_selected_pages(
    pdf_path: str | Path,
    page_indices: Sequence[int],
    *,
    dpi: float = DEFAULT_OCR_DPI,
    padding: float = DEFAULT_RECTANGLE_PADDING,
    layout_options: Mapping[int, PageLayoutOptions] | None = None,
    force_ocr: bool = False,
) -> QuestionAnalysisRun:
    """Run the existing Tasks 1–5 pipeline for selected source-page indices.

    Page indices are zero-based and refer to the original PDF. Embedded text is
    reuse when useful; otherwise local OCR is attempted for that selected page.
    When ``force_ocr`` is true, OCR is run for every selected page regardless of
    embedded-text availability. An OCR or later page-stage failure is recorded
    on that page and does not stop processing of the remaining selected pages.
    """

    path = str(pdf_path)
    document: DocumentAnalysis = analyze_pdf_document(path, page_indices=page_indices)
    run = QuestionAnalysisRun(
        document_path=path,
        page_count=document.page_count,
        selected_page_indices=tuple(page.page_index for page in document.pages),
    )
    if not document.valid:
        run.document_error = document.error or "PDF analysis failed."
        return run

    for embedded_page in document.pages:
        page_result = PageQuestionAnalysisResult(
            page_index=embedded_page.page_index,
            embedded_text_availability=embedded_page.availability,
            text_source="embedded text",
            page_analysis=embedded_page,
        )
        active_page = embedded_page

        if force_ocr:
            page_result.text_source = "OCR"
        if force_ocr or embedded_page.availability != TextAvailability.USEFUL:
            try:
                active_page = ocr_page_analysis(path, embedded_page.page_index, dpi=dpi)
                page_result.page_analysis = active_page
                page_result.text_source = "OCR"
            except Exception as exc:
                failure = f"OCR failed for page {embedded_page.page_index + 1}: {type(exc).__name__}: {exc}"
                page_result.failures.append(failure)
                if force_ocr:
                    run.pages.append(page_result)
                    continue
                page_result.warnings.append(f"{failure} Embedded text, if any, was retained.")

        try:
            options = (layout_options or {}).get(embedded_page.page_index, PageLayoutOptions())
            if options.mode == PageLayoutMode.AUTOMATIC:
                layout = analyze_page_layout(active_page)
            elif options.mode == PageLayoutMode.SINGLE_COLUMN:
                layout = analyze_page_layout(active_page, force_single_column=True)
            elif options.mode == PageLayoutMode.TWO_COLUMNS:
                if options.divider_normalized is None:
                    raise ValueError("Two-column mode requires a manually positioned divider.")
                layout = analyze_page_layout(
                    active_page,
                    divider=options.divider_normalized,
                    divider_is_normalized=True,
                )
            else:  # pragma: no cover - enum values are validated by PageLayoutOptions
                raise ValueError(f"Unsupported page layout mode: {options.mode!r}.")
            page_result.layout = layout
            page_result.warnings.extend(layout.warnings)
        except Exception as exc:
            page_result.failures.append(
                f"Layout analysis failed for page {embedded_page.page_index + 1}: {type(exc).__name__}: {exc}"
            )
            run.pages.append(page_result)
            continue

        try:
            questions = analyze_questions(layout)
            page_result.questions = questions
            page_result.warnings.extend(
                candidate_warning
                for candidate in questions.candidates
                for candidate_warning in candidate.warnings
            )
        except Exception as exc:
            page_result.failures.append(
                f"Question grouping failed for page {embedded_page.page_index + 1}: {type(exc).__name__}: {exc}"
            )
            run.pages.append(page_result)
            continue

        try:
            page_result.rectangles = propose_question_rectangles(
                questions,
                active_page.bounds,
                layout=layout,
                padding=padding,
            )
            page_result.warnings.extend(page_result.rectangles.warnings)
        except Exception as exc:
            page_result.failures.append(
                f"Rectangle generation failed for page {embedded_page.page_index + 1}: {type(exc).__name__}: {exc}"
            )

        run.pages.append(page_result)

    return run