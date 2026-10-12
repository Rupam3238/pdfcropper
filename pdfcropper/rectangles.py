from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from .layout import PageLayout
from .questions import QuestionAnalysis, QuestionCandidate

DEFAULT_RECTANGLE_PADDING = 6.0


@dataclass
class ProposedQuestionRectangle:
    """A reviewable page-space rectangle proposed for one question candidate."""

    page_index: int
    order: int
    label: str | None
    region_index: int
    region_name: str
    rect: fitz.Rect
    candidate_id: str
    source_candidate: QuestionCandidate
    confidence: float
    warnings: list[str] = field(default_factory=list)
    boundary_adjusted: bool = False
    fallback_applied: bool = False


@dataclass
class RectangleGenerationResult:
    """Proposals and diagnostics produced for one page."""

    page_index: int
    proposals: list[ProposedQuestionRectangle] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_candidate_orders: list[int] = field(default_factory=list)


def propose_question_rectangles(
    analysis: QuestionAnalysis,
    page_bounds: fitz.Rect,
    *,
    layout: PageLayout | None = None,
    padding: float = DEFAULT_RECTANGLE_PADDING,
) -> RectangleGenerationResult:
    """Create padded, clipped rectangle proposals for one page of candidates.

    Padding is expressed in PDF page points. Candidate word bounds are the only
    content geometry used; no diagrams or blank-space extents are inferred. If a
    matching split layout is supplied, left/right regions are clipped at its
    exact divider. Without a divider, their region content extent is used as a
    conservative horizontal limit because Task 3 does not expose a column model
    independent of its word bounds.
    """

    pad = _validate_padding(padding)
    page_rect = _validated_page_rect(page_bounds)
    result = RectangleGenerationResult(page_index=analysis.page_index)
    if layout is not None and layout.page_index != analysis.page_index:
        result.warnings.append(
            f"Layout page index {layout.page_index} does not match question analysis page index {analysis.page_index}; region clipping was not applied."
        )
        layout = None

    candidates = sorted(enumerate(analysis.candidates), key=lambda pair: (pair[1].order, pair[0]))
    for _, candidate in candidates:
        candidate_id = _candidate_id(candidate)
        candidate_rect = _validated_candidate_rect(candidate)
        if candidate_rect is None:
            result.skipped_candidate_orders.append(candidate.order)
            result.warnings.append(f"Skipped {candidate_id}: candidate bounds are empty, non-finite, or invalid.")
            continue

        word_rects = _usable_word_rects(candidate)
        if not word_rects:
            result.skipped_candidate_orders.append(candidate.order)
            result.warnings.append(f"Skipped {candidate_id}: candidate has no usable word geometry.")
            continue

        content_rect = _union_rect(word_rects)
        if content_rect is None:
            result.skipped_candidate_orders.append(candidate.order)
            result.warnings.append(f"Skipped {candidate_id}: usable word bounds could not be combined.")
            continue

        warnings = list(candidate.warnings)
        if len(word_rects) < len(candidate.words):
            warnings.append("Some words had invalid geometry and were excluded from the proposal bounds.")

        clip_rect, region_warning, fallback_applied = _region_clip_rect(layout, candidate, page_rect)
        if region_warning:
            warnings.append(region_warning)

        expanded_rect = fitz.Rect(
            content_rect.x0 - pad,
            content_rect.y0 - pad,
            content_rect.x1 + pad,
            content_rect.y1 + pad,
        )
        clipped_rect = _intersect_rect(expanded_rect, page_rect)
        if clipped_rect is not None and clip_rect != page_rect:
            clipped_rect = _intersect_rect(clipped_rect, clip_rect)

        if clipped_rect is None or clipped_rect.width <= 0 or clipped_rect.height <= 0:
            result.skipped_candidate_orders.append(candidate.order)
            result.warnings.append(f"Skipped {candidate_id}: clipping left no positive-area rectangle.")
            continue

        _normalize_rect(clipped_rect)
        _normalize_rect(expanded_rect)
        adjusted = not _rect_equal(clipped_rect, expanded_rect)
        if adjusted:
            warnings.append("Padding or content bounds were clipped to the page or region boundary.")

        result.proposals.append(
            ProposedQuestionRectangle(
                page_index=analysis.page_index,
                order=candidate.order,
                label=candidate.label,
                region_index=candidate.region_index,
                region_name=candidate.region_name,
                rect=clipped_rect,
                candidate_id=candidate_id,
                source_candidate=candidate,
                confidence=candidate.detection_confidence,
                warnings=warnings,
                boundary_adjusted=adjusted,
                fallback_applied=fallback_applied,
            )
        )

    return result


def propose_document_rectangles(
    analyses: Sequence[QuestionAnalysis],
    page_bounds: Mapping[int, fitz.Rect],
    *,
    layouts: Mapping[int, PageLayout] | None = None,
    padding: float = DEFAULT_RECTANGLE_PADDING,
) -> dict[int, RectangleGenerationResult]:
    """Generate proposals for multiple pages by reusing the single-page API.

    ``page_bounds`` and optional ``layouts`` are keyed by zero-based page index.
    The returned mapping follows the input analysis order.
    """

    _validate_padding(padding)
    results: dict[int, RectangleGenerationResult] = {}
    for analysis in analyses:
        if analysis.page_index not in page_bounds:
            raise ValueError(f"No page bounds were supplied for page {analysis.page_index}.")
        results[analysis.page_index] = propose_question_rectangles(
            analysis,
            page_bounds[analysis.page_index],
            layout=(layouts or {}).get(analysis.page_index),
            padding=padding,
        )
    return results


def _validate_padding(padding: float) -> float:
    try:
        value = float(padding)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Rectangle padding must be a finite non-negative number of page points.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Rectangle padding must be a finite non-negative number of page points.")
    return value


def _validated_page_rect(bounds: fitz.Rect) -> fitz.Rect:
    rect = fitz.Rect(bounds)
    values = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Page bounds must contain finite coordinates.")
    normalized = _normalize_rect(rect)
    if normalized.width <= 0 or normalized.height <= 0:
        raise ValueError("Page bounds must have positive width and height.")
    return normalized


def _validated_candidate_rect(candidate: QuestionCandidate) -> fitz.Rect | None:
    try:
        rect = fitz.Rect(candidate.rect)
        values = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    normalized = _normalize_rect(rect)
    if normalized.width <= 0 or normalized.height <= 0:
        return None
    return normalized


def _usable_word_rects(candidate: QuestionCandidate) -> list[fitz.Rect]:
    rects: list[fitz.Rect] = []
    for word in candidate.words:
        try:
            rect = fitz.Rect(word.rect)
            values = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in values):
            continue
        normalized = _normalize_rect(rect)
        if normalized.width > 0 and normalized.height > 0:
            rects.append(normalized)
    return rects


def _region_clip_rect(
    layout: PageLayout | None,
    candidate: QuestionCandidate,
    page_rect: fitz.Rect,
) -> tuple[fitz.Rect, str | None, bool]:
    if layout is None:
        return page_rect, None, False
    if not 0 <= candidate.region_index < len(layout.regions):
        return page_rect, "Matching layout region was unavailable; only page clipping was applied.", True

    region = layout.regions[candidate.region_index]
    if region.name != candidate.region_name:
        return page_rect, "Candidate region identity did not match the supplied layout; only page clipping was applied.", True

    if region.kind in {"left", "right"} and layout.split_x is not None:
        split_x = float(layout.split_x)
        if math.isfinite(split_x) and page_rect.x0 < split_x < page_rect.x1:
            if region.kind == "left":
                return fitz.Rect(page_rect.x0, page_rect.y0, split_x, page_rect.y1), None, False
            return fitz.Rect(split_x, page_rect.y0, page_rect.x1, page_rect.y1), None, False

    if region.kind in {"left", "right"}:
        try:
            region_rect = _validated_page_rect(region.rect)
        except ValueError:
            return page_rect, "Region bounds were invalid; only page clipping was applied.", True
        region_clip = fitz.Rect(region_rect.x0, page_rect.y0, region_rect.x1, page_rect.y1)
        clipped = _intersect_rect(region_clip, page_rect)
        if clipped is not None:
            return clipped, "No divider was available; the region content extent conservatively limits horizontal padding.", True
        return page_rect, "Region bounds did not overlap the page; only page clipping was applied.", True
    return page_rect, None, False


def _union_rect(rectangles: Sequence[fitz.Rect]) -> fitz.Rect | None:
    if not rectangles:
        return None
    return fitz.Rect(
        min(rect.x0 for rect in rectangles),
        min(rect.y0 for rect in rectangles),
        max(rect.x1 for rect in rectangles),
        max(rect.y1 for rect in rectangles),
    )


def _intersect_rect(first: fitz.Rect, second: fitz.Rect) -> fitz.Rect | None:
    result = fitz.Rect(
        max(first.x0, second.x0),
        max(first.y0, second.y0),
        min(first.x1, second.x1),
        min(first.y1, second.y1),
    )
    if result.width <= 0 or result.height <= 0:
        return None
    return result


def _normalize_rect(rect: fitz.Rect) -> fitz.Rect:
    x0, x1 = sorted((float(rect.x0), float(rect.x1)))
    y0, y1 = sorted((float(rect.y0), float(rect.y1)))
    rect.x0, rect.y0, rect.x1, rect.y1 = x0, y0, x1, y1
    return rect


def _rect_equal(first: fitz.Rect, second: fitz.Rect) -> bool:
    return all(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(first, second)
    )


def _candidate_id(candidate: QuestionCandidate) -> str:
    return f"p{candidate.page_index}:r{candidate.region_index}:{candidate.region_name}:q{candidate.order}"