from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Sequence

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from .analysis import ExtractedWord, PageAnalysis


@dataclass
class LayoutRegion:
    """A reading region on a page, usually one column or a full-width heading."""

    name: str
    rect: fitz.Rect
    words: list[ExtractedWord] = field(default_factory=list)
    kind: str = "content"
    split_source: str = "auto"
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rect = fitz.Rect(self.rect)
        self.words = sorted(self.words, key=lambda word: (word.rect.y0, word.rect.x0))

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass
class PageLayout:
    """A page-level reading-order model for subsequent detection tasks."""

    page_index: int
    bounds: fitz.Rect
    regions: list[LayoutRegion] = field(default_factory=list)
    split_x: float | None = None
    split_detected: bool = False
    split_source: str = "auto"
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.bounds = fitz.Rect(self.bounds)
        self.regions = list(self.regions)

    @property
    def reading_order(self) -> list[LayoutRegion]:
        return list(self.regions)


def analyze_page_layout(
    page_source: PageAnalysis | fitz.Page | Sequence[ExtractedWord] | fitz.Rect,
    *,
    words: Sequence[ExtractedWord] | None = None,
    page_index: int | None = None,
    divider: float | None = None,
    divider_is_normalized: bool = False,
    force_single_column: bool = False,
    bounds: fitz.Rect | None = None,
) -> PageLayout:
    """Inspect a page's word positions and return likely reading regions.

    The function works with either a PageAnalysis object, a PyMuPDF Page, or a raw
    word list. The resulting geometry remains in the same page-space coordinate
    system as Task 1 and Task 2.
    """

    resolved_words, resolved_index, resolved_bounds = _resolve_page_input(
        page_source,
        words=words,
        page_index=page_index,
        bounds=bounds,
    )

    if force_single_column and divider is not None:
        raise ValueError("A manual divider cannot be combined with forced single-column layout.")

    if force_single_column:
        heading_words, remaining_words = _extract_full_width_heading(resolved_words, resolved_bounds)
        warning = ["Single-column layout was explicitly selected; automatic column detection was skipped."]
        return _build_single_region_layout_with_heading(
            remaining_words,
            heading_words,
            resolved_index,
            resolved_bounds,
            warning,
        )

    if divider is not None:
        split_x = _normalize_divider(divider, resolved_bounds, divider_is_normalized)
        heading_words, resolved_words = _extract_full_width_heading(resolved_words, resolved_bounds, split_x)
        return _build_manual_layout_with_heading(resolved_words, heading_words, resolved_index, resolved_bounds, split_x)

    split_x, confidence, warnings = _detect_split(resolved_words, resolved_bounds)
    heading_words, resolved_words = _extract_full_width_heading(resolved_words, resolved_bounds, split_x)
    if split_x is None:
        return _build_single_region_layout_with_heading(resolved_words, heading_words, resolved_index, resolved_bounds, warnings)

    return _build_split_layout(resolved_words, heading_words, resolved_index, resolved_bounds, split_x, confidence, warnings)


def _resolve_page_input(
    page_source: PageAnalysis | fitz.Page | Sequence[ExtractedWord] | fitz.Rect,
    *,
    words: Sequence[ExtractedWord] | None,
    page_index: int | None,
    bounds: fitz.Rect | None,
) -> tuple[list[ExtractedWord], int, fitz.Rect]:
    if isinstance(page_source, PageAnalysis):
        resolved_index = page_source.page_index
        resolved_bounds = fitz.Rect(page_source.bounds)
        resolved_words = list(page_source.words if words is None else words)
        return _normalize_words(resolved_words), resolved_index, resolved_bounds

    if isinstance(page_source, fitz.Page):
        resolved_index = page_index if page_index is not None else 0
        resolved_bounds = fitz.Rect(page_source.rect)
        resolved_words = list(page_source.get_text("words"))
        if words is not None:
            resolved_words = list(words)
        return _extract_words_from_fitz(resolved_words, resolved_index), resolved_index, resolved_bounds

    if isinstance(page_source, fitz.Rect):
        resolved_bounds = fitz.Rect(page_source)
        resolved_index = page_index if page_index is not None else 0
        resolved_words = list(words or [])
        return _normalize_words(resolved_words), resolved_index, resolved_bounds

    resolved_words = list(page_source if words is None else words or [])
    resolved_index = page_index if page_index is not None else 0
    resolved_bounds = fitz.Rect(bounds) if bounds is not None else _default_page_bounds(resolved_words)
    return _normalize_words(resolved_words), resolved_index, resolved_bounds


def _default_page_bounds(words: Sequence[ExtractedWord]) -> fitz.Rect:
    if not words:
        return fitz.Rect(0, 0, 0, 0)
    all_x = [float(word.rect.x0) for word in words] + [float(word.rect.x1) for word in words]
    all_y = [float(word.rect.y0) for word in words] + [float(word.rect.y1) for word in words]
    return fitz.Rect(min(all_x), min(all_y), max(all_x), max(all_y))


def _extract_words_from_fitz(raw_words: Sequence[object], page_index: int) -> list[ExtractedWord]:
    converted: list[ExtractedWord] = []
    for idx, item in enumerate(raw_words):
        if not isinstance(item, (list, tuple)) or len(item) < 5:
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
        word_no = int(rest[2]) if len(rest) > 2 else idx
        converted.append(
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
    return _normalize_words(converted)


def _normalize_words(words: Sequence[ExtractedWord]) -> list[ExtractedWord]:
    normalized: list[ExtractedWord] = []
    for word in words:
        if not isinstance(word, ExtractedWord):
            continue
        rect = fitz.Rect(word.rect)
        if not _is_finite_rect(rect):
            continue
        normalized.append(
            ExtractedWord(
                text=word.text,
                rect=rect,
                page_index=int(word.page_index),
                block_no=int(word.block_no),
                line_no=int(word.line_no),
                word_no=int(word.word_no),
                metadata=dict(word.metadata),
            )
        )
    return sorted(normalized, key=lambda item: (item.rect.y0, item.rect.x0, item.text))


def _is_finite_rect(rect: fitz.Rect) -> bool:
    return all(math.isfinite(float(value)) for value in (rect.x0, rect.y0, rect.x1, rect.y1))


def _normalize_divider(divider: float, bounds: fitz.Rect, divider_is_normalized: bool) -> float:
    if not math.isfinite(float(divider)):
        raise ValueError("Divider value must be finite.")
    value = float(divider)
    if divider_is_normalized or 0.0 <= value <= 1.0:
        if not 0.0 < value < 1.0:
            raise ValueError("Divider value must be strictly between 0.0 and 1.0 when normalized.")
        split_x = bounds.x0 + value * bounds.width
    else:
        if value <= bounds.x0 or value >= bounds.x1:
            raise ValueError("Divider value must fall strictly inside the page bounds.")
        split_x = value
    if not (bounds.x0 < split_x < bounds.x1):
        raise ValueError("Divider value must fall inside the usable page area.")
    return split_x


def _build_single_region_layout(
    words: Sequence[ExtractedWord],
    page_index: int,
    bounds: fitz.Rect,
    warnings: list[str],
) -> PageLayout:
    region_words = list(words)
    region_area = _bounds_for_words(region_words, bounds)
    return PageLayout(
        page_index=page_index,
        bounds=bounds,
        regions=[
            LayoutRegion(
                name="single",
                rect=region_area,
                words=list(region_words),
                kind="single",
                split_source="auto",
                confidence=0.0,
                warnings=list(warnings),
            )
        ],
        split_x=None,
        split_detected=False,
        split_source="auto",
        warnings=list(warnings),
        confidence=0.0,
    )


def _build_single_region_layout_with_heading(
    words: Sequence[ExtractedWord],
    heading_words: Sequence[ExtractedWord],
    page_index: int,
    bounds: fitz.Rect,
    warnings: list[str],
) -> PageLayout:
    regions: list[LayoutRegion] = []
    if heading_words:
        regions.append(
            LayoutRegion(
                name="full_width",
                rect=_bounds_for_words(heading_words, bounds),
                words=list(heading_words),
                kind="full_width",
                split_source="auto",
                confidence=0.0,
                warnings=[],
            )
        )

    regions.append(
        LayoutRegion(
            name="single",
            rect=_bounds_for_words(words, bounds),
            words=list(words),
            kind="single",
            split_source="auto",
            confidence=0.0,
            warnings=list(warnings),
        )
    )

    return PageLayout(
        page_index=page_index,
        bounds=bounds,
        regions=regions,
        split_x=None,
        split_detected=False,
        split_source="auto",
        warnings=list(warnings),
        confidence=0.0,
    )


def _build_manual_layout(
    words: Sequence[ExtractedWord],
    page_index: int,
    bounds: fitz.Rect,
    split_x: float,
) -> PageLayout:
    left_words, right_words, crossing_words, left_warning = _assign_words_to_side(words, split_x)
    regions: list[LayoutRegion] = []
    if left_words:
        regions.append(
            LayoutRegion(
                name="left",
                rect=_bounds_for_words(left_words, bounds),
                words=list(left_words),
                kind="left",
                split_source="manual",
                confidence=1.0,
                warnings=[] if not crossing_words else [f"{len(crossing_words)} words cross the supplied divider; assigned to the dominant side."],
            )
        )
    if right_words:
        regions.append(
            LayoutRegion(
                name="right",
                rect=_bounds_for_words(right_words, bounds),
                words=list(right_words),
                kind="right",
                split_source="manual",
                confidence=1.0,
                warnings=[],
            )
        )
    warnings = list(left_warning)
    if not regions:
        warnings.append("No words were assigned to either side of the supplied divider.")
    return PageLayout(
        page_index=page_index,
        bounds=bounds,
        regions=regions,
        split_x=split_x,
        split_detected=True,
        split_source="manual",
        warnings=warnings,
        confidence=1.0,
    )


def _build_manual_layout_with_heading(
    words: Sequence[ExtractedWord],
    heading_words: Sequence[ExtractedWord],
    page_index: int,
    bounds: fitz.Rect,
    split_x: float,
) -> PageLayout:
    regions: list[LayoutRegion] = []
    if heading_words:
        regions.append(
            LayoutRegion(
                name="full_width",
                rect=_bounds_for_words(heading_words, bounds),
                words=list(heading_words),
                kind="full_width",
                split_source="manual",
                confidence=1.0,
                warnings=[],
            )
        )

    left_words, right_words, crossing_words, warnings = _assign_words_to_side(words, split_x)
    if left_words:
        regions.append(
            LayoutRegion(
                name="left",
                rect=_bounds_for_words(left_words, bounds),
                words=list(left_words),
                kind="left",
                split_source="manual",
                confidence=1.0,
                warnings=[f"{len(crossing_words)} words cross the supplied divider; assigned to the dominant side."] if crossing_words else [],
            )
        )
    if right_words:
        regions.append(
            LayoutRegion(
                name="right",
                rect=_bounds_for_words(right_words, bounds),
                words=list(right_words),
                kind="right",
                split_source="manual",
                confidence=1.0,
                warnings=[],
            )
        )

    if not regions:
        return _build_single_region_layout(words, page_index, bounds, warnings)

    return PageLayout(
        page_index=page_index,
        bounds=bounds,
        regions=regions,
        split_x=split_x,
        split_detected=True,
        split_source="manual",
        warnings=warnings,
        confidence=1.0,
    )


def _build_split_layout(
    words: Sequence[ExtractedWord],
    heading_words: Sequence[ExtractedWord],
    page_index: int,
    bounds: fitz.Rect,
    split_x: float,
    confidence: float,
    warnings: list[str],
) -> PageLayout:
    regions: list[LayoutRegion] = []
    if heading_words:
        regions.append(
            LayoutRegion(
                name="full_width",
                rect=_bounds_for_words(heading_words, bounds),
                words=list(heading_words),
                kind="full_width",
                split_source="auto",
                confidence=max(0.0, min(1.0, confidence)),
                warnings=[],
            )
        )

    left_words, right_words, crossing_words, side_warnings = _assign_words_to_side(words, split_x)
    if left_words:
        regions.append(
            LayoutRegion(
                name="left",
                rect=_bounds_for_words(left_words, bounds),
                words=list(left_words),
                kind="left",
                split_source="auto",
                confidence=max(0.0, min(1.0, confidence)),
                warnings=[f"{len(crossing_words)} words cross the detected divider and were assigned by overlap."] if crossing_words else [],
            )
        )
    if right_words:
        regions.append(
            LayoutRegion(
                name="right",
                rect=_bounds_for_words(right_words, bounds),
                words=list(right_words),
                kind="right",
                split_source="auto",
                confidence=max(0.0, min(1.0, confidence)),
                warnings=[],
            )
        )

    if not regions:
        return _build_single_region_layout(words, page_index, bounds, warnings + side_warnings)

    region_warnings = list(warnings)
    region_warnings.extend(side_warnings)
    if crossing_words:
        region_warnings.append(f"{len(crossing_words)} words intersect the detected divider; assigned by dominant overlap.")

    return PageLayout(
        page_index=page_index,
        bounds=bounds,
        regions=regions,
        split_x=split_x,
        split_detected=True,
        split_source="auto",
        warnings=region_warnings,
        confidence=max(0.0, min(1.0, confidence)),
    )


def _detect_split(words: Sequence[ExtractedWord], bounds: fitz.Rect) -> tuple[float | None, float, list[str]]:
    if not words:
        return None, 0.0, ["No words were available for split detection."]

    page_width = max(float(bounds.width), 1.0)
    page_height = max(float(bounds.height), 1.0)
    line_groups = _group_words_by_metadata(words)
    if line_groups:
        split_x, confidence = _detect_split_from_lines(line_groups, bounds)
        if split_x is not None:
            return split_x, confidence, []
        if len(line_groups) >= 6:
            return None, 0.0, [
                "Reliable text lines did not support separate columns; keeping one region instead of splitting at word-level gaps."
            ]

    split_x = _detect_split_from_word_bands(words, bounds)
    if split_x is not None:
        return split_x, 0.62, []

    return None, 0.0, ["No stable column divider was supported by repeated content on both sides; keeping one region."]


def _center_x(rect: fitz.Rect) -> float:
    return (float(rect.x0) + float(rect.x1)) / 2.0


def _center_y(rect: fitz.Rect) -> float:
    return (float(rect.y0) + float(rect.y1)) / 2.0


def _group_words_by_metadata(words: Sequence[ExtractedWord]) -> list[list[ExtractedWord]] | None:
    grouped: dict[tuple[int, int], list[ExtractedWord]] = {}
    for word in words:
        grouped.setdefault((word.block_no, word.line_no), []).append(word)
    if len(grouped) < 2 or not any(block_no or line_no for block_no, line_no in grouped):
        return None

    typical_height = median(max(float(word.rect.height), 0.1) for word in words)
    line_groups = list(grouped.values())
    for line_words in line_groups:
        tops = [float(word.rect.y0) for word in line_words]
        bottoms = [float(word.rect.y1) for word in line_words]
        if max(bottoms) - min(tops) > max(8.0, typical_height * 3.0):
            return None
    return line_groups


def _detect_split_from_lines(
    line_groups: Sequence[Sequence[ExtractedWord]], bounds: fitz.Rect
) -> tuple[float | None, float]:
    page_width = max(float(bounds.width), 1.0)
    page_height = max(float(bounds.height), 1.0)
    line_bounds = [
        (
            min(float(word.rect.x0) for word in group),
            max(float(word.rect.x1) for word in group),
            min(float(word.rect.y0) for word in group),
            max(float(word.rect.y1) for word in group),
        )
        for group in line_groups
        if group
    ]
    if len(line_bounds) < 12:
        return None, 0.0

    typical_height = median(max(0.1, bottom - top) for _, _, top, bottom in line_bounds)
    band_tolerance = max(45.0, typical_height * 3.0)
    minimum_span = min(0.30 * page_height, max(80.0, band_tolerance * 2.0))
    minimum_clearance = max(8.0, typical_height * 0.5)
    step = max(2.0, page_width / 300.0)
    candidate_xs = _candidate_divider_positions(bounds, step)
    valid: list[tuple[tuple[int, int, int, int], float, int]] = []

    for split_x in candidate_xs:
        left_lines = [(x0, x1, y0, y1) for x0, x1, y0, y1 in line_bounds if x1 <= split_x]
        right_lines = [(x0, x1, y0, y1) for x0, x1, y0, y1 in line_bounds if x0 >= split_x]
        crossing_count = sum(1 for x0, x1, _, _ in line_bounds if x0 < split_x < x1)
        if min(len(left_lines), len(right_lines)) < max(6, int(len(line_bounds) * 0.08)):
            continue
        if crossing_count > max(2, int(min(len(left_lines), len(right_lines)) * 0.12)):
            continue

        left_span = max(item[3] for item in left_lines) - min(item[2] for item in left_lines)
        right_span = max(item[3] for item in right_lines) - min(item[2] for item in right_lines)
        if min(left_span, right_span) < minimum_span:
            continue

        shared_bands = _shared_vertical_bands(left_lines, right_lines, band_tolerance)
        if shared_bands < 3:
            continue

        clearance = min(
            split_x - max(item[1] for item in left_lines),
            min(item[0] for item in right_lines) - split_x,
        )
        if clearance < minimum_clearance:
            continue
        rank = (crossing_count, -shared_bands, -int(clearance * 1000), -min(len(left_lines), len(right_lines)))
        valid.append((rank, split_x, shared_bands))

    if not valid:
        return None, 0.0

    best_rank = min(item[0] for item in valid)
    best_positions = sorted(item[1] for item in valid if item[0] == best_rank)
    split_x = median(best_positions)
    shared_bands = max(item[2] for item in valid if item[0] == best_rank)
    confidence = min(0.98, 0.68 + 0.035 * shared_bands)
    return float(split_x), confidence


def _detect_split_from_word_bands(words: Sequence[ExtractedWord], bounds: fitz.Rect) -> float | None:
    page_width = max(float(bounds.width), 1.0)
    page_height = max(float(bounds.height), 1.0)
    if len(words) == 2:
        left_word, right_word = sorted(words, key=lambda word: _center_x(word.rect))
        gap = float(right_word.rect.x0) - float(left_word.rect.x1)
        vertical_offset = abs(_center_y(left_word.rect) - _center_y(right_word.rect))
        if gap > max(0.15 * page_width, 20.0) and vertical_offset <= max(
            20.0, min(float(left_word.rect.height), float(right_word.rect.height)) * 1.5
        ):
            return (float(left_word.rect.x1) + float(right_word.rect.x0)) / 2.0
        return None

    typical_height = median(max(float(word.rect.height), 0.1) for word in words)
    band_tolerance = max(45.0, typical_height * 3.0)
    minimum_span = min(0.30 * page_height, max(80.0, band_tolerance * 2.0))
    minimum_clearance = max(8.0, typical_height * 0.5)
    step = max(2.0, page_width / 300.0)
    valid: list[tuple[tuple[int, int, int, int], float]] = []

    for split_x in _candidate_divider_positions(bounds, step):
        left_words = [word for word in words if word.rect.x1 <= split_x]
        right_words = [word for word in words if word.rect.x0 >= split_x]
        crossing_count = sum(1 for word in words if word.rect.x0 < split_x < word.rect.x1)
        if min(len(left_words), len(right_words)) < 2:
            continue
        if crossing_count > max(3, int(min(len(left_words), len(right_words)) * 0.15)):
            continue

        left_span = max(word.rect.y1 for word in left_words) - min(word.rect.y0 for word in left_words)
        right_span = max(word.rect.y1 for word in right_words) - min(word.rect.y0 for word in right_words)
        if min(left_span, right_span) < minimum_span:
            continue

        shared_bands = _shared_word_vertical_bands(left_words, right_words, band_tolerance)
        minimum_shared_bands = 2 if min(len(left_words), len(right_words)) <= 4 else 3
        if shared_bands < minimum_shared_bands:
            continue

        clearance = min(
            split_x - max(float(word.rect.x1) for word in left_words),
            min(float(word.rect.x0) for word in right_words) - split_x,
        )
        if clearance < minimum_clearance:
            continue
        rank = (crossing_count, -shared_bands, -int(clearance * 1000), -min(len(left_words), len(right_words)))
        valid.append((rank, split_x))

    if not valid:
        return None
    best_rank = min(item[0] for item in valid)
    return float(median(sorted(item[1] for item in valid if item[0] == best_rank)))


def _shared_vertical_bands(
    left_lines: Sequence[tuple[float, float, float, float]],
    right_lines: Sequence[tuple[float, float, float, float]],
    tolerance: float,
) -> int:
    positions = [((top + bottom) / 2.0, 0) for _, _, top, bottom in left_lines]
    positions.extend(((top + bottom) / 2.0, 1) for _, _, top, bottom in right_lines)
    return _count_shared_y_clusters(positions, tolerance)


def _shared_word_vertical_bands(
    left_words: Sequence[ExtractedWord],
    right_words: Sequence[ExtractedWord],
    tolerance: float,
) -> int:
    positions = [(_center_y(word.rect), 0) for word in left_words]
    positions.extend((_center_y(word.rect), 1) for word in right_words)
    return _count_shared_y_clusters(positions, tolerance)


def _count_shared_y_clusters(positions: Sequence[tuple[float, int]], tolerance: float) -> int:
    if not positions:
        return 0
    clusters: list[tuple[float, set[int]]] = []
    for y_position, side in sorted(positions):
        if not clusters or y_position - clusters[-1][0] > tolerance:
            clusters.append((y_position, {side}))
        else:
            clusters[-1][1].add(side)
    return sum(len(sides) == 2 for _, sides in clusters)


def _candidate_divider_positions(bounds: fitz.Rect, step: float) -> list[float]:
    start = float(bounds.x0) + 0.20 * float(bounds.width)
    end = float(bounds.x1) - 0.20 * float(bounds.width)
    count = max(1, int((end - start) / step))
    return [start + index * step for index in range(count + 1)]


def _extract_full_width_heading(
    words: Sequence[ExtractedWord], bounds: fitz.Rect, split_x: float | None = None
) -> tuple[list[ExtractedWord], list[ExtractedWord]]:
    if not words:
        return [], list(words)

    page_width = float(bounds.width)
    page_height = float(bounds.height)
    top_cutoff = max(0.0, 0.22 * page_height)
    top_words = [word for word in words if word.rect.y1 <= top_cutoff]
    full_width_words: list[ExtractedWord] = []

    line_groups = _group_words_by_metadata(top_words)
    if split_x is not None and line_groups:
        for line_words in line_groups:
            x_min = min(float(word.rect.x0) for word in line_words)
            x_max = max(float(word.rect.x1) for word in line_words)
            if x_min < split_x < x_max and x_max - x_min >= 0.15 * page_width:
                full_width_words.extend(line_words)

    if not full_width_words:
        wide_words = [word for word in top_words if float(word.rect.width) >= 0.55 * page_width]
        for anchor in wide_words:
            anchor_height = max(float(anchor.rect.height), 1.0)
            nearby_limit = min(float(bounds.y1), float(anchor.rect.y1) + 1.5 * anchor_height)
            full_width_words.extend(
                word for word in top_words
                if float(word.rect.y0) >= float(anchor.rect.y0) and float(word.rect.y1) <= nearby_limit
            )

    unique_words = {id(word): word for word in full_width_words}
    full_width_words = sorted(unique_words.values(), key=lambda item: (item.rect.y0, item.rect.x0))
    if not full_width_words:
        return [], list(words)

    heading_set = set(id(word) for word in full_width_words)
    remaining = [word for word in words if id(word) not in heading_set]
    return sorted(full_width_words, key=lambda item: (item.rect.y0, item.rect.x0)), remaining


def _bounds_for_words(words: Sequence[ExtractedWord], fallback: fitz.Rect | None = None) -> fitz.Rect:
    if not words:
        return fitz.Rect(fallback) if fallback is not None else fitz.Rect(0, 0, 0, 0)
    xs = [float(word.rect.x0) for word in words] + [float(word.rect.x1) for word in words]
    ys = [float(word.rect.y0) for word in words] + [float(word.rect.y1) for word in words]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))


def _assign_words_to_side(
    words: Sequence[ExtractedWord], split_x: float
) -> tuple[list[ExtractedWord], list[ExtractedWord], list[ExtractedWord], list[str]]:
    left_words: list[ExtractedWord] = []
    right_words: list[ExtractedWord] = []
    crossing_words: list[ExtractedWord] = []
    warnings: list[str] = []

    for word in words:
        rect = word.rect
        if rect.x1 <= split_x:
            left_words.append(word)
            continue
        if rect.x0 >= split_x:
            right_words.append(word)
            continue

        overlap_left = max(0.0, min(rect.x1, split_x) - rect.x0)
        overlap_right = max(0.0, rect.x1 - max(rect.x0, split_x))
        crossing_words.append(word)
        if overlap_left >= overlap_right:
            left_words.append(word)
        else:
            right_words.append(word)

    if crossing_words:
        warnings.append(f"{len(crossing_words)} words span the divider and were assigned by dominant overlap.")

    return left_words, right_words, crossing_words, warnings
