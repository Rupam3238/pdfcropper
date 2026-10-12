from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from statistics import median
from typing import Pattern, Sequence

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from .analysis import ExtractedWord
from .layout import LayoutRegion, PageLayout


@dataclass(frozen=True)
class QuestionLine:
    """A logical line reconstructed from words within one layout region."""

    text: str
    words: tuple[ExtractedWord, ...]
    rect: fitz.Rect
    region_index: int
    region_name: str
    line_index: int


@dataclass
class QuestionCandidate:
    """A likely top-level question and its assigned page-space content."""

    page_index: int
    order: int
    region_index: int
    region_name: str
    number: int | None
    label: str | None
    lines: list[QuestionLine] = field(default_factory=list)
    words: list[ExtractedWord] = field(default_factory=list)
    rect: fitz.Rect = field(default_factory=lambda: fitz.Rect(0, 0, 0, 0))
    detection_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AmbiguousQuestionLine:
    """A number-like line that was deliberately not accepted as a question start."""

    line: QuestionLine
    reason: str


@dataclass
class QuestionAnalysis:
    """Question candidates and retained page-level diagnostics for one layout."""

    page_index: int
    candidates: list[QuestionCandidate] = field(default_factory=list)
    preamble_lines: list[QuestionLine] = field(default_factory=list)
    ambiguous_lines: list[AmbiguousQuestionLine] = field(default_factory=list)
    question_start_count: int = 0
    no_reliable_starts: bool = True
    content_preceding_first_question_retained: bool = False


@dataclass(frozen=True)
class QuestionDetectionConfig:
    """Small set of tunable rules for local question-start recognition.

    ``custom_marker_patterns`` are regular expressions matched against the start
    of a line. A pattern may provide a named ``number`` group; otherwise its
    matched text is used as the label. Patterns should include their own marker
    boundaries and must not match ordinary mid-line numbers.
    """

    custom_marker_patterns: tuple[str, ...] = ()
    left_margin_ratio: float = 0.08
    minimum_left_margin_points: float = 18.0


@dataclass(frozen=True)
class _ParsedMarker:
    number: int | None
    label: str
    suffix: str
    style: str
    explicit: bool


@dataclass(frozen=True)
class _StartDecision:
    accepted: bool
    marker: _ParsedMarker | None = None
    confidence: float = 0.0
    reason: str = ""


_QUESTION_PREFIX = re.compile(r"^question\s*\.?\s*(\d+)\s*([.):])?\s*(.*)$", re.IGNORECASE)
_Q_PREFIX = re.compile(r"^q\s*\.?\s*(\d+)\s*([.):])?\s*(.*)$", re.IGNORECASE)
_NUMBER_PUNCTUATED = re.compile(r"^(\d+)\s*([.):])\s*(.*)$")
_NUMBER_BARE = re.compile(r"^(\d+)\s+(.+)$")
_NUMBER_ONLY = re.compile(r"^\d{1,4}\s*[.):]?$")
_ALPHA_WORD = re.compile(r"[A-Za-z]+")
_PROMPT_VERBS = frozenset(
    {
        "calculate", "compare", "compute", "define", "derive", "describe", "determine",
        "discuss", "explain", "find", "give", "how", "if", "identify", "let",
        "illustrate", "list", "prove", "show", "simplify", "solve", "state", "what",
        "when", "where", "which", "why", "write",
    }
)
_OPTION_CUES = frozenset({"answer", "choose", "correct", "following", "option", "options", "select"})
_MATH_OPERATORS = frozenset("=+*/^")


def analyze_questions(
    layout: PageLayout,
    *,
    config: QuestionDetectionConfig | None = None,
) -> QuestionAnalysis:
    """Detect likely question starts and group their following lines.

    Regions are processed independently and in ``PageLayout.reading_order``;
    therefore columns never interleave or merge. Text before a region's first
    accepted start is returned in ``preamble_lines`` rather than discarded.
    All word and rectangle geometry remains in Task 1 page coordinates.
    """

    settings = config or QuestionDetectionConfig()
    custom_patterns = _compile_custom_patterns(settings.custom_marker_patterns)
    result = QuestionAnalysis(page_index=layout.page_index)
    order = 0
    first_question_seen = False
    leading_preamble_retained = False

    for region_index, region in enumerate(layout.reading_order):
        lines = _build_region_lines(region, region_index)
        region_left: float | None = None
        region_width: float | None = None
        if layout.split_detected and layout.split_x is not None and region.kind in {"left", "right"}:
            text_inset = _dominant_line_start(lines, float(region.rect.width))
            if region.kind == "left":
                region_left = max(float(layout.bounds.x0), text_inset)
                region_width = float(layout.split_x - layout.bounds.x0)
            else:
                region_left = max(float(layout.split_x), text_inset)
                region_width = float(layout.bounds.x1 - layout.split_x)
        decisions = _classify_region_lines(
            lines,
            region,
            settings,
            custom_patterns,
            region_left=region_left,
            region_width=region_width,
        )
        accepted_indices = [index for index, decision in enumerate(decisions) if decision.accepted]
        first_start = accepted_indices[0] if accepted_indices else len(lines)
        result.preamble_lines.extend(lines[:first_start])
        if not first_question_seen and lines[:first_start]:
            leading_preamble_retained = True
        if accepted_indices:
            first_question_seen = True

        for index, decision in enumerate(decisions):
            if decision.reason:
                result.ambiguous_lines.append(AmbiguousQuestionLine(lines[index], decision.reason))

        for start_position, start_index in enumerate(accepted_indices):
            end_index = accepted_indices[start_position + 1] if start_position + 1 < len(accepted_indices) else len(lines)
            assigned_lines = lines[start_index:end_index]
            words = [word for line in assigned_lines for word in line.words]
            marker = decisions[start_index].marker
            order += 1
            warnings: list[str] = []
            if decisions[start_index].confidence < 0.75:
                warnings.append("Question start is inferred from sequence context and may be ambiguous.")
            result.candidates.append(
                QuestionCandidate(
                    page_index=layout.page_index,
                    order=order,
                    region_index=region_index,
                    region_name=region.name,
                    number=marker.number if marker else None,
                    label=marker.label if marker else None,
                    lines=assigned_lines,
                    words=words,
                    rect=_union_rect([word.rect for word in words]),
                    detection_confidence=decisions[start_index].confidence,
                    warnings=warnings,
                )
            )

    result.question_start_count = len(result.candidates)
    result.no_reliable_starts = not any(candidate.detection_confidence >= 0.75 for candidate in result.candidates)
    result.content_preceding_first_question_retained = (
        leading_preamble_retained if result.candidates else bool(result.preamble_lines)
    )
    return result


def _dominant_line_start(lines: Sequence[QuestionLine], region_width: float) -> float:
    starts = sorted(float(line.rect.x0) for line in lines)
    if not starts:
        return 0.0
    tolerance = max(10.0, min(24.0, region_width * 0.06))
    clusters = [
        [start for start in starts if abs(start - anchor) <= tolerance]
        for anchor in starts
    ]
    dominant = max(clusters, key=lambda cluster: (len(cluster), -median(cluster)))
    return float(median(dominant))


def _compile_custom_patterns(patterns: Sequence[str]) -> list[Pattern[str]]:
    compiled: list[Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(f"Invalid custom question-start pattern {pattern!r}: {exc}") from exc
    return compiled


def _build_region_lines(region: LayoutRegion, region_index: int) -> list[QuestionLine]:
    words = [word for word in region.words if word.text.strip() and _valid_rect(word.rect)]
    if not words:
        return []

    grouped = _group_by_metadata_if_reliable(words)
    if grouped is None:
        grouped = _group_geometrically(words)

    lines: list[QuestionLine] = []
    for line_index, line_words in enumerate(grouped):
        ordered_words = tuple(sorted(line_words, key=lambda word: (word.rect.x0, word.rect.y0, word.word_no, word.text)))
        lines.append(
            QuestionLine(
                text=" ".join(word.text.strip() for word in ordered_words),
                words=ordered_words,
                rect=_union_rect([word.rect for word in ordered_words]),
                region_index=region_index,
                region_name=region.name,
                line_index=line_index,
            )
        )
    return sorted(lines, key=lambda line: (line.rect.y0, line.rect.x0, line.line_index))


def _group_by_metadata_if_reliable(words: Sequence[ExtractedWord]) -> list[list[ExtractedWord]] | None:
    keys: dict[tuple[int, int], list[ExtractedWord]] = {}
    for word in words:
        keys.setdefault((word.block_no, word.line_no), []).append(word)
    if len(keys) < 2 or not any(block or line for block, line in keys):
        return None

    typical_height = median(max(float(word.rect.height), 0.1) for word in words)
    centers: list[float] = []
    for line_words in keys.values():
        ys = [_center_y(word.rect) for word in line_words]
        if max(ys) - min(ys) > max(3.0, typical_height * 0.75):
            return None
        centers.append(median(ys))
    centers.sort()
    if any(right - left < typical_height * 0.45 for left, right in zip(centers, centers[1:])):
        return None

    return sorted(keys.values(), key=lambda items: (min(word.rect.y0 for word in items), min(word.rect.x0 for word in items)))


def _group_geometrically(words: Sequence[ExtractedWord]) -> list[list[ExtractedWord]]:
    ordered = sorted(words, key=lambda word: (_center_y(word.rect), word.rect.y0, word.rect.x0, word.word_no, word.text))
    typical_height = median(max(float(word.rect.height), 0.1) for word in ordered)
    lines: list[list[ExtractedWord]] = []

    for word in ordered:
        best_line: list[ExtractedWord] | None = None
        best_distance = math.inf
        for line in lines:
            top = min(float(item.rect.y0) for item in line)
            bottom = max(float(item.rect.y1) for item in line)
            line_center = (top + bottom) / 2.0
            word_center = _center_y(word.rect)
            overlap = max(0.0, min(bottom, word.rect.y1) - max(top, word.rect.y0))
            min_height = max(min(bottom - top, float(word.rect.height)), 0.1)
            distance = abs(line_center - word_center)
            if overlap / min_height >= 0.45 or distance <= typical_height * 0.45:
                if distance < best_distance:
                    best_line = line
                    best_distance = distance
        if best_line is None:
            lines.append([word])
        else:
            best_line.append(word)

    return sorted(lines, key=lambda items: (min(word.rect.y0 for word in items), min(word.rect.x0 for word in items)))


def _classify_region_lines(
    lines: Sequence[QuestionLine],
    region: LayoutRegion,
    config: QuestionDetectionConfig,
    custom_patterns: Sequence[Pattern[str]],
    *,
    region_left: float | None = None,
    region_width: float | None = None,
) -> list[_StartDecision]:
    decisions: list[_StartDecision] = []
    last_number: int | None = None
    if region_left is None:
        region_left = min((float(line.rect.x0) for line in lines), default=float(region.rect.x0))
    if region_width is None:
        region_width = float(region.rect.width)
    region_width = max(region_width, 1.0)
    typical_line_height = median([float(line.rect.height) for line in lines]) if lines else 0.0
    margin = max(config.minimum_left_margin_points, config.left_margin_ratio * region_width) + 0.5 * typical_line_height

    for line in lines:
        marker = _parse_marker(line.text, custom_patterns)
        if marker is None:
            if _NUMBER_ONLY.fullmatch(line.text.strip()):
                decisions.append(_StartDecision(False, reason="A standalone number or page label has no question text."))
            else:
                decisions.append(_StartDecision(False))
            continue

        left_aligned = float(line.rect.x0) <= region_left + margin
        if not left_aligned:
            decisions.append(_StartDecision(False, marker, reason="The number-like line is indented from the region's question margin."))
            continue

        prompt_strength, prompt_reason = _prompt_evidence(marker.suffix)
        if not prompt_strength and marker.style == "question_prefix":
            prompt_strength = len(_ALPHA_WORD.findall(marker.suffix)) >= 2
        if not prompt_strength:
            decisions.append(_StartDecision(False, marker, reason=prompt_reason))
            continue

        if marker.style == "bare":
            follows_sequence = last_number is not None and marker.number == last_number + 1
            starts_region_sequence = last_number is None and marker.number == 1
            if not (follows_sequence or starts_region_sequence):
                decisions.append(_StartDecision(False, marker, reason="Unpunctuated numbering lacks a strong sequence context."))
                continue
            confidence = 0.68 if follows_sequence else 0.62
        else:
            confidence = 0.93 if marker.style == "question_prefix" else 0.86
            if last_number is not None and marker.number == last_number + 1:
                confidence = min(0.98, confidence + 0.05)

        if marker.style == "numeric_paren" and last_number is not None and _looks_like_option_line(marker, line, lines, decisions, config):
            decisions.append(_StartDecision(False, marker, reason="This looks like an answer choice under the preceding question."))
            continue

        decisions.append(_StartDecision(True, marker, confidence))
        last_number = marker.number

    return decisions


def _parse_marker(text: str, custom_patterns: Sequence[Pattern[str]]) -> _ParsedMarker | None:
    value = " ".join(text.strip().split())

    match = _QUESTION_PREFIX.match(value)
    if match:
        label = value[:match.start(3)].strip()
        return _ParsedMarker(int(match.group(1)), label, match.group(3).strip(), "question_prefix", True)

    match = _Q_PREFIX.match(value)
    if match:
        label = value[:match.start(3)].strip()
        return _ParsedMarker(int(match.group(1)), label, match.group(3).strip(), "question_prefix", True)

    for pattern in custom_patterns:
        match = pattern.match(value)
        if match:
            number_text = match.groupdict().get("number")
            label = match.groupdict().get("label") or match.group(0).strip()
            suffix = value[match.end():].strip()
            number = int(number_text) if number_text and number_text.isdigit() else None
            return _ParsedMarker(number, label, suffix, "custom", True)

    match = _NUMBER_PUNCTUATED.match(value)
    if match:
        punctuation = match.group(2)
        style = "numeric_paren" if punctuation == ")" else "numeric_punctuated"
        return _ParsedMarker(int(match.group(1)), f"{match.group(1)}{punctuation}", match.group(3).strip(), style, True)

    match = _NUMBER_BARE.match(value)
    if match:
        return _ParsedMarker(int(match.group(1)), match.group(1), match.group(2).strip(), "bare", False)
    return None


def _prompt_evidence(suffix: str) -> tuple[bool, str]:
    value = suffix.strip()
    alpha_words = [word.lower() for word in _ALPHA_WORD.findall(value)]
    if not alpha_words:
        return False, "The marker is followed by no question-like prose."
    if any(word in _PROMPT_VERBS for word in alpha_words[:5]):
        return True, ""
    if len(alpha_words) < 2:
        return False, "The marker is followed by too little prose to distinguish it from a label or mark count."
    if any(operator in value for operator in _MATH_OPERATORS) and not any(word in _PROMPT_VERBS for word in alpha_words):
        return False, "The number-like line appears to be a mathematical expression rather than a question prompt."
    return True, ""


def _looks_like_option_line(
    marker: _ParsedMarker,
    line: QuestionLine,
    lines: Sequence[QuestionLine],
    decisions: Sequence[_StartDecision],
    config: QuestionDetectionConfig,
) -> bool:
    suffix_words = _ALPHA_WORD.findall(marker.suffix)
    if len(suffix_words) > 6:
        return False
    prior_start = next((index for index in range(len(decisions) - 1, -1, -1) if decisions[index].accepted), None)
    if prior_start is None:
        return False
    prior_text = " ".join(item.text.lower() for item in lines[prior_start:lines.index(line)])
    if any(cue in prior_text for cue in _OPTION_CUES):
        return True
    region_left = min(float(item.rect.x0) for item in lines)
    margin = max(config.minimum_left_margin_points, config.left_margin_ratio * max(float(line.rect.width), 1.0))
    return float(line.rect.x0) > region_left + margin * 0.45


def _union_rect(rectangles: Sequence[fitz.Rect]) -> fitz.Rect:
    values = [fitz.Rect(rect) for rect in rectangles if _valid_rect(rect)]
    if not values:
        return fitz.Rect(0, 0, 0, 0)
    x0 = min(min(float(rect.x0), float(rect.x1)) for rect in values)
    y0 = min(min(float(rect.y0), float(rect.y1)) for rect in values)
    x1 = max(max(float(rect.x0), float(rect.x1)) for rect in values)
    y1 = max(max(float(rect.y0), float(rect.y1)) for rect in values)
    return fitz.Rect(x0, y0, x1, y1)


def _valid_rect(rect: fitz.Rect) -> bool:
    values = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
    return all(math.isfinite(value) for value in values)


def _center_y(rect: fitz.Rect) -> float:
    return (float(rect.y0) + float(rect.y1)) / 2.0