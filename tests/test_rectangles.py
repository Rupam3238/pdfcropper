from __future__ import annotations

import math

import fitz
import pytest

from pdfcropper.analysis import ExtractedWord
from pdfcropper.analysis import analyze_pdf_document
from pdfcropper.layout import LayoutRegion, PageLayout, analyze_page_layout
from pdfcropper.questions import QuestionAnalysis, QuestionCandidate, QuestionLine
from pdfcropper.questions import analyze_questions
from pdfcropper.rectangles import (
    DEFAULT_RECTANGLE_PADDING,
    propose_document_rectangles,
    propose_question_rectangles,
)


def _candidate(
    *,
    page_index=0,
    order=1,
    region_index=0,
    region_name="single",
    bounds=(100, 120, 180, 150),
    label="1.",
    confidence=0.9,
    include_word=True,
):
    rect = fitz.Rect(bounds)
    words = []
    lines = []
    if include_word:
        word = ExtractedWord("Find", rect, page_index)
        words.append(word)
        lines.append(QuestionLine("Find", (word,), rect, region_index, region_name, 0))
    return QuestionCandidate(
        page_index=page_index,
        order=order,
        region_index=region_index,
        region_name=region_name,
        number=order,
        label=label,
        lines=lines,
        words=words,
        rect=rect,
        detection_confidence=confidence,
    )


def _analysis(*candidates, page_index=0):
    return QuestionAnalysis(page_index=page_index, candidates=list(candidates), question_start_count=len(candidates))


def test_normal_candidate_uses_padded_word_bounds_and_preserves_metadata():
    candidate = _candidate()
    result = propose_question_rectangles(_analysis(candidate), fitz.Rect(0, 0, 400, 500))

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.rect == fitz.Rect(94, 114, 186, 156)
    assert proposal.page_index == 0
    assert proposal.order == 1
    assert proposal.label == "1."
    assert proposal.region_name == "single"
    assert proposal.source_candidate is candidate
    assert proposal.confidence == pytest.approx(0.9)
    assert proposal.boundary_adjusted is False
    assert proposal.fallback_applied is False


def test_default_padding_is_six_page_points():
    candidate = _candidate()
    proposal = propose_question_rectangles(_analysis(candidate), fitz.Rect(0, 0, 400, 500)).proposals[0]

    assert DEFAULT_RECTANGLE_PADDING == 6.0
    assert proposal.rect == fitz.Rect(94, 114, 186, 156)


def test_page_edges_clamp_padding_and_mark_boundary_adjustment():
    candidate = _candidate(bounds=(2, 3, 30, 40))
    proposal = propose_question_rectangles(
        _analysis(candidate), fitz.Rect(0, 0, 400, 500), padding=6
    ).proposals[0]

    assert proposal.rect == fitz.Rect(0, 0, 36, 46)
    assert proposal.boundary_adjusted
    assert any("clipped" in warning for warning in proposal.warnings)


@pytest.mark.parametrize(
    ("kind", "bounds", "expected"),
    [
        ("left", (210, 100, 245, 130), fitz.Rect(204, 94, 250, 136)),
        ("right", (255, 100, 290, 130), fitz.Rect(250, 94, 296, 136)),
    ],
)
def test_split_column_rectangles_respect_exact_divider(kind, bounds, expected):
    region = LayoutRegion(kind, fitz.Rect(bounds), [], kind=kind)
    layout = PageLayout(0, fitz.Rect(0, 0, 500, 500), [region], split_x=250, split_detected=True)
    candidate = _candidate(bounds=bounds, region_name=kind)

    proposal = propose_question_rectangles(_analysis(candidate), layout.bounds, layout=layout).proposals[0]

    assert proposal.rect == expected
    assert proposal.rect.x1 <= 250 if kind == "left" else proposal.rect.x0 >= 250


def test_manual_divider_uses_layout_split_coordinate_not_region_content_extent():
    region = LayoutRegion("right", fitz.Rect(280, 100, 330, 130), [], kind="right", split_source="manual")
    layout = PageLayout(0, fitz.Rect(0, 0, 500, 500), [region], split_x=275, split_detected=True, split_source="manual")
    candidate = _candidate(bounds=(280, 100, 330, 130), region_name="right")

    proposal = propose_question_rectangles(_analysis(candidate), layout.bounds, layout=layout, padding=6).proposals[0]

    assert proposal.rect == fitz.Rect(275, 94, 336, 136)
    assert proposal.rect.x0 == 275


def test_no_divider_uses_region_content_extent_conservatively():
    region = LayoutRegion("left", fitz.Rect(50, 50, 180, 400), [], kind="left")
    layout = PageLayout(0, fitz.Rect(0, 0, 500, 500), [region])
    candidate = _candidate(bounds=(100, 120, 180, 150), region_name="left")

    proposal = propose_question_rectangles(_analysis(candidate), layout.bounds, layout=layout).proposals[0]

    assert proposal.rect == fitz.Rect(94, 114, 180, 156)
    assert any("conservatively" in warning for warning in proposal.warnings)
    assert proposal.fallback_applied


def test_adjacent_question_proposals_remain_separate():
    first = _candidate(order=1, bounds=(100, 100, 150, 120), label="1.")
    second = _candidate(order=2, bounds=(100, 128, 150, 150), label="2.")

    proposals = propose_question_rectangles(
        _analysis(first, second), fitz.Rect(0, 0, 400, 500), padding=4
    ).proposals

    assert len(proposals) == 2
    assert proposals[0].rect == fitz.Rect(96, 96, 154, 124)
    assert proposals[1].rect == fitz.Rect(96, 124, 154, 154)
    assert proposals[0].candidate_id != proposals[1].candidate_id


def test_different_page_sizes_and_rotation_use_existing_page_space():
    candidate = _candidate(page_index=4, bounds=(70, 40, 90, 55))
    bounds = fitz.Rect(0, 0, 200, 300)

    result = propose_question_rectangles(_analysis(candidate, page_index=4), bounds, padding=2)

    assert result.proposals[0].rect == fitz.Rect(68, 38, 92, 57)
    assert result.proposals[0].page_index == 4


def test_generated_rotated_pdf_pipeline_keeps_page_space_geometry(tmp_path):
    source = tmp_path / "rotated_question.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=300)
    page.insert_text((30, 50), "1. Find the value")
    page.set_rotation(90)
    document.save(source)
    document.close()

    page_analysis = analyze_pdf_document(source).pages[0]
    layout = analyze_page_layout(page_analysis)
    questions = analyze_questions(layout)
    result = propose_question_rectangles(
        questions,
        page_analysis.bounds,
        layout=layout,
        padding=2,
    )

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert 0 <= proposal.rect.x0 <= proposal.rect.x1 <= page_analysis.bounds.width
    assert 0 <= proposal.rect.y0 <= proposal.rect.y1 <= page_analysis.bounds.height
    assert page_analysis.bounds.width == pytest.approx(300)
    assert page_analysis.bounds.height == pytest.approx(200)


@pytest.mark.parametrize(
    "bad_bounds",
    [
        (math.nan, 10, 30, 40),
        (10, 10, 10, 40),
        (10, 10, 40, 10),
    ],
)
def test_invalid_candidate_geometry_is_skipped_with_diagnostics(bad_bounds):
    candidate = _candidate(bounds=bad_bounds)
    result = propose_question_rectangles(_analysis(candidate), fitz.Rect(0, 0, 500, 500))

    assert result.proposals == []
    assert result.skipped_candidate_orders == [1]
    assert "invalid" in result.warnings[0]


def test_empty_candidate_words_are_skipped_with_diagnostics():
    candidate = _candidate(include_word=False)
    result = propose_question_rectangles(_analysis(candidate), fitz.Rect(0, 0, 500, 500))

    assert result.proposals == []
    assert result.skipped_candidate_orders == [1]
    assert "no usable word geometry" in result.warnings[0]


@pytest.mark.parametrize("padding", [-1, math.nan, math.inf, "bad"])
def test_invalid_padding_is_rejected(padding):
    with pytest.raises(ValueError, match="padding"):
        propose_question_rectangles(_analysis(_candidate()), fitz.Rect(0, 0, 500, 500), padding=padding)


def test_invalid_page_bounds_are_rejected():
    with pytest.raises(ValueError, match="Page bounds"):
        propose_question_rectangles(_analysis(_candidate()), fitz.Rect(0, 0, math.inf, 500))


def test_batch_matches_repeated_single_page_processing():
    first = _analysis(_candidate(page_index=0, order=1))
    second = _analysis(_candidate(page_index=1, order=1, bounds=(20, 30, 50, 70)), page_index=1)
    bounds = {0: fitz.Rect(0, 0, 400, 500), 1: fitz.Rect(0, 0, 100, 100)}

    batch = propose_document_rectangles([first, second], bounds, padding=3)
    individual = {
        analysis.page_index: propose_question_rectangles(analysis, bounds[analysis.page_index], padding=3)
        for analysis in (first, second)
    }

    assert {page: [item.rect for item in result.proposals] for page, result in batch.items()} == {
        page: [item.rect for item in result.proposals] for page, result in individual.items()
    }


def test_repeated_calls_are_deterministic():
    analysis = _analysis(_candidate(order=2), _candidate(order=1, bounds=(10, 20, 30, 40)))
    first = propose_question_rectangles(analysis, fitz.Rect(0, 0, 500, 500))
    second = propose_question_rectangles(analysis, fitz.Rect(0, 0, 500, 500))

    assert [(item.order, item.candidate_id, item.rect) for item in first.proposals] == [
        (item.order, item.candidate_id, item.rect) for item in second.proposals
    ]


def test_empty_word_with_nonfinite_geometry_is_not_used():
    candidate = _candidate()
    candidate.words[0] = ExtractedWord("bad", fitz.Rect(0, 0, 1, 1), 0)
    object.__setattr__(candidate.words[0], "rect", fitz.Rect(math.nan, 0, 1, 1))
    result = propose_question_rectangles(_analysis(candidate), fitz.Rect(0, 0, 500, 500))

    assert result.proposals == []
    assert result.skipped_candidate_orders == [1]