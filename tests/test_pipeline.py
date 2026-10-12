from __future__ import annotations

import fitz
import pytest

import pdfcropper.pipeline as pipeline_module
from pdfcropper.analysis import ExtractedWord, PageAnalysis, TextAvailability
from pdfcropper.layout import LayoutRegion, PageLayout
from pdfcropper.pipeline import PageLayoutMode, PageLayoutOptions, analyze_selected_pages
from pdfcropper.questions import QuestionAnalysis
from pdfcropper.rectangles import RectangleGenerationResult


def _write_pdf(path, page_texts):
    document = fitz.open()
    for lines in page_texts:
        page = document.new_page(width=500, height=700)
        for line_index, text in enumerate(lines):
            page.insert_text((40, 70 + line_index * 32), text)
    document.save(path)
    document.close()


def _ocr_page(page_index: int, text: str = "Q9 Solve the equation") -> PageAnalysis:
    document = fitz.open()
    page = document.new_page(width=500, height=700)
    page.insert_text((40, 70), text)
    raw_words = page.get_text("words")
    words = [
        ExtractedWord(
            text=str(item[4]),
            rect=fitz.Rect(item[:4]),
            page_index=page_index,
            metadata={"source": "ocr"},
        )
        for item in raw_words
    ]
    bounds = fitz.Rect(page.rect)
    document.close()
    return PageAnalysis(
        page_index=page_index,
        width=bounds.width,
        height=bounds.height,
        bounds=bounds,
        text=" ".join(word.text for word in words),
        words=words,
        availability=TextAvailability.USEFUL,
        has_usable_text=True,
    )


def test_orchestration_processes_only_selected_source_pages_and_preserves_indices(tmp_path, monkeypatch):
    source = tmp_path / "selected.pdf"
    _write_pdf(source, [[], ["Q2 Find the value of x"], [], ["Q4 Solve for y"]])
    ocr_calls = []

    def fake_ocr(path, page_index, *, dpi):
        ocr_calls.append(page_index)
        return _ocr_page(page_index)

    monkeypatch.setattr("pdfcropper.pipeline.ocr_page_analysis", fake_ocr)

    result = analyze_selected_pages(source, [3, 1])

    assert result.page_count == 4
    assert result.selected_page_indices == (1, 3)
    assert [page.page_index for page in result.pages] == [1, 3]
    assert [page.page_index for page in result.proposals] == [1, 3]
    assert result.proposal_count == 2
    assert [page.text_source for page in result.pages] == ["embedded text", "embedded text"]
    assert ocr_calls == []
    assert all(page.layout.page_index == page.page_index for page in result.pages)
    assert all(page.questions.page_index == page.page_index for page in result.pages)


def test_ocr_fallback_is_limited_to_selected_pages_without_useful_text(tmp_path, monkeypatch):
    source = tmp_path / "ocr_subset.pdf"
    _write_pdf(source, [["Q1 Find x"], [], ["Q3 Find z"], []])
    ocr_calls = []

    def fake_ocr(path, page_index, *, dpi):
        ocr_calls.append(page_index)
        return _ocr_page(page_index)

    monkeypatch.setattr("pdfcropper.pipeline.ocr_page_analysis", fake_ocr)

    result = analyze_selected_pages(source, [1, 2])

    assert result.selected_page_indices == (1, 2)
    assert ocr_calls == [1]
    assert [page.text_source for page in result.pages] == ["OCR", "embedded text"]
    assert [page.page_index for page in result.proposals] == [1, 2]


def test_explicit_ocr_replaces_useful_embedded_text_for_selected_source_pages(tmp_path, monkeypatch):
    source = tmp_path / "forced_ocr.pdf"
    _write_pdf(source, [["Q1 embedded text is useful"], ["Q2 stale text"], [], ["Q4 stale text"]])
    ocr_calls = []
    ocr_page = _ocr_page(1, "Q9 Solve the equation")

    def fake_ocr(path, page_index, *, dpi):
        ocr_calls.append(page_index)
        page = _ocr_page(page_index, "Q9 Solve the equation")
        if page_index == 3:
            page = _ocr_page(page_index, "Q4 Calculate the answer")
        return page

    monkeypatch.setattr("pdfcropper.pipeline.ocr_page_analysis", fake_ocr)
    result = analyze_selected_pages(source, [3, 1], force_ocr=True)

    assert result.selected_page_indices == (1, 3)
    assert ocr_calls == [1, 3]
    assert [page.page_index for page in result.pages] == [1, 3]
    assert all(page.text_source == "OCR" for page in result.pages)
    assert all(page.page_analysis.words[0].metadata["source"] == "ocr" for page in result.pages)
    assert all(page.layout.page_index == page.page_index for page in result.pages)
    assert all(page.questions.page_index == page.page_index for page in result.pages)
    assert all(page.rectangles.page_index == page.page_index for page in result.pages)
    assert result.proposal_count >= 1
    assert [proposal.page_index for proposal in result.proposals] == [1, 3]
    assert ocr_page.words[0].text == "Q9"


def test_explicit_ocr_failure_does_not_fall_back_to_embedded_words(tmp_path, monkeypatch):
    source = tmp_path / "forced_ocr_failure.pdf"
    _write_pdf(source, [["Q1 Embedded words should not be analyzed"]])
    monkeypatch.setattr(
        "pdfcropper.pipeline.ocr_page_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Tesseract unavailable")),
    )

    result = analyze_selected_pages(source, [0], force_ocr=True)

    assert result.pages[0].text_source == "OCR"
    assert result.pages[0].failures
    assert "Tesseract unavailable" in result.pages[0].failures[0]
    assert result.pages[0].layout is None
    assert result.pages[0].proposals == []


def test_ocr_failure_is_page_diagnostic_and_other_pages_continue(tmp_path, monkeypatch):
    source = tmp_path / "ocr_failure.pdf"
    _write_pdf(source, [["Q1 Find x"], [], ["Q3 Find z"]])

    def fake_ocr(path, page_index, *, dpi):
        raise RuntimeError("local OCR unavailable")

    monkeypatch.setattr("pdfcropper.pipeline.ocr_page_analysis", fake_ocr)

    result = analyze_selected_pages(source, [0, 1, 2])

    assert [page.page_index for page in result.pages] == [0, 1, 2]
    failed_page = result.pages[1]
    assert failed_page.failures
    assert "page 2" in failed_page.failures[0]
    assert "local OCR unavailable" in failed_page.failures[0]
    assert "OCR failed" in failed_page.warnings[0]
    assert any("page 2" in failure for failure in result.failures)
    assert result.pages[0].proposals
    assert result.pages[2].proposals


def test_pipeline_preserves_ambiguities_warnings_skips_and_aggregation(tmp_path, monkeypatch):
    source = tmp_path / "diagnostics.pdf"
    _write_pdf(source, [["Q1 Find x"], ["Q2 Solve y"]])
    original_layout = pipeline_module.analyze_page_layout
    original_questions = pipeline_module.analyze_questions
    layout_calls = []
    question_calls = []

    def traced_layout(page):
        layout_calls.append(page.page_index)
        result = original_layout(page)
        result.warnings.append(f"layout-note-{page.page_index}")
        return result

    def traced_questions(layout):
        question_calls.append(layout.page_index)
        result = original_questions(layout)
        result.ambiguous_lines.append(type("Ambiguity", (), {"reason": "test ambiguity"})())
        return result

    def fake_proposals(questions, page_bounds, *, layout, padding):
        return RectangleGenerationResult(
            page_index=questions.page_index,
            warnings=[f"proposal-note-{questions.page_index}"],
            skipped_candidate_orders=[99],
        )

    monkeypatch.setattr("pdfcropper.pipeline.analyze_page_layout", traced_layout)
    monkeypatch.setattr("pdfcropper.pipeline.analyze_questions", traced_questions)
    monkeypatch.setattr("pdfcropper.pipeline.propose_question_rectangles", fake_proposals)

    result = analyze_selected_pages(source, [1, 0])

    assert layout_calls == [0, 1]
    assert question_calls == [0, 1]
    assert result.proposal_count == 0
    assert [page.skipped_candidate_orders for page in result.pages] == [[99], [99]]
    assert all(page.ambiguous_lines[0].reason == "test ambiguity" for page in result.pages)
    assert result.warnings == [
        "No stable column divider was supported by repeated content on both sides; keeping one region.",
        "layout-note-0",
        "proposal-note-0",
        "No stable column divider was supported by repeated content on both sides; keeping one region.",
        "layout-note-1",
        "proposal-note-1",
    ]


def test_layout_failure_is_recorded_and_later_selected_pages_continue(tmp_path, monkeypatch):
    source = tmp_path / "layout_failure.pdf"
    _write_pdf(source, [["Q1 Find x"], ["Q2 Solve y"]])
    original_layout = pipeline_module.analyze_page_layout

    def fail_first_page(page):
        if page.page_index == 0:
            raise RuntimeError("bad layout")
        return original_layout(page)

    monkeypatch.setattr("pdfcropper.pipeline.analyze_page_layout", fail_first_page)

    result = analyze_selected_pages(source, [0, 1])

    assert result.pages[0].failures[0].startswith("Layout analysis failed for page 1")
    assert result.pages[0].questions is None
    assert result.pages[1].proposals


def test_layout_modes_route_to_automatic_single_and_manual_divider(tmp_path):
    source = tmp_path / "layout_modes.pdf"
    document = fitz.open()
    page = document.new_page(width=500, height=700)
    for line_index, y in enumerate((70, 145, 220, 295, 370, 445, 520, 595), start=1):
        page.insert_text((35, y), f"Left column line {line_index}")
        page.insert_text((325, y + 5), f"Right column line {line_index}")
    document.save(source)
    document.close()
    indices = [0]

    automatic = analyze_selected_pages(source, indices)
    single = analyze_selected_pages(
        source,
        indices,
        layout_options={0: PageLayoutOptions(PageLayoutMode.SINGLE_COLUMN)},
    )
    manual = analyze_selected_pages(
        source,
        indices,
        layout_options={
            0: PageLayoutOptions(PageLayoutMode.TWO_COLUMNS, divider_normalized=0.5)
        },
    )

    assert automatic.pages[0].layout is not None
    assert automatic.pages[0].layout.split_detected
    assert single.pages[0].layout is not None
    assert single.pages[0].layout.split_detected is False
    assert single.pages[0].layout.regions[-1].kind == "single"
    assert manual.pages[0].layout is not None
    assert manual.pages[0].layout.split_source == "manual"
    assert manual.pages[0].layout.split_x == pytest.approx(250.0)


def test_two_column_mode_without_divider_fails_instead_of_falling_back(tmp_path):
    source = tmp_path / "missing_divider.pdf"
    _write_pdf(source, [["Q1 A question with text"]])

    result = analyze_selected_pages(
        source,
        [0],
        layout_options={0: PageLayoutOptions(PageLayoutMode.TWO_COLUMNS)},
    )

    assert result.pages[0].failures
    assert "requires a manually positioned divider" in result.pages[0].failures[0]
    assert result.pages[0].layout is None


def test_invalid_document_returns_run_level_error(tmp_path):
    source = tmp_path / "invalid.pdf"
    source.write_text("not a pdf")

    result = analyze_selected_pages(source, [0])

    assert result.document_error
    assert result.pages == []
    assert result.proposal_count == 0