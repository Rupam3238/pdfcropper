from __future__ import annotations

import fitz
import pytest

from pdfcropper.analysis import ExtractedWord, PageAnalysis, analyze_pdf_document
from pdfcropper.layout import LayoutRegion, PageLayout, analyze_page_layout


def _word(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_index: int = 0,
    block_no: int = 0,
    line_no: int = 0,
) -> ExtractedWord:
    return ExtractedWord(
        text=text,
        rect=fitz.Rect(x0, y0, x1, y1),
        page_index=page_index,
        block_no=block_no,
        line_no=line_no,
    )


def _make_two_column_words(page_width: float = 600.0, page_height: float = 800.0):
    left = [
        _word("Alpha", 40, 80, 130, 100),
        _word("Beta", 50, 160, 140, 180),
        _word("Gamma", 35, 280, 125, 300),
        _word("Delta", 60, 420, 150, 440),
    ]
    right = [
        _word("Echo", 360, 90, 430, 110),
        _word("Foxtrot", 330, 200, 470, 220),
        _word("Golf", 340, 310, 400, 330),
        _word("Hotel", 350, 460, 440, 480),
    ]
    return left + right


def _make_full_width_heading_words():
    return [
        _word("Instructions", 60, 20, 520, 40),
        _word("Page", 100, 50, 160, 70),
        _word("notes", 420, 50, 470, 70),
    ] + _make_two_column_words()


def test_single_column_page_is_not_split():
    words = [
        _word("One", 40, 60, 90, 80),
        _word("Two", 50, 170, 100, 190),
        _word("Three", 55, 300, 130, 320),
        _word("Four", 35, 450, 120, 470),
    ]

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800))

    assert layout.split_detected is False
    assert len(layout.regions) == 1
    assert layout.regions[0].kind == "single"
    assert [word.text for word in layout.regions[0].words] == ["One", "Two", "Three", "Four"]
    assert any("keeping one region" in warning for warning in layout.warnings)


def test_dense_single_column_lines_are_not_split_by_an_interior_gap():
    words = []
    for index in range(8):
        y = 60 + index * 70
        words.extend([
            _word("This", 40, y, 120, y + 18, block_no=1, line_no=index + 1),
            _word("is continuous", 130, y, 300, y + 18, block_no=1, line_no=index + 1),
            _word("single-column text", 310, y, 560, y + 18, block_no=1, line_no=index + 1),
        ])

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 700))

    assert not layout.split_detected
    assert len(layout.regions) == 1
    assert any("Reliable text lines did not support separate columns" in warning for warning in layout.warnings)


def test_two_column_page_detects_columns_and_reading_order():
    words = _make_two_column_words()

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800))

    assert layout.split_detected is True
    assert len(layout.regions) == 2
    assert [region.kind for region in layout.regions] == ["left", "right"]
    assert [word.text for word in layout.regions[0].words] == ["Alpha", "Beta", "Gamma", "Delta"]
    assert [word.text for word in layout.regions[1].words] == ["Echo", "Foxtrot", "Golf", "Hotel"]
    assert layout.regions[0].rect.x1 <= layout.regions[1].rect.x0


def test_line_metadata_supports_a_repeated_gutter_and_separate_regions():
    words = []
    for line_index, y in enumerate((80, 170, 260, 350, 440, 530), start=1):
        words.append(_word(f"L{line_index}", 40, y, 130, y + 18, block_no=1, line_no=line_index))
        words.append(_word(f"R{line_index}", 360, y + 8, 450, y + 26, block_no=2, line_no=line_index))

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 700))

    assert layout.split_detected
    assert layout.split_x == pytest.approx(245.0, abs=20.0)
    assert [region.kind for region in layout.regions] == ["left", "right"]
    assert all(word.rect.x1 <= layout.split_x for word in layout.regions[0].words)
    assert all(word.rect.x0 >= layout.split_x for word in layout.regions[1].words)


def test_small_repeated_gap_inside_question_lines_is_not_used_as_divider():
    words = []
    for line_index, y in enumerate((80, 170, 260, 350, 440, 530), start=1):
        words.extend([
            _word("Q", 40, y, 65, y + 18, block_no=1, line_no=line_index),
            _word("value", 68, y, 115, y + 18, block_no=1, line_no=line_index),
            _word("solution", 132, y, 205, y + 18, block_no=1, line_no=line_index),
        ])

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 400, 700))

    assert not layout.split_detected
    assert len(layout.regions) == 1
    assert any("did not support separate columns" in warning for warning in layout.warnings)


def test_non_midpoint_split_is_detected_when_supported():
    words = [
        _word("L1", 30, 80, 90, 100),
        _word("L2", 40, 200, 100, 220),
        _word("L3", 45, 320, 110, 340),
        _word("R1", 250, 90, 320, 110),
        _word("R2", 260, 220, 330, 240),
        _word("R3", 255, 340, 325, 360),
    ]

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 420, 500))

    assert layout.split_detected is True
    assert layout.split_x is not None
    assert layout.split_x > 140.0
    assert layout.split_x < 260.0
    assert [region.kind for region in layout.regions] == ["left", "right"]


def test_full_width_heading_is_kept_before_columns():
    words = _make_full_width_heading_words()

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800))

    assert layout.split_detected is True
    assert layout.regions[0].kind == "full_width"
    assert [word.text for word in layout.regions[0].words] == ["Instructions", "Page", "notes"]
    assert [region.kind for region in layout.regions[1:]] == ["left", "right"]


def test_supplied_divider_creates_expected_regions():
    words = _make_two_column_words()

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800), divider=0.55, divider_is_normalized=True)

    assert layout.split_source == "manual"
    assert layout.split_detected is True
    assert layout.split_x == pytest.approx(330.0)
    assert [region.kind for region in layout.regions] == ["left", "right"]
    assert len(layout.regions[0].words) >= 2
    assert len(layout.regions[1].words) >= 2


def test_invalid_divider_values_are_rejected():
    words = _make_two_column_words()

    with pytest.raises(ValueError, match="Divider"):
        analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800), divider=1.0)

    with pytest.raises(ValueError, match="Divider"):
        analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800), divider=600.0, divider_is_normalized=False)


def test_words_crossing_divider_are_handled_consistently():
    words = [
        _word("Left", 40, 80, 120, 100),
        _word("Cross", 120, 80, 220, 100),
        _word("Right", 220, 80, 300, 100),
    ]

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 400, 200), divider=170.0, divider_is_normalized=False)

    assert layout.split_detected is True
    assert sum(len(region.words) for region in layout.regions) == 3
    assert any("Cross" in [word.text for region in layout.regions for word in region.words] for _ in [0])


def test_sparse_page_falls_back_to_single_region():
    words = [
        _word("Only", 60, 80, 120, 100),
        _word("three", 70, 220, 130, 240),
    ]

    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 300, 400))

    assert layout.split_detected is False
    assert len(layout.regions) == 1
    assert layout.regions[0].kind == "single"


def test_forced_single_column_skips_automatic_split_detection():
    words = _make_two_column_words()

    layout = analyze_page_layout(
        words,
        page_index=0,
        bounds=fitz.Rect(0, 0, 600, 800),
        force_single_column=True,
    )

    assert not layout.split_detected
    assert [region.kind for region in layout.regions] == ["single"]
    assert len(layout.regions[0].words) == len(words)
    assert any("explicitly selected" in warning for warning in layout.warnings)


def test_forced_single_column_cannot_be_combined_with_manual_divider():
    with pytest.raises(ValueError, match="cannot be combined"):
        analyze_page_layout(
            _make_two_column_words(),
            page_index=0,
            bounds=fitz.Rect(0, 0, 600, 800),
            divider=0.5,
            divider_is_normalized=True,
            force_single_column=True,
        )


def test_page_layout_works_for_rotated_pages_and_different_sizes():
    words = [
        _word("Left", 40, 80, 120, 100),
        _word("Right", 180, 80, 260, 100),
    ]
    page = PageAnalysis(page_index=0, width=300.0, height=200.0, rotation=90, bounds=fitz.Rect(0, 0, 300, 200), words=words)

    layout = analyze_page_layout(page)

    assert layout.page_index == 0
    assert layout.bounds.width == pytest.approx(300.0)
    assert layout.bounds.height == pytest.approx(200.0)
    assert layout.split_detected is True


def test_embedded_text_and_ocr_words_share_layout_api(tmp_path):
    source = tmp_path / "layout.pdf"
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((50, 80), "Alpha")
    page.insert_text((360, 90), "Echo")
    page.insert_text((50, 200), "Beta")
    page.insert_text((350, 220), "Foxtrot")
    doc.save(source)
    doc.close()

    analysis = analyze_pdf_document(source)
    embedded_layout = analyze_page_layout(analysis.pages[0])
    synthetic_words = [
        _word("Alpha", 50, 80, 110, 100),
        _word("Echo", 360, 90, 430, 110),
        _word("Beta", 50, 200, 110, 220),
        _word("Foxtrot", 350, 220, 470, 240),
    ]
    ocr_layout = analyze_page_layout(synthetic_words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800))

    assert embedded_layout.split_detected is True
    assert ocr_layout.split_detected is True
    assert [region.kind for region in embedded_layout.regions] == ["left", "right"]
    assert [region.kind for region in ocr_layout.regions] == ["left", "right"]


def test_layout_output_is_deterministic():
    words = _make_two_column_words()

    first = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800))
    second = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800))

    assert [region.name for region in first.regions] == [region.name for region in second.regions]
    assert [word.text for region in first.regions for word in region.words] == [word.text for region in second.regions for word in region.words]


def test_layout_region_model_is_usable_for_later_tasks():
    words = _make_two_column_words()
    layout = analyze_page_layout(words, page_index=0, bounds=fitz.Rect(0, 0, 600, 800))

    assert isinstance(layout, PageLayout)
    assert all(isinstance(region, LayoutRegion) for region in layout.regions)
    assert layout.reading_order == layout.regions
    assert layout.regions[0].word_count > 0
