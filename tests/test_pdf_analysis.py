from __future__ import annotations

import fitz
import pytest
import pdfcropper.analysis as analysis_module

from pdfcropper.analysis import TextAvailability, analyze_pdf_document, page_indices_for_range


def _write_pdf_with_text(path, page_texts=None, *, rotations=None):
    doc = fitz.open()
    page_texts = page_texts or [[]]
    rotations = rotations or [0] * len(page_texts)

    for page_index, texts in enumerate(page_texts):
        page = doc.new_page(width=595, height=842)
        if rotations[page_index]:
            page.set_rotation(rotations[page_index])
        for offset, text in enumerate(texts):
            page.insert_text((50, 80 + offset * 40), text)
    doc.save(path)
    doc.close()


def test_pdf_analysis_extracts_embedded_text_and_words(tmp_path):
    source = tmp_path / "text.pdf"
    _write_pdf_with_text(source, [["Question 1", "Alpha beta gamma"]])

    analysis = analyze_pdf_document(source)

    assert analysis.valid
    assert analysis.page_count == 1
    page = analysis.pages[0]
    assert page.page_index == 0
    assert page.width == pytest.approx(595.0)
    assert page.height == pytest.approx(842.0)
    assert page.rotation == 0
    assert page.text.strip()
    assert page.availability == TextAvailability.USEFUL
    assert page.has_usable_text
    assert len(page.words) >= 5
    assert all(word.page_index == 0 for word in page.words)
    assert all(word.rect.x0 <= word.rect.x1 for word in page.words)
    assert all(word.rect.y0 <= word.rect.y1 for word in page.words)
    assert all(word.rect.x0 == pytest.approx(word.rect.x0) for word in page.words)


def test_pdf_analysis_tracks_multiple_lines_and_word_boxes(tmp_path):
    source = tmp_path / "multi_line.pdf"
    _write_pdf_with_text(source, [["First line", "Second line", "Third line"]])

    analysis = analyze_pdf_document(source)
    page = analysis.pages[0]

    assert len(page.text_blocks) >= 1
    assert len(page.words) >= 6
    assert page.words[0].text
    assert page.words[0].rect.width > 0
    assert page.words[0].rect.height > 0


def test_pdf_analysis_reports_empty_pages_and_mixed_documents(tmp_path):
    source = tmp_path / "mixed.pdf"
    doc = fitz.open()
    first = doc.new_page(width=595, height=842)
    first.insert_text((80, 110), "Useful text on page one")
    doc.new_page(width=595, height=842)
    third = doc.new_page(width=595, height=842)
    third.insert_text((80, 110), "Useful text on page three")
    doc.save(source)
    doc.close()

    analysis = analyze_pdf_document(source)

    assert analysis.page_count == 3
    assert analysis.pages[0].availability == TextAvailability.USEFUL
    assert analysis.pages[1].availability == TextAvailability.NO_TEXT
    assert analysis.pages[2].availability == TextAvailability.USEFUL
    assert analysis.pages[1].has_usable_text is False


def test_pdf_analysis_handles_different_page_sizes_and_rotations(tmp_path):
    source = tmp_path / "rotated.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=300)
    page.insert_text((25, 30), "Rotated")
    page.set_rotation(90)
    doc.new_page(width=400, height=700)
    doc.save(source)
    doc.close()

    analysis = analyze_pdf_document(source)

    assert analysis.pages[0].rotation == 90
    assert analysis.pages[0].width == pytest.approx(300.0)
    assert analysis.pages[0].height == pytest.approx(200.0)
    assert analysis.pages[0].bounds.width == pytest.approx(300.0)
    assert analysis.pages[0].bounds.height == pytest.approx(200.0)
    assert analysis.pages[1].width == pytest.approx(400.0)
    assert analysis.pages[1].height == pytest.approx(700.0)


def test_pdf_analysis_classifies_low_text_as_insufficient_for_auto_processing(tmp_path):
    source = tmp_path / "low_text.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 80), "A")
    doc.save(source)
    doc.close()

    analysis = analyze_pdf_document(source)

    assert analysis.pages[0].availability == TextAvailability.LOW_TEXT
    assert analysis.pages[0].has_usable_text is False
    assert len(analysis.pages[0].words) == 1


def test_pdf_analysis_handles_invalid_documents_gracefully(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_text("this is not a valid pdf\n")

    analysis = analyze_pdf_document(bad)

    assert not analysis.valid
    assert analysis.error
    assert analysis.pages == []
    assert analysis.page_count == 0


def test_pdf_analysis_uses_finite_word_bounding_boxes(tmp_path):
    source = tmp_path / "boxes.pdf"
    _write_pdf_with_text(source, [["Alpha beta", "Gamma"]])

    analysis = analyze_pdf_document(source)
    word = analysis.pages[0].words[0]

    assert all(value == value for value in (word.rect.x0, word.rect.y0, word.rect.x1, word.rect.y1))
    assert all(float(value) for value in (word.rect.x0, word.rect.y0, word.rect.x1, word.rect.y1))
    assert word.page_index == 0


def test_pdf_analysis_ignores_truly_empty_pages_but_keeps_valid_document(tmp_path):
    source = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.new_page(width=300, height=500)
    doc.save(source)
    doc.close()

    analysis = analyze_pdf_document(source)

    assert analysis.valid
    assert analysis.page_count == 2
    assert all(page.availability == TextAvailability.NO_TEXT for page in analysis.pages)
    assert not any(page.has_usable_text for page in analysis.pages)


@pytest.mark.parametrize(
    ("first_page", "last_page", "page_count", "expected"),
    [
        (1, 4, 4, [0, 1, 2, 3]),
        (2, 2, 5, [1]),
        (1, 3, 3, [0, 1, 2]),
        (4, 4, 4, [3]),
    ],
)
def test_page_indices_for_inclusive_range(first_page, last_page, page_count, expected):
    assert list(page_indices_for_range(first_page, last_page, page_count)) == expected


@pytest.mark.parametrize(
    ("first_page", "last_page", "page_count", "message"),
    [
        (3, 2, 4, "First page"),
        (0, 2, 4, "at least 1"),
        (1, 5, 4, "must not exceed"),
        (1, 1, 0, "Page count"),
        (True, 1, 4, "integers"),
    ],
)
def test_page_indices_for_invalid_range_is_explicit(first_page, last_page, page_count, message):
    with pytest.raises(ValueError, match=message):
        page_indices_for_range(first_page, last_page, page_count)


def test_selected_page_analysis_preserves_source_indices_and_order(tmp_path):
    source = tmp_path / "selected_pages.pdf"
    _write_pdf_with_text(source, [["Page zero"], ["Page one"], ["Page two"], ["Page three"]])

    analysis = analyze_pdf_document(source, page_indices=[3, 1])

    assert analysis.valid
    assert analysis.page_count == 4
    assert [page.page_index for page in analysis.pages] == [1, 3]
    assert [page.text for page in analysis.pages] == ["Page one", "Page three"]


def test_selected_page_analysis_does_not_extract_excluded_pages(tmp_path, monkeypatch):
    source = tmp_path / "selective_extraction.pdf"
    _write_pdf_with_text(source, [["Page zero"], ["Page one"], ["Page two"], ["Page three"]])
    extracted_word_pages = []
    extracted_block_pages = []
    original_extract_words = analysis_module._extract_words
    original_extract_blocks = analysis_module._extract_blocks

    def track_words(page, page_index):
        extracted_word_pages.append(page_index)
        return original_extract_words(page, page_index)

    def track_blocks(page):
        extracted_block_pages.append(page.number)
        return original_extract_blocks(page)

    monkeypatch.setattr("pdfcropper.analysis._extract_words", track_words)
    monkeypatch.setattr("pdfcropper.analysis._extract_blocks", track_blocks)

    analysis = analyze_pdf_document(source, page_indices=range(1, 3))

    assert analysis.page_count == 4
    assert extracted_word_pages == [1, 2]
    assert extracted_block_pages == [1, 2]
    assert [page.page_index for page in analysis.pages] == [1, 2]


@pytest.mark.parametrize("indices", [[], [-1], [4], [1, 1], [1.0], 2])
def test_selected_page_analysis_rejects_invalid_indices(tmp_path, indices):
    source = tmp_path / "invalid_selected_pages.pdf"
    _write_pdf_with_text(source, [["Page zero"], ["Page one"], ["Page two"]])

    with pytest.raises(ValueError):
        analyze_pdf_document(source, page_indices=indices)
