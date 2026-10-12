from __future__ import annotations

import math
from pathlib import Path

import fitz
import pytest

from pdfcropper.analysis import analyze_pdf_document
from pdfcropper.ocr import _image_to_page_rect, ocr_page_analysis, ocr_page_words


def _write_pdf(path: Path, *, width: float = 595, height: float = 842, rotation: int = 0, text_lines: list[str] | None = None):
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    if rotation:
        page.set_rotation(rotation)
    for offset, text in enumerate(text_lines or []):
        page.insert_text((30, 80 + offset * 35), text)
    doc.save(path)
    doc.close()


def _tsv_text(lefts=None, tops=None, texts=None):
    rows = [
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    ]
    texts = texts or ["Hello", "World"]
    lefts = lefts or [30 + index * 120 for index in range(len(texts))]
    tops = tops or [80 for _ in texts]
    for index, (left, top, text) in enumerate(zip(lefts, tops, texts), start=1):
        rows.append(f"1\t1\t1\t1\t1\t{index}\t{left}\t{top}\t90\t20\t95\t{text}")
    return "\n".join(rows)


def _tsv_text_for_boxes(boxes):
    rows = [
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    ]
    for index, (left, top, width, height, text) in enumerate(boxes, start=1):
        rows.append(f"1\t1\t1\t1\t1\t{index}\t{left}\t{top}\t{width}\t{height}\t95\t{text}")
    return "\n".join(rows)


def test_ocr_parses_words_and_metadata(monkeypatch, tmp_path):
    source = tmp_path / "ocr.pdf"
    _write_pdf(source, text_lines=["Hello World"])
    monkeypatch.setattr("pdfcropper.ocr._run_tesseract_ocr", lambda image_path: _tsv_text())

    words = ocr_page_words(source, 0, dpi=72.0)

    assert len(words) == 2
    assert [word.text for word in words] == ["Hello", "World"]
    assert words[0].page_index == 0
    assert words[0].metadata["source"] == "ocr"
    assert words[0].metadata["confidence"] == pytest.approx(95.0)
    assert words[0].rect.width > 0
    assert words[0].rect.height > 0


def test_image_to_page_rect_converts_pixel_coordinates_to_pdf_points():
    rect = _image_to_page_rect(10, 20, 110, 40, 2.0)

    assert rect.x0 == pytest.approx(5.0)
    assert rect.y0 == pytest.approx(10.0)
    assert rect.x1 == pytest.approx(55.0)
    assert rect.y1 == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("page_size", "rotation", "pixel_box", "expected_rect"),
    [
        ((200.0, 300.0), 0, (80, 60, 40, 30, "Alpha"), fitz.Rect(40.0, 30.0, 60.0, 45.0)),
        ((200.0, 300.0), 90, (120, 50, 80, 40, "Beta"), fitz.Rect(60.0, 25.0, 100.0, 45.0)),
        ((200.0, 300.0), 180, (30, 40, 90, 35, "Gamma"), fitz.Rect(15.0, 20.0, 60.0, 37.5)),
        ((300.0, 200.0), 270, (100, 70, 60, 25, "Delta"), fitz.Rect(50.0, 35.0, 80.0, 47.5)),
    ],
)
def test_ocr_conversion_matches_exact_page_space_geometry(monkeypatch, tmp_path, page_size, rotation, pixel_box, expected_rect):
    source = tmp_path / f"page_{rotation}.pdf"
    doc = fitz.open()
    page = doc.new_page(width=page_size[0], height=page_size[1])
    page.set_rotation(rotation)
    expected_width = float(page.rect.width)
    expected_height = float(page.rect.height)
    doc.save(source)
    doc.close()

    monkeypatch.setattr("pdfcropper.ocr._run_tesseract_ocr", lambda image_path: _tsv_text_for_boxes([pixel_box]))
    words = ocr_page_words(source, 0, dpi=144.0)

    assert len(words) == 1
    rect = words[0].rect
    assert rect.x0 == pytest.approx(expected_rect.x0)
    assert rect.y0 == pytest.approx(expected_rect.y0)
    assert rect.x1 == pytest.approx(expected_rect.x1)
    assert rect.y1 == pytest.approx(expected_rect.y1)
    assert rect.width == pytest.approx(expected_rect.width)
    assert rect.height == pytest.approx(expected_rect.height)
    assert 0.0 <= rect.x0 <= expected_width
    assert 0.0 <= rect.x1 <= expected_width
    assert 0.0 <= rect.y0 <= expected_height
    assert 0.0 <= rect.y1 <= expected_height


@pytest.mark.parametrize("rotation,width,height", [(0, 595, 842), (90, 842, 595), (180, 595, 842), (270, 842, 595)])
def test_ocr_handles_page_rotations_and_geometry(monkeypatch, tmp_path, rotation, width, height):
    source = tmp_path / f"rotated_{rotation}.pdf"
    _write_pdf(source, width=width, height=height, rotation=rotation, text_lines=["Sample"])
    monkeypatch.setattr("pdfcropper.ocr._run_tesseract_ocr", lambda image_path: _tsv_text())

    words = ocr_page_words(source, 0, dpi=144.0)

    assert words
    assert all(word.page_index == 0 for word in words)
    assert all(math.isfinite(value) for word in words for value in (word.rect.x0, word.rect.y0, word.rect.x1, word.rect.y1))
    assert all(word.rect.width > 0 for word in words)


def test_ocr_uses_requested_page_only(monkeypatch, tmp_path):
    source = tmp_path / "page_select.pdf"
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    doc.new_page(width=300, height=300)
    doc.save(source)
    doc.close()
    monkeypatch.setattr("pdfcropper.ocr._run_tesseract_ocr", lambda image_path: _tsv_text())

    words = ocr_page_words(source, 1, dpi=72.0)

    assert words
    assert all(word.page_index == 1 for word in words)


def test_ocr_reports_invalid_pdf_and_out_of_range_pages(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf")

    with pytest.raises(ValueError, match="Invalid or unreadable PDF"):
        ocr_page_words(bad, 0)

    valid = tmp_path / "good.pdf"
    _write_pdf(valid, text_lines=["One"])

    with pytest.raises(IndexError):
        ocr_page_words(valid, 99)


def test_ocr_handles_missing_tesseract(monkeypatch, tmp_path):
    source = tmp_path / "missing.pdf"
    _write_pdf(source, text_lines=["One"])
    monkeypatch.setattr("pdfcropper.ocr.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="Tesseract"):
        ocr_page_words(source, 0)


def test_ocr_rejects_invalid_dpi(tmp_path):
    source = tmp_path / "dpi.pdf"
    _write_pdf(source, text_lines=["One"])

    with pytest.raises(ValueError, match="positive finite"):
        ocr_page_words(source, 0, dpi=0)


def test_ocr_handles_empty_output(monkeypatch, tmp_path):
    source = tmp_path / "empty.pdf"
    _write_pdf(source, text_lines=["One"])
    monkeypatch.setattr("pdfcropper.ocr._run_tesseract_ocr", lambda image_path: "")

    assert ocr_page_words(source, 0) == []


def test_ocr_surfaces_execution_failures(monkeypatch, tmp_path):
    source = tmp_path / "failure.pdf"
    _write_pdf(source, text_lines=["One"])
    monkeypatch.setattr("pdfcropper.ocr._run_tesseract_ocr", lambda image_path: (_ for _ in ()).throw(RuntimeError("failure")))

    with pytest.raises(RuntimeError, match="failure"):
        ocr_page_words(source, 0)


def test_ocr_page_analysis_uses_task_1_models(monkeypatch, tmp_path):
    source = tmp_path / "analysis.pdf"
    _write_pdf(source, text_lines=["Question", "Alpha beta"])
    monkeypatch.setattr("pdfcropper.ocr._run_tesseract_ocr", lambda image_path: _tsv_text(texts=["Question", "Alpha", "beta"]))

    result = ocr_page_analysis(source, 0, dpi=72.0)

    assert result.page_index == 0
    assert result.width == pytest.approx(595.0)
    assert result.height == pytest.approx(842.0)
    assert result.text
    assert result.words[0].metadata["source"] == "ocr"
    assert result.has_usable_text is True


def test_analysis_api_remains_compatible_with_embedded_text(monkeypatch, tmp_path):
    source = tmp_path / "embedded.pdf"
    _write_pdf(source, text_lines=["Alpha beta", "Gamma"])

    analysis = analyze_pdf_document(source)

    assert analysis.valid
    assert analysis.page_count == 1
    assert analysis.pages[0].words
    assert analysis.pages[0].availability.name in {"USEFUL", "LOW_TEXT"}
