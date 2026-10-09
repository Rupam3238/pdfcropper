from pathlib import Path

import pytest

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from pdfcropper.exporter import export_question_regions
from pdfcropper.models import QuestionRegion


def _write_test_pdf(path: Path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 80), "Question 1")
    page.insert_text((50, 130), "Alpha beta gamma")
    page.draw_circle((210, 120), 24, color=(0, 0, 0), width=2)
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 90), "Question 2")
    page.insert_text((50, 150), "Delta epsilon zeta")
    doc.save(path)
    doc.close()


def test_export_creates_one_page_per_question(tmp_path):
    source = tmp_path / "source.pdf"
    output = tmp_path / "result.pdf"
    _write_test_pdf(source)

    regions = [
        QuestionRegion(page_index=0, rect=fitz.Rect(40, 50, 300, 220), label="1"),
        QuestionRegion(page_index=1, rect=fitz.Rect(60, 80, 320, 230), label="2"),
    ]

    export_question_regions(str(source), regions, str(output), page_size="A4")
    assert output.exists()
    exported = fitz.open(output)
    assert exported.page_count == 2
    assert exported[0].get_images()
    assert exported[1].get_images()
    assert exported[0].rect.width == pytest.approx(595.28, abs=1)
    assert exported[0].rect.height == pytest.approx(841.89, abs=1)
    assert not exported[0].get_text().strip()
    rendered = exported[0].get_pixmap()
    assert rendered.pixel(570, 800) == (0, 0, 0)
    assert rendered.pixel(50, 50) == (255, 255, 255)
    exported.close()


def test_export_rejects_invalid_crop_rectangles(tmp_path):
    source = tmp_path / "source.pdf"
    output = tmp_path / "result.pdf"
    _write_test_pdf(source)

    invalid_regions = [
        QuestionRegion(page_index=0, rect=fitz.Rect(0, 0, 0, 0), label="bad"),
        QuestionRegion(page_index=0, rect=fitz.Rect(-500, -500, -20, -10), label="out"),
    ]

    for region in invalid_regions:
        with pytest.raises(ValueError):
            export_question_regions(str(source), [region], str(output), page_size="A4")
