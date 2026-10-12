from pathlib import Path

import pytest

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from pdfcropper.exporter import (
    FALLBACK_PAGE_SIZE,
    QUESTION_IMAGE_INSET_PT,
    QUESTION_IMAGE_MAX_HEIGHT_RATIO,
    QUESTION_IMAGE_MAX_WIDTH_RATIO,
    display_dimensions_to_page_size,
    export_question_regions,
    resolve_workspace_page_size,
)
from pdfcropper.main_window import _workspace_export_filename
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


@pytest.mark.parametrize(
    ("source_path", "expected"),
    [
        ("Laws of motion.pdf", "Laws of motion_workspaces.pdf"),
        ("Relations and Functions.pdf", "Relations and Functions_workspaces.pdf"),
        ("Physics Chapter 1.pdf", "Physics Chapter 1_workspaces.pdf"),
        ("Chapter 1.v2.pdf", "Chapter 1.v2_workspaces.pdf"),
    ],
)
def test_workspace_export_filename_uses_source_stem(source_path, expected):
    assert _workspace_export_filename(source_path) == expected


def test_workspace_export_filename_uses_fallback_without_source():
    assert _workspace_export_filename(None) == "question_workspaces.pdf"


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
    expected_crop_ratios = (260 / 170, 260 / 150)
    for page, expected_ratio in zip(exported, expected_crop_ratios):
        image_objects = page.get_images(full=True)
        assert len(image_objects) == 1
        image_xref, _, pixel_width, pixel_height = image_objects[0][:4]
        assert "/Subtype /Image" in exported.xref_object(image_xref)
        image_rects = page.get_image_rects(image_xref)
        assert len(image_rects) == 1
        image_rect = image_rects[0]
        assert image_rect.x0 == pytest.approx(QUESTION_IMAGE_INSET_PT, abs=0.001)
        assert image_rect.y0 == pytest.approx(QUESTION_IMAGE_INSET_PT, abs=0.001)
        assert image_rect.width < page.rect.width
        assert image_rect.height < page.rect.height
        assert image_rect.width / image_rect.height == pytest.approx(expected_ratio, rel=1e-6)
        assert pixel_width / pixel_height == pytest.approx(expected_ratio, rel=1e-12)
        assert image_rect.width / page.rect.width <= QUESTION_IMAGE_MAX_WIDTH_RATIO + 1e-9
        assert image_rect.height / page.rect.height <= QUESTION_IMAGE_MAX_HEIGHT_RATIO + 1e-9
        assert any(
            drawing["rect"] == page.rect and drawing["fill"] == (0.0, 0.0, 0.0)
            for drawing in page.get_drawings()
        )
    rendered = exported[0].get_pixmap()
    assert rendered.pixel(570, 800) == (0, 0, 0)
    assert rendered.pixel(30, 30) == (255, 255, 255)
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


def test_export_uses_automatically_generated_region_through_existing_path(tmp_path):
    source = tmp_path / "source.pdf"
    output = tmp_path / "automatic_region.pdf"
    _write_test_pdf(source)
    generated_region = QuestionRegion(
        page_index=1,
        rect=fitz.Rect(40, 70, 300, 230),
        label="1",
        source="automatic",
    )

    export_question_regions(str(source), [generated_region], str(output), page_size="A4")

    exported = fitz.open(output)
    assert exported.page_count == 1
    assert exported[0].get_images()
    exported.close()


@pytest.mark.parametrize(
    ("width", "height", "device_pixel_ratio", "logical_dpi", "physical_dpi"),
    [
        (1920, 1080, 1.0, 96.0, 96.0),
        (1440, 2560, 2.0, 96.0, 192.0),
    ],
)
def test_display_page_size_preserves_screen_ratio(
    width, height, device_pixel_ratio, logical_dpi, physical_dpi
):
    page_width, page_height = display_dimensions_to_page_size(
        width, height, device_pixel_ratio, logical_dpi, physical_dpi
    )

    assert page_width / page_height == pytest.approx(width / height, rel=1e-12)
    assert page_width == pytest.approx(width * 72 / logical_dpi)


def test_unreliable_physical_dpi_uses_logical_pixel_conversion():
    page_size = display_dimensions_to_page_size(2560, 1440, 2.0, 96.0, 96.0)

    assert page_size == pytest.approx((1920, 1080))


def test_missing_screen_uses_documented_fallback():
    assert resolve_workspace_page_size(screen=None) == FALLBACK_PAGE_SIZE


def test_default_export_uses_identical_page_dimensions_for_batch(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    output = tmp_path / "automatic.pdf"
    _write_test_pdf(source)
    monkeypatch.setattr("pdfcropper.exporter.QGuiApplication.instance", lambda: None)

    regions = [
        QuestionRegion(page_index=0, rect=fitz.Rect(40, 50, 300, 220), label="1"),
        QuestionRegion(page_index=1, rect=fitz.Rect(60, 80, 320, 230), label="2"),
    ]
    export_question_regions(str(source), regions, str(output))

    exported = fitz.open(output)
    assert exported.page_count == 2
    assert (exported[0].rect.width, exported[0].rect.height) == pytest.approx(FALLBACK_PAGE_SIZE)
    assert (exported[1].rect.width, exported[1].rect.height) == pytest.approx(FALLBACK_PAGE_SIZE)
    exported.close()
