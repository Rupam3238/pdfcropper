from pathlib import Path

import pytest

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from pdfcropper.exporter import (
    FALLBACK_PAGE_SIZE,
    DEFAULT_EXPORT_SCALING_MODE,
    CONSISTENT_MAX_HEIGHT_RATIO,
    CONSISTENT_TEXT_SIZE,
    DEFAULT_ASSUMED_TEXT_SIZE_PT,
    FIT_TO_AVAILABLE_AREA,
    MAX_CONSISTENT_SCALE,
    MIN_CONSISTENT_SCALE,
    QUESTION_IMAGE_INSET_PT,
    QUESTION_IMAGE_MAX_HEIGHT_RATIO,
    QUESTION_IMAGE_MAX_WIDTH_RATIO,
    TARGET_TEXT_SIZE_PT,
    display_dimensions_to_page_size,
    calculate_consistent_scale,
    estimate_region_text_size,
    export_question_regions,
    resolve_question_image_placement,
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


def _image_annotations(document, page):
    """Return (rect, pixel_width, pixel_height) for each image annotation on a page."""
    found = []
    for annot in page.annots() or []:
        assert annot.type[1] == "Square"
        kind, ap_ref = document.xref_get_key(annot.xref, "AP/N")
        assert kind == "xref"
        ap_xref = int(ap_ref.split()[0])
        kind, image_ref = document.xref_get_key(ap_xref, "Resources/XObject/Im0")
        assert kind == "xref"
        image_xref = int(image_ref.split()[0])
        assert "/Subtype /Image" in document.xref_object(image_xref)
        pixel_width = int(document.xref_get_key(image_xref, "Width")[1])
        pixel_height = int(document.xref_get_key(image_xref, "Height")[1])
        found.append((fitz.Rect(annot.rect), pixel_width, pixel_height))
    return sorted(found, key=lambda item: item[0].y0)


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

    export_question_regions(
        str(source), regions, str(output), page_size="A4", scaling_mode=FIT_TO_AVAILABLE_AREA
    )
    assert output.exists()
    exported = fitz.open(output)
    assert exported.page_count == 2
    assert exported[0].rect.width == pytest.approx(595.28, abs=1)
    assert exported[0].rect.height == pytest.approx(841.89, abs=1)
    assert not exported[0].get_text().strip()
    expected_crop_ratios = (260 / 170, 260 / 150)
    for page, expected_ratio in zip(exported, expected_crop_ratios):
        # The image is a movable annotation, not flattened into the page content.
        assert not page.get_images()
        image_objects = _image_annotations(exported, page)
        assert len(image_objects) == 1
        image_rect, pixel_width, pixel_height = image_objects[0]
        (annot,) = list(page.annots())
        assert annot.flags == 4
        assert exported.xref_get_key(annot.xref, "BS/W")[1] == "0"
        expected_x, expected_y, expected_width, expected_height = resolve_question_image_placement(
            page.rect.width, page.rect.height, pixel_width, pixel_height, QUESTION_IMAGE_INSET_PT
        )
        assert image_rect.x0 == pytest.approx(QUESTION_IMAGE_INSET_PT, abs=0.001)
        assert image_rect.y0 == pytest.approx(QUESTION_IMAGE_INSET_PT, abs=0.001)
        assert image_rect.x0 == pytest.approx(expected_x, abs=0.001)
        assert image_rect.y0 == pytest.approx(page.rect.height - expected_y - expected_height, abs=0.001)
        assert image_rect.width == pytest.approx(expected_width, abs=0.001)
        assert image_rect.height == pytest.approx(expected_height, abs=0.001)
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


def test_consistent_scaling_uses_pdf_text_span_geometry(tmp_path):
    assert DEFAULT_EXPORT_SCALING_MODE == CONSISTENT_TEXT_SIZE
    source = tmp_path / "font_sizes.pdf"
    document = fitz.open()
    page = document.new_page(width=500, height=700)
    page.insert_text((40, 80), "Smaller source text", fontsize=10)
    page.insert_text((40, 160), "Larger source text", fontsize=20)
    document.save(source)
    document.close()

    pdf = fitz.open(source)
    small_region = QuestionRegion(page_index=0, rect=fitz.Rect(20, 40, 350, 100))
    large_region = QuestionRegion(page_index=0, rect=fitz.Rect(20, 120, 450, 180))
    small_text_size = estimate_region_text_size(pdf, small_region)
    large_text_size = estimate_region_text_size(pdf, large_region)
    small_scale = calculate_consistent_scale(100, 40, small_text_size, 250, 260)
    large_scale = calculate_consistent_scale(100, 40, large_text_size, 250, 260)
    pdf.close()

    assert small_text_size == pytest.approx(10)
    assert large_text_size == pytest.approx(20)
    assert small_scale == pytest.approx(1.5)
    assert large_scale == pytest.approx(0.8)
    assert small_text_size * small_scale == pytest.approx(15)
    assert large_text_size * large_scale == pytest.approx(16)


def test_consistent_scaling_uses_assumed_text_size_without_text_geometry(tmp_path):
    source = tmp_path / "scanned_page.pdf"
    document = fitz.open()
    page = document.new_page(width=500, height=700)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), 0)
    pixmap.clear_with(180)
    page.insert_image(fitz.Rect(20, 20, 320, 320), stream=pixmap.tobytes("png"))
    document.save(source)
    document.close()

    pdf = fitz.open(source)
    scanned_region = QuestionRegion(page_index=0, rect=fitz.Rect(20, 20, 320, 320))
    text_size = estimate_region_text_size(pdf, scanned_region)
    fallback_scale = calculate_consistent_scale(300, 300, text_size, 250, 260)
    small_fallback = calculate_consistent_scale(30, 40, None, 250, 260)
    large_fallback = calculate_consistent_scale(500, 400, None, 250, 260)
    pdf.close()

    assumed_scale = TARGET_TEXT_SIZE_PT / DEFAULT_ASSUMED_TEXT_SIZE_PT

    assert text_size is None
    # Without text the scale no longer depends on the crop's height: big crops get big space.
    assert small_fallback == pytest.approx(assumed_scale)
    assert fallback_scale == pytest.approx(250 / 300)  # only the area cap shrinks it
    assert large_fallback == pytest.approx(MIN_CONSISTENT_SCALE)  # 250 / 500


def test_consistent_scaling_respects_area_caps_and_preserves_aspect_ratio(tmp_path):
    source = tmp_path / "scaling_bounds.pdf"
    output = tmp_path / "consistent_bounds.pdf"
    document = fitz.open()
    page = document.new_page(width=1200, height=1400)
    page.insert_text((30, 45), "Small text", fontsize=10)
    page.insert_text((30, 1100), "Large crop text", fontsize=10)
    document.save(source)
    document.close()
    regions = [
        QuestionRegion(page_index=0, rect=fitz.Rect(20, 20, 120, 60)),
        QuestionRegion(page_index=0, rect=fitz.Rect(20, 100, 1120, 1250)),
    ]

    export_question_regions(
        str(source), regions, str(output), page_size="A4", scaling_mode=CONSISTENT_TEXT_SIZE
    )

    exported = fitz.open(output)
    for page_index, region in enumerate(regions):
        workspace_page = exported[page_index]
        image_rect = _image_annotations(exported, workspace_page)[0][0]
        assert image_rect.width / image_rect.height == pytest.approx(
            region.rect.width / region.rect.height, rel=1e-3
        )
        assert image_rect.x0 >= QUESTION_IMAGE_INSET_PT - 0.01
        assert image_rect.y0 >= QUESTION_IMAGE_INSET_PT - 0.01
        assert image_rect.x1 <= workspace_page.rect.width - QUESTION_IMAGE_INSET_PT + 0.01
        assert image_rect.y1 <= QUESTION_IMAGE_INSET_PT + workspace_page.rect.height * CONSISTENT_MAX_HEIGHT_RATIO + 0.01
    # A large crop may use far more than the old 42% x 32% corner.
    big_rect = _image_annotations(exported, exported[1])[0][0]
    assert big_rect.width > exported[1].rect.width * QUESTION_IMAGE_MAX_WIDTH_RATIO
    assert big_rect.height > exported[1].rect.height * QUESTION_IMAGE_MAX_HEIGHT_RATIO
    exported.close()

    too_large_scale = calculate_consistent_scale(1000, 900, 10, 200, 180)
    assert too_large_scale == pytest.approx(0.2)
    assert 1000 * too_large_scale <= 200
    assert 900 * too_large_scale <= 180


def test_large_grouped_crop_does_not_shrink_small_sibling_unnecessarily(tmp_path):
    source = tmp_path / "uneven_group.pdf"
    output = tmp_path / "uneven_group_workspace.pdf"
    document = fitz.open()
    page = document.new_page(width=500, height=900)
    page.insert_text((40, 60), "Tall region text", fontsize=10)
    page.insert_text((40, 710), "Short region text", fontsize=10)
    document.save(source)
    document.close()

    question_id = "uneven-group"
    tall_region = QuestionRegion(
        page_index=0,
        rect=fitz.Rect(20, 20, 120, 620),
        question_id=question_id,
        part_order=0,
    )
    short_region = QuestionRegion(
        page_index=0,
        rect=fitz.Rect(20, 670, 120, 740),
        question_id=question_id,
        part_order=1,
    )
    export_question_regions(
        str(source), [tall_region, short_region], str(output), page_size="A4",
        scaling_mode=CONSISTENT_TEXT_SIZE,
    )

    exported = fitz.open(output)
    assert exported.page_count == 1
    workspace_page = exported[0]
    image_rects = [rect for rect, _, _ in _image_annotations(exported, workspace_page)]

    assert len(image_rects) == 2
    # The tall crop takes the space the short one does not need...
    assert image_rects[0].height > workspace_page.rect.height * CONSISTENT_MAX_HEIGHT_RATIO / 2
    # ...while the short sibling keeps its full requested scale.
    assert image_rects[1].height == pytest.approx(70 * MAX_CONSISTENT_SCALE)
    assert image_rects[0].y1 <= image_rects[1].y0
    assert image_rects[1].y1 <= (
        QUESTION_IMAGE_INSET_PT + workspace_page.rect.height * CONSISTENT_MAX_HEIGHT_RATIO + 0.01
    )
    exported.close()


def test_exported_images_are_annotations_not_flattened_page_content(tmp_path):
    source = tmp_path / "source.pdf"
    output = tmp_path / "result.pdf"
    _write_test_pdf(source)
    regions = [QuestionRegion(page_index=0, rect=fitz.Rect(40, 50, 300, 220), label="1")]

    export_question_regions(str(source), regions, str(output), page_size="A4")

    exported = fitz.open(output)
    page = exported[0]
    assert page.get_images() == []
    annots = list(page.annots())
    assert len(annots) == 1
    assert annots[0].type[1] == "Square"
    assert len(_image_annotations(exported, page)) == 1
    exported.close()


def test_same_text_size_crops_get_the_same_scale_regardless_of_crop_size(tmp_path):
    source = tmp_path / "mixed_sizes.pdf"
    output = tmp_path / "mixed_sizes_workspace.pdf"
    document = fitz.open()
    page = document.new_page(width=800, height=1000)
    page.insert_text((30, 50), "Small question", fontsize=10)
    page.insert_text((30, 450), "Big question", fontsize=10)
    document.save(source)
    document.close()
    regions = [
        QuestionRegion(page_index=0, rect=fitz.Rect(20, 30, 220, 130)),
        QuestionRegion(page_index=0, rect=fitz.Rect(20, 420, 320, 720)),
    ]

    export_question_regions(
        str(source), regions, str(output), page_size="A4", scaling_mode=CONSISTENT_TEXT_SIZE
    )

    exported = fitz.open(output)
    for page_index, region in enumerate(regions):
        rect = _image_annotations(exported, exported[page_index])[0][0]
        assert rect.width / region.rect.width == pytest.approx(MAX_CONSISTENT_SCALE, rel=1e-3)
        assert rect.height / region.rect.height == pytest.approx(MAX_CONSISTENT_SCALE, rel=1e-3)
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
    assert len(_image_annotations(exported, exported[0])) == 1
    exported.close()


def test_multiple_regions_from_different_pages_export_in_order_on_one_workspace_page(tmp_path):
    source = tmp_path / "multi_part_source.pdf"
    output = tmp_path / "multi_part_workspace.pdf"
    fit_output = tmp_path / "multi_part_fit_workspace.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=500)
    page.draw_rect(fitz.Rect(20, 20, 220, 80), color=None, fill=(1, 0, 0))
    page.draw_rect(fitz.Rect(20, 100, 100, 260), color=None, fill=(0, 1, 0))
    page = document.new_page(width=400, height=500)
    page.draw_rect(fitz.Rect(50, 50, 150, 150), color=None, fill=(0, 0, 1))
    page = document.new_page(width=400, height=500)
    page.draw_rect(fitz.Rect(80, 80, 180, 180), color=None, fill=(1, 1, 0))
    document.save(source)
    document.close()

    question_id = "question-identity-not-label"
    red_region = QuestionRegion(
        page_index=0,
        rect=fitz.Rect(20, 20, 220, 80),
        label="misrecognized label",
        question_id=question_id,
        part_order=0,
    )
    blue_region = QuestionRegion(
        page_index=1,
        rect=fitz.Rect(50, 50, 150, 150),
        label="duplicate label",
        question_id=question_id,
        part_order=1,
    )
    green_region = QuestionRegion(
        page_index=0,
        rect=fitz.Rect(20, 100, 100, 260),
        label="",
        question_id=question_id,
        part_order=2,
    )
    other_question = QuestionRegion(
        page_index=2,
        rect=fitz.Rect(80, 80, 180, 180),
        label="duplicate label",
    )

    export_question_regions(
        str(source),
        [green_region, red_region, other_question, blue_region],
        str(output),
        page_size="A4",
    )

    exported = fitz.open(output)
    assert exported.page_count == 2
    first_page = exported[0]
    image_rects = [rect for rect, _, _ in _image_annotations(exported, first_page)]
    assert len(image_rects) == 3
    expected_ratios = [200 / 60, 100 / 100, 80 / 160]
    for rect, expected_ratio in zip(image_rects, expected_ratios):
        assert rect.width / rect.height == pytest.approx(expected_ratio, rel=1e-6)
        assert rect.x0 == pytest.approx(QUESTION_IMAGE_INSET_PT)
        assert rect.x1 <= first_page.rect.width - QUESTION_IMAGE_INSET_PT + 0.01
        assert rect.y1 <= QUESTION_IMAGE_INSET_PT + first_page.rect.height * CONSISTENT_MAX_HEIGHT_RATIO + 0.01
    assert all(
        next_rect.y0 - previous_rect.y1 >= 11
        for previous_rect, next_rect in zip(image_rects, image_rects[1:])
    )

    pixmap = first_page.get_pixmap()
    expected_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
    for rect, expected_color in zip(image_rects, expected_colors):
        center_x = round((rect.x0 + rect.x1) / 2)
        center_y = round((rect.y0 + rect.y1) / 2)
        actual_color = pixmap.pixel(center_x, center_y)
        assert actual_color == expected_color

    second_page_images = _image_annotations(exported, exported[1])
    assert len(second_page_images) == 1
    second_page_rect = second_page_images[0][0]
    second_pixmap = exported[1].get_pixmap()
    second_color = second_pixmap.pixel(
        round((second_page_rect.x0 + second_page_rect.x1) / 2),
        round((second_page_rect.y0 + second_page_rect.y1) / 2),
    )
    assert second_color == (255, 255, 0)
    exported.close()

    export_question_regions(
        str(source),
        [green_region, red_region, other_question, blue_region],
        str(fit_output),
        page_size="A4",
        scaling_mode=FIT_TO_AVAILABLE_AREA,
    )
    fit_exported = fitz.open(fit_output)
    assert fit_exported.page_count == 2
    fit_first_images = _image_annotations(fit_exported, fit_exported[0])
    assert len(fit_first_images) == 3
    fit_first_rects = [rect for rect, _, _ in fit_first_images]
    assert all(
        rect.width / rect.height == pytest.approx(ratio, rel=1e-6)
        for rect, ratio in zip(fit_first_rects, expected_ratios)
    )
    fit_exported.close()


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
