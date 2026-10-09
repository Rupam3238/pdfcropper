try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from pdfcropper.geometry import page_to_scene_rect, scene_to_page_point


def test_page_to_scene_rect_round_trip():
    rect = fitz.Rect(20, 30, 120, 200)
    scene_rect = page_to_scene_rect(rect, zoom=2.0)
    assert scene_rect.x() == 40.0
    assert scene_rect.y() == 60.0
    assert scene_rect.width() == 200.0
    assert scene_rect.height() == 340.0

    point = scene_to_page_point((scene_rect.x() + 10, scene_rect.y() + 15), zoom=2.0)
    assert abs(point[0] - 25.0) < 1e-6
    assert abs(point[1] - 37.5) < 1e-6
