from __future__ import annotations

from io import BytesIO

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz
from PyQt6.QtWidgets import QApplication
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _page_size_tuple(page_size: str):
    key = (page_size or "A4").lower()
    if key == "a4":
        return A4
    if key in {"letter", "us-letter"}:
        return letter
    return A4


def resolve_workspace_page_size(page_size: str | None = None):
    """Return a workspace page size, using screen aspect ratio when available."""
    if page_size and page_size.lower() not in {"screen", "auto", "default"}:
        return _page_size_tuple(page_size)

    app = QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is not None:
        geometry = screen.availableGeometry()
        screen_w = max(1, geometry.width())
        screen_h = max(1, geometry.height())
        aspect_ratio = screen_w / screen_h
        if aspect_ratio >= 1.0:
            width = max(595, min(1800, screen_w * 0.72))
            height = width / aspect_ratio
        else:
            height = max(842, min(1600, screen_h * 0.72))
            width = height * aspect_ratio
        return width, height

    return A4


def export_question_regions(document_path: str, regions, output_path: str, page_size: str | None = None, margin: int = 36):
    """Export each question region as a page image in a new PDF, preserving order."""
    if not regions:
        raise ValueError("No question rectangles selected for export.")

    pdf = fitz.open(document_path)
    page_width, page_height = resolve_workspace_page_size(page_size)
    output = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

    for region in regions:
        page = pdf[region.page_index]
        clip = fitz.Rect(region.rect)
        render_scale = max(2.0, min(4.0, 2400.0 / max(clip.width, clip.height, 1.0)))
        pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), clip=clip, alpha=False)

        png_data = pix.tobytes("png")
        image_reader = ImageReader(BytesIO(png_data))

        target_width = max(120, page_width - (margin * 2))
        target_height = max(120, page_height - (margin * 2))
        crop_aspect = pix.width / max(pix.height, 1)
        image_width = min(target_width, target_height * crop_aspect)
        image_height = image_width / crop_aspect
        if image_height > target_height:
            image_height = target_height
            image_width = image_height * crop_aspect

        x = margin
        y = page_height - margin - image_height
        output.drawImage(
            image_reader,
            x,
            y,
            width=image_width,
            height=image_height,
            preserveAspectRatio=True,
        )
        output.showPage()

    output.save()
    pdf.close()
    return output_path
