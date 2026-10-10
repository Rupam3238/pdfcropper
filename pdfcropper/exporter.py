from __future__ import annotations

from io import BytesIO
import math

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz
from PyQt6.QtGui import QGuiApplication
from reportlab.lib.colors import black
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

FALLBACK_PAGE_SIZE = A4
DEFAULT_LOGICAL_DPI = 96.0
_AUTO_SCREEN = object()


def _page_size_tuple(page_size: str):
    key = (page_size or "A4").lower()
    if key == "a4":
        return A4
    if key in {"letter", "us-letter"}:
        return letter
    return A4


def display_dimensions_to_page_size(
    logical_width: float,
    logical_height: float,
    device_pixel_ratio: float = 1.0,
    logical_dpi: float = DEFAULT_LOGICAL_DPI,
    physical_dpi: float | None = None,
):
    """Convert full screen dimensions to PDF points while preserving aspect ratio.

    Screen geometry is in Qt logical pixels. When physical DPI is finite and
    agrees with Qt's logical DPI after device scaling, physical pixels are
    converted at 72 PDF points per physical inch. Otherwise the logical pixel
    dimensions are converted at logical DPI. This avoids trusting broken
    monitor DPI metadata while accounting for high-DPI scaling when reliable.
    """
    width = float(logical_width)
    height = float(logical_height)
    ratio = float(device_pixel_ratio)
    logical_dpi = float(logical_dpi)
    if not all(math.isfinite(value) and value > 0 for value in (width, height, ratio)):
        return FALLBACK_PAGE_SIZE

    if not math.isfinite(logical_dpi) or not 30 <= logical_dpi <= 600:
        logical_dpi = DEFAULT_LOGICAL_DPI

    physical_is_reliable = False
    if physical_dpi is not None:
        physical_dpi = float(physical_dpi)
        if math.isfinite(physical_dpi) and 30 <= physical_dpi <= 1000:
            effective_logical_dpi = physical_dpi / ratio
            physical_is_reliable = (
                50 <= effective_logical_dpi <= 300
                and abs(effective_logical_dpi - logical_dpi) / logical_dpi <= 0.25
            )

    if physical_is_reliable:
        return width * ratio * 72 / physical_dpi, height * ratio * 72 / physical_dpi
    return width * 72 / logical_dpi, height * 72 / logical_dpi


def _screen_to_workspace_page_size(screen):
    if screen is None:
        return FALLBACK_PAGE_SIZE
    try:
        geometry = screen.geometry()
        return display_dimensions_to_page_size(
            geometry.width(),
            geometry.height(),
            screen.devicePixelRatio(),
            screen.logicalDotsPerInch(),
            screen.physicalDotsPerInch(),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        return FALLBACK_PAGE_SIZE


def resolve_workspace_page_size(page_size: str | None = None, screen=_AUTO_SCREEN):
    """Return explicit paper size or a full-display page size; headless fallback is A4."""
    if page_size and page_size.lower() not in {"screen", "auto", "default"}:
        return _page_size_tuple(page_size)

    if screen is _AUTO_SCREEN:
        app = QGuiApplication.instance()
        screen = app.primaryScreen() if app is not None else None
    return _screen_to_workspace_page_size(screen)


def export_question_regions(document_path: str, regions, output_path: str, page_size: str | None = None, margin: int = 36):
    """Export each question region as a page image in a new PDF, preserving order."""
    if not regions:
        raise ValueError("No question rectangles selected for export.")

    pdf = fitz.open(document_path)
    page_width, page_height = resolve_workspace_page_size(page_size)
    output = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

    for region in regions:
        page = pdf[region.page_index]
        clip = fitz.Rect(region.rect).normalize()
        if clip.is_empty or clip.width <= 0 or clip.height <= 0:
            raise ValueError(f"Invalid region for page {region.page_index}: {region.rect}")
        clip = clip & page.rect
        if clip.is_empty or clip.width <= 0 or clip.height <= 0:
            raise ValueError(f"Crop rectangle falls outside page {region.page_index}: {region.rect}")

        render_scale = max(2.0, min(4.0, 2400.0 / max(clip.width, clip.height, 1.0)))
        pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), clip=clip, alpha=False, colorspace=fitz.csRGB)

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
        output.setFillColor(black)
        output.rect(0, 0, page_width, page_height, fill=1, stroke=0)
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
