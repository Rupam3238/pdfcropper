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
QUESTION_IMAGE_INSET_PT = 24.0
QUESTION_IMAGE_MAX_WIDTH_RATIO = 0.42
QUESTION_IMAGE_MAX_HEIGHT_RATIO = 0.32


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


def resolve_question_image_placement(
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
    inset: float = QUESTION_IMAGE_INSET_PT,
):
    """Return a top-left inset image rectangle in ReportLab's bottom-left coordinates."""
    inset = max(0.0, float(inset))
    left = min(inset, page_width / 2)
    top = min(inset, page_height / 2)
    max_width = min(page_width - left, page_width * QUESTION_IMAGE_MAX_WIDTH_RATIO)
    max_height = min(page_height - top, page_height * QUESTION_IMAGE_MAX_HEIGHT_RATIO)
    scale = min(max_width / image_width, max_height / image_height)
    width = image_width * scale
    height = image_height * scale
    x = left
    y = page_height - top - height
    return x, y, width, height


def _question_region_groups(regions):
    groups: dict[object, list[object]] = {}
    for index, region in enumerate(regions):
        question_id = getattr(region, "question_id", None)
        key = ("question", question_id) if question_id is not None else ("region", index)
        groups.setdefault(key, []).append(region)
    return [
        sorted(parts, key=lambda region: getattr(region, "part_order", 0))
        for parts in groups.values()
    ]


def _render_region(pdf, region):
    page = pdf[region.page_index]
    clip = fitz.Rect(region.rect).normalize()
    if clip.is_empty or clip.width <= 0 or clip.height <= 0:
        raise ValueError(f"Invalid region for page {region.page_index}: {region.rect}")
    clip = clip & page.rect
    if clip.is_empty or clip.width <= 0 or clip.height <= 0:
        raise ValueError(f"Crop rectangle falls outside page {region.page_index}: {region.rect}")

    render_scale = max(2.0, min(4.0, 2400.0 / max(clip.width, clip.height, 1.0)))
    pix = page.get_pixmap(
        matrix=fitz.Matrix(render_scale, render_scale),
        clip=clip,
        alpha=False,
        colorspace=fitz.csRGB,
    )
    image_reader = ImageReader(BytesIO(pix.tobytes("png")))
    return clip, pix, image_reader


def _draw_region_group(canvas, pdf, page_width, page_height, parts, margin):
    rendered = [_render_region(pdf, region) for region in parts]
    inset = max(0.0, float(margin))
    top = min(inset, page_height / 2)
    left = min(inset, page_width / 2)
    max_width = min(page_width - 2 * left, page_width * QUESTION_IMAGE_MAX_WIDTH_RATIO)
    max_height = min(page_height - top, page_height * QUESTION_IMAGE_MAX_HEIGHT_RATIO)
    gap = min(12.0, max_height / max(2 * len(rendered), 1))
    gap_total = gap * (len(rendered) - 1)
    max_source_width = max(clip.width for clip, _, _ in rendered)
    total_source_height = sum(clip.height for clip, _, _ in rendered)
    scale = min(max_width / max_source_width, (max_height - gap_total) / total_source_height)

    canvas.setFillColor(black)
    canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
    cursor_top = page_height - top
    for clip, _, image_reader in rendered:
        width = clip.width * scale
        height = clip.height * scale
        cursor_top -= height
        canvas.drawImage(
            image_reader,
            left,
            cursor_top,
            width=width,
            height=height,
            preserveAspectRatio=True,
        )
        cursor_top -= gap


def export_question_regions(
    document_path: str,
    regions,
    output_path: str,
    page_size: str | None = None,
    margin: float = QUESTION_IMAGE_INSET_PT,
):
    """Export each question as one black workspace page with its crops stacked at the upper-left."""
    if not regions:
        raise ValueError("No question rectangles selected for export.")

    pdf = fitz.open(document_path)
    page_width, page_height = resolve_workspace_page_size(page_size)
    output = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

    for parts in _question_region_groups(regions):
        if len(parts) > 1:
            _draw_region_group(output, pdf, page_width, page_height, parts, margin)
        else:
            _, pix, image_reader = _render_region(pdf, parts[0])
            x, y, image_width, image_height = resolve_question_image_placement(
                page_width, page_height, pix.width, pix.height, margin
            )
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
