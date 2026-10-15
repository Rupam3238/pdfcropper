from __future__ import annotations

from io import BytesIO
import math
from statistics import median
from uuid import uuid4

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz
from PyQt6.QtGui import QGuiApplication
from reportlab.lib.colors import black
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas

FALLBACK_PAGE_SIZE = A4
DEFAULT_LOGICAL_DPI = 96.0
_AUTO_SCREEN = object()
QUESTION_IMAGE_INSET_PT = 24.0
QUESTION_IMAGE_MAX_WIDTH_RATIO = 0.42
QUESTION_IMAGE_MAX_HEIGHT_RATIO = 0.32
FIT_TO_AVAILABLE_AREA = "fit_to_available_area"
CONSISTENT_TEXT_SIZE = "consistent_text_size"
DEFAULT_EXPORT_SCALING_MODE = CONSISTENT_TEXT_SIZE
MIN_CONSISTENT_SCALE = 0.5
MAX_CONSISTENT_SCALE = 1.5
TARGET_TEXT_SIZE_PT = 16.0
# Text size assumed for crops with no text layer (e.g. scanned pages) when no
# other crop in the export has text to borrow a size from.
DEFAULT_ASSUMED_TEXT_SIZE_PT = 11.0
MULTI_REGION_GAP_PT = 12.0
# In consistent-text-size mode a crop may use the full page width and this
# fraction of the page height, so large crops keep their text size instead of
# being squeezed into a small corner. Only fit-to-area mode uses the 0.42/0.32
# ratios above.
CONSISTENT_MAX_HEIGHT_RATIO = 0.85


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
    return clip, pix, pix.tobytes("png")


def estimate_region_text_size(pdf, region) -> float | None:
    """Estimate text size from PDF spans intersecting a source crop, if available."""
    page = pdf[region.page_index]
    clip = fitz.Rect(region.rect).normalize() & page.rect
    if clip.is_empty:
        return None
    sizes = []
    try:
        blocks = page.get_text("dict", clip=clip).get("blocks", [])
    except (RuntimeError, ValueError):
        return None
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not str(span.get("text", "")).strip():
                    continue
                try:
                    size = float(span.get("size", 0.0))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(size) and 1.0 <= size <= 96.0:
                    sizes.append(size)
    if not sizes:
        return None
    sizes.sort()
    middle = len(sizes) // 2
    if len(sizes) % 2:
        return sizes[middle]
    return (sizes[middle - 1] + sizes[middle]) / 2


def calculate_consistent_scale(
    crop_width: float,
    crop_height: float,
    text_size: float | None,
    max_width: float,
    max_height: float,
) -> float:
    """Choose the scale that brings the crop's text to the target size, capped by the area."""
    requested = requested_text_scale(text_size)
    area_limit = min(float(max_width) / crop_width, float(max_height) / crop_height)
    return min(requested, area_limit)


def requested_text_scale(text_size: float | None) -> float:
    """Bounded scale that would render text of ``text_size`` at the target size."""
    if text_size is None or not math.isfinite(text_size) or text_size <= 0:
        text_size = DEFAULT_ASSUMED_TEXT_SIZE_PT
    requested = TARGET_TEXT_SIZE_PT / text_size
    return min(MAX_CONSISTENT_SCALE, max(MIN_CONSISTENT_SCALE, requested))


def _resolve_text_sizes(pdf, regions) -> dict[int, float | None]:
    """Map id(region) to its text size; crops without text borrow the median of the others."""
    sizes = {id(region): estimate_region_text_size(pdf, region) for region in regions}
    known = [size for size in sizes.values() if size is not None]
    reference = median(known) if known else None
    return {key: (size if size is not None else reference) for key, size in sizes.items()}


def _allocate_group_scales(crop_heights, wanted_scales, max_height):
    """Fit stacked crops into ``max_height`` without shrinking crops that already fit.

    Each crop keeps its wanted scale if it fits in an equal share of the height.
    Height left over by the small crops goes to the larger ones, which are only
    shrunk when they still do not fit.
    """
    scales = list(wanted_scales)
    heights = [height * scale for height, scale in zip(crop_heights, scales)]
    pending = set(range(len(scales)))
    remaining = float(max_height)
    while pending:
        share = remaining / len(pending)
        fitting = [index for index in pending if heights[index] <= share + 1e-9]
        if not fitting:
            for index in pending:
                scales[index] = share / crop_heights[index]
            break
        for index in fitting:
            remaining -= heights[index]
            pending.discard(index)
    return scales


def _workspace_crop_area(page_width, page_height, margin):
    inset = max(0.0, float(margin))
    left = min(inset, page_width / 2)
    top = min(inset, page_height / 2)
    max_width = page_width - 2 * left
    max_height = min(page_height - top, page_height * CONSISTENT_MAX_HEIGHT_RATIO)
    return left, top, max_width, max_height


def _top_left_rect(x, y_bottom, width, height, page_height):
    """Convert a bottom-left-origin placement to a top-left-origin fitz.Rect."""
    return fitz.Rect(x, page_height - y_bottom - height, x + width, page_height - y_bottom)


def _add_image_annotation(document, page, rect, png_bytes):
    """Add the image as a movable/resizable annotation instead of flattening it into the page.

    This mirrors what Charta/Okular write for an inserted image: a /Stamp
    annotation whose normal appearance stream draws the image.
    """
    pixmap = fitz.Pixmap(png_bytes)
    image_xref = document.get_new_xref()
    document.update_object(
        image_xref,
        f"<</Type/XObject/Subtype/Image/Width {pixmap.width}/Height {pixmap.height}"
        "/ColorSpace/DeviceRGB/BitsPerComponent 8>>",
    )
    document.update_stream(image_xref, pixmap.samples, new=True)

    width, height = rect.width, rect.height
    form_xref = document.get_new_xref()
    document.update_object(
        form_xref,
        "<</Type/XObject/Subtype/Form/FormType 1"
        f"/BBox[0 0 {width:.4f} {height:.4f}]/Matrix[1 0 0 1 0 0]"
        f"/Resources<</XObject<</Im0 {image_xref} 0 R>>>>>>",
    )
    document.update_stream(
        form_xref, f"q {width:.4f} 0 0 {height:.4f} 0 0 cm /Im0 Do Q".encode(), new=True
    )

    # Create a text annotation as base for our custom stamp
    # We'll change its subtype to /Stamp and set up the appearance
    text_annot = page.add_text_annot(rect.top_left, "")
    annot_xref = text_annot.xref

    # Change subtype from /Text to /Stamp to make it a stamp annotation
    document.xref_set_key(annot_xref, "Subtype", "/Stamp")

    # Set the name to "Draft" to match Charta's clipboard image annotations
    text_annot.set_name("Draft")

    # Set up the appearance stream and rectangle (matching the image bounds)
    page_height = page.rect.height
    document.xref_set_key(
        annot_xref,
        "Rect",
        f"[{rect.x0:.4f} {page_height - rect.y1:.4f} {rect.x1:.4f} {page_height - rect.y0:.4f}]",
    )
    document.xref_set_key(annot_xref, "AP", f"<</N {form_xref} 0 R>>")
    document.xref_set_key(annot_xref, "BS", "<</W 0>>")
    document.xref_set_key(annot_xref, "F", "4")
    # Set NM to create the proper ID field that Charta recognizes
    # The ID field in the annotation will be the NM value without parentheses
    chama_id = f"charta-clipboard-image-{uuid4().hex}"
    document.xref_set_key(annot_xref, "NM", f"({chama_id})")
    for key in ("C", "IC", "CA", "Border", "RD"):
        document.xref_set_key(annot_xref, key, "null")


def _draw_region_group_fit(canvas, pdf, page_width, page_height, parts, margin):
    """Draw the black page and return the (rect, png) placements for its images."""
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
    placements = []
    cursor_top = page_height - top
    for clip, _, png in rendered:
        width = clip.width * scale
        height = clip.height * scale
        cursor_top -= height
        placements.append((_top_left_rect(left, cursor_top, width, height, page_height), png))
        cursor_top -= gap
    return placements


def _draw_region_group_consistent(canvas, pdf, page_width, page_height, parts, margin, text_sizes):
    """Draw the black page and return the (rect, png) placements for its images."""
    rendered = [_render_region(pdf, region) for region in parts]
    left, top, max_width, max_height = _workspace_crop_area(page_width, page_height, margin)
    gap = min(MULTI_REGION_GAP_PT, max_height / max(2 * len(rendered), 1))
    available_height = max_height - gap * (len(rendered) - 1)
    wanted = [
        min(requested_text_scale(text_sizes[id(region)]), max_width / clip.width)
        for region, (clip, _, _) in zip(parts, rendered)
    ]
    scales = _allocate_group_scales(
        [clip.height for clip, _, _ in rendered], wanted, available_height
    )

    canvas.setFillColor(black)
    canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
    placements = []
    cursor_top = page_height - top
    for (clip, _, png), scale in zip(rendered, scales):
        width = clip.width * scale
        height = clip.height * scale
        cursor_top -= height
        placements.append((_top_left_rect(left, cursor_top, width, height, page_height), png))
        cursor_top -= gap
    return placements


def export_question_regions(
    document_path: str,
    regions,
    output_path: str,
    page_size: str | None = None,
    margin: float = QUESTION_IMAGE_INSET_PT,
    scaling_mode: str = DEFAULT_EXPORT_SCALING_MODE,
):
    """Export each question as one black workspace page with its crops stacked at the upper-left.

    The crops are not flattened into the page: each one is a separate image
    annotation, so it can be moved and resized in Charta/Okular.
    """
    if not regions:
        raise ValueError("No question rectangles selected for export.")
    if scaling_mode not in {FIT_TO_AVAILABLE_AREA, CONSISTENT_TEXT_SIZE}:
        raise ValueError(f"Unsupported export scaling mode: {scaling_mode!r}.")

    pdf = fitz.open(document_path)
    page_width, page_height = resolve_workspace_page_size(page_size)
    background = BytesIO()
    output = canvas.Canvas(background, pagesize=(page_width, page_height))
    text_sizes = (
        _resolve_text_sizes(pdf, regions) if scaling_mode == CONSISTENT_TEXT_SIZE else {}
    )

    page_placements = []
    for parts in _question_region_groups(regions):
        if len(parts) > 1:
            if scaling_mode == FIT_TO_AVAILABLE_AREA:
                placements = _draw_region_group_fit(
                    output, pdf, page_width, page_height, parts, margin
                )
            else:
                placements = _draw_region_group_consistent(
                    output, pdf, page_width, page_height, parts, margin, text_sizes
                )
        else:
            region = parts[0]
            clip, pix, png = _render_region(pdf, region)
            if scaling_mode == FIT_TO_AVAILABLE_AREA:
                x, y, image_width, image_height = resolve_question_image_placement(
                    page_width, page_height, pix.width, pix.height, margin
                )
            else:
                left, top, max_width, max_height = _workspace_crop_area(page_width, page_height, margin)
                scale = calculate_consistent_scale(
                    clip.width,
                    clip.height,
                    text_sizes[id(region)],
                    max_width,
                    max_height,
                )
                image_width = clip.width * scale
                image_height = clip.height * scale
                x = left
                y = page_height - top - image_height
            output.setFillColor(black)
            output.rect(0, 0, page_width, page_height, fill=1, stroke=0)
            placements = [(_top_left_rect(x, y, image_width, image_height, page_height), png)]
        page_placements.append(placements)
        output.showPage()

    output.save()
    pdf.close()

    workspace = fitz.open("pdf", background.getvalue())
    for page, placements in zip(workspace, page_placements):
        for rect, png in placements:
            _add_image_annotation(workspace, page, rect, png)
    workspace.save(str(output_path), garbage=3, deflate=True)
    workspace.close()
    return output_path
