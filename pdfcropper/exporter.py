from __future__ import annotations

from io import BytesIO

import fitz
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


def export_question_regions(document_path: str, regions, output_path: str, page_size: str = "A4", margin: int = 36):
    """Export each question region as a page in a new PDF, preserving order."""
    if not regions:
        raise ValueError("No question rectangles selected for export.")

    pdf = fitz.open(document_path)
    page_width, page_height = _page_size_tuple(page_size)
    output = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

    for region in regions:
        page = pdf[region.page_index]
        clip = fitz.Rect(region.rect)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=clip, alpha=False)

        png_data = pix.tobytes("png")
        image_reader = ImageReader(BytesIO(png_data))

        scale = min(1.0, (page_width - 2 * margin) / max(pix.width, 1))
        image_width = max(40, pix.width * scale)
        image_height = pix.height * scale
        if image_height > page_height * 0.78:
            image_height = page_height * 0.78
            image_width = image_height * (pix.width / max(pix.height, 1))
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
