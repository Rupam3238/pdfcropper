from __future__ import annotations

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz
from PyQt6.QtGui import QImage, QPixmap


class PdfService:
    """Small PDF loader/rendering utility around PyMuPDF."""

    def __init__(self):
        self.document = None
        self.path = None

    def open(self, pdf_path: str):
        self.close()
        self.path = pdf_path
        self.document = fitz.open(pdf_path)
        return self.document

    def close(self):
        if self.document is not None:
            self.document.close()
            self.document = None
        self.path = None

    def page_count(self) -> int:
        return self.document.page_count if self.document is not None else 0

    def page_rect(self, page_index: int):
        if self.document is None:
            raise ValueError("No PDF loaded")
        return self.document[page_index].rect

    def render_page_pixmap(self, page_index: int, zoom: float = 1.0) -> QPixmap:
        if self.document is None:
            raise ValueError("No PDF loaded")
        page = self.document[page_index]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)

    def render_crop_pixmap(self, page_index: int, rect, zoom: float = 2.0) -> QPixmap:
        if self.document is None:
            raise ValueError("No PDF loaded")
        page = self.document[page_index]
        clip = fitz.Rect(rect)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        image = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)
