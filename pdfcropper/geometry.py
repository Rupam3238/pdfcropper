from __future__ import annotations

from typing import Tuple

from PyQt6.QtCore import QRectF


def page_to_scene_rect(page_rect, zoom: float = 1.0, offset: Tuple[float, float] = (0.0, 0.0)) -> QRectF:
    """Convert a source-page rectangle to scene coordinates."""
    x0 = float(page_rect.x0) * zoom + offset[0]
    y0 = float(page_rect.y0) * zoom + offset[1]
    x1 = float(page_rect.x1) * zoom + offset[0]
    y1 = float(page_rect.y1) * zoom + offset[1]
    return QRectF(x0, y0, x1 - x0, y1 - y0)


def scene_to_page_point(scene_point, zoom: float = 1.0, offset: Tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """Convert a scene point to page coordinates."""
    if isinstance(scene_point, tuple):
        x_val, y_val = scene_point
    else:
        x_val = float(scene_point.x())
        y_val = float(scene_point.y())
    x = (x_val - offset[0]) / zoom
    y = (y_val - offset[1]) / zoom
    return x, y


def clamp_rect(raw_rect, page_width: float, page_height: float):
    """Clamp a rectangle to the page boundaries."""
    x0 = max(0.0, min(float(raw_rect.x0), page_width))
    y0 = max(0.0, min(float(raw_rect.y0), page_height))
    x1 = max(0.0, min(float(raw_rect.x1), page_width))
    y1 = max(0.0, min(float(raw_rect.y1), page_height))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)
