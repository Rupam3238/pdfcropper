from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import fitz


@dataclass
class QuestionRegion:
    """A single question rectangle in source PDF page coordinates."""

    page_index: int
    rect: fitz.Rect
    label: str = "1"
    source: str = "manual"
    order: int = 0

    def width(self) -> float:
        return float(self.rect.x1 - self.rect.x0)

    def height(self) -> float:
        return float(self.rect.y1 - self.rect.y0)


@dataclass
class DocumentSession:
    """Simple state container for the current PDF and selected question regions."""

    path: str | None = None
    page_count: int = 0
    current_page_index: int = 0
    regions: List[QuestionRegion] = field(default_factory=list)
    selected_region_index: int | None = None

    def add_region(self, region: QuestionRegion) -> None:
        self.regions.append(region)
        self._refresh_order()

    def remove_region(self, region: QuestionRegion) -> None:
        if region in self.regions:
            self.regions.remove(region)
            self._refresh_order()

    def move_region(self, current_index: int, new_index: int) -> None:
        if not (0 <= current_index < len(self.regions) and 0 <= new_index < len(self.regions)):
            return
        item = self.regions.pop(current_index)
        self.regions.insert(new_index, item)
        self._refresh_order()

    def _refresh_order(self) -> None:
        for idx, region in enumerate(self.regions):
            region.order = idx
            region.label = str(idx + 1)

    def sorted_regions(self) -> List[QuestionRegion]:
        return list(sorted(self.regions, key=lambda r: (r.order, r.page_index)))
