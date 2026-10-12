from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz


@dataclass
class QuestionRegion:
    """One ordered source-page rectangle belonging to an exported question."""

    page_index: int
    rect: fitz.Rect
    label: str = "1"
    source: str = "manual"
    order: int = 0
    question_id: str = field(default_factory=lambda: uuid4().hex)
    part_order: int = 0

    def __post_init__(self) -> None:
        self.rect = fitz.Rect(self.rect)

    def width(self) -> float:
        return float(self.rect.x1 - self.rect.x0)

    def height(self) -> float:
        return float(self.rect.y1 - self.rect.y0)


@dataclass
class DocumentSession:
    """Document state containing ordered questions and their source rectangles."""

    path: str | None = None
    page_count: int = 0
    current_page_index: int = 0
    regions: list[QuestionRegion] = field(default_factory=list)
    selected_region_index: int | None = None

    def add_region(self, region: QuestionRegion) -> None:
        self.regions.append(region)
        self._refresh_order()

    def regions_for_question(self, question_id: str) -> list[QuestionRegion]:
        return sorted(
            (region for region in self.regions if region.question_id == question_id),
            key=lambda region: region.part_order,
        )

    def question_root(self, question_id: str) -> QuestionRegion | None:
        parts = self.regions_for_question(question_id)
        if not parts:
            return None
        return next((part for part in parts if part.part_order == 0), parts[0])

    def question_roots(self) -> list[QuestionRegion]:
        roots: list[QuestionRegion] = []
        seen: set[str] = set()
        for region in self.regions:
            if region.question_id in seen:
                continue
            seen.add(region.question_id)
            root = self.question_root(region.question_id)
            if root is not None:
                roots.append(root)
        return roots

    def add_additional_region(self, question_id: str, region: QuestionRegion) -> None:
        parts = self.regions_for_question(question_id)
        if not parts:
            raise ValueError(f"Question {question_id!r} does not exist.")
        region.question_id = question_id
        region.part_order = max(part.part_order for part in parts) + 1
        self.regions.append(region)
        self._refresh_order()

    def remove_region(self, region: QuestionRegion) -> None:
        if region in self.regions:
            self.regions.remove(region)
            self._refresh_order()

    def remove_question(self, question_id: str) -> None:
        self.regions[:] = [region for region in self.regions if region.question_id != question_id]
        self._refresh_order()

    def move_region(self, current_index: int, new_index: int) -> None:
        roots = self.question_roots()
        if not (0 <= current_index < len(roots) and 0 <= new_index < len(roots)):
            return
        moving_id = roots[current_index].question_id
        remaining = [root for root in roots if root.question_id != moving_id]
        remaining.insert(new_index, roots[current_index])
        reordered = [
            part
            for root in remaining
            for part in self.regions_for_question(root.question_id)
        ]
        self.regions[:] = reordered
        self._refresh_order()

    def _refresh_order(self) -> None:
        grouped: dict[str, list[QuestionRegion]] = {}
        for region in self.regions:
            grouped.setdefault(region.question_id, []).append(region)

        ordered: list[QuestionRegion] = []
        for question_index, parts in enumerate(grouped.values()):
            parts.sort(key=lambda region: region.part_order)
            parts[0].part_order = 0
            root = parts[0]
            root.label = str(question_index + 1)
            root.order = question_index
            for part_index, region in enumerate(parts):
                region.part_order = part_index
                ordered.append(region)
        self.regions[:] = ordered

    def sorted_regions(self) -> list[QuestionRegion]:
        return list(self.regions)
