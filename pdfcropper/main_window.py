from __future__ import annotations

from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .exporter import export_question_regions
from .geometry import clamp_rect, page_to_scene_rect, scene_to_page_point
from .models import DocumentSession, QuestionRegion
from .pdf_service import PdfService


class ResizeHandle(QGraphicsRectItem):
    def __init__(self, owner, handle_name):
        super().__init__(-6, -6, 12, 12)
        self.owner = owner
        self.handle_name = handle_name
        self.setBrush(QColor("#f4b942"))
        self.setPen(QPen(QColor("#f4b942"), 1))
        self.setZValue(30)
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._drag_start_scene = None
        self._start_rect = None

    def mousePressEvent(self, event):
        self._drag_start_scene = event.scenePos()
        self._start_rect = self.owner.rect().getRect()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.owner is None:
            return
        if self._drag_start_scene is None or self._start_rect is None:
            return

        handler = self.handle_name
        scene_point = event.scenePos()
        local_point = self.owner.mapFromScene(scene_point)
        rect = self.owner.rect().getRect()
        x0, y0, x1, y1 = rect

        if handler in {"top-left", "left", "bottom-left"}:
            x0 = min(max(local_point.x(), 0), x1 - 2)
        if handler in {"top-right", "right", "bottom-right"}:
            x1 = max(min(local_point.x(), self.owner.page_width), x0 + 2)
        if handler in {"top-left", "top", "top-right"}:
            y0 = min(max(local_point.y(), 0), y1 - 2)
        if handler in {"bottom-left", "bottom", "bottom-right"}:
            y1 = max(min(local_point.y(), self.owner.page_height), y0 + 2)

        self.owner.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        self.owner.update_region_from_rect()
        self.owner.sync_handles()
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_start_scene = None
        self._start_rect = None
        event.accept()


class QuestionRectItem(QGraphicsRectItem):
    def __init__(self, region, page_width, page_height):
        super().__init__()
        self.region = region
        self.page_width = page_width
        self.page_height = page_height
        self.setRect(QRectF(0, 0, max(1.0, region.rect.x1 - region.rect.x0), max(1.0, region.rect.y1 - region.rect.y0)))
        self.setPos(region.rect.x0, region.rect.y0)
        self.setPen(QPen(QColor("#ff5a36"), 2))
        self.setBrush(QColor(255, 90, 54, 80))
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setZValue(10)
        self.label_background = QGraphicsRectItem(self)
        self.label_background.setPen(QPen(Qt.PenStyle.NoPen))
        self.label_background.setBrush(QColor("#202020"))
        self.label_background.setZValue(20)
        self.label_background.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.label_item = QGraphicsSimpleTextItem(self.label_background)
        self.label_item.setText(str(region.label))
        self.label_item.setBrush(QColor("#ffffff"))
        self.label_item.setPos(4, 2)
        self.label_item.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._sync_label()
        self.handle_items = {}
        for name in ["top-left", "top", "top-right", "right", "bottom-right", "bottom", "bottom-left", "left"]:
            handle = ResizeHandle(self, name)
            handle.setParentItem(self)
            handle.setVisible(False)
            self.handle_items[name] = handle
        self.sync_handles()

    def update_region_from_rect(self):
        origin = self.pos()
        rect = self.rect()
        x0 = origin.x() + rect.x()
        y0 = origin.y() + rect.y()
        x1 = origin.x() + rect.right()
        y1 = origin.y() + rect.bottom()
        self.region.rect = fitz.Rect(x0, y0, x1, y1)
        self.label_item.setText(str(self.region.label))
        self._sync_label()

    def _sync_label(self):
        bounds = self.label_item.boundingRect()
        self.label_background.setRect(0, 0, bounds.width() + 8, bounds.height() + 4)

    def sync_handles(self):
        for name, item in self.handle_items.items():
            if name == "top-left":
                item.setPos(self.rect().topLeft())
            elif name == "top":
                item.setPos(self.rect().left() + self.rect().width() / 2, self.rect().top())
            elif name == "top-right":
                item.setPos(self.rect().topRight())
            elif name == "right":
                item.setPos(self.rect().right(), self.rect().top() + self.rect().height() / 2)
            elif name == "bottom-right":
                item.setPos(self.rect().bottomRight())
            elif name == "bottom":
                item.setPos(self.rect().left() + self.rect().width() / 2, self.rect().bottom())
            elif name == "bottom-left":
                item.setPos(self.rect().bottomLeft())
            elif name == "left":
                item.setPos(self.rect().left(), self.rect().top() + self.rect().height() / 2)

    def update_selection_state(self):
        visible = self.isSelected()
        self.setPen(QPen(QColor("#36d6c5" if visible else "#ff5a36"), 3 if visible else 2))
        self.label_background.setBrush(QColor("#f4b942" if visible else "#202020"))
        self.label_item.setBrush(QColor("#202020" if visible else "#ffffff"))
        for handle in self.handle_items.values():
            handle.setVisible(visible)

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemSelectedHasChanged:
            self.update_selection_state()
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.update_region_from_rect()
        self.sync_handles()


class PdfPageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.page_item = None
        self.page_index = 0
        self.page_size = (0, 0)
        self.zoom = 1.0
        self.drawing_mode = False
        self.rubber_band_rect = None
        self.start_scene_pos = None
        self.controller = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_controller(self, controller):
        self.controller = controller

    def set_zoom(self, factor: float):
        self.zoom = max(0.25, min(6.0, factor))
        self.resetTransform()
        self.scale(self.zoom, self.zoom)

    def clear_scene(self):
        self.scene.clear()
        self.page_item = None
        self.rubber_band_rect = None

    def render_page(self, page_index: int, page_pixmap, page_rect):
        self.clear_scene()
        self.page_index = page_index
        self.page_size = (page_rect.width, page_rect.height)
        self.page_item = self.scene.addPixmap(page_pixmap)
        self.page_item.setPos(0, 0)
        self.page_item.setZValue(0)
        self.scene.setSceneRect(0, 0, page_rect.width, page_rect.height)

        for region in self.controller.session.regions:
            if region.page_index != page_index:
                continue
            item = QuestionRectItem(region, page_rect.width, page_rect.height)
            self.scene.addItem(item)
            item.sync_handles()
            if self.controller.selected_region and self.controller.selected_region is region:
                item.setSelected(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_1 and self.controller is not None:
            self.controller.activate_drawing_mode()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self.rubber_band_rect is not None:
            self.scene.removeItem(self.rubber_band_rect)
            self.rubber_band_rect = None
            self.start_scene_pos = None
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if self.drawing_mode and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if not (0 <= scene_pos.x() <= self.page_size[0] and 0 <= scene_pos.y() <= self.page_size[1]):
                return
            self.start_scene_pos = scene_pos
            self.rubber_band_rect = QGraphicsRectItem()
            self.rubber_band_rect.setPen(QPen(QColor("#43a047"), 2, Qt.PenStyle.DashLine))
            self.rubber_band_rect.setBrush(QColor(67, 160, 71, 40))
            self.rubber_band_rect.setRect(QRectF(scene_pos, scene_pos))
            self.scene.addItem(self.rubber_band_rect)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_mode and self.rubber_band_rect is not None:
            scene_pos = self.mapToScene(event.pos())
            scene_pos.setX(min(max(scene_pos.x(), 0), self.page_size[0]))
            scene_pos.setY(min(max(scene_pos.y(), 0), self.page_size[1]))
            rect = QRectF(self.start_scene_pos, scene_pos).normalized()
            self.rubber_band_rect.setRect(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_mode and self.rubber_band_rect is not None:
            rubber_band_rect = self.rubber_band_rect
            scene_rect = rubber_band_rect.rect()
            rect = None
            if scene_rect.width() > 2 and scene_rect.height() > 2:
                x0, y0 = scene_to_page_point(scene_rect.topLeft(), zoom=1.0)
                x1, y1 = scene_to_page_point(scene_rect.bottomRight(), zoom=1.0)
                rect = fitz.Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.scene.removeItem(rubber_band_rect)
            self.rubber_band_rect = None
            self.start_scene_pos = None
            if rect is not None:
                self.controller.add_region_from_rect(self.page_index, rect)
            self.controller.update_question_list()
            return
        if not self.drawing_mode and event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            while item is not None and not isinstance(item, QuestionRectItem):
                item = item.parentItem()
            if item is None:
                self.controller.clear_canvas_selection()
            else:
                self.controller.select_region_from_canvas(item.region)
        super().mouseReleaseEvent(event)


class PdfCropperMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pdf_service = PdfService()
        self.session = DocumentSession()
        self.selected_region = None
        self.current_file = None
        self.build_ui()

    def build_ui(self):
        self.setWindowTitle("PDF Question Cropper")
        self.resize(1200, 820)

        self.open_action = QAction("Open PDF", self)
        self.open_action.triggered.connect(self.open_pdf)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)

        self.prev_page_action = QAction("Previous", self)
        self.prev_page_action.triggered.connect(self.previous_page)
        self.next_page_action = QAction("Next", self)
        self.next_page_action.triggered.connect(self.next_page)
        self.zoom_in_action = QAction("Zoom +", self)
        self.zoom_in_action.triggered.connect(lambda: self.zoom(1.25))
        self.zoom_out_action = QAction("Zoom -", self)
        self.zoom_out_action.triggered.connect(lambda: self.zoom(1.0 / 1.25))
        self.draw_rect_action = QAction("Draw rectangle", self)
        self.draw_rect_action.setCheckable(True)
        self.draw_rect_action.triggered.connect(self.activate_drawing_mode)
        self.select_action = QAction("Select / edit", self)
        self.select_action.setCheckable(True)
        self.select_action.setChecked(True)
        self.select_action.triggered.connect(self.activate_selection_mode)
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_group.addAction(self.select_action)
        self.tool_group.addAction(self.draw_rect_action)
        self.delete_action = QAction("Delete", self)
        self.delete_action.triggered.connect(self.delete_selected_region)
        self.add_before_action = QAction("Add before", self)
        self.add_before_action.triggered.connect(self.add_before_selected)
        self.add_after_action = QAction("Add after", self)
        self.add_after_action.triggered.connect(self.add_after_selected)
        self.move_up_action = QAction("Move up", self)
        self.move_up_action.triggered.connect(self.move_selected_up)
        self.move_down_action = QAction("Move down", self)
        self.move_down_action.triggered.connect(self.move_selected_down)
        self.export_action = QAction("Export PDF", self)
        self.export_action.triggered.connect(self.export_selected)

        toolbar = self.addToolBar("Main")
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.prev_page_action)
        toolbar.addAction(self.next_page_action)
        toolbar.addSeparator()
        toolbar.addAction(self.zoom_in_action)
        toolbar.addAction(self.zoom_out_action)
        toolbar.addAction(self.select_action)
        toolbar.addAction(self.draw_rect_action)
        toolbar.addAction(self.add_before_action)
        toolbar.addAction(self.add_after_action)
        toolbar.addAction(self.delete_action)
        toolbar.addAction(self.move_up_action)
        toolbar.addAction(self.move_down_action)
        toolbar.addSeparator()
        toolbar.addAction(self.export_action)

        central = QWidget()
        layout = QHBoxLayout()
        central.setLayout(layout)

        self.viewer = PdfPageView()
        self.viewer.set_controller(self)
        layout.addWidget(self.viewer, 3)

        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_layout.addWidget(QLabel("Questions"))
        self.question_list = QListWidget()
        self.question_list.itemClicked.connect(self.select_region_from_list)
        right_layout.addWidget(self.question_list)
        self.page_label = QLabel("No PDF loaded")
        right_layout.addWidget(self.page_label)
        layout.addWidget(right_panel, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Mode: Select / edit")

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF files (*.pdf)")
        if not path:
            return
        try:
            self.pdf_service.open(path)
            self.current_file = path
            self.session = DocumentSession(path=path, page_count=self.pdf_service.page_count())
            self.selected_region = None
            self.viewer.page_index = 0
            self.show_page(0)
            self.update_question_list()
        except Exception as exc:
            QMessageBox.critical(self, "Unable to open PDF", str(exc))

    def show_page(self, page_index: int):
        self.session.current_page_index = page_index
        if self.pdf_service.document is None:
            return
        page = self.pdf_service.document[page_index]
        pixmap = self.pdf_service.render_page_pixmap(page_index, zoom=1.0)
        self.viewer.render_page(page_index, pixmap, page.rect)
        self.page_label.setText(f"Page {page_index + 1} of {self.session.page_count}")

    def previous_page(self):
        if self.session.current_page_index > 0:
            self.show_page(self.session.current_page_index - 1)

    def next_page(self):
        if self.session.current_page_index + 1 < self.session.page_count:
            self.show_page(self.session.current_page_index + 1)

    def zoom(self, factor: float):
        current = self.viewer.zoom * factor
        self.viewer.set_zoom(current)

    def activate_drawing_mode(self):
        self.viewer.drawing_mode = True
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.draw_rect_action.setChecked(True)
        self.statusBar().showMessage("Mode: Draw rectangles (1) | Escape cancels the current rectangle")

    def activate_selection_mode(self):
        self.viewer.drawing_mode = False
        self.viewer.setCursor(Qt.CursorShape.ArrowCursor)
        self.select_action.setChecked(True)
        self.statusBar().showMessage("Mode: Select / edit")

    def toggle_drawing_mode(self):
        if self.viewer.drawing_mode:
            self.activate_selection_mode()
        else:
            self.activate_drawing_mode()

    def add_region_from_rect(self, page_index: int, rect):
        if rect.width <= 0 or rect.height <= 0:
            return
        region = QuestionRegion(page_index=page_index, rect=rect)
        self.session.regions.append(region)
        self.session._refresh_order()
        self.session.selected_region_index = len(self.session.regions) - 1
        self.selected_region = region
        self.update_question_list()
        self.show_page(page_index)

    def select_region_from_canvas(self, region):
        self.selected_region = region
        self.session.selected_region_index = self.session.regions.index(region)
        self.question_list.setCurrentRow(self.session.selected_region_index)

    def clear_canvas_selection(self):
        self.selected_region = None
        self.session.selected_region_index = None
        self.question_list.setCurrentRow(-1)

    def update_question_list(self):
        self.question_list.clear()
        for region in self.session.regions:
            item = QListWidgetItem(f"Question {region.label} — page {region.page_index + 1}")
            item.setData(32, region)
            self.question_list.addItem(item)

    def select_region_from_list(self, item):
        region = item.data(32)
        self.selected_region = region
        self.session.selected_region_index = self.session.regions.index(region)
        self.show_page(region.page_index)

    def delete_selected_region(self):
        if self.selected_region is None:
            return
        self.session.regions.remove(self.selected_region)
        self.selected_region = None
        self.session._refresh_order()
        self.update_question_list()
        self.show_page(self.session.current_page_index)

    def add_before_selected(self):
        if not self.selected_region:
            QMessageBox.information(self, "Select a question", "Select a question before adding one before it.")
            return
        index = self.session.regions.index(self.selected_region)
        self._insert_region_at(index)

    def add_after_selected(self):
        if not self.selected_region:
            QMessageBox.information(self, "Select a question", "Select a question before adding one after it.")
            return
        index = self.session.regions.index(self.selected_region)
        self._insert_region_at(index + 1)

    def _insert_region_at(self, index):
        selected = self.selected_region
        page_rect = self.pdf_service.page_rect(selected.page_index)
        width = min(page_rect.width, max(40.0, selected.rect.width))
        height = min(page_rect.height, max(40.0, selected.rect.height))
        occupied = [region.rect for region in self.session.regions if region.page_index == selected.page_index]
        step_x = max(1.0, min(width, height) / 2)
        step_y = step_x
        max_x = page_rect.width - width
        max_y = page_rect.height - height
        x_positions = list(range(0, int(max_x) + 1, int(step_x)))
        y_positions = list(range(0, int(max_y) + 1, int(step_y)))
        if not x_positions or x_positions[-1] < max_x:
            x_positions.append(max_x)
        if not y_positions or y_positions[-1] < max_y:
            y_positions.append(max_y)
        candidates = []
        for y in y_positions:
            for x in x_positions:
                candidates.append(fitz.Rect(x, y, x + width, y + height))
        new_rect = next(
            (candidate for candidate in candidates if not any(candidate.intersects(other) for other in occupied)),
            None,
        )
        if new_rect is None:
            QMessageBox.information(self, "No room on page", "There is no free space on this page for another question region.")
            return

        new_region = QuestionRegion(page_index=selected.page_index, rect=new_rect)
        self.session.regions.insert(index, new_region)
        self.session._refresh_order()
        self.selected_region = new_region
        self.session.selected_region_index = index
        self.update_question_list()
        self.question_list.setCurrentRow(index)
        self.show_page(selected.page_index)

    def move_selected_up(self):
        if self.selected_region is None:
            return
        idx = self.session.regions.index(self.selected_region)
        if idx > 0:
            self.session.move_region(idx, idx - 1)
            self.selected_region = self.session.regions[idx - 1]
            self.update_question_list()
            self.show_page(self.session.current_page_index)

    def move_selected_down(self):
        if self.selected_region is None:
            return
        idx = self.session.regions.index(self.selected_region)
        if idx < len(self.session.regions) - 1:
            self.session.move_region(idx, idx + 1)
            self.selected_region = self.session.regions[idx + 1]
            self.update_question_list()
            self.show_page(self.session.current_page_index)

    def export_selected(self):
        if self.pdf_service.document is None:
            QMessageBox.warning(self, "No PDF loaded", "Open a PDF before exporting.")
            return
        if not self.session.regions:
            QMessageBox.warning(self, "No questions", "Add at least one question rectangle before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save export", "question_workspaces.pdf", "PDF files (*.pdf)")
        if not path:
            return
        try:
            export_question_regions(self.current_file, self.session.regions, path)
            QMessageBox.information(self, "Export complete", f"Saved export to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


def run_app():
    app = QApplication([])
    window = PdfCropperMainWindow()
    window.show()
    app.exec()
