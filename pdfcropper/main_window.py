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
    QComboBox,
    QFileDialog,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .analysis import page_indices_for_range
from .exporter import export_question_regions
from .geometry import clamp_rect, page_to_scene_rect, scene_to_page_point
from .models import DocumentSession, QuestionRegion
from .pipeline import (
    PageLayoutMode,
    PageLayoutOptions,
    QuestionAnalysisRun,
    analyze_selected_pages as run_selected_page_analysis,
)
from .pdf_service import PdfService


def _workspace_export_filename(source_pdf_path: str | Path | None) -> str:
    if not source_pdf_path:
        return "question_workspaces.pdf"
    return f"{Path(source_pdf_path).stem}_workspaces.pdf"


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


class ManualDividerItem(QGraphicsLineItem):
    """Draggable, non-exported vertical guide stored as normalized page x."""

    def __init__(self, page_width, page_height, normalized_x, on_moved):
        super().__init__()
        self.page_width = float(page_width)
        self.page_height = float(page_height)
        self.normalized_x = float(normalized_x)
        self.on_moved = on_moved
        self.setPen(QPen(QColor("#c026d3"), 2, Qt.PenStyle.DashLine))
        self.setZValue(25)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, False)
        self.set_normalized_x(self.normalized_x, notify=False)

    def set_normalized_x(self, normalized_x: float, *, notify: bool = True):
        self.normalized_x = min(0.95, max(0.05, float(normalized_x)))
        x = self.normalized_x * self.page_width
        self.setLine(x, 0.0, x, self.page_height)
        if notify:
            self.on_moved(self.normalized_x)

    def mousePressEvent(self, event):
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.page_width > 0:
            self.set_normalized_x(event.scenePos().x() / self.page_width)
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()


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
        self.manual_divider_item: ManualDividerItem | None = None
        self.divider_placement_mode = False
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
        self.manual_divider_item = None
        self.divider_placement_mode = False

    def render_page(self, page_index: int, page_pixmap, page_rect):
        self.clear_scene()
        self.page_index = page_index
        self.page_size = (page_rect.width, page_rect.height)
        self.page_item = self.scene.addPixmap(page_pixmap)
        self.page_item.setPos(0, 0)
        self.page_item.setZValue(0)
        self.scene.setSceneRect(0, 0, page_rect.width, page_rect.height)

        self.manual_divider_item = None
        layout_options = self.controller.page_layout_settings.get(page_index, PageLayoutOptions())
        if layout_options.mode == PageLayoutMode.TWO_COLUMNS and layout_options.divider_normalized is not None:
            self._add_manual_divider(layout_options.divider_normalized)

        for region in self.controller.session.regions:
            if region.page_index != page_index:
                continue
            item = QuestionRectItem(region, page_rect.width, page_rect.height)
            self.scene.addItem(item)
            item.sync_handles()
            if self.controller.selected_region and self.controller.selected_region is region:
                item.setSelected(True)

    def _add_manual_divider(self, normalized_x: float):
        self.manual_divider_item = ManualDividerItem(
            self.page_size[0],
            self.page_size[1],
            normalized_x,
            lambda position: self.controller.set_manual_divider_position(self.page_index, position),
        )
        self.scene.addItem(self.manual_divider_item)

    def set_manual_divider_item_position(self, normalized_x: float):
        if self.manual_divider_item is None:
            self._add_manual_divider(normalized_x)
        else:
            self.manual_divider_item.set_normalized_x(normalized_x, notify=False)

    def _normalized_page_x(self, scene_x: float) -> float:
        return min(0.95, max(0.05, float(scene_x) / max(self.page_size[0], 1.0)))

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
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if not (0 <= scene_pos.x() <= self.page_size[0] and 0 <= scene_pos.y() <= self.page_size[1]):
                return
            if self.manual_divider_item is not None and (not self.drawing_mode or self.divider_placement_mode):
                divider_x = self.manual_divider_item.line().x1()
                if abs(scene_pos.x() - divider_x) <= max(8.0, self.page_size[0] * 0.015):
                    super().mousePressEvent(event)
                    return
            if self.divider_placement_mode:
                normalized_x = self._normalized_page_x(scene_pos.x())
                self.controller.set_manual_divider_position(self.page_index, normalized_x)
                self.set_manual_divider_item_position(normalized_x)
                self.divider_placement_mode = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.controller.finish_divider_placement()
                event.accept()
                return
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
            if isinstance(item, ManualDividerItem):
                event.accept()
                return
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
        self.analysis_result: QuestionAnalysisRun | None = None
        self.page_layout_settings: dict[int, PageLayoutOptions] = {}
        self._syncing_layout_controls = False
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
        right_layout.addWidget(QLabel("Pages to analyze"))
        page_range_layout = QHBoxLayout()
        page_range_layout.addWidget(QLabel("From"))
        self.analysis_from_spinbox = QSpinBox()
        self.analysis_from_spinbox.setRange(1, 1)
        self.analysis_from_spinbox.setEnabled(False)
        page_range_layout.addWidget(self.analysis_from_spinbox)
        page_range_layout.addWidget(QLabel("To"))
        self.analysis_to_spinbox = QSpinBox()
        self.analysis_to_spinbox.setRange(1, 1)
        self.analysis_to_spinbox.setEnabled(False)
        page_range_layout.addWidget(self.analysis_to_spinbox)
        right_layout.addLayout(page_range_layout)
        self.analysis_range_status_label = QLabel("Open a PDF to select pages")
        right_layout.addWidget(self.analysis_range_status_label)
        self.analysis_from_spinbox.valueChanged.connect(self._update_analysis_range_status)
        self.analysis_to_spinbox.valueChanged.connect(self._update_analysis_range_status)
        self.analyze_pages_button = QPushButton("Analyze pages")
        self.analyze_pages_button.setEnabled(False)
        self.analyze_pages_button.clicked.connect(self.analyze_selected_pages)
        right_layout.addWidget(self.analyze_pages_button)
        self.run_ocr_button = QPushButton("Run OCR")
        self.run_ocr_button.setEnabled(False)
        self.run_ocr_button.clicked.connect(self.run_ocr_selected_pages)
        right_layout.addWidget(self.run_ocr_button)
        self.analysis_summary_label = QLabel("Analysis has not run")
        right_layout.addWidget(self.analysis_summary_label)
        self.analysis_diagnostics = QPlainTextEdit()
        self.analysis_diagnostics.setReadOnly(True)
        self.analysis_diagnostics.setMaximumHeight(150)
        self.analysis_diagnostics.setPlaceholderText("Page warnings and ambiguities will appear here")
        right_layout.addWidget(self.analysis_diagnostics)
        right_layout.addWidget(QLabel("Page layout"))
        self.page_layout_combo = QComboBox()
        self.page_layout_combo.addItem("Automatic", PageLayoutMode.AUTOMATIC)
        self.page_layout_combo.addItem("Single column", PageLayoutMode.SINGLE_COLUMN)
        self.page_layout_combo.addItem("Two columns", PageLayoutMode.TWO_COLUMNS)
        self.page_layout_combo.currentIndexChanged.connect(self._on_page_layout_mode_changed)
        right_layout.addWidget(self.page_layout_combo)
        layout_actions = QHBoxLayout()
        self.place_divider_button = QPushButton("Place divider")
        self.place_divider_button.setEnabled(False)
        self.place_divider_button.clicked.connect(self.start_divider_placement)
        self.clear_divider_button = QPushButton("Clear divider")
        self.clear_divider_button.setEnabled(False)
        self.clear_divider_button.clicked.connect(self.clear_current_page_divider)
        layout_actions.addWidget(self.place_divider_button)
        layout_actions.addWidget(self.clear_divider_button)
        right_layout.addLayout(layout_actions)
        self.apply_layout_range_button = QPushButton("Apply layout to selected range")
        self.apply_layout_range_button.setEnabled(False)
        self.apply_layout_range_button.clicked.connect(self.apply_layout_to_analysis_range)
        right_layout.addWidget(self.apply_layout_range_button)
        self.page_layout_status_label = QLabel("Automatic layout")
        right_layout.addWidget(self.page_layout_status_label)
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
            page_count = self.pdf_service.page_count()
            self.session = DocumentSession(path=path, page_count=page_count)
            self.selected_region = None
            self.analysis_result = None
            self.page_layout_settings.clear()
            self.analysis_summary_label.setText("Analysis has not run")
            self.analysis_diagnostics.clear()
            self._sync_layout_controls_for_page(0)
            self._reset_analysis_page_range(page_count)
            self.viewer.page_index = 0
            self.show_page(0)
            self.update_question_list()
        except Exception as exc:
            QMessageBox.critical(self, "Unable to open PDF", str(exc))

    def _reset_analysis_page_range(self, page_count: int):
        control_maximum = max(1, page_count)
        controls = (self.analysis_from_spinbox, self.analysis_to_spinbox)
        previous_signal_states = [control.blockSignals(True) for control in controls]
        try:
            self.analysis_from_spinbox.setRange(1, control_maximum)
            self.analysis_to_spinbox.setRange(1, control_maximum)
            self.analysis_from_spinbox.setValue(1)
            self.analysis_to_spinbox.setValue(control_maximum)
            for control in controls:
                control.setEnabled(page_count > 0)
            self.analyze_pages_button.setEnabled(page_count > 0)
            self.run_ocr_button.setEnabled(page_count > 0)
        finally:
            for control, previous_state in zip(controls, previous_signal_states):
                control.blockSignals(previous_state)
        self._update_analysis_range_status()

    def selected_analysis_page_indices(self) -> range:
        """Return validated zero-based source-page indices selected for analysis."""

        if self.pdf_service.document is None:
            raise ValueError("Open a PDF before selecting pages for analysis.")
        return page_indices_for_range(
            self.analysis_from_spinbox.value(),
            self.analysis_to_spinbox.value(),
            self.session.page_count,
        )

    def _update_analysis_range_status(self, *_):
        try:
            self.selected_analysis_page_indices()
        except ValueError:
            self.analysis_range_status_label.setText("Select a valid page range")
            return
        first_page = self.analysis_from_spinbox.value()
        last_page = self.analysis_to_spinbox.value()
        self.analysis_range_status_label.setText(f"Pages {first_page} to {last_page} selected")

    def analyze_selected_pages(self):
        self._run_selected_page_analysis(force_ocr=False)

    def run_ocr_selected_pages(self):
        self._run_selected_page_analysis(force_ocr=True)

    def _run_selected_page_analysis(self, *, force_ocr: bool):
        if self.pdf_service.document is None or not self.current_file:
            action = "running OCR" if force_ocr else "analyzing pages"
            self.analysis_summary_label.setText(f"Open a PDF before {action}")
            return
        try:
            page_indices = self.selected_analysis_page_indices()
        except ValueError as exc:
            self.analysis_summary_label.setText(str(exc))
            self.analysis_diagnostics.setPlainText("Choose a valid inclusive page range.")
            return

        layout_options = {
            page_index: self.page_layout_settings.get(page_index, PageLayoutOptions())
            for page_index in page_indices
        }

        self.analyze_pages_button.setEnabled(False)
        self.run_ocr_button.setEnabled(False)
        self.statusBar().showMessage(
            "Running OCR on selected pages..." if force_ocr else "Analyzing selected pages..."
        )
        try:
            result = run_selected_page_analysis(
                self.current_file,
                page_indices,
                layout_options=layout_options,
                force_ocr=force_ocr,
            )
        except Exception as exc:
            self.analysis_result = None
            self.analysis_summary_label.setText("Page analysis could not be completed")
            self.analysis_diagnostics.setPlainText(f"{type(exc).__name__}: {exc}")
            return
        finally:
            self.analyze_pages_button.setEnabled(self.session.page_count > 0)
            self.run_ocr_button.setEnabled(self.session.page_count > 0)

        self.analysis_result = result
        if result.document_error:
            self.analysis_summary_label.setText("Page analysis could not be completed")
            self.analysis_diagnostics.setPlainText(result.document_error)
            return

        created_regions = self._replace_regions_from_successful_pages(result)

        selected_count = len(result.selected_page_indices)
        warning_count = sum(len(page.warnings) for page in result.pages)
        failure_count = sum(len(page.failures) for page in result.pages)
        page_label = "page" if selected_count == 1 else "pages"
        action = f"Ran OCR on {selected_count} {page_label}" if force_ocr else f"Analyzed {selected_count} {page_label}"
        self.analysis_summary_label.setText(
            f"{action}; created {created_regions} editable rectangles; "
            f"{warning_count} warnings; {failure_count} failures"
        )
        self.statusBar().showMessage(self.analysis_summary_label.text())

        details: list[str] = []
        for page in result.pages:
            ambiguous = page.ambiguous_lines
            details.append(
                f"Page {page.page_index + 1} ({page.text_source}): {len(page.proposals)} proposals, "
                f"{len(ambiguous)} ambiguous line(s), {len(page.skipped_candidate_orders)} skipped"
            )
            details.extend(f"  Warning: {warning}" for warning in page.warnings)
            details.extend(f"  Failure: {failure}" for failure in page.failures)
            if page.failures and any(
                region.source == "automatic" and region.page_index == page.page_index
                for region in self.session.regions
            ):
                details.append("  Existing generated rectangles were retained because this page did not complete successfully.")
            if page.rectangles is not None:
                details.extend(
                    f"  Skipped candidate {order}"
                    for order in page.rectangles.skipped_candidate_orders
                )
            details.extend(
                f"  Ambiguous: {item.line.text} ({item.reason})"
                for item in ambiguous
            )
        self.analysis_diagnostics.setPlainText("\n".join(details))
        self.update_question_list()
        first_page_with_new_regions = next(
            (
                page.page_index
                for page in result.pages
                if not page.failures and page.rectangles is not None and page.proposals
            ),
            None,
        )
        if first_page_with_new_regions is not None:
            self.show_page(first_page_with_new_regions)
        else:
            self.show_page(self.session.current_page_index)

    def _replace_regions_from_successful_pages(self, result: QuestionAnalysisRun) -> int:
        successful_pages = {
            page.page_index
            for page in result.pages
            if not page.failures and page.rectangles is not None
        }
        removed_regions = [
            region
            for region in self.session.regions
            if region.source == "automatic" and region.page_index in successful_pages
        ]
        for region in removed_regions:
            self.session.remove_region(region)
        if self.selected_region in removed_regions:
            self.selected_region = None
            self.session.selected_region_index = None

        created = 0
        for page in result.pages:
            if page.page_index not in successful_pages:
                continue
            for proposal in page.proposals:
                if proposal.page_index != page.page_index:
                    continue
                self.session.add_region(
                    QuestionRegion(
                        page_index=proposal.page_index,
                        rect=fitz.Rect(proposal.rect),
                        source="automatic",
                    )
                )
                created += 1
        return created

    def _page_layout_options(self, page_index: int) -> PageLayoutOptions:
        return self.page_layout_settings.get(page_index, PageLayoutOptions())

    def _sync_layout_controls_for_page(self, page_index: int):
        options = self._page_layout_options(page_index)
        self._syncing_layout_controls = True
        try:
            combo_index = self.page_layout_combo.findData(options.mode)
            if combo_index >= 0:
                self.page_layout_combo.setCurrentIndex(combo_index)
        finally:
            self._syncing_layout_controls = False
        is_two_columns = options.mode == PageLayoutMode.TWO_COLUMNS
        self.place_divider_button.setEnabled(is_two_columns)
        self.clear_divider_button.setEnabled(is_two_columns and options.divider_normalized is not None)
        self.apply_layout_range_button.setEnabled(self.pdf_service.document is not None)
        if is_two_columns and options.divider_normalized is None:
            self.page_layout_status_label.setText("Two columns · place a divider before analysis")
        elif is_two_columns:
            self.page_layout_status_label.setText(
                f"Two columns · divider at {options.divider_normalized:.0%} of page width"
            )
        else:
            self.page_layout_status_label.setText(
                "Single column · automatic splitting disabled"
                if options.mode == PageLayoutMode.SINGLE_COLUMN
                else "Automatic layout detection"
            )

    def _on_page_layout_mode_changed(self, *_):
        if self._syncing_layout_controls or self.pdf_service.document is None:
            return
        mode = self.page_layout_combo.currentData()
        previous = self._page_layout_options(self.session.current_page_index)
        self.page_layout_settings[self.session.current_page_index] = PageLayoutOptions(
            mode=mode,
            divider_normalized=previous.divider_normalized,
        )
        if mode != PageLayoutMode.TWO_COLUMNS:
            self._cancel_divider_placement()
        self._sync_layout_controls_for_page(self.session.current_page_index)
        self.show_page(self.session.current_page_index)

    def start_divider_placement(self):
        if self.pdf_service.document is None:
            return
        if self._page_layout_options(self.session.current_page_index).mode != PageLayoutMode.TWO_COLUMNS:
            self.page_layout_combo.setCurrentIndex(self.page_layout_combo.findData(PageLayoutMode.TWO_COLUMNS))
        self.viewer.divider_placement_mode = True
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.page_layout_status_label.setText("Click the page to place a divider, or drag the existing line")

    def finish_divider_placement(self):
        self._cancel_divider_placement()
        self._sync_layout_controls_for_page(self.session.current_page_index)

    def _cancel_divider_placement(self):
        self.viewer.divider_placement_mode = False
        self.viewer.setCursor(Qt.CursorShape.ArrowCursor)

    def set_manual_divider_position(self, page_index: int, normalized_x: float):
        self.page_layout_settings[page_index] = PageLayoutOptions(
            mode=PageLayoutMode.TWO_COLUMNS,
            divider_normalized=normalized_x,
        )
        if page_index == self.session.current_page_index:
            self._sync_layout_controls_for_page(page_index)

    def clear_current_page_divider(self):
        page_index = self.session.current_page_index
        self.page_layout_settings[page_index] = PageLayoutOptions(
            mode=PageLayoutMode.TWO_COLUMNS,
            divider_normalized=None,
        )
        self._cancel_divider_placement()
        self.show_page(page_index)

    def apply_layout_to_analysis_range(self):
        try:
            page_indices = self.selected_analysis_page_indices()
        except ValueError as exc:
            self.page_layout_status_label.setText(str(exc))
            return
        current_options = self._page_layout_options(self.session.current_page_index)
        for page_index in page_indices:
            self.page_layout_settings[page_index] = PageLayoutOptions(
                mode=current_options.mode,
                divider_normalized=current_options.divider_normalized,
            )
        self._sync_layout_controls_for_page(self.session.current_page_index)
        self.show_page(self.session.current_page_index)

    def show_page(self, page_index: int):
        self._cancel_divider_placement()
        self.session.current_page_index = page_index
        if self.pdf_service.document is None:
            return
        self._sync_layout_controls_for_page(page_index)
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
        default_filename = _workspace_export_filename(self.current_file)
        path, _ = QFileDialog.getSaveFileName(self, "Save export", default_filename, "PDF files (*.pdf)")
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
