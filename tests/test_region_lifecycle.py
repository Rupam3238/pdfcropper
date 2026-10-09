try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication, QGraphicsScene

from pdfcropper.main_window import PdfCropperMainWindow, PdfPageView, QuestionRectItem
from pdfcropper.models import DocumentSession, QuestionRegion


def test_question_rect_item_creation_and_selection_does_not_crash():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    region = type("R", (), {"page_index": 0, "rect": fitz.Rect(20, 30, 200, 250), "label": "1"})()

    item = QuestionRectItem(region, 595, 842)
    scene.addItem(item)
    item.setSelected(True)
    item.setPos(10, 15)
    item.update_region_from_rect()
    item.sync_handles()

    assert item.label_item.text() == "1"
    assert len(item.handle_items) == 8
    assert all(handle.parentItem() is item for handle in item.handle_items.values())

    scene.removeItem(item)


def test_question_rect_item_move_and_resize_updates_geometry():
    app = QApplication.instance() or QApplication([])
    region = type("R", (), {"page_index": 0, "rect": fitz.Rect(40, 60, 140, 180), "label": "2"})()
    item = QuestionRectItem(region, 595, 842)

    item.setPos(15, 20)
    item.setRect(item.rect().adjusted(0, 0, 50, 20))
    item.update_region_from_rect()
    item.sync_handles()

    assert item.region.rect.width > 0
    assert item.region.rect.height > 0
    assert item.region.rect.x0 >= 0
    assert item.region.rect.y0 >= 0
    assert item.label_item.text() == "2"


def test_draw_mode_stays_active_until_selection_mode_is_chosen():
    app = QApplication.instance() or QApplication([])
    window = PdfCropperMainWindow()

    window.activate_drawing_mode()
    assert window.viewer.drawing_mode
    assert window.draw_rect_action.isChecked()
    assert "Draw rectangles" in window.statusBar().currentMessage()

    window.activate_selection_mode()
    assert not window.viewer.drawing_mode
    assert window.select_action.isChecked()
    window.close()


def test_drawing_release_survives_scene_redraw_and_keeps_mode_active():
    app = QApplication.instance() or QApplication([])
    view = PdfPageView()
    view.page_size = (595, 842)
    view.drawing_mode = True
    view.start_scene_pos = QPointF(20, 30)
    view.rubber_band_rect = view.scene.addRect(20, 30, 120, 140)

    class RedrawingController:
        def __init__(self):
            self.added = []

        def add_region_from_rect(self, page_index, rect):
            self.added.append((page_index, rect))
            view.clear_scene()

        def update_question_list(self):
            pass

    controller = RedrawingController()
    view.set_controller(controller)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(140, 170),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    view.mouseReleaseEvent(event)

    assert len(controller.added) == 1
    assert view.drawing_mode
    assert view.rubber_band_rect is None
    view.close()


def test_draw_shortcut_is_scoped_to_the_pdf_viewer():
    app = QApplication.instance() or QApplication([])
    window = PdfCropperMainWindow()
    window.show()

    key_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(window.viewer.viewport(), key_event)
    assert window.viewer.drawing_mode

    window.activate_selection_mode()
    key_event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(window.question_list, key_event)
    assert not window.viewer.drawing_mode
    window.close()


def test_add_before_and_after_insert_in_order_without_overwriting(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "source.pdf"
    document = fitz.open()
    document.new_page(width=595, height=842)
    document.save(source)
    document.close()

    window = PdfCropperMainWindow()
    window.pdf_service.open(str(source))
    first = QuestionRegion(page_index=0, rect=fitz.Rect(20, 20, 120, 120), label="1")
    selected = QuestionRegion(page_index=0, rect=fitz.Rect(200, 200, 300, 300), label="2")
    last = QuestionRegion(page_index=0, rect=fitz.Rect(400, 500, 500, 600), label="3")
    window.session = DocumentSession(path=str(source), page_count=1, regions=[first, selected, last])
    window.selected_region = selected
    monkeypatch.setattr(window, "show_page", lambda page_index: None)

    window.add_before_selected()
    inserted_before = window.selected_region
    assert window.session.regions == [first, inserted_before, selected, last]
    assert [region.label for region in window.session.regions] == ["1", "2", "3", "4"]
    assert not inserted_before.rect.intersects(selected.rect)
    assert inserted_before.rect.width > 0 and inserted_before.rect.height > 0

    window.selected_region = selected
    window.add_after_selected()
    inserted_after = window.selected_region
    assert window.session.regions == [first, inserted_before, selected, inserted_after, last]
    assert [region.label for region in window.session.regions] == ["1", "2", "3", "4", "5"]
    assert not inserted_after.rect.intersects(selected.rect)
    window.pdf_service.close()
    window.close()
