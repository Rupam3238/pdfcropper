import fitz
from PyQt6.QtWidgets import QApplication, QGraphicsScene

from pdfcropper.main_window import QuestionRectItem


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
