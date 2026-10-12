try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication, QFileDialog, QGraphicsScene

from pdfcropper.main_window import (
    ManualDividerItem,
    PdfCropperMainWindow,
    PdfPageView,
    QuestionRectItem,
)
from pdfcropper.models import DocumentSession, QuestionRegion
from pdfcropper.pipeline import PageLayoutMode, PageLayoutOptions


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


def test_page_range_resets_on_open_and_does_not_limit_navigation_or_manual_regions(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    first_pdf = tmp_path / "first.pdf"
    first_document = fitz.open()
    for _ in range(3):
        first_document.new_page(width=595, height=842)
    first_document.save(first_pdf)
    first_document.close()

    second_pdf = tmp_path / "second.pdf"
    second_document = fitz.open()
    for _ in range(2):
        second_document.new_page(width=595, height=842)
    second_document.save(second_pdf)
    second_document.close()

    paths = iter((str(first_pdf), str(second_pdf)))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (next(paths), "PDF files (*.pdf)"))
    window = PdfCropperMainWindow()

    window.open_pdf()
    assert window.analysis_from_spinbox.value() == 1
    assert window.analysis_to_spinbox.value() == 3
    assert window.analysis_from_spinbox.maximum() == 3
    assert window.analysis_to_spinbox.maximum() == 3
    assert list(window.selected_analysis_page_indices()) == [0, 1, 2]

    window.analysis_from_spinbox.setValue(2)
    window.analysis_to_spinbox.setValue(2)
    assert list(window.selected_analysis_page_indices()) == [1]
    window.next_page()
    window.next_page()
    assert window.session.current_page_index == 2

    window.add_region_from_rect(2, fitz.Rect(20, 30, 200, 220))
    assert window.session.regions[-1].page_index == 2
    assert window.session.regions[-1].rect == fitz.Rect(20, 30, 200, 220)

    window.analysis_result = object()
    window.analysis_summary_label.setText("Previous document result")
    window.analysis_diagnostics.setPlainText("Previous document warning")
    window.open_pdf()
    assert window.session.page_count == 2
    assert window.analysis_from_spinbox.value() == 1
    assert window.analysis_to_spinbox.value() == 2
    assert window.analysis_from_spinbox.maximum() == 2
    assert window.analysis_to_spinbox.maximum() == 2
    assert list(window.selected_analysis_page_indices()) == [0, 1]
    assert window.analysis_result is None
    assert window.analysis_summary_label.text() == "Analysis has not run"
    assert window.analysis_diagnostics.toPlainText() == ""
    window.pdf_service.close()
    window.close()


def test_export_dialog_default_updates_when_source_pdf_changes(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source_paths = [tmp_path / "Laws of motion.pdf", tmp_path / "Chapter 1.v2.pdf"]
    for source in source_paths:
        document = fitz.open()
        document.new_page(width=595, height=842)
        document.save(source)
        document.close()

    imports = iter(str(source) for source in source_paths)
    dialog_defaults = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args: (next(imports), "PDF files (*.pdf)"),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args: (dialog_defaults.append(args[2]) or "", "PDF files (*.pdf)"),
    )
    window = PdfCropperMainWindow()

    for _ in source_paths:
        window.open_pdf()
        window.add_region_from_rect(0, fitz.Rect(20, 30, 200, 220))
        window.export_selected()

    assert dialog_defaults == [
        "Laws of motion_workspaces.pdf",
        "Chapter 1.v2_workspaces.pdf",
    ]
    window.pdf_service.close()
    window.close()


def test_invalid_gui_page_range_is_not_silently_reordered(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "range.pdf"
    document = fitz.open()
    for _ in range(3):
        document.new_page(width=595, height=842)
    document.save(source)
    document.close()

    window = PdfCropperMainWindow()
    window.pdf_service.open(str(source))
    window.session = DocumentSession(path=str(source), page_count=3)
    window._reset_analysis_page_range(3)
    window.analysis_from_spinbox.setValue(3)
    window.analysis_to_spinbox.setValue(2)

    assert window.analysis_range_status_label.text() == "Select a valid page range"
    try:
        window.selected_analysis_page_indices()
    except ValueError as exc:
        assert "First page" in str(exc)
    else:
        raise AssertionError("A reversed page range must be rejected")

    window.pdf_service.close()
    window.close()


def test_analyze_pages_button_uses_selected_range_without_mutating_manual_regions(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "analysis_button.pdf"
    document = fitz.open()
    page = document.new_page(width=500, height=700)
    page.insert_text((40, 70), "Q1 Find the value")
    page = document.new_page(width=500, height=700)
    page.insert_text((40, 70), "Q2 Find the value")
    page.insert_text((40, 160), "Q3 Solve for y")
    page = document.new_page(width=500, height=700)
    page.insert_text((40, 70), "Q4 Calculate z")
    document.save(source)
    document.close()

    window = PdfCropperMainWindow()
    window.pdf_service.open(str(source))
    window.current_file = str(source)
    manual_region = QuestionRegion(page_index=0, rect=fitz.Rect(20, 30, 180, 220))
    window.session = DocumentSession(path=str(source), page_count=3, regions=[manual_region])
    window._reset_analysis_page_range(3)
    window.analysis_from_spinbox.setValue(2)
    window.analysis_to_spinbox.setValue(3)
    assert window.page_layout_combo.currentData() == PageLayoutMode.AUTOMATIC

    import pdfcropper.pipeline as pipeline_module

    original_analyze_selected_pages = pipeline_module.analyze_selected_pages
    calls = []

    def capture_call(path, page_indices, **kwargs):
        calls.append((path, tuple(page_indices)))
        assert set(kwargs["layout_options"]) == {1, 2}
        assert all(option.mode == PageLayoutMode.AUTOMATIC for option in kwargs["layout_options"].values())
        return original_analyze_selected_pages(path, page_indices, **kwargs)

    monkeypatch.setattr("pdfcropper.main_window.run_selected_page_analysis", capture_call)

    window.analyze_pages_button.click()

    assert calls == [(str(source), (1, 2))]
    assert window.analysis_result.selected_page_indices == (1, 2)
    assert window.analysis_result.proposal_count == 3
    assert "Analyzed 2 pages" in window.analysis_summary_label.text()
    assert "created 3 editable rectangles" in window.analysis_summary_label.text()
    generated_regions = [region for region in window.session.regions if region.source == "automatic"]
    assert len(generated_regions) == 3
    assert [(region.page_index, region.rect) for region in generated_regions] == [
        (proposal.page_index, proposal.rect) for proposal in window.analysis_result.proposals
    ]
    assert window.session.current_page_index == 1
    manual_overlays = [item for item in window.viewer.scene.items() if isinstance(item, QuestionRectItem)]
    assert len(manual_overlays) == 2
    assert all(item.region.source in {"manual", "automatic"} for item in manual_overlays)

    assert window.session.regions[0] is manual_region
    assert len(window.session.regions) == 4
    assert [region for region in window.session.regions if region.source == "automatic"] == generated_regions
    assert manual_region.page_index == 0
    assert manual_region.rect == fitz.Rect(20, 30, 180, 220)

    window.pdf_service.close()
    window.close()


def test_analysis_with_no_proposals_creates_no_generated_regions(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "no_proposals.pdf"
    document = fitz.open()
    page = document.new_page(width=500, height=700)
    page.insert_text((40, 70), "General notes without numbered questions")
    document.save(source)
    document.close()

    window = PdfCropperMainWindow()
    window.pdf_service.open(str(source))
    window.current_file = str(source)
    window.session = DocumentSession(path=str(source), page_count=1)
    window._reset_analysis_page_range(1)

    window.analyze_pages_button.click()

    assert window.analysis_result is not None
    assert window.analysis_result.proposal_count == 0
    assert "created 0 editable rectangles" in window.analysis_summary_label.text()
    assert not [region for region in window.session.regions if region.source == "automatic"]

    window.pdf_service.close()
    window.close()


def test_generated_region_overlay_stays_in_rotated_page_space_when_view_is_zoomed(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "rotated_proposal.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=300)
    page.insert_text((30, 50), "Q1 Find the value of x")
    page.set_rotation(90)
    document.save(source)
    document.close()

    window = PdfCropperMainWindow()
    window.pdf_service.open(str(source))
    window.current_file = str(source)
    window.session = DocumentSession(path=str(source), page_count=1)
    window._reset_analysis_page_range(1)
    window.analyze_pages_button.click()

    generated_regions = [region for region in window.session.regions if region.source == "automatic"]
    assert len(generated_regions) == 1
    generated_region = generated_regions[0]
    proposal = window.analysis_result.proposals[0]
    source_page = window.pdf_service.document[0]
    assert source_page.rect.width == pytest.approx(300)
    assert source_page.rect.height == pytest.approx(200)
    assert generated_region.page_index == proposal.page_index == 0
    assert generated_region.rect == proposal.rect
    overlay = next(
        item for item in window.viewer.scene.items()
        if isinstance(item, QuestionRectItem) and item.region is generated_region
    )
    assert overlay.pos().x() == pytest.approx(proposal.rect.x0)
    assert overlay.pos().y() == pytest.approx(proposal.rect.y0)

    window.viewer.set_zoom(2.0)
    assert window.viewer.transform().m11() == pytest.approx(2.0)
    assert overlay.pos().x() == pytest.approx(proposal.rect.x0)
    assert overlay.pos().y() == pytest.approx(proposal.rect.y0)

    window.pdf_service.close()
    window.close()


def test_rerun_replaces_generated_regions_only_on_successful_analyzed_pages(tmp_path, monkeypatch):
    from types import SimpleNamespace

    app = QApplication.instance() or QApplication([])
    source = tmp_path / "automatic_region_reruns.pdf"
    document = fitz.open()
    for page_index in range(4):
        page = document.new_page(width=500, height=700)
        page.insert_text((40, 70), f"Q{page_index + 1} Find the answer")
    document.save(source)
    document.close()

    window = PdfCropperMainWindow()
    window.pdf_service.open(str(source))
    window.current_file = str(source)
    manual_page_zero = QuestionRegion(page_index=0, rect=fitz.Rect(10, 20, 100, 110))
    old_auto_page_zero = QuestionRegion(page_index=0, rect=fitz.Rect(120, 20, 220, 110), source="automatic")
    old_auto_failed_page = QuestionRegion(page_index=1, rect=fitz.Rect(10, 130, 110, 220), source="automatic")
    old_auto_outside_range = QuestionRegion(page_index=3, rect=fitz.Rect(10, 240, 110, 330), source="automatic")
    window.session = DocumentSession(
        path=str(source),
        page_count=4,
        regions=[manual_page_zero, old_auto_page_zero, old_auto_failed_page, old_auto_outside_range],
    )
    window._reset_analysis_page_range(4)
    window.analysis_from_spinbox.setValue(1)
    window.analysis_to_spinbox.setValue(2)

    def page_result(page_index, proposals, failures=()):
        return SimpleNamespace(
            page_index=page_index,
            proposals=proposals,
            failures=list(failures),
            warnings=[],
            ambiguous_lines=[],
            skipped_candidate_orders=[],
            rectangles=SimpleNamespace(skipped_candidate_orders=[]),
            text_source="embedded text",
        )

    def run_result(pages):
        return SimpleNamespace(
            document_error=None,
            selected_page_indices=(0, 1),
            pages=pages,
            proposal_count=sum(len(page.proposals) for page in pages),
        )

    successful_proposal = SimpleNamespace(page_index=0, rect=fitz.Rect(20, 120, 180, 240))
    failed_page_proposal = SimpleNamespace(page_index=1, rect=fitz.Rect(120, 130, 220, 240))
    runs = iter((
        run_result([
            page_result(0, [successful_proposal]),
            page_result(1, [failed_page_proposal], failures=["simulated OCR failure"]),
        ]),
        run_result([page_result(0, []), page_result(1, [])]),
    ))
    monkeypatch.setattr("pdfcropper.main_window.run_selected_page_analysis", lambda *args, **kwargs: next(runs))

    window.analyze_selected_pages()

    auto_page_zero = [region for region in window.session.regions if region.source == "automatic" and region.page_index == 0]
    assert len(auto_page_zero) == 1
    assert auto_page_zero[0].rect == successful_proposal.rect
    assert manual_page_zero in window.session.regions
    assert old_auto_failed_page in window.session.regions
    assert old_auto_outside_range in window.session.regions
    assert "retained because this page did not complete successfully" in window.analysis_diagnostics.toPlainText()

    window.analyze_selected_pages()

    assert manual_page_zero in window.session.regions
    assert not any(region.source == "automatic" and region.page_index == 0 for region in window.session.regions)
    assert not any(region.source == "automatic" and region.page_index == 1 for region in window.session.regions)
    assert old_auto_outside_range in window.session.regions

    window.pdf_service.close()
    window.close()


def test_layout_modes_and_normalized_divider_are_per_page_and_apply_to_range(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "per_page_layout.pdf"
    document = fitz.open()
    document.new_page(width=500, height=700)
    rotated = document.new_page(width=200, height=300)
    rotated.set_rotation(90)
    document.new_page(width=400, height=600)
    document.save(source)
    document.close()

    window = PdfCropperMainWindow()
    window.pdf_service.open(str(source))
    window.current_file = str(source)
    window.session = DocumentSession(path=str(source), page_count=3)
    window._reset_analysis_page_range(3)
    two_columns_index = window.page_layout_combo.findData(PageLayoutMode.TWO_COLUMNS)
    window.page_layout_combo.setCurrentIndex(two_columns_index)
    window.start_divider_placement()

    assert window.viewer.divider_placement_mode
    assert window.viewer._normalized_page_x(200.0) == pytest.approx(0.4)
    window.set_manual_divider_position(0, 0.4)
    window.viewer.set_manual_divider_item_position(0.4)
    assert isinstance(window.viewer.manual_divider_item, ManualDividerItem)
    assert window.viewer.manual_divider_item.line().x1() == pytest.approx(200.0)
    window.viewer.set_zoom(2.0)
    assert window.viewer.manual_divider_item.line().x1() == pytest.approx(200.0)
    window.finish_divider_placement()

    window.analysis_from_spinbox.setValue(2)
    window.analysis_to_spinbox.setValue(3)
    window.apply_layout_to_analysis_range()
    assert window.page_layout_settings[0] == PageLayoutOptions(PageLayoutMode.TWO_COLUMNS, 0.4)
    assert window.page_layout_settings[1] == PageLayoutOptions(PageLayoutMode.TWO_COLUMNS, 0.4)
    assert window.page_layout_settings[2] == PageLayoutOptions(PageLayoutMode.TWO_COLUMNS, 0.4)

    window.show_page(1)
    assert window.page_layout_combo.currentData() == PageLayoutMode.TWO_COLUMNS
    assert window.viewer.page_size == pytest.approx((300.0, 200.0))
    assert window.viewer.manual_divider_item.line().x1() == pytest.approx(120.0)
    assert window.viewer._normalized_page_x(120.0) == pytest.approx(0.4)
    window.viewer.manual_divider_item.set_normalized_x(0.6)
    assert window.page_layout_settings[1].divider_normalized == pytest.approx(0.6)

    window.show_page(2)
    assert window.viewer.page_size == pytest.approx((400.0, 600.0))
    assert window.viewer.manual_divider_item.line().x1() == pytest.approx(160.0)
    window.show_page(0)
    assert window.page_layout_settings[0].divider_normalized == pytest.approx(0.4)
    assert window.viewer.manual_divider_item.line().x1() == pytest.approx(200.0)

    window.page_layout_combo.setCurrentIndex(window.page_layout_combo.findData(PageLayoutMode.SINGLE_COLUMN))
    assert window.page_layout_settings[0].mode == PageLayoutMode.SINGLE_COLUMN
    assert window.viewer.manual_divider_item is None

    calls = []

    def capture_analysis(path, page_indices, *, layout_options, force_ocr):
        calls.append((tuple(page_indices), dict(layout_options), force_ocr))
        return type("Run", (), {
            "document_error": None,
            "selected_page_indices": (1, 2),
            "proposal_count": 0,
            "pages": [],
        })()

    monkeypatch.setattr("pdfcropper.main_window.run_selected_page_analysis", capture_analysis)
    window.analyze_selected_pages()
    assert calls == [(
        (1, 2),
        {
            1: PageLayoutOptions(PageLayoutMode.TWO_COLUMNS, 0.6),
            2: PageLayoutOptions(PageLayoutMode.TWO_COLUMNS, 0.4),
        },
        False,
    )]
    window.pdf_service.close()
    window.close()


def test_run_ocr_button_creates_exportable_region_from_selected_page(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "explicit_ocr.pdf"
    output = tmp_path / "explicit_ocr_export.pdf"
    document = fitz.open()
    document.new_page(width=500, height=700).insert_text((40, 70), "Q1 embedded question text")
    document.new_page(width=500, height=700).insert_text((40, 70), "Q2 stale embedded text")
    document.save(source)
    document.close()

    window = PdfCropperMainWindow()
    window.pdf_service.open(str(source))
    window.current_file = str(source)
    window.session = DocumentSession(path=str(source), page_count=2)
    window._reset_analysis_page_range(2)
    window.analysis_from_spinbox.setValue(2)
    window.analysis_to_spinbox.setValue(2)
    ocr_calls = []

    def fake_ocr(path, page_index, *, dpi):
        ocr_calls.append(page_index)
        document = fitz.open()
        page = document.new_page(width=500, height=700)
        page.insert_text((40, 70), "Q9 Solve the equation")
        words = page.get_text("words")
        bounds = fitz.Rect(page.rect)
        document.close()
        from pdfcropper.analysis import ExtractedWord, PageAnalysis, TextAvailability

        return PageAnalysis(
            page_index=page_index,
            width=bounds.width,
            height=bounds.height,
            bounds=bounds,
            words=[
                ExtractedWord(str(word[4]), fitz.Rect(word[:4]), page_index, metadata={"source": "ocr"})
                for word in words
            ],
            availability=TextAvailability.USEFUL,
            has_usable_text=True,
        )

    monkeypatch.setattr("pdfcropper.pipeline.ocr_page_analysis", fake_ocr)

    window.run_ocr_button.click()

    generated = [region for region in window.session.regions if region.source == "automatic"]
    assert ocr_calls == [1]
    assert window.analysis_result.pages[0].text_source == "OCR"
    assert window.analysis_result.pages[0].page_index == 1
    assert generated
    assert all(region.page_index == 1 for region in generated)
    assert "Ran OCR on 1 page" in window.analysis_summary_label.text()
    from pdfcropper.exporter import export_question_regions

    export_question_regions(str(source), generated, str(output))
    exported = fitz.open(output)
    assert exported.page_count == len(generated)
    assert all(page.get_images() for page in exported)
    exported.close()

    window.pdf_service.close()
    window.close()


def test_explicit_ocr_rerun_clears_successful_empty_page_but_retains_failed_and_unselected_regions(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    app = QApplication.instance() or QApplication([])
    source = tmp_path / "explicit_ocr_region_lifecycle.pdf"
    document = fitz.open()
    for _ in range(3):
        document.new_page(width=500, height=700)
    document.save(source)
    document.close()

    window = PdfCropperMainWindow()
    window.pdf_service.open(str(source))
    window.current_file = str(source)
    manual = QuestionRegion(page_index=0, rect=fitz.Rect(10, 20, 100, 110))
    stale_empty_page = QuestionRegion(page_index=0, rect=fitz.Rect(120, 20, 220, 110), source="automatic")
    retained_failed_page = QuestionRegion(page_index=1, rect=fitz.Rect(10, 130, 110, 220), source="automatic")
    retained_outside_range = QuestionRegion(page_index=2, rect=fitz.Rect(10, 240, 110, 330), source="automatic")
    window.session = DocumentSession(
        path=str(source),
        page_count=3,
        regions=[manual, stale_empty_page, retained_failed_page, retained_outside_range],
    )
    window._reset_analysis_page_range(3)
    window.analysis_from_spinbox.setValue(1)
    window.analysis_to_spinbox.setValue(2)

    success_empty = SimpleNamespace(
        page_index=0,
        proposals=[],
        failures=[],
        warnings=[],
        ambiguous_lines=[],
        skipped_candidate_orders=[],
        rectangles=SimpleNamespace(skipped_candidate_orders=[]),
        text_source="OCR",
    )
    failed = SimpleNamespace(
        page_index=1,
        proposals=[],
        failures=["OCR failed for page 2: RuntimeError: Tesseract unavailable"],
        warnings=[],
        ambiguous_lines=[],
        skipped_candidate_orders=[],
        rectangles=None,
        text_source="OCR",
    )
    run = SimpleNamespace(
        document_error=None,
        selected_page_indices=(0, 1),
        proposal_count=0,
        pages=[success_empty, failed],
    )

    def capture_run(path, page_indices, *, layout_options, force_ocr):
        assert tuple(page_indices) == (0, 1)
        assert force_ocr is True
        return run

    monkeypatch.setattr("pdfcropper.main_window.run_selected_page_analysis", capture_run)
    window.run_ocr_button.click()

    assert manual in window.session.regions
    assert stale_empty_page not in window.session.regions
    assert retained_failed_page in window.session.regions
    assert retained_outside_range in window.session.regions
    diagnostics = window.analysis_diagnostics.toPlainText()
    assert "OCR failed for page 2" in diagnostics
    assert "retained because this page did not complete successfully" in diagnostics
    assert "1 failures" in window.analysis_summary_label.text()

    window.pdf_service.close()
    window.close()
