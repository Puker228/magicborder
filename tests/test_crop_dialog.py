from __future__ import annotations

from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtTest import QTest

from magicborder.crop_dialog import CropDialog, CropSelectionView
from magicborder.image_crop import CropBox


def _pixmap(width: int, height: int) -> QPixmap:
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#4a7c3a"))
    return pixmap


def _shown_dialog(qapp, width: int = 400, height: int = 300) -> CropDialog:
    dialog = CropDialog(_pixmap(width, height), display_name="leaf.png")
    dialog.resize(900, 700)
    dialog.show()
    qapp.processEvents()
    return dialog


def _view_point(view: CropSelectionView, x: float, y: float) -> QPoint:
    return view.mapFromScene(QPointF(x, y))


class TestCropDialog:
    def test_initial_selection_is_whole_image_in_original_resolution(
        self, qapp
    ) -> None:
        dialog = _shown_dialog(qapp, 4000, 3000)
        try:
            assert dialog.crop_box() == CropBox(0, 0, 4000, 3000)
            assert not hasattr(dialog, "preset")
            assert "4000×3000" in dialog.area_label.text()
            # Отказ от обрезки — «Оставить как есть»; «Обрезать» ждёт рамку.
            assert not dialog.crop_button.isEnabled()
            assert dialog.keep_button.isEnabled()
        finally:
            dialog.close()

    def test_crop_button_is_enabled_only_for_partial_frame(self, qapp) -> None:
        dialog = _shown_dialog(qapp, 400, 300)
        try:
            assert not dialog.crop_button.isEnabled()
            dialog.selection_view.set_crop_box(CropBox(10, 10, 200, 200))
            assert dialog.crop_button.isEnabled()
            dialog.reset_button.click()
            assert not dialog.crop_button.isEnabled()
        finally:
            dialog.close()

    def test_dragging_corner_handle_resizes_selection(self, qapp) -> None:
        dialog = _shown_dialog(qapp)
        view = dialog.selection_view
        try:
            viewport = view.viewport()
            start = _view_point(view, 400, 300)
            end = _view_point(view, 200, 150)
            assert view.handle_at(start) == "br"

            QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, start)
            QTest.mouseMove(viewport, end)
            QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, end)

            box = dialog.crop_box()
            assert (box.left, box.top) == (0, 0)
            assert abs(box.width - 200) <= 2
            assert abs(box.height - 150) <= 2
            assert dialog.crop_button.isEnabled()
            assert f"{box.width}×{box.height}" in dialog.area_label.text()
        finally:
            dialog.close()

    def test_dragging_inside_moves_selection_within_image(self, qapp) -> None:
        dialog = _shown_dialog(qapp)
        view = dialog.selection_view
        try:
            view.set_crop_box(CropBox(100, 100, 100, 100))
            viewport = view.viewport()
            start = _view_point(view, 150, 150)
            end = _view_point(view, 600, 150)
            assert view.handle_at(start) == "move"

            QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, start)
            QTest.mouseMove(viewport, end)
            QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, end)

            box = dialog.crop_box()
            assert box.size == (100, 100)
            assert box.right == 400
        finally:
            dialog.close()

    def test_reset_restores_whole_image(self, qapp) -> None:
        dialog = _shown_dialog(qapp, 3200, 2400)
        try:
            dialog.selection_view.set_crop_box(CropBox(0, 0, 1000, 500))
            assert "1000×500" in dialog.area_label.text()

            dialog.reset_button.click()

            assert dialog.crop_box() == CropBox(0, 0, 3200, 2400)
        finally:
            dialog.close()

    def test_keep_as_is_rejects_dialog(self, qapp) -> None:
        dialog = _shown_dialog(qapp)
        dialog.keep_button.click()
        assert dialog.result() == CropDialog.Rejected
