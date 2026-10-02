from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QApplication

from magicborder import main_window as main_window_module
from magicborder.canvas import ImageCanvas
from magicborder.contour_analysis import (
    build_contour_analysis,
    contour_color_sums,
    contour_mask_from_points,
    contour_signature,
)
from magicborder.image_crop import CropBox, apply_crop_to_record
from magicborder.io_utils import load_project, save_project
from magicborder.main_window import MainWindow, _exclusion_start_points
from magicborder.models import Annotation, Point, ProjectDocument, ProjectImageRecord

CONTOUR = [Point(0, 0), Point(40, 0), Point(40, 30), Point(0, 30)]
EXCLUSION = [Point(10, 10), Point(20, 10), Point(20, 20), Point(10, 20)]


def _annotation(**kwargs: Any) -> Annotation:
    return Annotation(
        image_path="images/leaf.png",
        image_width=40,
        image_height=30,
        points=list(CONTOUR),
        **kwargs,
    )


class TestAnnotationExclusions:
    def test_round_trip(self) -> None:
        annotation = _annotation(exclusions=[list(EXCLUSION)])

        restored = Annotation.from_dict(annotation.to_dict())

        assert restored.exclusions == [EXCLUSION]

    def test_empty_exclusions_are_not_written(self) -> None:
        assert "exclusions" not in _annotation().to_dict()

    def test_annotation_without_exclusions_is_backward_compatible(self) -> None:
        payload = _annotation().to_dict()

        assert Annotation.from_dict(payload).exclusions == []

    def test_broken_exclusions_are_skipped_without_breaking_contour(self) -> None:
        payload = _annotation().to_dict()
        payload["exclusions"] = [
            {"points": [point.to_dict() for point in EXCLUSION]},
            {"points": [{"x": 1, "y": 1}, {"x": 2, "y": 2}]},
            {"points": [{"x": "много", "y": 1}] * 3},
            "мусор",
        ]

        restored = Annotation.from_dict(payload)

        assert restored.points == CONTOUR
        assert restored.exclusions == [EXCLUSION]


class TestExclusionCalculations:
    def test_mask_subtracts_exclusion(self) -> None:
        mask = contour_mask_from_points((30, 40), CONTOUR, [EXCLUSION])

        assert mask[15, 15] == 0
        assert mask[5, 5] == 255

    def test_excluded_pixels_do_not_affect_mean_color(self) -> None:
        rgb = np.zeros((30, 40, 3), dtype=np.uint8)
        rgb[:, :] = (0, 200, 0)
        # «Кусок фона» внутри объекта: белый квадрат.
        rgb[9:22, 9:22] = (255, 255, 255)
        background_hole = [Point(9, 9), Point(21, 9), Point(21, 21), Point(9, 21)]

        with_hole = build_contour_analysis(rgb, CONTOUR, [background_hole])
        without_hole = build_contour_analysis(rgb, CONTOUR)

        assert with_hole is not None and without_hole is not None
        assert with_hole.stats.mean_rgb == (0, 200, 0)
        assert without_hole.stats.mean_rgb != (0, 200, 0)
        assert with_hole.stats.pixel_count < without_hole.stats.pixel_count

    def test_color_sums_respect_exclusions(self) -> None:
        rgb = np.full((30, 40, 3), 100, dtype=np.uint8)

        full = contour_color_sums(rgb, CONTOUR)
        reduced = contour_color_sums(rgb, CONTOUR, [EXCLUSION])

        assert full is not None and reduced is not None
        assert reduced.pixel_count == full.pixel_count - 11 * 11

    def test_signature_changes_with_exclusions(self) -> None:
        assert contour_signature(CONTOUR) == contour_signature(CONTOUR, [])
        assert contour_signature(CONTOUR) != contour_signature(CONTOUR, [EXCLUSION])
        moved = [Point(point.x + 1, point.y) for point in EXCLUSION]
        assert contour_signature(CONTOUR, [EXCLUSION]) != contour_signature(
            CONTOUR, [moved]
        )


class TestExclusionCrop:
    def test_crop_moves_exclusions_and_drops_degenerate(self) -> None:
        outside = [Point(1, 1), Point(3, 1), Point(3, 3)]
        record = ProjectImageRecord(
            id="a",
            relative_path="images/a.png",
            annotation=_annotation(exclusions=[list(EXCLUSION), outside]),
        )

        apply_crop_to_record(
            record, CropBox(left=5, top=5, width=30, height=20), (30, 20)
        )

        assert record.annotation is not None
        assert record.annotation.exclusions == [
            [Point(5, 5), Point(15, 5), Point(15, 15), Point(5, 15)]
        ]


class TestCanvasExclusions:
    @pytest.fixture()
    def loaded_canvas(self, canvas: ImageCanvas) -> ImageCanvas:
        from magicborder.io_utils import loaded_image_from_rgb_array

        rgb = np.zeros((30, 40, 3), dtype=np.uint8)
        canvas.set_loaded_image(loaded_image_from_rgb_array(Path("leaf.png"), rgb))
        return canvas

    def test_add_requires_contour(self, loaded_canvas: ImageCanvas) -> None:
        with pytest.raises(ValueError, match="контур"):
            loaded_canvas.add_exclusion(EXCLUSION)

    def test_add_and_read_back(self, loaded_canvas: ImageCanvas) -> None:
        loaded_canvas.set_contour(CONTOUR)
        geometry_events: list[None] = []
        loaded_canvas.contour_geometry_changed.connect(
            lambda: geometry_events.append(None)
        )

        loaded_canvas.add_exclusion(EXCLUSION)

        assert loaded_canvas.has_exclusions()
        assert loaded_canvas.exclusion_polygons() == [EXCLUSION]
        assert geometry_events

    def test_exclusion_is_drawn_in_its_own_color(
        self, loaded_canvas: ImageCanvas
    ) -> None:
        from magicborder.canvas import EXCLUSION_LINE_COLOR

        loaded_canvas.set_contour(CONTOUR)
        loaded_canvas.add_exclusion(EXCLUSION)

        path_item = loaded_canvas._exclusion_path_items[0]
        assert path_item.pen().color().name() == EXCLUSION_LINE_COLOR
        assert EXCLUSION_LINE_COLOR != loaded_canvas.contour_line_color()

    def test_contour_pixels_exclude_selection(self, loaded_canvas: ImageCanvas) -> None:
        loaded_canvas.set_contour(CONTOUR)
        before = loaded_canvas.contour_rgb_pixels().shape[0]

        loaded_canvas.add_exclusion(EXCLUSION)

        assert loaded_canvas.contour_rgb_pixels().shape[0] == before - 11 * 11

    def test_clear_contour_clears_exclusions(self, loaded_canvas: ImageCanvas) -> None:
        loaded_canvas.set_contour(CONTOUR)
        loaded_canvas.add_exclusion(EXCLUSION)

        loaded_canvas.clear_contour()

        assert loaded_canvas.exclusion_polygons() == []
        assert loaded_canvas._exclusion_path_items == []

    def test_visibility_follows_contour(self, loaded_canvas: ImageCanvas) -> None:
        loaded_canvas.set_contour(CONTOUR)
        loaded_canvas.add_exclusion(EXCLUSION)

        loaded_canvas.set_contour_visible(False)
        assert not loaded_canvas._exclusion_path_items[0].isVisible()

        loaded_canvas.set_contour_visible(True)
        assert loaded_canvas._exclusion_path_items[0].isVisible()

    def test_node_editing_like_contour(self, loaded_canvas: ImageCanvas) -> None:
        loaded_canvas.set_contour(CONTOUR)
        loaded_canvas.add_exclusion(EXCLUSION)

        assert loaded_canvas.insert_exclusion_node_near(QPointF(15, 10.5))
        assert len(loaded_canvas.exclusion_polygons()[0]) == 5

        assert loaded_canvas.remove_exclusion_node(0, 0)
        assert len(loaded_canvas.exclusion_polygons()[0]) == 4

        handle = loaded_canvas._exclusion_handles[0][0]
        handle.setPos(QPointF(12, 12))
        assert loaded_canvas.exclusion_polygons()[0][0] == Point(12, 12)

    def test_polygon_keeps_at_least_three_nodes(
        self, loaded_canvas: ImageCanvas
    ) -> None:
        loaded_canvas.set_contour(CONTOUR)
        loaded_canvas.add_exclusion(EXCLUSION[:3])

        assert not loaded_canvas.remove_exclusion_node(0, 0)
        for handle in loaded_canvas._exclusion_handles[0]:
            handle.setSelected(True)
        assert not loaded_canvas.delete_selected_exclusion_nodes()
        assert len(loaded_canvas.exclusion_polygons()[0]) == 3

    def test_insert_far_from_exclusion_falls_through(
        self, loaded_canvas: ImageCanvas
    ) -> None:
        loaded_canvas.set_contour(CONTOUR)
        loaded_canvas.add_exclusion(EXCLUSION)

        assert not loaded_canvas.insert_exclusion_node_near(QPointF(35, 28))

    def test_delete_selected_exclusion(self, loaded_canvas: ImageCanvas) -> None:
        loaded_canvas.set_contour(CONTOUR)
        loaded_canvas.add_exclusion(EXCLUSION)
        loaded_canvas.add_exclusion([Point(25, 5), Point(35, 5), Point(30, 12)])

        assert not loaded_canvas.delete_selected_exclusion()
        loaded_canvas._exclusion_handles[1][0].setSelected(True)

        assert loaded_canvas.delete_selected_exclusion()
        assert loaded_canvas.exclusion_polygons() == [EXCLUSION]


def test_exclusion_start_points_lie_inside_contour_bounds() -> None:
    points = _exclusion_start_points(CONTOUR, 6)

    assert len(points) == 6
    assert all(0 < point.x < 40 and 0 < point.y < 30 for point in points)


@pytest.fixture()
def window_with_image(
    qapp: QApplication,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        staticmethod(lambda _p, title, *a, **k: warnings.append(title)),
    )
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getInt",
        staticmethod(lambda *a, **k: (4, True)),
    )
    root = tmp_path / "project"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (40, 30), (0, 200, 0)).save(root / "images" / "leaf.png")
    project_path = root / "project.json"
    save_project(
        project_path,
        ProjectDocument(
            name="project",
            images=[
                ProjectImageRecord(
                    id="image-0",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    image_width=40,
                    image_height=30,
                )
            ],
        ),
    )
    window = MainWindow()
    window._set_project(project_path, load_project(project_path))
    window.warnings = warnings
    yield window
    window.close()
    window.deleteLater()


class TestMainWindowNegativeSelection:
    def test_requires_contour(self, window_with_image) -> None:
        window_with_image.add_negative_selection()

        assert window_with_image.warnings == ["Нет контура"]
        assert not window_with_image.canvas.has_exclusions()

    def test_is_saved_into_project_file(self, window_with_image) -> None:
        window = window_with_image
        window.canvas.set_contour(CONTOUR)
        assert window.add_exclusion_action.isEnabled()

        window.add_negative_selection()
        window._save_project_silently()

        payload = json.loads(window.project_path.read_text(encoding="utf-8"))
        exclusions = payload["images"][0]["contour"]["annotation"]["exclusions"]
        assert len(exclusions) == 1
        assert len(exclusions[0]["points"]) == 4
        assert window.delete_exclusion_action.isEnabled()

    def test_analysis_excludes_selection(self, window_with_image) -> None:
        window = window_with_image
        window.canvas.set_contour(CONTOUR)
        window.refresh_analysis()
        full_pixels = int(window.property_contour_pixels.text())

        window.canvas.add_exclusion(EXCLUSION)
        window.refresh_analysis()

        assert int(window.property_contour_pixels.text()) == full_pixels - 11 * 11

    def test_delete_without_selection_warns(self, window_with_image) -> None:
        window = window_with_image
        window.canvas.set_contour(CONTOUR)
        window.canvas.add_exclusion(EXCLUSION)

        window.delete_negative_selection()

        assert window.warnings == ["Негативное выделение не выбрано"]
        assert window.canvas.has_exclusions()

    def test_delete_selected_updates_record(self, window_with_image) -> None:
        window = window_with_image
        window.canvas.set_contour(CONTOUR)
        window.canvas.add_exclusion(EXCLUSION)
        window.canvas._exclusion_handles[0][0].setSelected(True)

        window.delete_negative_selection()

        record = window._current_project_image()
        assert record.annotation is not None
        assert record.annotation.exclusions == []
        assert not window.delete_exclusion_action.isEnabled()
