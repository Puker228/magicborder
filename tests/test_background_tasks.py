"""Тяжёлые команды над большими изображениями не блокируют GUI-поток."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from PyQt5.QtWidgets import QMessageBox

from magicborder import main_window as main_window_module
from magicborder.detector import detect_leaf_contour
from magicborder.io_utils import load_project, save_project
from magicborder.main_window import (
    CONTOUR_ANALYSIS_PENDING_TEXT,
    CONTOUR_ANALYSIS_SYNC_PIXEL_LIMIT,
    MainWindow,
)
from magicborder.models import Annotation, Point, ProjectDocument, ProjectImageRecord

LARGE_SIZE = (800, 600)
LEAF_CONTOUR = [Point(100, 100), Point(700, 100), Point(700, 500), Point(100, 500)]
DETECTED_CONTOUR = [Point(10, 10), Point(300, 10), Point(300, 200), Point(10, 200)]

assert LARGE_SIZE[0] * LARGE_SIZE[1] > CONTOUR_ANALYSIS_SYNC_PIXEL_LIMIT


class FakeThreadPool:
    """Копит воркеры, чтобы тест сам решал, когда «поток» завершится."""

    def __init__(self) -> None:
        self.workers: list[Any] = []

    def start(self, worker: Any) -> None:
        self.workers.append(worker)

    def run_all(self) -> None:
        workers, self.workers = self.workers, []
        for worker in workers:
            worker.run()


@pytest.fixture()
def messages(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[tuple[str, str]]]:
    record: dict[str, list[tuple[str, str]]] = {"critical": [], "warning": []}
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, text, *_args: record["critical"].append((title, text)),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, text, *_args: record["warning"].append((title, text)),
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes
    )
    return record


@pytest.fixture()
def large_window(qapp, tmp_path: Path, messages):  # noqa: ARG001
    windows: list[MainWindow] = []

    def factory(*, annotated: bool = False) -> MainWindow:
        image_dir = tmp_path / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for index, color in enumerate(((120, 80, 40), (20, 60, 100))):
            name = f"leaf-{index}.png"
            Image.new("RGB", LARGE_SIZE, color).save(image_dir / name)
            annotation = (
                Annotation(
                    image_path=f"images/{name}",
                    image_width=LARGE_SIZE[0],
                    image_height=LARGE_SIZE[1],
                    points=list(LEAF_CONTOUR),
                )
                if annotated
                else None
            )
            records.append(
                ProjectImageRecord(
                    id=f"leaf-{index}",
                    relative_path=f"images/{name}",
                    display_name=name,
                    image_width=LARGE_SIZE[0],
                    image_height=LARGE_SIZE[1],
                    annotation=annotation,
                )
            )
        project_path = tmp_path / "large.json"
        save_project(project_path, ProjectDocument(name="large", images=records))

        window = MainWindow()
        window._contour_analysis_thread_pool = FakeThreadPool()
        window._contour_detection_thread_pool = FakeThreadPool()
        window._project_color_stats_thread_pool = FakeThreadPool()
        window._flatten_background_thread_pool = FakeThreadPool()
        window._set_project(project_path, load_project(project_path))
        windows.append(window)
        return window

    yield factory

    for window in windows:
        window.close()
        window.deleteLater()


def _busy(window: MainWindow) -> bool:
    return not window.busy_progress.isHidden()


class TestContourDetectionWorker:
    def test_large_image_is_detected_in_worker(
        self, large_window, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = large_window()
        monkeypatch.setattr(
            main_window_module, "detect_leaf_contour", lambda _rgb: DETECTED_CONTOUR
        )

        window.detect_contour()

        # Команда вернулась сразу: контура ещё нет, работа в пуле потоков.
        assert window.canvas.has_contour() is False
        assert len(window._contour_detection_thread_pool.workers) == 1
        assert _busy(window)
        assert window.detect_contour_action.isEnabled() is False
        assert window.flatten_background_action.isEnabled() is False

        window._contour_detection_thread_pool.run_all()

        assert len(window.canvas.contour_points()) == 4
        assert "4" in window.statusBar().currentMessage()
        assert not _busy(window)
        assert window.detect_contour_action.isEnabled() is True
        assert window._contour_detection_workers == {}

    def test_repeated_command_while_pending_starts_no_second_worker(
        self, large_window, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = large_window()
        monkeypatch.setattr(
            main_window_module, "detect_leaf_contour", lambda _rgb: DETECTED_CONTOUR
        )

        window.detect_contour()
        window.detect_contour()

        assert len(window._contour_detection_thread_pool.workers) == 1

    def test_result_for_previous_image_is_ignored(
        self, large_window, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = large_window()
        monkeypatch.setattr(
            main_window_module, "detect_leaf_contour", lambda _rgb: DETECTED_CONTOUR
        )

        window.detect_contour()
        window.project_list.setCurrentRow(1)
        window._contour_detection_thread_pool.run_all()

        assert window._current_project_image_id == "leaf-1"
        assert window.canvas.has_contour() is False
        assert not _busy(window)
        assert window.detect_contour_action.isEnabled() is True

    def test_worker_error_is_shown_in_gui_thread(
        self,
        large_window,
        messages,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = large_window()

        def failing_detector(_rgb):
            raise ValueError("Не удалось определить контур листа.")

        monkeypatch.setattr(main_window_module, "detect_leaf_contour", failing_detector)

        window.detect_contour()
        window._contour_detection_thread_pool.run_all()

        assert messages["critical"][-1] == (
            "Не удалось определить контур",
            "Не удалось определить контур листа.",
        )
        assert window.canvas.has_contour() is False
        assert not _busy(window)

    def test_worker_gives_the_same_contour_as_direct_detection(
        self, qapp, tmp_path: Path, messages
    ) -> None:
        # Алгоритм детекции не меняется: воркер только переносит вызов в поток.
        height, width = 600, 800
        yy, xx = np.mgrid[0:height, 0:width]
        rgb = np.full((height, width, 3), 235, dtype=np.uint8)
        rgb[((xx - 400) / 250.0) ** 2 + ((yy - 300) / 180.0) ** 2 < 1] = (60, 140, 50)
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        Image.fromarray(rgb).save(image_dir / "leaf.png")
        project_path = tmp_path / "real.json"
        save_project(
            project_path,
            ProjectDocument(
                name="real",
                images=[
                    ProjectImageRecord(
                        id="leaf",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    )
                ],
            ),
        )
        window = MainWindow()
        try:
            pool = FakeThreadPool()
            window._contour_detection_thread_pool = pool
            window._set_project(project_path, load_project(project_path))

            window.detect_contour()
            pool.run_all()

            expected = detect_leaf_contour(rgb)
            assert window.canvas.contour_points() == pytest.approx(expected)
        finally:
            window.close()
            window.deleteLater()


class TestProjectColorStatsWorker:
    def test_refresh_computes_large_project_stats_in_worker(self, large_window) -> None:
        window = large_window(annotated=True)
        progress: list[tuple[int, int]] = []

        window.refresh_analysis()

        assert window.project_mean_red.text() == CONTOUR_ANALYSIS_PENDING_TEXT
        (worker,) = window._project_color_stats_thread_pool.workers
        worker.signals.progress.connect(
            lambda _request_id, done, total: progress.append((done, total))
        )
        assert _busy(window)

        window._project_color_stats_thread_pool.run_all()

        assert progress == [(1, 2), (2, 2)]
        # Оба кадра одного размера и с одинаковым контуром: среднее — середина цветов.
        assert window.project_mean_red.text() == "70"
        assert window.project_mean_green.text() == "70"
        assert window.project_mean_blue.text() == "70"
        assert not _busy(window)

    def test_repeated_refresh_reuses_cached_sums(
        self, large_window, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = large_window(annotated=True)
        window.refresh_analysis()
        window._project_color_stats_thread_pool.run_all()

        def unexpected_load(_path):
            raise AssertionError("неизменённые изображения не должны декодироваться")

        monkeypatch.setattr(main_window_module, "load_rgb_array", unexpected_load)
        window.refresh_analysis()

        assert window._project_color_stats_thread_pool.workers == []
        assert window.project_mean_red.text() == "70"

    def test_changed_file_is_recomputed_alone(
        self, large_window, tmp_path: Path
    ) -> None:
        window = large_window(annotated=True)
        window.refresh_analysis()
        window._project_color_stats_thread_pool.run_all()

        changed = tmp_path / "images" / "leaf-1.png"
        Image.new("RGB", LARGE_SIZE, (220, 60, 100)).save(changed)
        stat = changed.stat()
        os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        window.refresh_analysis()

        (worker,) = window._project_color_stats_thread_pool.workers
        assert [entry.image_path.name for entry in worker._entries] == ["leaf-1.png"]
        window._project_color_stats_thread_pool.run_all()
        assert window.project_mean_red.text() == "170"

    def test_closing_project_cancels_pending_stats(self, large_window) -> None:
        window = large_window(annotated=True)
        window.refresh_analysis()
        (worker,) = window._project_color_stats_thread_pool.workers

        window.close_project()
        worker.run()

        assert worker._cancel_event.is_set()
        assert window._pending_project_color_stats is None
        assert window.project_mean_red.text() == "-"
        assert not _busy(window)


class TestFlattenBackgroundWorker:
    def test_large_flatten_saves_in_worker_without_reloading_file(
        self, large_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = large_window()
        window.canvas.set_contour(LEAF_CONTOUR)
        image_path = tmp_path / "images" / "leaf-0.png"

        def unexpected_reload(_path):
            raise AssertionError("сохранённый файл не должен декодироваться повторно")

        monkeypatch.setattr(main_window_module, "load_raster_image", unexpected_reload)

        window.flatten_background()

        # До завершения воркера ни файл, ни канвас не меняются.
        assert tuple(window.canvas.current_rgb_array()[0, 0]) == (120, 80, 40)
        with Image.open(image_path) as image:
            assert image.getpixel((0, 0)) == (120, 80, 40)
        assert window.flatten_background_action.isEnabled() is False
        assert window.detect_contour_action.isEnabled() is False
        assert _busy(window)

        window._flatten_background_thread_pool.run_all()

        with Image.open(image_path) as image:
            assert image.getpixel((0, 0)) == (255, 255, 255)
            assert image.getpixel((400, 300)) == (120, 80, 40)
        canvas_rgb = window.canvas.current_rgb_array()
        assert tuple(canvas_rgb[0, 0]) == (255, 255, 255)
        assert tuple(canvas_rgb[300, 400]) == (120, 80, 40)
        assert window.flatten_background_action.isEnabled() is True
        assert not _busy(window)
        assert not list(image_path.parent.glob(".*magicborder-*"))

    def test_result_after_switching_image_is_saved_but_not_shown(
        self, large_window, tmp_path: Path
    ) -> None:
        window = large_window()
        window.canvas.set_contour(LEAF_CONTOUR)

        window.flatten_background()
        window.project_list.setCurrentRow(1)
        window._flatten_background_thread_pool.run_all()

        assert tuple(window.canvas.current_rgb_array()[0, 0]) == (20, 60, 100)
        with Image.open(tmp_path / "images" / "leaf-0.png") as image:
            assert image.getpixel((0, 0)) == (255, 255, 255)

        window.project_list.setCurrentRow(0)
        assert tuple(window.canvas.current_rgb_array()[0, 0]) == (255, 255, 255)

    def test_save_error_is_reported_and_canvas_is_untouched(
        self,
        large_window,
        messages,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = large_window()
        window.canvas.set_contour(LEAF_CONTOUR)

        def failing_save(_rgb, _path):
            raise OSError("диск переполнен")

        monkeypatch.setattr(
            main_window_module, "save_rgb_array_atomically", failing_save
        )

        window.flatten_background()
        window._flatten_background_thread_pool.run_all()

        assert messages["critical"][-1] == (
            "Не удалось выровнять фон",
            "диск переполнен",
        )
        assert tuple(window.canvas.current_rgb_array()[0, 0]) == (120, 80, 40)
        assert not _busy(window)
