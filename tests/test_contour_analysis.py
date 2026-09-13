from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from magicborder.contour_analysis import (  # noqa: E402
    build_contour_analysis,
    color_stats_from_sums,
    combine_color_sums,
    contour_color_sums,
    contour_rgb_pixels_from_points,
    contour_signature,
    flatten_background_outside_contour,
)
from magicborder.histograms import rgb_to_lms  # noqa: E402
from magicborder.io_utils import load_project, save_project  # noqa: E402
from magicborder.main_window import (  # noqa: E402
    CONTOUR_ANALYSIS_OUTDATED_TEXT,
    CONTOUR_ANALYSIS_PENDING_TEXT,
    ContourAnalysisWorkResult,
    MainWindow,
)
from magicborder.models import Point, ProjectDocument, ProjectImageRecord  # noqa: E402

_APP: QApplication | None = None


class FakeThreadPool:
    def __init__(self) -> None:
        self.workers: list[object] = []

    def start(self, worker: object) -> None:
        self.workers.append(worker)


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


class ContourAnalysisTests(unittest.TestCase):
    def test_means_are_exact_on_full_hd_frame(self) -> None:
        # Раньше средние копились в float32 по миллионам пикселей и уплывали
        # на 1–2 единицы. Эталон здесь — прямой расчёт в float64.
        rng = np.random.default_rng(7)
        rgb = np.clip(rng.normal((70, 150, 60), 25, (1080, 1920, 3)), 0, 255).astype(
            np.uint8
        )
        points = [Point(100, 80), Point(1800, 120), Point(1700, 1000), Point(150, 950)]
        pixels = contour_rgb_pixels_from_points(rgb, points)
        column = pixels.reshape((-1, 1, 3))
        lab = cv2.cvtColor(column, cv2.COLOR_RGB2LAB).reshape((-1, 3)).astype(float)
        hsv = cv2.cvtColor(column, cv2.COLOR_RGB2HSV).reshape((-1, 3)).astype(float)
        yuv = cv2.cvtColor(column, cv2.COLOR_RGB2YUV).reshape((-1, 3)).astype(float)
        lms = rgb_to_lms(pixels).astype(float)

        def rounded(values: np.ndarray) -> tuple[int, int, int]:
            return tuple(int(v) for v in np.rint(values))

        analysis = build_contour_analysis(rgb, points)

        assert analysis is not None
        stats = analysis.stats
        self.assertEqual(stats.pixel_count, pixels.shape[0])
        self.assertEqual(stats.mean_rgb, rounded(pixels.astype(float).mean(axis=0)))
        self.assertEqual(
            stats.mean_lab,
            rounded(
                np.array(
                    [
                        lab[:, 0].mean() * 100.0 / 255.0,
                        lab[:, 1].mean() - 128.0,
                        lab[:, 2].mean() - 128.0,
                    ]
                )
            ),
        )
        self.assertEqual(
            stats.mean_hsv,
            rounded(
                np.array([hsv[:, 0].mean() * 2.0, hsv[:, 1].mean(), hsv[:, 2].mean()])
            ),
        )
        self.assertEqual(stats.mean_yuv, rounded(yuv.mean(axis=0)))
        self.assertEqual(
            stats.mean_lms,
            rounded(lms.mean(axis=0) / lms.max(axis=0) * 255.0),
        )

    def test_combined_sums_give_pixel_weighted_project_mean(self) -> None:
        small = np.full((10, 10, 3), (200, 0, 0), dtype=np.uint8)
        large = np.full((30, 30, 3), (0, 0, 100), dtype=np.uint8)
        small_points = [Point(0, 0), Point(9, 0), Point(9, 9), Point(0, 9)]
        large_points = [Point(0, 0), Point(29, 0), Point(29, 29), Point(0, 29)]

        combined = combine_color_sums(
            [
                contour_color_sums(small, small_points),
                contour_color_sums(large, large_points),
            ]
        )

        assert combined is not None
        stats = color_stats_from_sums(combined)
        self.assertEqual(stats.pixel_count, 100 + 900)
        self.assertEqual(stats.mean_rgb, (20, 0, 90))
        self.assertIsNone(combine_color_sums([]))

    def test_flatten_background_outside_contour_returns_new_array(self) -> None:
        source = np.full((20, 30, 3), 60, dtype=np.uint8)
        source.flags.writeable = False
        points = [Point(5, 5), Point(20, 5), Point(20, 15), Point(5, 15)]

        flattened = flatten_background_outside_contour(source, points)

        self.assertEqual(tuple(flattened[0, 0]), (255, 255, 255))
        self.assertEqual(tuple(flattened[10, 10]), (60, 60, 60))
        self.assertEqual(tuple(source[0, 0]), (60, 60, 60))
        with self.assertRaises(ValueError):
            flatten_background_outside_contour(source, points[:2])

    def test_build_contour_analysis_reuses_one_contour_pixel_selection(self) -> None:
        rgb_array = Image.new("RGB", (20, 20), (120, 80, 40))
        points = [
            Point(1, 1),
            Point(18, 1),
            Point(18, 18),
            Point(1, 18),
        ]

        analysis = build_contour_analysis(
            rgb_array=np.asarray(rgb_array),
            points=points,
        )

        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis.stats.pixel_count, 324)
        self.assertEqual(analysis.stats.mean_rgb, (120, 80, 40))
        self.assertEqual(analysis.histograms.rgb.sample_count, 324)
        self.assertEqual(analysis.histograms.lab.sample_count, 324)
        self.assertEqual(analysis.histograms.hsv.sample_count, 324)
        self.assertEqual(analysis.histograms.yuv.sample_count, 324)
        self.assertEqual(analysis.histograms.lms.sample_count, 324)

    def test_large_contour_analysis_is_deferred_and_ignores_stale_result(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (700, 700), (120, 80, 40)).save(image_dir / "leaf.png")
            project = ProjectDocument(
                name="large",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    )
                ],
            )
            project_path = root / "large.json"
            save_project(project_path, project)

            window = MainWindow()
            fake_pool = FakeThreadPool()
            window._contour_analysis_thread_pool = fake_pool
            window._set_project(project_path, load_project(project_path))

            first_points = [
                Point(10, 10),
                Point(600, 10),
                Point(600, 600),
                Point(10, 600),
            ]
            window.canvas.set_contour(first_points)

            # Без команды «Обновить» расчёт не стартует.
            self.assertEqual(fake_pool.workers, [])
            self.assertEqual(
                window.property_contour_pixels.text(), CONTOUR_ANALYSIS_OUTDATED_TEXT
            )

            window.refresh_analysis()
            self.assertEqual(len(fake_pool.workers), 1)
            self.assertEqual(
                window.property_contour_pixels.text(), CONTOUR_ANALYSIS_PENDING_TEXT
            )
            first_result = ContourAnalysisWorkResult(
                request_id=window._contour_analysis_request_id,
                record_id="leaf-1",
                image_path=str(window.canvas.current_image_path()),
                signature=contour_signature(first_points),
                analysis=None,
            )

            window.canvas.set_contour(
                [
                    Point(20, 20),
                    Point(620, 20),
                    Point(620, 620),
                    Point(20, 620),
                ]
            )
            window._handle_contour_analysis_finished(first_result)

            self.assertIsNone(window._contour_analysis_cache)


if __name__ == "__main__":
    unittest.main()
