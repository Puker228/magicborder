from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from magicborder.contour_analysis import (  # noqa: E402
    build_contour_analysis,
    contour_signature,
)
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
