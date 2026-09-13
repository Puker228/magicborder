from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Должно быть выставлено до первого импорта PyQt5 в любом тестовом модуле.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from magicborder.canvas import ImageCanvas  # noqa: E402
from magicborder.io_utils import (  # noqa: E402
    load_project,
    loaded_image_from_rgb_array,
    save_project,
)
from magicborder.models import ProjectDocument, ProjectImageRecord  # noqa: E402


@pytest.fixture(autouse=True)
def _no_automatic_crop_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Автопоказ модального диалога обрезки заблокировал бы тесты.

    Тесты автопоказа вызывают MainWindow._maybe_prompt_crop напрямую.
    """
    from magicborder.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_schedule_crop_prompt", lambda *_args: None)


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    """Единственный QApplication на весь прогон тестов."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture()
def canvas(qapp: QApplication) -> Iterator[ImageCanvas]:  # noqa: ARG001
    """Пустой канвас без загруженного изображения."""
    widget = ImageCanvas()
    widget.resize(320, 240)
    yield widget
    widget.deleteLater()


@pytest.fixture()
def canvas_with_image(canvas: ImageCanvas) -> ImageCanvas:
    """Канвас с загруженным изображением 100x80."""
    rgb_array = np.zeros((80, 100, 3), dtype=np.uint8)
    canvas.set_loaded_image(loaded_image_from_rgb_array(Path("leaf.png"), rgb_array))
    return canvas


@pytest.fixture()
def make_project_window(
    qapp: QApplication,  # noqa: ARG001
    tmp_path: Path,
) -> Iterator[Callable[..., object]]:
    """Фабрика MainWindow с сохранённым на диск проектом."""
    from magicborder.main_window import MainWindow

    created: list[object] = []

    def factory(
        images: list[ProjectImageRecord] | None = None,
        *,
        name: str = "project",
        root: Path | None = None,
    ):
        project_root = root or tmp_path
        project_root.mkdir(parents=True, exist_ok=True)
        document = ProjectDocument(name=name, images=list(images or []))
        project_path = project_root / f"{name}.json"
        save_project(project_path, document)

        window = MainWindow()
        window._set_project(project_path, load_project(project_path))
        created.append(window)
        return window

    yield factory

    for window in created:
        window.close()
        window.deleteLater()
