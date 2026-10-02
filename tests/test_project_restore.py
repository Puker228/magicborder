from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from PyQt5.QtWidgets import QApplication

from magicborder import main_window as main_window_module
from magicborder.io_utils import load_project, save_project
from magicborder.main_window import MainWindow
from magicborder.models import (
    PROJECT_FORMAT_VERSION,
    Point,
    ProjectDocument,
    ProjectImageRecord,
)

CONTOUR = [Point(4, 4), Point(36, 4), Point(36, 26), Point(4, 26)]
EXCLUSION = [Point(10, 10), Point(20, 10), Point(20, 20), Point(10, 20)]


def _make_image(path: Path, size: tuple[int, int] = (40, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 80, 40)).save(path)
    return path


@pytest.fixture()
def dialogs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    record: dict[str, Any] = {"warning": [], "critical": [], "open_file": ("", "")}
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        staticmethod(lambda _p, title, text, *a, **k: record["warning"].append(title)),
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        staticmethod(
            lambda _p, title, text, *a, **k: record["critical"].append((title, text))
        ),
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: record["open_file"]),
    )
    return record


def _legacy_project_payload() -> dict[str, Any]:
    """Проект формата версии 1: поля записи на верхнем уровне, без групп."""
    return {
        "version": 1,
        "name": "old",
        "images_dir": "images",
        "images": [
            {
                "id": "legacy-id",
                "path": "images/leaf.png",
                "display_name": "leaf.png",
                "image_size": {"width": 40, "height": 30},
                "annotation": {
                    "version": 1,
                    "image_path": "images/leaf.png",
                    "image_size": {"width": 40, "height": 30},
                    "closed": True,
                    "points": [point.to_dict() for point in CONTOUR],
                },
                "metadata": {
                    "added_at": "2026-05-01T10:00:00",
                    "diagnosis": "здоров",
                    "notes": "старая заметка",
                    "sample_id": "S-1",
                },
            }
        ],
    }


class TestLegacyProjectFormat:
    def test_legacy_records_are_migrated_with_contour_and_metadata(
        self, tmp_path: Path
    ) -> None:
        project_path = tmp_path / "old.json"
        project_path.write_text(
            json.dumps(_legacy_project_payload(), ensure_ascii=False),
            encoding="utf-8",
        )

        document = load_project(project_path)

        assert [record.id for record in document.images] == ["legacy-id"]
        record = document.images[0]
        assert record.relative_path == "images/leaf.png"
        assert (record.image_width, record.image_height) == (40, 30)
        assert record.annotation is not None
        assert record.annotation.points == CONTOUR
        assert record.metadata["added_at"] == "2026-05-01T10:00:00"
        assert record.metadata["diagnosis"] == "здоров"
        assert record.metadata["notes"] == "старая заметка"
        assert record.metadata["sample_id"] == "S-1"

    def test_migrated_project_is_saved_in_current_format(self, tmp_path: Path) -> None:
        project_path = tmp_path / "old.json"
        project_path.write_text(json.dumps(_legacy_project_payload()), "utf-8")

        save_project(project_path, load_project(project_path))
        payload = json.loads(project_path.read_text(encoding="utf-8"))

        assert payload["version"] == PROJECT_FORMAT_VERSION
        assert payload["images"][0]["file"]["id"] == "legacy-id"
        assert payload["images"][0]["contour"]["annotation"]["points"]
        reloaded = load_project(project_path)
        assert [record.id for record in reloaded.images] == ["legacy-id"]

    def test_legacy_id_falls_back_to_sample_id_and_path(self) -> None:
        document = ProjectDocument.from_dict(
            {
                "images": [
                    {"path": "images/a.png", "metadata": {"sample_id": "S-7"}},
                    {"relative_path": "images\\b.png"},
                ]
            }
        )

        assert [record.id for record in document.images] == ["S-7", "images/b.png"]
        assert document.images[1].display_name == "b.png"


class TestUnreadableRecords:
    def test_unreadable_records_are_kept_in_the_saved_file(
        self, tmp_path: Path
    ) -> None:
        project_path = tmp_path / "project.json"
        project_path.write_text(
            json.dumps(
                {
                    "images": [
                        {"file": {"id": "a", "path": "images/a.png"}},
                        {"file": {"id": "", "path": ""}},
                        "мусор",
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        document = load_project(project_path)
        save_project(project_path, document)
        payload = json.loads(project_path.read_text(encoding="utf-8"))

        assert [record.id for record in document.images] == ["a"]
        assert len(document.unreadable_images) == 2
        assert payload["images"][1:] == [{"file": {"id": "", "path": ""}}, "мусор"]


class TestNonProjectJson:
    def test_annotation_file_is_rejected_as_project(self, tmp_path: Path) -> None:
        annotation_path = tmp_path / "leaf.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "image_size": {"width": 40, "height": 30},
                    "points": [point.to_dict() for point in CONTOUR],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="не является проектом"):
            load_project(annotation_path)

    def test_open_project_does_not_overwrite_annotation_file(
        self,
        qapp: QApplication,  # noqa: ARG002
        tmp_path: Path,
        dialogs: dict[str, Any],
    ) -> None:
        annotation_path = tmp_path / "leaf.json"
        original = json.dumps({"points": [point.to_dict() for point in CONTOUR]})
        annotation_path.write_text(original, encoding="utf-8")
        dialogs["open_file"] = (str(annotation_path), "")

        window = MainWindow()
        try:
            window.open_project()
            QApplication.processEvents()

            assert window.project_document is None
            assert dialogs["critical"]
            assert annotation_path.read_text(encoding="utf-8") == original
        finally:
            window.close()
            window.deleteLater()


class TestProjectRestoreAfterRestart:
    def test_images_contour_and_exclusions_survive_restart(
        self,
        qapp: QApplication,  # noqa: ARG002
        tmp_path: Path,
        dialogs: dict[str, Any],
    ) -> None:
        project_root = tmp_path / "project"
        _make_image(project_root / "images" / "leaf.png")
        project_path = project_root / "project.json"
        save_project(project_path, ProjectDocument(name="project", images=[]))

        first = MainWindow()
        try:
            first._set_project(project_path, load_project(project_path))
            document = first.project_document
            assert document is not None
            document.images.append(
                ProjectImageRecord(
                    id="image-0",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    image_width=40,
                    image_height=30,
                )
            )
            first._refresh_project_list()
            first._select_project_image("image-0")
            first.canvas.set_contour(CONTOUR)
            first.canvas.add_exclusion(EXCLUSION)
            first.save_project_file()
        finally:
            first.close()
            first.deleteLater()

        dialogs["open_file"] = (str(project_path), "")
        second = MainWindow()
        try:
            second.open_project()

            assert second.project_document is not None
            assert [record.id for record in second.project_document.images] == [
                "image-0"
            ]
            assert second.canvas.has_image()
            assert second.canvas.contour_points() == CONTOUR
            assert second.canvas.exclusion_polygons() == [EXCLUSION]
            assert dialogs["warning"] == []
        finally:
            second.close()
            second.deleteLater()

    def test_unreadable_records_are_reported_on_open(
        self,
        qapp: QApplication,  # noqa: ARG002
        tmp_path: Path,
        dialogs: dict[str, Any],
    ) -> None:
        project_path = tmp_path / "project.json"
        project_path.write_text(json.dumps({"images": ["мусор"]}), "utf-8")
        dialogs["open_file"] = (str(project_path), "")

        window = MainWindow()
        try:
            window.open_project()

            assert window.project_document is not None
            assert dialogs["warning"] == ["Не все изображения прочитаны"]
        finally:
            window.close()
            window.deleteLater()
