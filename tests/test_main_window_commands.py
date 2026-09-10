from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMessageBox

from magicborder import main_window as main_window_module
from magicborder.io_utils import load_project, save_project
from magicborder.main_window import (
    ANALYSIS_OUTDATED_STATUS_TEXT,
    CONTOUR_ANALYSIS_OUTDATED_TEXT,
    HISTOGRAM_DEFAULT_SIZES,
    HISTOGRAM_MANUAL_REFRESH_TEXT,
    WORKSPACE_DEFAULT_SIZES,
    MainWindow,
    _unique_destination_path,
)
from magicborder.models import (
    Annotation,
    Point,
    ProjectAngleMeasurement,
    ProjectDocument,
    ProjectImageMeasurements,
    ProjectImageRecord,
)

CONTOUR = [Point(4, 4), Point(36, 4), Point(36, 26), Point(4, 26)]


def _make_image(
    path: Path, size: tuple[int, int] = (40, 30), color=(120, 80, 40)
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture()
def dialogs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Перехватывает все модальные диалоги MainWindow."""
    record: dict[str, Any] = {
        "warning": [],
        "critical": [],
        "about": [],
        "question": [],
        "question_answer": QMessageBox.No,
        "input_text": ("", False),
        "existing_directory": "",
        "open_file": ("", ""),
        "open_files": ([], ""),
        "save_file": ("", ""),
        "message_box_choice": None,
    }

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        staticmethod(
            lambda _p, title, text, *a, **k: record["warning"].append((title, text))
        ),
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        staticmethod(
            lambda _p, title, text, *a, **k: record["critical"].append((title, text))
        ),
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "about",
        staticmethod(lambda _p, title, text: record["about"].append((title, text))),
    )

    def fake_question(_parent, title, text, *args, **kwargs):  # noqa: ARG001
        record["question"].append((title, text))
        return record["question_answer"]

    monkeypatch.setattr(
        main_window_module.QMessageBox, "question", staticmethod(fake_question)
    )
    monkeypatch.setattr(
        main_window_module.QInputDialog,
        "getText",
        staticmethod(lambda *a, **k: record["input_text"]),
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *a, **k: record["existing_directory"]),
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: record["open_file"]),
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *a, **k: record["open_files"]),
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: record["save_file"]),
    )

    def fake_exec(self) -> int:
        wanted = record["message_box_choice"]
        for button in self.buttons():
            if button.text().replace("&", "") == wanted:
                self._chosen_button = button
                return 0
        self._chosen_button = None
        return 0

    monkeypatch.setattr(main_window_module.QMessageBox, "exec_", fake_exec)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "clickedButton",
        lambda self: getattr(self, "_chosen_button", None),
    )
    return record


@pytest.fixture()
def project_window(qapp, tmp_path: Path, dialogs: dict[str, Any]):  # noqa: ARG001
    """Проект с одним изображением 40x30 на диске."""
    windows: list[MainWindow] = []

    def factory(
        *,
        image_names: tuple[str, ...] = ("leaf.png",),
        records: list[ProjectImageRecord] | None = None,
        root: Path | None = None,
        name: str = "project",
    ) -> MainWindow:
        project_root = root or (tmp_path / name)
        project_root.mkdir(parents=True, exist_ok=True)
        images: list[ProjectImageRecord] = []
        if records is None:
            for index, image_name in enumerate(image_names):
                _make_image(project_root / "images" / image_name)
                images.append(
                    ProjectImageRecord(
                        id=f"image-{index}",
                        relative_path=f"images/{image_name}",
                        display_name=image_name,
                        image_width=40,
                        image_height=30,
                    )
                )
        else:
            images = records

        project_path = project_root / f"{name}.json"
        save_project(project_path, ProjectDocument(name=name, images=images))

        window = MainWindow()
        window._set_project(project_path, load_project(project_path))
        windows.append(window)
        return window

    yield factory

    for window in windows:
        window.close()
        window.deleteLater()


class TestOpenProject:
    def test_cancelled_dialog_keeps_current_state(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window()
        original_path = window.project_path
        dialogs["open_file"] = ("", "")

        window.open_project()

        assert window.project_path == original_path
        assert dialogs["critical"] == []

    def test_broken_json_keeps_previous_project_open(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        original_path = window.project_path
        broken_path = tmp_path / "broken.json"
        broken_path.write_text("{ сломано", encoding="utf-8")
        dialogs["open_file"] = (str(broken_path), "")

        window.open_project()

        assert window.project_path == original_path
        assert window.project_document is not None
        assert dialogs["critical"][0][0] == "Ошибка открытия проекта"

    def test_failed_save_of_previous_project_aborts_opening(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        original_path = window.project_path
        other_path = tmp_path / "other.json"
        save_project(other_path, ProjectDocument(name="other", images=[]))
        dialogs["open_file"] = (str(other_path), "")
        monkeypatch.setattr(window, "_save_project_silently", lambda **kwargs: False)

        window.open_project()

        assert window.project_path == original_path

    def test_valid_project_is_opened(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        other_path = tmp_path / "другой.json"
        save_project(other_path, ProjectDocument(name="другой", images=[]))
        dialogs["open_file"] = (str(other_path), "")

        window.open_project()

        assert window.project_path == other_path.resolve()
        assert window.project_document is not None
        assert window.project_document.name == "другой"


class TestCloseProject:
    def test_without_project_is_noop(self, qapp, dialogs: dict[str, Any]) -> None:
        window = MainWindow()
        try:
            window.close_project()

            assert window.project_document is None
            assert dialogs["critical"] == []
        finally:
            window.close()
            window.deleteLater()

    def test_annotation_and_measurements_are_saved(self, project_window) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)
        window.canvas.set_angle_measurements(
            [[Point(5, 5), Point(5, 15), Point(15, 15)]]
        )
        project_path = window.project_path
        assert project_path is not None

        window.close_project()

        payload = json.loads(project_path.read_text(encoding="utf-8"))
        record = payload["images"][0]
        assert record["contour"]["annotation"] is not None
        assert len(record["contour"]["annotation"]["points"]) == 4
        assert len(record["measurements"]["angles"]) == 1

    def test_state_is_cleared(self, project_window) -> None:
        window = project_window()

        window.close_project()

        assert window.project_document is None
        assert window.project_path is None
        assert window.project_list.count() == 0
        assert window.canvas.has_image() is False
        assert window._current_project_image_id is None

    def test_failed_save_keeps_project_open(
        self,
        project_window,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        monkeypatch.setattr(window, "_save_project_silently", lambda **kwargs: False)

        window.close_project()

        assert window.project_document is not None


class TestNewProject:
    def test_non_empty_existing_directory_is_rejected(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        parent_dir = tmp_path / "родитель"
        occupied = parent_dir / "новый"
        occupied.mkdir(parents=True)
        (occupied / "занято.txt").write_text("данные", encoding="utf-8")
        dialogs["input_text"] = ("новый", True)
        dialogs["existing_directory"] = str(parent_dir)

        window.new_project()

        assert (
            "Папка уже существует",
            "Выберите другое имя проекта или пустую папку.",
        ) in dialogs["warning"]

    def test_blank_name_is_rejected(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window()
        dialogs["input_text"] = ("   ", True)

        window.new_project()

        assert dialogs["warning"][-1][0] == "Некорректное имя"

    def test_mkdir_error_is_reported(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        parent_dir = tmp_path / "родитель"
        parent_dir.mkdir()
        dialogs["input_text"] = ("новый", True)
        dialogs["existing_directory"] = str(parent_dir)

        original_mkdir = Path.mkdir

        def failing_mkdir(self, *args, **kwargs):
            if self.name == "новый":
                raise OSError("нет прав")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", failing_mkdir)

        window.new_project()

        assert dialogs["critical"][-1][0] == "Ошибка создания проекта"
        assert "нет прав" in dialogs["critical"][-1][1]

    def test_save_error_is_reported(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        parent_dir = tmp_path / "родитель"
        parent_dir.mkdir()
        dialogs["input_text"] = ("новый", True)
        dialogs["existing_directory"] = str(parent_dir)

        original_save = main_window_module.save_project

        def failing_save(path, document):
            # Сохранение текущего проекта должно пройти: падает только новый файл.
            if Path(path).is_relative_to(parent_dir):
                raise OSError("диск переполнен")
            return original_save(path, document)

        monkeypatch.setattr(main_window_module, "save_project", failing_save)

        window.new_project()

        assert dialogs["critical"][-1][0] == "Ошибка создания проекта"
        assert "диск переполнен" in dialogs["critical"][-1][1]

    def test_successful_creation(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        parent_dir = tmp_path / "родитель"
        parent_dir.mkdir()
        dialogs["input_text"] = ("виноград", True)
        dialogs["existing_directory"] = str(parent_dir)

        window.new_project()

        assert (parent_dir / "виноград" / "виноград.json").is_file()
        assert (parent_dir / "виноград" / "images").is_dir()
        assert window.project_document is not None
        assert window.project_document.name == "виноград"


class TestRenameProjectFile:
    def test_directory_collision_is_reported(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window(name="исходный")
        assert window.project_path is not None
        (window.project_path.parent.parent / "занятый").mkdir()

        assert window._rename_project_file("занятый") is False
        assert dialogs["warning"][-1][0] == "Папка уже существует"

    def test_os_error_restores_previous_state(
        self,
        project_window,
        dialogs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window(name="исходный")
        original_path = window.project_path
        assert original_path is not None

        def failing_rename(self, target):  # noqa: ARG001
            raise OSError("файл занят")

        monkeypatch.setattr(Path, "rename", failing_rename)

        assert window._rename_project_file("новый") is False
        assert window.project_path == original_path
        assert window.project_document is not None
        assert window.project_document.name == "исходный"
        assert dialogs["critical"][-1][0] == "Ошибка переименования проекта"

    def test_file_collision_inside_the_same_directory_is_reported(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        # Папка называется "проект", а файл — "старый.json": переименование
        # файла не трогает папку, поэтому срабатывает проверка коллизии файлов.
        project_root = tmp_path / "проект"
        window = project_window(name="старый", root=project_root)
        (project_root / "проект.json").write_text("{}", encoding="utf-8")

        assert window._rename_project_file("проект") is False
        assert dialogs["warning"][-1][0] == "Файл уже существует"
        assert (project_root / "старый.json").is_file()

    def test_same_name_only_updates_document(self, project_window) -> None:
        window = project_window(name="исходный")
        original_path = window.project_path

        assert window._rename_project_file("исходный") is True
        assert window.project_path == original_path

    def test_successful_rename_moves_file_and_directory(self, project_window) -> None:
        window = project_window(name="исходный")
        assert window.project_path is not None
        root = window.project_path.parent.parent

        assert window._rename_project_file("переименованный") is True

        assert (root / "переименованный" / "переименованный.json").is_file()
        assert window.project_document is not None
        assert window.project_document.name == "переименованный"


class TestAddImagesToProject:
    def test_without_project_warns(self, qapp, dialogs: dict[str, Any]) -> None:
        window = MainWindow()
        try:
            window.add_images_to_project()

            assert dialogs["warning"][-1][0] == "Нет проекта"
        finally:
            window.close()
            window.deleteLater()

    def test_cancelled_dialog_adds_nothing(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window()
        dialogs["open_files"] = ([], "")

        window.add_images_to_project()

        assert window.project_document is not None
        assert len(window.project_document.images) == 1

    def test_images_are_copied_into_project(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        source = _make_image(tmp_path / "источник" / "новый.png", size=(20, 15))
        dialogs["open_files"] = ([str(source)], "")
        assert window.project_path is not None
        image_dir = window.project_path.parent / "images"

        window.add_images_to_project()

        assert (image_dir / "новый.png").is_file()
        assert window.project_document is not None
        added = window.project_document.images[-1]
        assert added.relative_path == "images/новый.png"
        assert (added.image_width, added.image_height) == (20, 15)

    def test_name_collision_gets_a_suffix(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        source = _make_image(tmp_path / "источник" / "leaf.png", size=(20, 15))
        dialogs["open_files"] = ([str(source)], "")

        window.add_images_to_project()

        assert window.project_document is not None
        assert window.project_document.images[-1].relative_path == "images/leaf_1.png"

    def test_unsupported_file_is_reported_but_others_are_added(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        good = _make_image(tmp_path / "источник" / "хороший.png")
        bad = tmp_path / "источник" / "плохой.gif"
        bad.write_bytes(b"not an image")
        dialogs["open_files"] = ([str(bad), str(good)], "")

        window.add_images_to_project()

        assert window.project_document is not None
        assert [record.display_name for record in window.project_document.images] == [
            "leaf.png",
            "хороший.png",
        ]
        assert dialogs["warning"][-1][0] == "Не все изображения добавлены"
        assert "плохой.gif" in dialogs["warning"][-1][1]

    def test_metadata_is_filled(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "источник" / "снимок.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        exif = Image.Exif()
        exif[36867] = "2021:01:02 03:04:05"
        Image.new("RGB", (20, 15), (10, 10, 10)).save(source, exif=exif)

        window = project_window()
        dialogs["open_files"] = ([str(source)], "")

        window.add_images_to_project()

        assert window.project_document is not None
        metadata = window.project_document.images[-1].metadata
        assert metadata["captured_at"] == "2021-01-02T03:04:05"
        assert metadata["added_at"]

    def test_selection_moves_to_first_added_image(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        first = _make_image(tmp_path / "источник" / "первый.png")
        second = _make_image(tmp_path / "источник" / "второй.png")
        dialogs["open_files"] = ([str(first), str(second)], "")

        window.add_images_to_project()

        assert window.project_document is not None
        added_first = window.project_document.images[1]
        assert window._selected_project_image_id() == added_first.id
        assert added_first.display_name == "первый.png"

    def test_image_dir_error_is_reported(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        source = _make_image(tmp_path / "источник" / "новый.png")
        dialogs["open_files"] = ([str(source)], "")

        original_mkdir = Path.mkdir

        def failing_mkdir(self, *args, **kwargs):
            if self.name == "images":
                raise OSError("нет прав")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", failing_mkdir)

        window.add_images_to_project()

        assert dialogs["critical"][-1][0] == "Ошибка добавления"
        assert window.project_document is not None
        assert len(window.project_document.images) == 1


class TestUniqueDestinationPath:
    def test_free_name_is_used_as_is(self, tmp_path: Path) -> None:
        assert _unique_destination_path(tmp_path, "leaf.png") == tmp_path / "leaf.png"

    def test_taken_names_get_incremental_suffixes(self, tmp_path: Path) -> None:
        (tmp_path / "leaf.png").write_bytes(b"a")
        (tmp_path / "leaf_1.png").write_bytes(b"a")

        assert _unique_destination_path(tmp_path, "leaf.png") == tmp_path / "leaf_2.png"

    def test_name_without_stem_falls_back(self, tmp_path: Path) -> None:
        assert _unique_destination_path(tmp_path, ".png").name == ".png"


class TestRemoveSelectedProjectImage:
    def test_without_selection_warns(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window()
        window.project_list.setCurrentRow(-1)
        window.project_list.clearSelection()

        window.remove_selected_project_image()

        assert dialogs["warning"][-1][0] == "Изображение не выбрано"

    def test_cancel_changes_nothing(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window()
        dialogs["message_box_choice"] = "Отмена"
        assert window.project_path is not None
        image_path = window.project_path.parent / "images" / "leaf.png"

        window.remove_selected_project_image()

        assert window.project_document is not None
        assert len(window.project_document.images) == 1
        assert image_path.is_file()

    def test_remove_only_keeps_file_on_disk(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window()
        dialogs["message_box_choice"] = "Только убрать из проекта"
        assert window.project_path is not None
        image_path = window.project_path.parent / "images" / "leaf.png"

        window.remove_selected_project_image()

        assert window.project_document is not None
        assert window.project_document.images == []
        assert image_path.is_file()

    def test_delete_file_removes_it_from_disk(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window()
        dialogs["message_box_choice"] = "Удалить файл"
        assert window.project_path is not None
        image_path = window.project_path.parent / "images" / "leaf.png"

        window.remove_selected_project_image()

        assert window.project_document is not None
        assert window.project_document.images == []
        assert image_path.exists() is False

    def test_unlink_error_keeps_record(
        self,
        project_window,
        dialogs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        dialogs["message_box_choice"] = "Удалить файл"

        def failing_unlink(self, *args, **kwargs):  # noqa: ARG001
            raise OSError("файл занят")

        monkeypatch.setattr(Path, "unlink", failing_unlink)

        window.remove_selected_project_image()

        assert dialogs["critical"][-1][0] == "Ошибка удаления"
        assert window.project_document is not None
        assert len(window.project_document.images) == 1

    def test_selection_moves_to_neighbour_row(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window(image_names=("первый.png", "второй.png", "третий.png"))
        dialogs["message_box_choice"] = "Только убрать из проекта"
        window.project_list.setCurrentRow(1)

        window.remove_selected_project_image()

        assert window.project_document is not None
        assert [record.display_name for record in window.project_document.images] == [
            "первый.png",
            "третий.png",
        ]
        assert window.project_list.currentRow() == 1

    def test_removing_the_last_image_clears_the_canvas(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window()
        dialogs["message_box_choice"] = "Только убрать из проекта"

        window.remove_selected_project_image()

        assert window.canvas.has_image() is False
        assert window.project_list.count() == 0


class TestDetectContour:
    def test_without_image_warns(self, qapp, dialogs: dict[str, Any]) -> None:
        window = MainWindow()
        try:
            window.detect_contour()

            assert dialogs["warning"][-1][0] == "Нет изображения"
        finally:
            window.close()
            window.deleteLater()

    def test_successful_detection_sets_contour(
        self,
        project_window,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        monkeypatch.setattr(
            main_window_module,
            "detect_leaf_contour",
            lambda _rgb: [Point(2, 2), Point(30, 2), Point(30, 20), Point(2, 20)],
        )

        window.detect_contour()

        assert len(window.canvas.contour_points()) == 4
        assert "4" in window.statusBar().currentMessage()
        assert QApplication.overrideCursor() is None

    def test_detector_error_shows_dialog(
        self,
        project_window,
        dialogs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()

        def failing_detector(_rgb):
            raise ValueError("Не удалось определить контур листа.")

        monkeypatch.setattr(main_window_module, "detect_leaf_contour", failing_detector)

        window.detect_contour()

        assert dialogs["critical"][-1][0] == "Не удалось определить контур"
        assert window.canvas.has_contour() is False

    def test_override_cursor_is_balanced_on_error(
        self,
        project_window,
        dialogs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        monkeypatch.setattr(
            main_window_module,
            "detect_leaf_contour",
            lambda _rgb: (_ for _ in ()).throw(ValueError("нет контура")),
        )

        window.detect_contour()

        # finally-блок выполняется даже при `return` внутри except.
        assert QApplication.overrideCursor() is None

    def test_capture_modes_are_cancelled(
        self,
        project_window,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        window.canvas.begin_angle_measurement()
        monkeypatch.setattr(
            main_window_module,
            "detect_leaf_contour",
            lambda _rgb: [Point(2, 2), Point(30, 2), Point(30, 20)],
        )

        window.detect_contour()

        assert window.canvas._angle_capture_active is False


class TestSaveAnnotationFile:
    def test_without_contour_warns(
        self, project_window, dialogs: dict[str, Any]
    ) -> None:
        window = project_window()

        window.save_annotation_file()

        assert dialogs["warning"][-1][0] == "Нет контура"

    def test_cancelled_dialog_writes_nothing(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)
        dialogs["save_file"] = ("", "")

        window.save_annotation_file()

        assert window._current_annotation_path is None

    def test_json_suffix_is_appended(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)
        target = tmp_path / "аннотация"
        dialogs["save_file"] = (str(target), "")

        window.save_annotation_file()

        assert (tmp_path / "аннотация.json").is_file()
        assert window._current_annotation_path == tmp_path / "аннотация.json"

    def test_os_error_is_reported(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)
        dialogs["save_file"] = (str(tmp_path / "аннотация.json"), "")

        def failing_save(path, annotation):  # noqa: ARG001
            raise OSError("нет места")

        monkeypatch.setattr(main_window_module, "save_annotation", failing_save)

        window.save_annotation_file()

        assert dialogs["critical"][-1][0] == "Ошибка сохранения"
        assert window._current_annotation_path is None

    def test_remembered_path_is_offered_next_time(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)
        annotation_path = tmp_path / "аннотация.json"
        dialogs["save_file"] = (str(annotation_path), "")

        window.save_annotation_file()

        offered: list[str] = []
        monkeypatch.setattr(
            main_window_module.QFileDialog,
            "getSaveFileName",
            staticmethod(
                lambda _p, _t, initial, _f: offered.append(initial) or ("", "")
            ),
        )

        window.save_annotation_file()

        assert offered == [str(annotation_path)]


class TestOpenAnnotationFile:
    def test_without_selected_image_warns(self, qapp, dialogs: dict[str, Any]) -> None:
        window = MainWindow()
        try:
            window.open_annotation_file()

            assert dialogs["warning"][-1][0] == "Нет выбранного изображения"
        finally:
            window.close()
            window.deleteLater()

    def test_cancelled_dialog_does_nothing(
        self,
        project_window,
        dialogs: dict[str, Any],
    ) -> None:
        window = project_window()
        dialogs["open_file"] = ("", "")

        window.open_annotation_file()

        assert window.canvas.has_contour() is False

    def test_size_mismatch_is_reported(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        annotation_path = tmp_path / "аннотация.json"
        annotation_path.write_text(
            json.dumps(
                Annotation(
                    image_path="images/leaf.png",
                    image_width=100,
                    image_height=80,
                    points=CONTOUR,
                ).to_dict(),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        dialogs["open_file"] = (str(annotation_path), "")

        window.open_annotation_file()

        assert dialogs["warning"][-1][0] == "Аннотация не подходит"
        assert window.canvas.has_contour() is False

    def test_contour_with_less_than_three_points_is_reported(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        annotation_path = tmp_path / "аннотация.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "image_path": "images/leaf.png",
                    "image_size": {"width": 40, "height": 30},
                    "points": [{"x": 1, "y": 1}, {"x": 2, "y": 2}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        dialogs["open_file"] = (str(annotation_path), "")

        window.open_annotation_file()

        assert dialogs["critical"][-1][0] == "Ошибка загрузки"
        assert "минимум 3 точки" in dialogs["critical"][-1][1]

    def test_matching_annotation_is_loaded(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        annotation_path = tmp_path / "аннотация.json"
        annotation_path.write_text(
            json.dumps(
                Annotation(
                    image_path="images/leaf.png",
                    image_width=40,
                    image_height=30,
                    points=CONTOUR,
                    line_color="#123456",
                ).to_dict(),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        dialogs["open_file"] = (str(annotation_path), "")

        window.open_annotation_file()

        assert len(window.canvas.contour_points()) == 4
        assert window.canvas.contour_line_color() == "#123456"
        assert window._current_annotation_path == annotation_path

    @pytest.mark.parametrize("image_reference", ["images/leaf.png", "images\\leaf.png"])
    def test_prepare_image_accepts_relative_and_windows_paths(
        self,
        project_window,
        tmp_path: Path,
        image_reference: str,
    ) -> None:
        window = project_window()
        annotation = Annotation(
            image_path=image_reference,
            image_width=40,
            image_height=30,
            points=CONTOUR,
        )

        assert (
            window._prepare_image_for_annotation(annotation, tmp_path / "а.json")
            is True
        )

    def test_prepare_image_rejects_other_sizes(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        window = project_window()
        annotation = Annotation(
            image_path="images/leaf.png",
            image_width=400,
            image_height=300,
            points=CONTOUR,
        )

        assert (
            window._prepare_image_for_annotation(annotation, tmp_path / "а.json")
            is False
        )
        assert dialogs["warning"][-1][0] == "Аннотация не подходит"


class TestMiscCommands:
    def test_export_project_csv_is_an_alias(
        self,
        project_window,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        calls: list[int] = []
        monkeypatch.setattr(window, "export_project_excel", lambda: calls.append(1))

        window.export_project_csv()

        assert calls == [1]

    def test_csv_and_excel_actions_are_the_same_object(self, project_window) -> None:
        window = project_window()

        assert window.export_project_csv_action is window.export_project_excel_action

    def test_show_about_dialog(self, project_window, dialogs: dict[str, Any]) -> None:
        window = project_window()

        window.show_about_dialog()

        assert dialogs["about"][-1][0] == "О программе"
        assert "MagicBorder" in dialogs["about"][-1][1]

    def test_restore_default_view_resets_splitters(self, project_window) -> None:
        window = project_window()
        window.workspace_splitter.setSizes([10, 10, 10])
        window.histogram_splitter.setSizes([5, 5, 5, 5, 5])

        window.restore_default_view()

        assert len(window.workspace_splitter.sizes()) == len(WORKSPACE_DEFAULT_SIZES)
        assert len(window.histogram_splitter.sizes()) == len(HISTOGRAM_DEFAULT_SIZES)
        assert window.statusBar().currentMessage() == "Вид по умолчанию восстановлен."

    def test_start_scale_calibration_cancels_other_modes(self, project_window) -> None:
        window = project_window()
        window.canvas.begin_angle_measurement()

        window.start_scale_calibration()

        assert window.canvas._calibration_capture_active is True
        assert window.canvas._angle_capture_active is False

    def test_start_angle_measurement_cancels_other_modes(self, project_window) -> None:
        window = project_window()
        window.canvas.begin_calibration()

        window.start_angle_measurement()

        assert window.canvas._angle_capture_active is True
        assert window.canvas._calibration_capture_active is False

    def test_start_segment_measurement_cancels_other_modes(
        self, project_window
    ) -> None:
        window = project_window()
        window.canvas.begin_angle_measurement()

        window.start_segment_measurement()

        assert window.canvas._segment_capture_active is True
        assert window.canvas._angle_capture_active is False

    @pytest.mark.parametrize(
        "command",
        [
            "start_scale_calibration",
            "start_angle_measurement",
            "start_segment_measurement",
        ],
    )
    def test_measurement_commands_require_an_image(
        self,
        qapp,
        dialogs: dict[str, Any],
        command: str,
    ) -> None:
        window = MainWindow()
        try:
            getattr(window, command)()

            assert dialogs["warning"][-1][0] == "Нет изображения"
        finally:
            window.close()
            window.deleteLater()

    def test_metadata_notes_are_stored(self, project_window) -> None:
        window = project_window()

        window.metadata_notes.setPlainText("первая строка\nвторая строка")
        window._handle_metadata_notes_changed()

        record = window._selected_project_image()
        assert record is not None
        assert record.metadata["notes"] == "первая строка\nвторая строка"

    def test_metadata_notes_without_selection_are_ignored(
        self,
        qapp,
        dialogs: dict[str, Any],
    ) -> None:
        window = MainWindow()
        try:
            window.metadata_notes.setPlainText("текст")

            window._handle_metadata_notes_changed()  # не должно падать
        finally:
            window.close()
            window.deleteLater()


class TestCloseEvent:
    class _Event:
        def __init__(self) -> None:
            self.accepted: bool | None = None

        def accept(self) -> None:
            self.accepted = True

        def ignore(self) -> None:
            self.accepted = False

    def test_successful_save_accepts_close(self, project_window) -> None:
        window = project_window()
        event = self._Event()

        window.closeEvent(event)

        assert event.accepted is True

    def test_failed_save_and_yes_accepts_close(
        self,
        project_window,
        dialogs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        monkeypatch.setattr(window, "_save_project_silently", lambda **kwargs: False)
        dialogs["question_answer"] = QMessageBox.Yes
        event = self._Event()

        window.closeEvent(event)

        assert dialogs["question"][-1][0] == "Закрыть без сохранения?"
        assert event.accepted is True

    def test_failed_save_and_no_ignores_close(
        self,
        project_window,
        dialogs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window()
        monkeypatch.setattr(window, "_save_project_silently", lambda **kwargs: False)
        dialogs["question_answer"] = QMessageBox.No
        event = self._Event()

        window.closeEvent(event)

        assert event.accepted is False


class TestHistograms:
    HISTOGRAM_PANELS = (
        "rgb_histogram_panel",
        "lab_histogram_panel",
        "hsv_histogram_panel",
        "yuv_histogram_panel",
        "lms_histogram_panel",
    )

    def _panels(self, window: MainWindow) -> list[Any]:
        return [getattr(window, name) for name in self.HISTOGRAM_PANELS]

    def test_contour_fills_every_panel(self, project_window) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)

        window._refresh_histograms()

        for panel in self._panels(window):
            assert panel.canvas.has_plot_data() is True
            assert panel._save_button.isEnabled() is True

    def test_panels_are_cleared_without_contour(self, project_window) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)
        window._refresh_histograms()

        window.canvas.clear_contour()
        window._invalidate_current_contour_analysis()
        window._refresh_histograms()

        for panel in self._panels(window):
            assert panel.canvas.has_plot_data() is False
            assert (
                panel.canvas._empty_message
                == "Создайте основной контур, чтобы увидеть гистограмму."
            )

    def test_message_without_image(self, qapp, dialogs: dict[str, Any]) -> None:
        window = MainWindow()
        try:
            window._refresh_histograms()

            assert window.rgb_histogram_panel.canvas._empty_message == (
                "Откройте изображение и создайте контур, чтобы увидеть гистограмму."
            )
        finally:
            window.close()
            window.deleteLater()

    def test_analysis_failure_shows_message(self, project_window) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)
        window._pending_contour_analysis_request_id = 42

        window._handle_contour_analysis_failed(42, "Ошибка расчёта гистограмм.")

        for panel in self._panels(window):
            assert panel.canvas.has_plot_data() is False
            assert panel.canvas._empty_message == "Ошибка расчёта гистограмм."

    def test_analysis_failure_with_empty_message_uses_fallback(
        self, project_window
    ) -> None:
        window = project_window()
        window._pending_contour_analysis_request_id = 7

        window._handle_contour_analysis_failed(7, "")

        assert window.rgb_histogram_panel.canvas._empty_message == (
            "Не удалось рассчитать гистограммы."
        )

    def test_stale_analysis_failure_is_ignored(self, project_window) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)
        window._refresh_histograms()
        window._pending_contour_analysis_request_id = 5

        window._handle_contour_analysis_failed(4, "устаревшая ошибка")

        assert window.rgb_histogram_panel.canvas.has_plot_data() is True

    def test_analysis_cache_is_reused(self, project_window) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)

        window.refresh_analysis()

        first = window._contour_analysis_cache
        second = window._ensure_current_contour_analysis(defer_large_async=False)

        assert first is not None
        assert second is first


class TestManualAnalysisRefresh:
    HISTOGRAM_PANELS = TestHistograms.HISTOGRAM_PANELS

    def _panels(self, window: MainWindow) -> list[Any]:
        return [getattr(window, name) for name in self.HISTOGRAM_PANELS]

    def test_loading_an_image_does_not_recalculate(self, project_window) -> None:
        window = project_window()

        assert window._contour_analysis_cache is None
        for panel in self._panels(window):
            assert panel.canvas.has_plot_data() is False
            assert panel.canvas._empty_message == HISTOGRAM_MANUAL_REFRESH_TEXT

    def test_new_contour_only_marks_the_analysis_as_outdated(
        self, project_window
    ) -> None:
        window = project_window()

        window.canvas.set_contour(CONTOUR)

        assert window._contour_analysis_cache is None
        assert window._is_current_contour_analysis_outdated() is True
        assert window.analysis_status_label.text() == ANALYSIS_OUTDATED_STATUS_TEXT
        assert window.property_contour_pixels.text() == CONTOUR_ANALYSIS_OUTDATED_TEXT

    def test_refresh_button_recalculates_every_panel(self, project_window) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)

        window.refresh_analysis_button.click()

        assert window._contour_analysis_cache is not None
        for panel in self._panels(window):
            assert panel.canvas.has_plot_data() is True
        assert window.analysis_status_label.text() == ""
        assert window.property_contour_pixels.text() != CONTOUR_ANALYSIS_OUTDATED_TEXT
        assert int(window.property_contour_pixels.text()) > 0

    def test_refresh_action_recalculates_project_summary(self, project_window) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)

        assert window.project_mean_red.text() == "-"

        window.refresh_analysis_action.trigger()

        assert window.project_mean_red.text() != "-"

    def test_moving_a_node_marks_the_analysis_as_outdated(self, project_window) -> None:
        window = project_window()
        window.canvas.set_contour(CONTOUR)
        window.refresh_analysis()

        assert window.analysis_status_label.text() == ""

        window.canvas.set_contour([Point(2, 2), Point(30, 2), Point(30, 20)])

        assert window.analysis_status_label.text() == ANALYSIS_OUTDATED_STATUS_TEXT
        assert window.property_contour_pixels.text() == CONTOUR_ANALYSIS_OUTDATED_TEXT

    def test_refresh_is_disabled_without_a_project(
        self, qapp, dialogs: dict[str, Any]
    ) -> None:
        window = MainWindow()
        try:
            assert window.refresh_analysis_action.isEnabled() is False
            assert window.refresh_analysis_button.isEnabled() is False
        finally:
            window.close()
            window.deleteLater()


class TestHistogramFileNames:
    def test_default_histogram_file_name_follows_image(self, project_window) -> None:
        window = project_window()

        name = window._default_histogram_file_name("RGB")

        assert name.endswith("leaf_RGB_histogram.png")

    def test_default_histogram_file_name_without_image(
        self,
        qapp,
        dialogs: dict[str, Any],
    ) -> None:
        window = MainWindow()
        try:
            assert window._default_histogram_file_name("RGB") == "RGB_histogram.png"
        finally:
            window.close()
            window.deleteLater()


def test_measurements_are_restored_from_project(project_window) -> None:
    record = ProjectImageRecord(
        id="leaf",
        relative_path="images/leaf.png",
        display_name="leaf.png",
        image_width=40,
        image_height=30,
        measurements=ProjectImageMeasurements(
            angles=[
                ProjectAngleMeasurement(
                    id="angle-a",
                    first=Point(2, 10),
                    vertex=Point(2, 2),
                    second=Point(10, 2),
                )
            ]
        ),
    )
    window = project_window(records=[record])
    assert window.project_path is not None
    _make_image(window.project_path.parent / "images" / "leaf.png")
    window._load_project_image(window.project_document.images[0])

    assert window.canvas.has_angle_measurements() is True
