from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from PIL import Image
from PyQt5.QtWidgets import QApplication

from magicborder import main_window as main_window_module
from magicborder.io_utils import load_project, save_project
from magicborder.main_window import MainWindow
from magicborder.models import (
    Annotation,
    ImageCalibration,
    Point,
    ProjectAngleMeasurement,
    ProjectDocument,
    ProjectImageMeasurements,
    ProjectImageRecord,
)

CONTOUR = [Point(0, 0), Point(40, 0), Point(40, 30), Point(0, 30)]
EXCLUSION = [Point(10, 10), Point(20, 10), Point(20, 20), Point(10, 20)]
_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as workbook:
        root = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    table = [
        [
            "".join(node.text or "" for node in cell.iterfind("s:is/s:t", _NS))
            for cell in row.findall("s:c", _NS)
        ]
        for row in root.findall("s:sheetData/s:row", _NS)
    ]
    headers = table[0]
    return [dict(zip(headers, row, strict=True)) for row in table[1:]]


def _annotation(*, exclusions: list[list[Point]] | None = None) -> Annotation:
    return Annotation(
        image_path="",
        image_width=40,
        image_height=30,
        points=list(CONTOUR),
        exclusions=exclusions or [],
    )


@pytest.fixture()
def dialogs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    record: dict[str, Any] = {"warning": [], "critical": [], "save_file": ("", "")}
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        staticmethod(lambda _p, title, *a, **k: record["warning"].append(title)),
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        staticmethod(lambda _p, title, *a, **k: record["critical"].append(title)),
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: record["save_file"]),
    )
    return record


@pytest.fixture()
def project_window(
    qapp: QApplication,  # noqa: ARG001
    tmp_path: Path,
    dialogs: dict[str, Any],  # noqa: ARG001
):
    windows: list[MainWindow] = []

    def factory(records: list[ProjectImageRecord]) -> MainWindow:
        root = tmp_path / "project"
        (root / "images").mkdir(parents=True, exist_ok=True)
        for record in records:
            if record.display_name != "missing.png":
                Image.new("RGB", (40, 30), (0, 200, 0)).save(
                    root / record.relative_path
                )
        project_path = root / "project.json"
        save_project(project_path, ProjectDocument(name="project", images=records))
        window = MainWindow()
        window._set_project(project_path, load_project(project_path))
        windows.append(window)
        return window

    yield factory
    for window in windows:
        window.close()
        window.deleteLater()


def _record(record_id: str, name: str, **kwargs: Any) -> ProjectImageRecord:
    return ProjectImageRecord(
        id=record_id,
        relative_path=f"images/{name}",
        display_name=name,
        image_width=40,
        image_height=30,
        **kwargs,
    )


class TestExportAllImagesExcel:
    def test_one_row_per_photo_with_fresh_data(
        self, project_window, dialogs: dict[str, Any], tmp_path: Path
    ) -> None:
        window = project_window(
            [
                _record("a", "a.png", annotation=_annotation()),
                _record(
                    "b",
                    "b.png",
                    annotation=_annotation(exclusions=[list(EXCLUSION)]),
                    calibration=ImageCalibration(Point(0, 0), Point(10, 0), 1.0),
                ),
                _record("c", "c.png"),
                _record("d", "missing.png"),
            ]
        )
        output_path = tmp_path / "all.xlsx"
        dialogs["save_file"] = (str(output_path), "")

        window.export_all_images_excel()

        rows = _read_rows(output_path)
        assert [row["ID"] for row in rows] == ["a", "b", "c", "d"]
        full_pixels = int(rows[0]["Количество пикселов контура"])
        assert int(rows[1]["Количество пикселов контура"]) == full_pixels - 11 * 11
        assert rows[1]["Негативных выделений"] == "1"
        assert rows[1]["Площадь контура, мм²"].endswith("мм²")
        assert rows[0]["RGB Зелёный"] == "200"
        assert rows[0]["RGB Средний цвет"] == "RGB(0, 200, 0)"
        assert rows[0]["Статус расчёта"] == "ok"
        assert rows[2]["Статус расчёта"] == "нет контура"
        assert rows[3]["Статус расчёта"] == "файл не найден"
        assert rows[3]["Статус"] == "отсутствует"
        assert "Lab L" in rows[0] and "LMS L" in rows[0]
        assert dialogs["critical"] == []

    def test_measurements_with_the_same_name_share_a_column(
        self, project_window, dialogs: dict[str, Any], tmp_path: Path
    ) -> None:
        def angle(angle_id: str) -> ProjectAngleMeasurement:
            return ProjectAngleMeasurement(
                id=angle_id,
                first=Point(10, 0),
                vertex=Point(0, 0),
                second=Point(0, 10),
            )

        window = project_window(
            [
                _record(
                    "a",
                    "a.png",
                    measurements=ProjectImageMeasurements(angles=[angle("x")]),
                ),
                _record(
                    "b",
                    "b.png",
                    measurements=ProjectImageMeasurements(angles=[angle("y")]),
                ),
            ]
        )
        output_path = tmp_path / "all.xlsx"
        dialogs["save_file"] = (str(output_path), "")

        window.export_all_images_excel()

        rows = _read_rows(output_path)
        assert rows[0]["Угол 1: Значение"]
        assert rows[0]["Угол 1: Значение"] == rows[1]["Угол 1: Значение"]

    def test_cancelled_file_dialog_writes_nothing(
        self, project_window, dialogs: dict[str, Any], tmp_path: Path
    ) -> None:
        window = project_window([_record("a", "a.png", annotation=_annotation())])

        window.export_all_images_excel()

        assert not list(tmp_path.glob("**/*.xlsx"))
        assert dialogs["critical"] == []

    def test_write_error_is_reported(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window([_record("a", "a.png", annotation=_annotation())])
        dialogs["save_file"] = (str(tmp_path / "all.xlsx"), "")

        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise PermissionError("файл открыт в Excel")

        monkeypatch.setattr(main_window_module, "write_xlsx_table", fail)

        window.export_all_images_excel()

        assert dialogs["critical"] == ["Ошибка экспорта данных всех фотографий в Excel"]

    def test_cancelled_calculation_aborts_export(
        self,
        project_window,
        dialogs: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window = project_window([_record("a", "a.png", annotation=_annotation())])
        output_path = tmp_path / "all.xlsx"
        dialogs["save_file"] = (str(output_path), "")
        monkeypatch.setattr(
            MainWindow, "_run_export_color_sums_worker", lambda *_args: None
        )

        window.export_all_images_excel()

        assert not output_path.exists()
        assert dialogs["critical"] == []

    def test_xlsx_suffix_is_added(
        self, project_window, dialogs: dict[str, Any], tmp_path: Path
    ) -> None:
        window = project_window([_record("a", "a.png", annotation=_annotation())])
        dialogs["save_file"] = (str(tmp_path / "all"), "")

        window.export_all_images_excel()

        assert (tmp_path / "all.xlsx").exists()

    def test_without_project_warns(
        self,
        qapp: QApplication,  # noqa: ARG002
        dialogs: dict[str, Any],
    ) -> None:
        window = MainWindow()
        try:
            window.export_all_images_excel()

            assert dialogs["warning"] == ["Нет проекта"]
            assert not window.export_all_images_excel_button.isEnabled()
        finally:
            window.close()
            window.deleteLater()

    def test_button_is_enabled_and_existing_buttons_are_kept(
        self, project_window
    ) -> None:
        window = project_window([_record("a", "a.png")])

        assert window.export_all_images_excel_button.isEnabled()
        assert window.export_all_images_excel_action.isEnabled()
        assert window.export_project_excel_button.text() == "Excel"
        assert window.refresh_analysis_button.text() == "Обновить"
