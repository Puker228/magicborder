from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import UUID
from xml.etree import ElementTree

from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QColor  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QAction,
    QApplication,
    QDialog,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QWidgetAction,
)

from magicborder.io_utils import load_project, save_project  # noqa: E402
from magicborder.main_window import (  # noqa: E402
    PROJECT_EXPORT_FIELDNAMES,
    MainWindow,
    _circle_contour_points,
    _qdatetime_from_text,
)
from magicborder.models import (  # noqa: E402
    ANGLE_LABEL_COLOR,
    ANGLE_LINE_COLOR,
    CONTOUR_LINE_COLOR,
    SEGMENT_LABEL_COLOR,
    SEGMENT_LINE_COLOR,
    Annotation,
    ImageCalibration,
    Point,
    ProjectAngleMeasurement,
    ProjectDocument,
    ProjectImageMeasurements,
    ProjectImageRecord,
    ProjectMeasurementAssessment,
    ProjectSegmentMeasurement,
)
from magicborder.property_browser import (  # noqa: E402
    PROPERTY_BROWSER_STYLE,
    PROPERTY_GRID_HORIZONTAL_COLOR,
    PROPERTY_GRID_VERTICAL_COLOR,
    PropertyBrowser,
    PropertyGridOverlay,
    PropertyValueLabel,
)

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


def _assert_uuid4(test_case: unittest.TestCase, value: str) -> None:
    parsed_uuid = UUID(value)
    test_case.assertEqual(str(parsed_uuid), value)
    test_case.assertEqual(parsed_uuid.version, 4)


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        sheet_xml = workbook.read("xl/worksheets/sheet1.xml")

    root = ElementTree.fromstring(sheet_xml)
    values: list[list[str]] = []
    for row in root.findall("s:sheetData/s:row", namespace):
        row_values: dict[int, str] = {}
        for cell in row.findall("s:c", namespace):
            cell_ref = cell.attrib["r"]
            column_name = "".join(char for char in cell_ref if char.isalpha())
            text_node = cell.find("s:is/s:t", namespace)
            row_values[_xlsx_column_index(column_name)] = (
                "" if text_node is None else (text_node.text or "")
            )
        values.append(
            [
                row_values.get(index, "")
                for index in range(max(row_values, default=-1) + 1)
            ]
        )
    return values


def _read_xlsx_dict_rows(path: Path) -> list[dict[str, str]]:
    values = _read_xlsx_rows(path)

    if not values:
        return []
    headers = values[0]
    return [
        dict(zip(headers, row + [""] * (len(headers) - len(row)), strict=False))
        for row in values[1:]
    ]


def _read_xlsx_cell_styles(path: Path) -> dict[str, str]:
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        sheet_xml = workbook.read("xl/worksheets/sheet1.xml")

    root = ElementTree.fromstring(sheet_xml)
    return {
        cell.attrib["r"]: cell.attrib.get("s", "")
        for cell in root.findall("s:sheetData/s:row/s:c", namespace)
    }


def _read_xlsx_styles_xml(path: Path) -> str:
    with zipfile.ZipFile(path) as workbook:
        return workbook.read("xl/styles.xml").decode("utf-8")


def _xlsx_column_index(column_name: str) -> int:
    index = 0
    for char in column_name:
        index = index * 26 + ord(char.upper()) - 64
    return index - 1


def _minimal_export_window(root: Path) -> MainWindow:
    project = ProjectDocument(
        name="export",
        images=[
            ProjectImageRecord(
                id="row-1",
                relative_path="images/missing.png",
                display_name="missing.png",
                metadata={"diagnosis": "class_x"},
            )
        ],
    )
    project_path = root / "export.json"
    save_project(project_path, project)

    window = MainWindow()
    window._set_project(project_path, load_project(project_path))
    return window


def _measurement_export_window(root: Path) -> tuple[MainWindow, Path]:
    image_dir = root / "images"
    image_dir.mkdir()
    Image.new("RGB", (40, 30), (120, 80, 40)).save(image_dir / "leaf.png")

    project = ProjectDocument(
        name="measurement_export",
        images=[
            ProjectImageRecord(
                id="leaf",
                relative_path="images/leaf.png",
                display_name="leaf.png",
                image_width=40,
                image_height=30,
                calibration=ImageCalibration(
                    start=Point(0, 0),
                    end=Point(10, 0),
                    length_mm=5,
                ),
                measurements=ProjectImageMeasurements(
                    angles=[
                        ProjectAngleMeasurement(
                            id="angle-a",
                            first=Point(2, 10),
                            vertex=Point(2, 2),
                            second=Point(10, 2),
                            name="Контрольный угол",
                            note="контрольный угол",
                        ),
                        ProjectAngleMeasurement(
                            id="angle-b",
                            first=Point(20, 10),
                            vertex=Point(20, 2),
                            second=Point(28, 2),
                            name="Второй угол",
                            note="",
                        ),
                    ],
                    segments=[
                        ProjectSegmentMeasurement(
                            id="segment-a",
                            start=Point(2, 2),
                            end=Point(12, 2),
                            name="Контрольный отрезок",
                            start_label="Начало",
                            end_label="Конец",
                            note="измерить повторно",
                        )
                    ],
                ),
            )
        ],
    )
    project_path = root / "measurement_export.json"
    save_project(project_path, project)

    window = MainWindow()
    window._set_project(project_path, load_project(project_path))
    return window, project_path


def _add_test_contour(window: MainWindow) -> None:
    record = window._selected_project_image()
    assert record is not None
    points = [
        Point(4, 4),
        Point(36, 4),
        Point(36, 26),
        Point(4, 26),
    ]
    record.annotation = Annotation(
        image_path=record.relative_path,
        image_width=40,
        image_height=30,
        points=points,
    )
    window.canvas.set_contour(points)
    # Аналитика считается только по команде «Обновить».
    window.refresh_analysis()
    window._update_project_properties()
    window._update_action_states()


def _list_item_color_name(window: MainWindow, row: int = 0) -> str:
    return window.project_list.item(row).foreground().color().name()


def _property_panel_rows(window: MainWindow) -> list[str]:
    return window.properties_browser.rows()


def _project_property_panel_rows(window: MainWindow) -> list[str]:
    return window.project_properties_browser.rows()


def _property_group(browser: PropertyBrowser, title: str):
    return browser.group_item(title)


def _main_toolbar(window: MainWindow) -> QToolBar:
    for toolbar in window.findChildren(QToolBar):
        if toolbar.windowTitle() == "Основная панель":
            return toolbar
    raise AssertionError("Основная панель не найдена")


def _tree_item_by_text(tree: QTreeWidget, text: str) -> QTreeWidgetItem:
    def visit(item: QTreeWidgetItem) -> QTreeWidgetItem | None:
        if item.text(0) == text:
            return item
        for index in range(item.childCount()):
            found = visit(item.child(index))
            if found is not None:
                return found
        return None

    for index in range(tree.topLevelItemCount()):
        found = visit(tree.topLevelItem(index))
        if found is not None:
            return found
    raise AssertionError(f"Tree item not found: {text}")


def _tree_item_by_key(tree: QTreeWidget, key: str) -> QTreeWidgetItem:
    def visit(item: QTreeWidgetItem) -> QTreeWidgetItem | None:
        if item.data(0, Qt.UserRole) == key:
            return item
        for index in range(item.childCount()):
            found = visit(item.child(index))
            if found is not None:
                return found
        return None

    for index in range(tree.topLevelItemCount()):
        found = visit(tree.topLevelItem(index))
        if found is not None:
            return found
    raise AssertionError(f"Tree item not found for key: {key}")


def _combo_item_data(widget) -> list[str]:
    return [str(widget.itemData(index) or "") for index in range(widget.count())]


class ProjectPropertiesTest(unittest.TestCase):
    def test_single_image_open_save_actions_are_removed(self) -> None:
        _app()
        window = MainWindow()

        self.assertFalse(hasattr(window, "open_image_action"))
        self.assertFalse(hasattr(window, "save_image_action"))

        actions = window.findChildren(QAction)
        action_texts = {action.text() for action in actions}
        shortcuts = {
            action.shortcut().toString()
            for action in actions
            if not action.shortcut().isEmpty()
        }

        self.assertNotIn("Открыть изображение...", action_texts)
        self.assertNotIn("Сохранить изображение...", action_texts)
        self.assertNotIn("Ctrl+O", shortcuts)
        self.assertNotIn("Ctrl+S", shortcuts)

    def test_exit_button_is_right_aligned_on_toolbar(self) -> None:
        _app()
        window = MainWindow()
        toolbar = _main_toolbar(window)

        actions = toolbar.actions()
        self.assertIs(actions[-1], window.exit_action)
        self.assertIsInstance(actions[-2], QWidgetAction)
        spacer = actions[-2].defaultWidget()
        self.assertIsNotNone(spacer)
        self.assertEqual(spacer.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)

    def test_canvas_visibility_actions_exist_and_follow_image_state(self) -> None:
        _app()
        window = MainWindow()

        self.assertIsInstance(window.show_all_canvas_elements_action, QAction)
        self.assertIsInstance(window.hide_all_canvas_elements_action, QAction)
        self.assertEqual(
            window.show_all_canvas_elements_action.text(), "Показать все элементы"
        )
        self.assertEqual(
            window.hide_all_canvas_elements_action.text(), "Скрыть все элементы"
        )
        self.assertFalse(window.show_all_canvas_elements_action.isEnabled())
        self.assertFalse(window.hide_all_canvas_elements_action.isEnabled())

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loaded_window, _project_path = _measurement_export_window(root)

            self.assertTrue(loaded_window.show_all_canvas_elements_action.isEnabled())
            self.assertTrue(loaded_window.hide_all_canvas_elements_action.isEnabled())

    def test_toolbar_groups_canvas_visibility_and_annotation_actions(self) -> None:
        _app()
        window = MainWindow()
        toolbar = _main_toolbar(window)
        actions = toolbar.actions()

        show_index = actions.index(window.show_all_canvas_elements_action)
        hide_index = actions.index(window.hide_all_canvas_elements_action)
        save_annotation_index = actions.index(window.save_annotation_action)
        open_annotation_index = actions.index(window.open_annotation_action)
        about_index = actions.index(window.about_action)

        self.assertEqual(hide_index, show_index + 1)
        self.assertTrue(actions[show_index - 1].isSeparator())
        self.assertTrue(actions[hide_index + 1].isSeparator())
        self.assertTrue(actions[save_annotation_index - 1].isSeparator())
        self.assertEqual(open_annotation_index, save_annotation_index + 1)
        self.assertFalse(actions[open_annotation_index].isSeparator())
        self.assertTrue(actions[about_index - 1].isSeparator())
        self.assertIs(actions[-1], window.exit_action)
        self.assertIsInstance(actions[-2], QWidgetAction)

    def test_image_properties_panel_uses_grouped_order(self) -> None:
        _app()
        window = MainWindow()

        self.assertEqual(
            _property_panel_rows(window),
            [
                "--- Общая информация о файле",
                "ID",
                "Файл",
                "Путь",
                "Размер",
                "Статус",
                "Дата добавления",
                "Дата съёмки",
                "--- Калибровка",
                "Длина калибровки, мм",
                "Масштаб",
                "--- Информация о главном контуре",
                "Аннотация",
                "Показывать контур",
                "Цвет линии",
                "Количество узлов контура",
                "Количество пикселов контура",
                "Площадь контура, мм²",
                "--- Измерения",
                "--- Углы",
                "Нет углов",
                "--- Отрезки",
                "Нет отрезков",
                "--- Цветовые пространства",
                "--- Цветовое пространство RGB",
                "Красный",
                "Зелёный",
                "Синий",
                "Средний цвет",
                "--- Цветовое пространство Lab",
                "L",
                "a",
                "b",
                "--- Цветовое пространство HSV",
                "H",
                "S",
                "V",
                "--- Цветовое пространство YUV",
                "Y",
                "U",
                "V",
                "--- Цветовое пространство LMS",
                "L",
                "M",
                "S",
                "--- Локация",
                "Освещённость",
                "Влажность, %",
                "Скорость ветра",
                "Направление ветра",
                "Широта",
                "Долгота",
                "--- Дополнительно",
                "Диагноз",
                "Дополнительные сведения",
            ],
        )
        calibration_group = _property_group(window.properties_browser, "Калибровка")
        contour_group = _property_group(
            window.properties_browser, "Информация о главном контуре"
        )
        color_spaces_group = _property_group(
            window.properties_browser, "Цветовые пространства"
        )
        rgb_group = _property_group(
            window.properties_browser, "Цветовое пространство RGB"
        )
        lab_group = _property_group(
            window.properties_browser, "Цветовое пространство Lab"
        )
        hsv_group = _property_group(
            window.properties_browser, "Цветовое пространство HSV"
        )
        yuv_group = _property_group(
            window.properties_browser, "Цветовое пространство YUV"
        )
        lms_group = _property_group(
            window.properties_browser, "Цветовое пространство LMS"
        )

        self.assertIs(
            window.properties_browser.property_item("Длина калибровки, мм").parent(),
            calibration_group,
        )
        self.assertIs(
            window.properties_browser.property_item("Масштаб").parent(),
            calibration_group,
        )
        self.assertIsNot(
            window.properties_browser.property_item("Длина калибровки, мм").parent(),
            contour_group,
        )
        self.assertIsNot(
            window.properties_browser.property_item("Масштаб").parent(),
            contour_group,
        )
        self.assertLess(
            window.properties_browser.indexOfTopLevelItem(calibration_group),
            window.properties_browser.indexOfTopLevelItem(contour_group),
        )
        for group in (rgb_group, lab_group, hsv_group, yuv_group, lms_group):
            self.assertIs(group.parent(), color_spaces_group)
        self.assertIs(
            window.properties_browser.property_item("Красный").parent(), rgb_group
        )
        self.assertEqual(window.properties_browser.indexOfTopLevelItem(rgb_group), -1)
        self.assertLess(
            window.properties_browser.indexOfTopLevelItem(
                window.measurements_group_item
            ),
            window.properties_browser.indexOfTopLevelItem(color_spaces_group),
        )
        self.assertLess(
            window.properties_browser.indexOfTopLevelItem(color_spaces_group),
            window.properties_browser.indexOfTopLevelItem(
                _property_group(window.properties_browser, "Локация")
            ),
        )

    def test_project_properties_panel_is_between_images_and_image_properties(
        self,
    ) -> None:
        _app()
        window = MainWindow()

        self.assertEqual(
            window.project_splitter.widget(0).objectName(), "projectImagesPanel"
        )
        self.assertEqual(
            window.project_splitter.widget(1).objectName(), "projectPropertiesPanel"
        )
        self.assertEqual(
            window.project_splitter.widget(2).objectName(), "imagePropertiesPanel"
        )
        self.assertEqual(
            _project_property_panel_rows(window),
            [
                "--- Общие свойства",
                "Имя проекта",
                "Путь",
                "Количество файлов",
                "Общая информация",
                "--- Цветовые пространства",
                "--- Цветовое пространство RGB",
                "Средний R",
                "Средний G",
                "Средний B",
                "--- Цветовое пространство Lab",
                "Средний L",
                "Средний a",
                "Средний b",
                "--- Цветовое пространство HSV",
                "Средний H",
                "Средний S",
                "Средний V",
                "--- Цветовое пространство YUV",
                "Средний Y",
                "Средний U",
                "Средний V",
                "--- Цветовое пространство LMS",
                "Средний L",
                "Средний M",
                "Средний S",
            ],
        )
        project_color_spaces_group = _property_group(
            window.project_properties_browser, "Цветовые пространства"
        )
        for group_title in (
            "Цветовое пространство RGB",
            "Цветовое пространство Lab",
            "Цветовое пространство HSV",
            "Цветовое пространство YUV",
            "Цветовое пространство LMS",
        ):
            group = _property_group(window.project_properties_browser, group_title)
            self.assertIs(group.parent(), project_color_spaces_group)
            self.assertEqual(
                window.project_properties_browser.indexOfTopLevelItem(group), -1
            )

    def test_project_properties_show_project_name_and_path(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_path = root / "display" / "display.json"
            save_project(project_path, ProjectDocument(name="stale_name", images=[]))

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            window._project_autosave_timer.stop()

            self.assertIsInstance(window.project_name, QLineEdit)
            self.assertIsInstance(window.project_path_field, QLineEdit)
            self.assertEqual(window.project_name.text(), "display")
            self.assertEqual(window.project_document.name, "display")
            self.assertEqual(
                window.project_path_field.text(), str(project_path.resolve())
            )
            self.assertTrue(window._save_project_silently(show_error=True))
            self.assertEqual(load_project(project_path).name, "display")
            self.assertTrue(window.project_path_field.isReadOnly())
            self.assertTrue(window.project_path_field.isEnabled())
            self.assertIs(
                window.project_properties_browser.property_item(
                    "project_name"
                ).parent(),
                _property_group(window.project_properties_browser, "Общие свойства"),
            )
            self.assertIs(
                window.project_properties_browser.property_item(
                    "project_path"
                ).parent(),
                _property_group(window.project_properties_browser, "Общие свойства"),
            )

    def test_project_name_edit_renames_project_file(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "original"
            project_path = project_dir / "original.json"
            save_project(
                project_path,
                ProjectDocument(
                    name="original",
                    images=[
                        ProjectImageRecord(
                            id="leaf-1",
                            relative_path="images/leaf.png",
                            display_name="leaf.png",
                        )
                    ],
                ),
            )
            (project_dir / "images").mkdir()

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            window.project_name.setText("Renamed Project.json")
            window.project_name.editingFinished.emit()
            new_project_dir = root / "Renamed_Project"
            new_project_path = new_project_dir / "Renamed_Project.json"

            self.assertFalse(project_dir.exists())
            self.assertFalse(project_path.exists())
            self.assertTrue(new_project_dir.exists())
            self.assertTrue((new_project_dir / "images").is_dir())
            self.assertTrue(new_project_path.exists())
            self.assertEqual(window.project_path, new_project_path.resolve())
            self.assertEqual(
                window.project_path_field.text(), str(new_project_path.resolve())
            )
            self.assertEqual(window.project_document.name, "Renamed_Project")
            self.assertEqual(window.project_name.text(), "Renamed_Project")
            self.assertEqual(
                window.project_document.images[0].relative_path, "images/leaf.png"
            )
            self.assertEqual(load_project(new_project_path).name, "Renamed_Project")
            self.assertEqual(
                window.windowTitle(), "MagicBorder - Renamed_Project - leaf.png"
            )

            window.project_general_info.setPlainText("После переименования")
            window._project_autosave_timer.stop()
            self.assertTrue(window._save_project_silently(show_error=True))
            self.assertFalse(project_path.exists())
            self.assertEqual(
                load_project(new_project_path).project_info.general_info,
                "После переименования",
            )

    def test_project_name_edit_rejects_existing_project_directory(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "safe"
            project_path = project_dir / "safe.json"
            existing_dir = root / "taken"
            existing_path = existing_dir / "taken.json"
            save_project(project_path, ProjectDocument(name="safe", images=[]))
            save_project(existing_path, ProjectDocument(name="taken", images=[]))
            existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            with patch.object(window, "_show_warning") as show_warning:
                window.project_name.setText("taken")
                window.project_name.editingFinished.emit()

                self.assertTrue(project_path.exists())
                self.assertTrue(project_dir.exists())
                self.assertTrue(existing_dir.exists())
                self.assertTrue(existing_path.exists())
                self.assertEqual(
                    json.loads(existing_path.read_text(encoding="utf-8")),
                    existing_payload,
                )
                self.assertEqual(window.project_path, project_path.resolve())
                self.assertEqual(window.project_document.name, "safe")
                self.assertEqual(window.project_name.text(), "safe")
                self.assertEqual(
                    window.project_path_field.text(), str(project_path.resolve())
                )
                show_warning.assert_called_once()

    def test_project_name_edit_rejects_empty_name_and_paths(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_path = root / "safe" / "safe.json"
            save_project(project_path, ProjectDocument(name="safe", images=[]))

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            for invalid_name in ("   ", ".", "..", "folder/unsafe", "folder\\unsafe"):
                with self.subTest(invalid_name=invalid_name):
                    with patch.object(window, "_show_warning") as show_warning:
                        window.project_name.setText(invalid_name)
                        window.project_name.editingFinished.emit()

                        self.assertEqual(window.project_path, project_path.resolve())
                        self.assertEqual(window.project_document.name, "safe")
                        self.assertEqual(window.project_name.text(), "safe")
                        self.assertTrue(project_path.exists())
                        show_warning.assert_called_once()

    def test_left_project_sections_use_distinct_light_backgrounds(self) -> None:
        _app()
        window = MainWindow()
        style_sheet = window.project_splitter.parentWidget().styleSheet()

        self.assertIn(
            "QWidget#projectImagesPanel { background: #f7fbff; }", style_sheet
        )
        self.assertIn(
            "QWidget#projectPropertiesPanel { background: #f8fcf6; }", style_sheet
        )
        self.assertIn(
            "QWidget#imagePropertiesPanel { background: #fffaf4; }", style_sheet
        )

    def test_property_groups_are_collapsed_by_default_and_toggle_independently(
        self,
    ) -> None:
        _app()
        window = MainWindow()

        browser = window.properties_browser
        self.assertIsInstance(browser, PropertyBrowser)
        file_group = _property_group(browser, "Общая информация о файле")
        contour_group = _property_group(browser, "Информация о главном контуре")

        self.assertFalse(file_group.isExpanded())
        self.assertFalse(browser.is_property_visible("ID"))
        self.assertFalse(browser.is_property_visible("Аннотация"))

        file_group.setExpanded(True)

        self.assertTrue(file_group.isExpanded())
        self.assertTrue(browser.is_property_visible("ID"))
        self.assertFalse(contour_group.isExpanded())
        self.assertFalse(browser.is_property_visible("Аннотация"))

        file_group.setExpanded(False)

        self.assertFalse(file_group.isExpanded())
        self.assertFalse(browser.is_property_visible("ID"))

    def test_project_property_groups_are_collapsed_by_default_and_toggle(self) -> None:
        _app()
        window = MainWindow()

        browser = window.project_properties_browser
        self.assertIsInstance(browser, PropertyBrowser)
        general_group = _property_group(browser, "Общие свойства")
        rgb_group = _property_group(browser, "Цветовое пространство RGB")

        self.assertFalse(general_group.isExpanded())
        self.assertFalse(browser.is_property_visible("Общая информация"))
        self.assertFalse(browser.is_property_visible("Средний R"))

        general_group.setExpanded(True)

        self.assertTrue(general_group.isExpanded())
        self.assertTrue(browser.is_property_visible("Общая информация"))
        self.assertFalse(rgb_group.isExpanded())
        self.assertFalse(browser.is_property_visible("Средний R"))

    def test_property_browsers_allow_key_column_width_adjustment(self) -> None:
        _app()
        window = MainWindow()

        for browser in (window.project_properties_browser, window.properties_browser):
            browser.resize(320, 220)

            self.assertIsInstance(browser, PropertyBrowser)
            self.assertFalse(browser.isHeaderHidden())
            self.assertEqual(
                browser.header().sectionResizeMode(0), QHeaderView.Interactive
            )
            self.assertEqual(browser.header().sectionResizeMode(1), QHeaderView.Stretch)

            browser.set_key_column_width(110)
            self.assertEqual(browser.key_column_width(), 110)
            browser.set_key_column_width(180)
            self.assertEqual(browser.key_column_width(), 180)

        self.assertIsInstance(window.project_image_count, PropertyValueLabel)
        self.assertIsInstance(window.property_path, PropertyValueLabel)

    def test_property_browsers_use_shared_grid_lines(self) -> None:
        _app()
        window = MainWindow()

        for browser in (window.project_properties_browser, window.properties_browser):
            overlay = browser.findChild(PropertyGridOverlay)

            self.assertIsNotNone(overlay)
            self.assertTrue(overlay.testAttribute(Qt.WA_TransparentForMouseEvents))
            self.assertEqual(browser.styleSheet(), PROPERTY_BROWSER_STYLE)
            self.assertIn(
                f"border-right: 1px solid {PROPERTY_GRID_VERTICAL_COLOR}",
                browser.styleSheet(),
            )
            self.assertIn(
                f"border-bottom: 1px solid {PROPERTY_GRID_HORIZONTAL_COLOR}",
                browser.styleSheet(),
            )

    def test_property_browser_grid_keeps_group_rows_spanned(self) -> None:
        _app()
        browser = PropertyBrowser()
        group = browser.add_group("Группа", expanded=True)
        subgroup = browser.add_group("Подгруппа", expanded=True, parent=group)

        browser.add_property_to_item(
            subgroup, "Свойство", PropertyValueLabel("Значение")
        )

        self.assertTrue(group.isFirstColumnSpanned())
        self.assertTrue(subgroup.isFirstColumnSpanned())

    def test_property_browser_long_values_remain_available_and_resize_rows(
        self,
    ) -> None:
        _app()
        browser = PropertyBrowser()
        browser.resize(260, 220)
        browser.add_group("Группа", expanded=True)
        label = PropertyValueLabel()
        item = browser.add_property("Группа", "Длинное свойство", label)

        browser.set_key_column_width(96)
        compact_height = item.sizeHint(1).height()
        long_text = "длинное значение свойства " * 20
        label.setText(long_text)
        browser.refresh_layout()

        self.assertTrue(label.wordWrap())
        self.assertTrue(label.textInteractionFlags() & Qt.TextSelectableByMouse)
        self.assertEqual(label.toolTip(), long_text)
        self.assertGreater(item.sizeHint(1).height(), compact_height)

    def test_image_properties_panel_has_no_refresh_button(self) -> None:
        _app()
        window = MainWindow()

        property_buttons = window.image_properties_panel.findChildren(QToolButton)
        button_texts = {button.text() for button in property_buttons}
        button_tooltips = {button.toolTip() for button in property_buttons}

        self.assertNotIn("Обновить", button_texts)
        self.assertNotIn("Обновить свойства выбранного изображения", button_tooltips)

    def test_project_panel_header_buttons_share_icon_text_style(self) -> None:
        _app()
        window = MainWindow()

        buttons = [
            window.export_project_excel_button,
            window.export_image_properties_excel_button,
        ]

        for button in buttons:
            self.assertEqual(button.text(), "Excel")
            self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonTextBesideIcon)
            self.assertFalse(button.icon().isNull())
            self.assertFalse(button.autoRaise())

    def test_new_project_prompts_for_name_then_parent_directory(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            call_order: list[str] = []

            def get_project_name(*_args, **_kwargs) -> tuple[str, bool]:
                call_order.append("name")
                return ("Leaves 2026", True)

            def get_parent_directory(*_args, **_kwargs) -> str:
                call_order.append("directory")
                return str(root)

            window = MainWindow()
            with (
                patch(
                    "magicborder.main_window.QInputDialog.getText",
                    side_effect=get_project_name,
                ),
                patch(
                    "magicborder.main_window.QFileDialog.getExistingDirectory",
                    side_effect=get_parent_directory,
                ),
            ):
                window.new_project()

            project_dir = root / "Leaves_2026"
            project_path = project_dir / "Leaves_2026.json"

            self.assertEqual(call_order, ["name", "directory"])
            self.assertEqual(window.project_path, project_path.resolve())
            self.assertTrue(project_dir.is_dir())
            self.assertTrue((project_dir / "images").is_dir())
            self.assertTrue(project_path.is_file())
            self.assertEqual(load_project(project_path).name, "Leaves_2026")

    def test_new_project_cancel_name_does_not_open_directory_dialog(self) -> None:
        _app()
        window = MainWindow()

        with (
            patch(
                "magicborder.main_window.QInputDialog.getText", return_value=("", False)
            ),
            patch(
                "magicborder.main_window.QFileDialog.getExistingDirectory",
            ) as get_directory,
        ):
            window.new_project()

        get_directory.assert_not_called()
        self.assertIsNone(window.project_path)
        self.assertIsNone(window.project_document)

    def test_new_project_cancel_directory_creates_nothing(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = MainWindow()

            with (
                patch(
                    "magicborder.main_window.QInputDialog.getText",
                    return_value=("Leaves 2026", True),
                ),
                patch(
                    "magicborder.main_window.QFileDialog.getExistingDirectory",
                    return_value="",
                ),
            ):
                window.new_project()

            self.assertFalse((root / "Leaves_2026").exists())
            self.assertIsNone(window.project_path)
            self.assertIsNone(window.project_document)

    def test_circle_contour_points_are_centered_and_counted(self) -> None:
        points = _circle_contour_points(20, 20, 5)

        self.assertEqual(len(points), 5)
        self.assertAlmostEqual(points[0].x, 10.0)
        self.assertAlmostEqual(points[0].y, 3.0)
        self.assertTrue(all(0.0 <= point.x <= 20.0 for point in points))
        self.assertTrue(all(0.0 <= point.y <= 20.0 for point in points))

    def test_property_point_count_matches_selected_contour(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            annotation = Annotation(
                image_path="images/leaf.png",
                image_width=20,
                image_height=20,
                points=[
                    Point(2, 2),
                    Point(15, 2),
                    Point(17, 8),
                    Point(12, 17),
                    Point(3, 14),
                ],
            )
            project = ProjectDocument(
                name="check_points",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                        annotation=annotation,
                    )
                ],
            )
            project_path = root / "check_points.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            # Аналитика считается только по команде «Обновить».
            window.refresh_analysis()

            self.assertTrue(window.canvas.has_image())
            # Канвас хранит разрешённый путь: на macOS /var — симлинк на /private/var.
            self.assertEqual(
                window.canvas.current_image_path(),
                (image_dir / "leaf.png").resolve(),
            )
            self.assertEqual(window.property_points.text(), "5")
            self.assertEqual(
                window.property_contour_area_mm2.text(), "калибровка не произведена"
            )
            self.assertEqual(
                window.property_calibration_length.text(), "калибровка не произведена"
            )
            self.assertFalse(window.property_calibration_length.isEnabled())
            self.assertEqual(window.property_calibration_scale.text(), "-")

            window.canvas.set_contour(
                [
                    Point(1, 1),
                    Point(18, 1),
                    Point(18, 18),
                    Point(1, 18),
                ]
            )
            window.refresh_analysis()

            self.assertEqual(window.property_points.text(), "4")
            self.assertEqual(window.property_contour_pixels.text(), "324")
            self.assertEqual(
                window.property_contour_area_mm2.text(), "калибровка не произведена"
            )
            self.assertEqual(
                window.property_calibration_length.text(), "калибровка не произведена"
            )
            self.assertEqual(
                len(window.project_document.images[0].annotation.points), 4
            )

            window.delete_current_contour()

            self.assertFalse(window.canvas.has_contour())
            self.assertIsNone(window.project_document.images[0].annotation)
            self.assertEqual(window.property_annotation.text(), "нет")
            self.assertEqual(window.property_points.text(), "-")
            self.assertEqual(window.property_contour_pixels.text(), "-")
            self.assertEqual(window.property_contour_area_mm2.text(), "-")
            self.assertEqual(
                window.property_calibration_length.text(), "калибровка не произведена"
            )
            self.assertEqual(window.property_calibration_scale.text(), "-")
            self.assertEqual(window.property_lab_l.text(), "-")
            self.assertEqual(window.property_lab_a.text(), "-")
            self.assertEqual(window.property_lab_b.text(), "-")
            self.assertEqual(window.property_hsv_h.text(), "-")
            self.assertEqual(window.property_hsv_s.text(), "-")
            self.assertEqual(window.property_hsv_v.text(), "-")
            self.assertEqual(window.property_yuv_y.text(), "-")
            self.assertEqual(window.property_yuv_u.text(), "-")
            self.assertEqual(window.property_yuv_v.text(), "-")
            self.assertEqual(window.property_lms_l.text(), "-")
            self.assertEqual(window.property_lms_m.text(), "-")
            self.assertEqual(window.property_lms_s.text(), "-")

    def test_contour_visibility_property_toggles_canvas_without_changing_annotation(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "empty.png")
            Image.new("RGB", (20, 20), (60, 120, 80)).save(image_dir / "leaf.png")

            annotation = Annotation(
                image_path="images/leaf.png",
                image_width=20,
                image_height=20,
                points=[
                    Point(2, 2),
                    Point(17, 2),
                    Point(17, 17),
                    Point(2, 17),
                ],
            )
            project = ProjectDocument(
                name="contour_visibility",
                images=[
                    ProjectImageRecord(
                        id="empty",
                        relative_path="images/empty.png",
                        display_name="empty.png",
                        image_width=20,
                        image_height=20,
                    ),
                    ProjectImageRecord(
                        id="leaf",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                        annotation=annotation,
                    ),
                ],
            )
            project_path = root / "contour_visibility.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertIn("Показывать контур", _property_panel_rows(window))
            self.assertFalse(window.property_contour_visible.isEnabled())
            self.assertFalse(window.property_contour_visible.isChecked())

            window.project_list.setCurrentRow(1)

            record = window._project_image_by_id("leaf")
            self.assertIsNotNone(record)
            self.assertTrue(window.property_contour_visible.isEnabled())
            self.assertTrue(window.property_contour_visible.isChecked())
            self.assertTrue(window.canvas._path_item.isVisible())
            self.assertTrue(
                all(handle.isVisible() for handle in window.canvas._handles)
            )

            window._project_autosave_timer.stop()
            window.property_contour_visible.setChecked(False)

            self.assertFalse(window.canvas.is_contour_visible())
            self.assertFalse(window.canvas._path_item.isVisible())
            self.assertTrue(
                all(not handle.isVisible() for handle in window.canvas._handles)
            )
            self.assertIsNotNone(record.annotation)
            self.assertEqual(record.annotation.points, annotation.points)
            self.assertFalse(window._project_autosave_timer.isActive())

            window.property_contour_visible.setChecked(True)

            self.assertTrue(window.canvas.is_contour_visible())
            self.assertTrue(window.canvas._path_item.isVisible())
            self.assertTrue(
                all(handle.isVisible() for handle in window.canvas._handles)
            )

    def test_line_color_properties_update_canvas_save_and_restore(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, project_path = _measurement_export_window(root)
            _add_test_contour(window)
            record = window._selected_project_image()
            self.assertIsNotNone(record)

            contour_item = window.properties_browser.property_item("contour:line_color")
            angle_item = window.properties_browser.property_item(
                "angle:angle-a:line_color"
            )
            angle_label_item = window.properties_browser.property_item(
                "angle:angle-a:label_color"
            )
            segment_item = window.properties_browser.property_item(
                "segment:segment-a:line_color"
            )
            segment_label_item = window.properties_browser.property_item(
                "segment:segment-a:label_color"
            )
            self.assertIsNotNone(contour_item)
            self.assertIsNotNone(angle_item)
            self.assertIsNotNone(angle_label_item)
            self.assertIsNotNone(segment_item)
            self.assertIsNotNone(segment_label_item)
            self.assertEqual(
                angle_item.parent().indexOfChild(angle_label_item),
                angle_item.parent().indexOfChild(angle_item) + 1,
            )
            self.assertEqual(
                segment_item.parent().indexOfChild(segment_label_item),
                segment_item.parent().indexOfChild(segment_item) + 1,
            )
            self.assertEqual(
                window.property_contour_line_color.text(), CONTOUR_LINE_COLOR
            )
            self.assertEqual(
                window._angle_line_color_fields["angle-a"].text(), ANGLE_LINE_COLOR
            )
            self.assertEqual(
                window._angle_label_color_fields["angle-a"].text(), ANGLE_LABEL_COLOR
            )
            self.assertEqual(
                window._segment_line_color_fields["segment-a"].text(),
                SEGMENT_LINE_COLOR,
            )
            self.assertEqual(
                window._segment_label_color_fields["segment-a"].text(),
                SEGMENT_LABEL_COLOR,
            )

            window._project_autosave_timer.stop()
            with patch(
                "magicborder.main_window.QColorDialog.getColor",
                return_value=QColor("#dc2626"),
            ):
                window.property_contour_line_color.click()

            self.assertEqual(record.annotation.line_color, "#dc2626")
            self.assertEqual(window.canvas.contour_line_color(), "#dc2626")
            self.assertEqual(window.canvas._path_item.pen().color().name(), "#dc2626")
            self.assertEqual(window.property_contour_line_color.text(), "#dc2626")
            self.assertTrue(window._project_autosave_timer.isActive())

            window._project_autosave_timer.stop()
            with patch(
                "magicborder.main_window.QColorDialog.getColor",
                return_value=QColor("#2563eb"),
            ):
                window._angle_line_color_fields["angle-a"].click()

            self.assertEqual(record.measurements.angles[0].line_color, "#2563eb")
            self.assertEqual(
                window.canvas._angle_graphics[0].first_line.pen().color().name(),
                "#2563eb",
            )
            self.assertEqual(
                window.canvas._angle_graphics[0].second_line.pen().color().name(),
                "#2563eb",
            )
            self.assertEqual(
                window._angle_line_color_fields["angle-a"].text(),
                "#2563eb",
            )
            self.assertTrue(window._project_autosave_timer.isActive())

            window._project_autosave_timer.stop()
            with patch(
                "magicborder.main_window.QColorDialog.getColor",
                return_value=QColor("#0f766e"),
            ):
                window._angle_label_color_fields["angle-a"].click()

            self.assertEqual(record.measurements.angles[0].label_color, "#0f766e")
            self.assertEqual(
                window.canvas._angle_graphics[0].label.defaultTextColor().name(),
                "#0f766e",
            )
            self.assertEqual(
                window._angle_label_color_fields["angle-a"].text(),
                "#0f766e",
            )
            self.assertTrue(window._project_autosave_timer.isActive())

            window._project_autosave_timer.stop()
            with patch(
                "magicborder.main_window.QColorDialog.getColor",
                return_value=QColor("#a855f7"),
            ):
                window._segment_line_color_fields["segment-a"].click()

            self.assertEqual(record.measurements.segments[0].line_color, "#a855f7")
            self.assertEqual(
                window.canvas._segment_graphics[0].line.pen().color().name(),
                "#a855f7",
            )
            self.assertEqual(
                window._segment_line_color_fields["segment-a"].text(),
                "#a855f7",
            )
            self.assertTrue(window._project_autosave_timer.isActive())

            window._project_autosave_timer.stop()
            with patch(
                "magicborder.main_window.QColorDialog.getColor",
                return_value=QColor("#c2410c"),
            ):
                window._segment_label_color_fields["segment-a"].click()

            self.assertEqual(record.measurements.segments[0].label_color, "#c2410c")
            self.assertEqual(
                window.canvas._segment_graphics[0]
                .length_label.defaultTextColor()
                .name(),
                "#c2410c",
            )
            self.assertEqual(
                window.canvas._segment_graphics[0]
                .start_label.defaultTextColor()
                .name(),
                "#c2410c",
            )
            self.assertEqual(
                window.canvas._segment_graphics[0].end_label.defaultTextColor().name(),
                "#c2410c",
            )
            self.assertEqual(
                window._segment_label_color_fields["segment-a"].text(),
                "#c2410c",
            )
            self.assertTrue(window._project_autosave_timer.isActive())

            window._project_autosave_timer.stop()
            with patch(
                "magicborder.main_window.QColorDialog.getColor",
                return_value=QColor(),
            ):
                window._angle_label_color_fields["angle-a"].click()

            self.assertEqual(record.measurements.angles[0].line_color, "#2563eb")
            self.assertEqual(record.measurements.angles[0].label_color, "#0f766e")
            self.assertFalse(window._project_autosave_timer.isActive())

            window.canvas.angle_handle_moved(0, 2, QPointF(12, 12))
            window.canvas.segment_handle_moved(0, 1, QPointF(14, 2))
            self.assertEqual(record.measurements.angles[0].line_color, "#2563eb")
            self.assertEqual(record.measurements.angles[0].label_color, "#0f766e")
            self.assertEqual(record.measurements.segments[0].line_color, "#a855f7")
            self.assertEqual(record.measurements.segments[0].label_color, "#c2410c")

            export_keys = window._image_property_export_leaf_keys(record)
            self.assertNotIn("contour:line_color", export_keys)
            self.assertNotIn("angle:angle-a:line_color", export_keys)
            self.assertNotIn("angle:angle-a:label_color", export_keys)
            self.assertNotIn("segment:segment-a:line_color", export_keys)
            self.assertNotIn("segment:segment-a:label_color", export_keys)

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            image_payload = payload["images"][0]
            self.assertEqual(
                image_payload["contour"]["annotation"]["line_color"],
                "#dc2626",
            )
            self.assertEqual(
                image_payload["measurements"]["angles"][0]["line_color"],
                "#2563eb",
            )
            self.assertEqual(
                image_payload["measurements"]["angles"][0]["label_color"],
                "#0f766e",
            )
            self.assertEqual(
                image_payload["measurements"]["segments"][0]["line_color"],
                "#a855f7",
            )
            self.assertEqual(
                image_payload["measurements"]["segments"][0]["label_color"],
                "#c2410c",
            )

            restored = MainWindow()
            restored._set_project(project_path, load_project(project_path))
            self.assertEqual(restored.canvas.contour_line_color(), "#dc2626")
            self.assertEqual(
                restored.canvas.angle_measurement_line_color("angle-a"),
                "#2563eb",
            )
            self.assertEqual(
                restored.canvas.angle_measurement_label_color("angle-a"),
                "#0f766e",
            )
            self.assertEqual(
                restored.canvas.segment_measurement_line_color("segment-a"),
                "#a855f7",
            )
            self.assertEqual(
                restored.canvas.segment_measurement_label_color("segment-a"),
                "#c2410c",
            )
            self.assertEqual(restored.property_contour_line_color.text(), "#dc2626")
            self.assertEqual(
                restored._angle_line_color_fields["angle-a"].text(),
                "#2563eb",
            )
            self.assertEqual(
                restored._angle_label_color_fields["angle-a"].text(),
                "#0f766e",
            )
            self.assertEqual(
                restored._segment_line_color_fields["segment-a"].text(),
                "#a855f7",
            )
            self.assertEqual(
                restored._segment_label_color_fields["segment-a"].text(),
                "#c2410c",
            )

    def test_line_colors_follow_selected_image_without_state_leak(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (40, 30), (120, 80, 40)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (40, 30), (60, 120, 80)).save(image_dir / "leaf_b.png")
            points = [Point(4, 4), Point(36, 4), Point(36, 26), Point(4, 26)]

            project = ProjectDocument(
                name="line_colors",
                images=[
                    ProjectImageRecord(
                        id="leaf-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        annotation=Annotation(
                            image_path="images/leaf_a.png",
                            image_width=40,
                            image_height=30,
                            points=points,
                            line_color="#111111",
                        ),
                        measurements=ProjectImageMeasurements(
                            angles=[
                                ProjectAngleMeasurement(
                                    id="angle-a",
                                    first=Point(2, 10),
                                    vertex=Point(2, 2),
                                    second=Point(10, 2),
                                    line_color="#222222",
                                    label_color="#223344",
                                )
                            ],
                            segments=[
                                ProjectSegmentMeasurement(
                                    id="segment-a",
                                    start=Point(2, 2),
                                    end=Point(12, 2),
                                    line_color="#333333",
                                    label_color="#334455",
                                )
                            ],
                        ),
                    ),
                    ProjectImageRecord(
                        id="leaf-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                        annotation=Annotation(
                            image_path="images/leaf_b.png",
                            image_width=40,
                            image_height=30,
                            points=points,
                            line_color="#444444",
                        ),
                        measurements=ProjectImageMeasurements(
                            angles=[
                                ProjectAngleMeasurement(
                                    id="angle-b",
                                    first=Point(20, 10),
                                    vertex=Point(20, 2),
                                    second=Point(28, 2),
                                    line_color="#555555",
                                    label_color="#556677",
                                )
                            ],
                            segments=[
                                ProjectSegmentMeasurement(
                                    id="segment-b",
                                    start=Point(20, 2),
                                    end=Point(30, 2),
                                    line_color="#666666",
                                    label_color="#667788",
                                )
                            ],
                        ),
                    ),
                ],
            )
            project_path = root / "line_colors.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            self.assertEqual(window.canvas.contour_line_color(), "#111111")
            self.assertEqual(
                window.canvas.angle_measurement_line_color("angle-a"), "#222222"
            )
            self.assertEqual(
                window.canvas.angle_measurement_label_color("angle-a"), "#223344"
            )
            self.assertEqual(
                window.canvas.segment_measurement_line_color("segment-a"),
                "#333333",
            )
            self.assertEqual(
                window.canvas.segment_measurement_label_color("segment-a"),
                "#334455",
            )

            window.project_list.setCurrentRow(1)

            self.assertEqual(window.canvas.contour_line_color(), "#444444")
            self.assertEqual(
                window.canvas.angle_measurement_line_color("angle-b"), "#555555"
            )
            self.assertEqual(
                window.canvas.angle_measurement_label_color("angle-b"), "#556677"
            )
            self.assertEqual(
                window.canvas.segment_measurement_line_color("segment-b"),
                "#666666",
            )
            self.assertEqual(
                window.canvas.segment_measurement_label_color("segment-b"),
                "#667788",
            )
            self.assertEqual(window.property_contour_line_color.text(), "#444444")
            self.assertEqual(
                window._angle_label_color_fields["angle-b"].text(), "#556677"
            )
            self.assertEqual(
                window._segment_label_color_fields["segment-b"].text(), "#667788"
            )

            window.project_list.setCurrentRow(0)

            self.assertEqual(window.canvas.contour_line_color(), "#111111")
            self.assertEqual(
                window.canvas.angle_measurement_line_color("angle-a"), "#222222"
            )
            self.assertEqual(
                window.canvas.angle_measurement_label_color("angle-a"), "#223344"
            )
            self.assertEqual(
                window.canvas.segment_measurement_line_color("segment-a"),
                "#333333",
            )
            self.assertEqual(
                window.canvas.segment_measurement_label_color("segment-a"),
                "#334455",
            )
            self.assertEqual(window.property_contour_line_color.text(), "#111111")
            self.assertEqual(
                window._angle_label_color_fields["angle-a"].text(), "#223344"
            )
            self.assertEqual(
                window._segment_label_color_fields["segment-a"].text(), "#334455"
            )

    def test_open_annotation_restores_contour_line_color(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            annotation_path = root / "imported_annotation.json"
            annotation = Annotation(
                image_path="images/leaf.png",
                image_width=40,
                image_height=30,
                points=[Point(4, 4), Point(36, 4), Point(36, 26), Point(4, 26)],
                line_color="#0f766e",
            )
            annotation_path.write_text(
                json.dumps(annotation.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )

            with patch(
                "magicborder.main_window.QFileDialog.getOpenFileName",
                return_value=(str(annotation_path), "JSON (*.json)"),
            ):
                window.open_annotation_file()

            record = window._selected_project_image()
            self.assertIsNotNone(record.annotation)
            self.assertEqual(record.annotation.line_color, "#0f766e")
            self.assertEqual(window.canvas.contour_line_color(), "#0f766e")
            self.assertEqual(window.canvas._path_item.pen().color().name(), "#0f766e")
            self.assertEqual(window.property_contour_line_color.text(), "#0f766e")

    def test_canvas_visibility_commands_toggle_all_elements_and_fields(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, project_path = _measurement_export_window(root)
            _add_test_contour(window)
            window._project_autosave_timer.stop()

            window.hide_all_canvas_elements_action.trigger()

            self.assertFalse(window.canvas.is_contour_visible())
            self.assertFalse(window.canvas.is_angle_measurement_visible("angle-a"))
            self.assertFalse(window.canvas.is_angle_measurement_visible("angle-b"))
            self.assertFalse(window.canvas.is_segment_measurement_visible("segment-a"))
            self.assertFalse(window.property_contour_visible.isChecked())
            self.assertFalse(window._angle_visibility_fields["angle-a"].isChecked())
            self.assertFalse(window._angle_visibility_fields["angle-b"].isChecked())
            self.assertFalse(window._segment_visibility_fields["segment-a"].isChecked())
            self.assertTrue(window._angle_visibility_fields["angle-a"].isEnabled())
            self.assertTrue(window._segment_visibility_fields["segment-a"].isEnabled())
            self.assertFalse(window._project_autosave_timer.isActive())

            window._angle_visibility_fields["angle-a"].setChecked(True)

            self.assertTrue(window.canvas.is_angle_measurement_visible("angle-a"))
            self.assertFalse(window.canvas.is_angle_measurement_visible("angle-b"))
            self.assertFalse(window.canvas.is_segment_measurement_visible("segment-a"))

            window.show_all_canvas_elements_action.trigger()

            self.assertTrue(window.canvas.is_contour_visible())
            self.assertTrue(window.canvas.is_angle_measurement_visible("angle-a"))
            self.assertTrue(window.canvas.is_angle_measurement_visible("angle-b"))
            self.assertTrue(window.canvas.is_segment_measurement_visible("segment-a"))
            self.assertTrue(window.property_contour_visible.isChecked())
            self.assertTrue(window._angle_visibility_fields["angle-a"].isChecked())
            self.assertTrue(window._angle_visibility_fields["angle-b"].isChecked())
            self.assertTrue(window._segment_visibility_fields["segment-a"].isChecked())
            self.assertFalse(window._project_autosave_timer.isActive())

            window.hide_all_canvas_elements()
            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            image_payload = payload["images"][0]
            serialized_image = json.dumps(image_payload, ensure_ascii=False)
            self.assertNotIn('"visible"', serialized_image)

    def test_image_property_measurement_click_highlights_canvas_item_without_selecting_it(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)

            first_angle_width = (
                window.canvas._angle_graphics[0].first_line.pen().widthF()
            )
            second_angle_width = (
                window.canvas._angle_graphics[1].first_line.pen().widthF()
            )
            segment_width = window.canvas._segment_graphics[0].line.pen().widthF()

            window._handle_image_property_item_clicked(
                window.properties_browser.property_item("angle:angle-a:value"),
                0,
            )

            self.assertEqual(window.canvas.highlighted_angle_id(), "angle-a")
            self.assertIsNone(window.canvas.highlighted_segment_id())
            self.assertGreater(
                window.canvas._angle_graphics[0].first_line.pen().widthF(),
                first_angle_width,
            )
            self.assertEqual(
                window.canvas._angle_graphics[1].first_line.pen().widthF(),
                second_angle_width,
            )
            self.assertFalse(window.canvas.has_selected_angle_vertex())
            self.assertFalse(window.delete_angle_action.isEnabled())

            window._handle_image_property_item_clicked(
                window.properties_browser.group_item("segment:segment-a"),
                0,
            )

            self.assertIsNone(window.canvas.highlighted_angle_id())
            self.assertEqual(window.canvas.highlighted_segment_id(), "segment-a")
            self.assertEqual(
                window.canvas._angle_graphics[0].first_line.pen().widthF(),
                first_angle_width,
            )
            self.assertGreater(
                window.canvas._segment_graphics[0].line.pen().widthF(), segment_width
            )
            self.assertFalse(window.canvas.has_selected_segment_endpoint())
            self.assertFalse(window.delete_segment_action.isEnabled())

            window._handle_image_property_item_clicked(
                window.properties_browser.property_item("Файл"), 0
            )

            self.assertIsNone(window.canvas.highlighted_angle_id())
            self.assertIsNone(window.canvas.highlighted_segment_id())
            self.assertEqual(
                window.canvas._segment_graphics[0].line.pen().widthF(), segment_width
            )

    def test_image_property_contour_click_highlights_canvas_contour(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            _add_test_contour(window)
            normal_width = window.canvas._path_item.pen().widthF()

            contour_group = window.properties_browser.group_item(
                "Информация о главном контуре"
            )
            self.assertEqual(contour_group.data(0, Qt.UserRole + 1), "contour")

            window._handle_image_property_item_clicked(contour_group, 0)

            self.assertTrue(window.canvas.is_contour_highlighted())
            self.assertGreater(window.canvas._path_item.pen().widthF(), normal_width)
            self.assertIsNone(window.canvas.highlighted_angle_id())
            self.assertIsNone(window.canvas.highlighted_segment_id())

            window._handle_image_property_item_clicked(
                window.properties_browser.property_item("contour:point_count"),
                0,
            )

            self.assertTrue(window.canvas.is_contour_highlighted())
            self.assertGreater(window.canvas._path_item.pen().widthF(), normal_width)

            window._handle_image_property_item_clicked(
                window.properties_browser.property_item("angle:angle-a:value"),
                0,
            )

            self.assertFalse(window.canvas.is_contour_highlighted())
            self.assertEqual(window.canvas._path_item.pen().widthF(), normal_width)
            self.assertEqual(window.canvas.highlighted_angle_id(), "angle-a")

            window._handle_image_property_item_clicked(
                window.properties_browser.group_item("segment:segment-a"),
                0,
            )

            self.assertFalse(window.canvas.is_contour_highlighted())
            self.assertIsNone(window.canvas.highlighted_angle_id())
            self.assertEqual(window.canvas.highlighted_segment_id(), "segment-a")

            window.canvas.highlight_contour()
            window._handle_image_property_item_clicked(
                window.properties_browser.property_item("Файл"), 0
            )

            self.assertFalse(window.canvas.is_contour_highlighted())
            self.assertIsNone(window.canvas.highlighted_angle_id())
            self.assertIsNone(window.canvas.highlighted_segment_id())
            self.assertEqual(window.canvas._path_item.pen().widthF(), normal_width)

            window.canvas.highlight_contour()
            window._clear_image_property_canvas_highlight()

            self.assertFalse(window.canvas.is_contour_highlighted())
            self.assertEqual(window.canvas._path_item.pen().widthF(), normal_width)

    def test_angle_measurements_are_project_entities_and_saved(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "leaf_b.png")

            project = ProjectDocument(
                name="measurements",
                images=[
                    ProjectImageRecord(
                        id="leaf-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        image_width=20,
                        image_height=20,
                    ),
                    ProjectImageRecord(
                        id="leaf-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                        image_width=20,
                        image_height=20,
                    ),
                ],
            )
            project_path = root / "angles.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.measure_angle_action.isEnabled())
            self.assertFalse(window.delete_angle_action.isEnabled())

            window.canvas.set_angle_measurements(
                [
                    (
                        Point(2, 10),
                        Point(2, 2),
                        Point(10, 2),
                    )
                ]
            )

            self.assertTrue(window._project_autosave_timer.isActive())
            record = window._project_image_by_id("leaf-a")
            self.assertIsNotNone(record)
            self.assertEqual(len(record.measurements.angles), 1)
            angle_id = record.measurements.angles[0].id
            self.assertIn("--- Измерения", _property_panel_rows(window))
            self.assertIn("--- Углы", _property_panel_rows(window))
            self.assertIn("--- Угол 1", _property_panel_rows(window))
            self.assertIn("Имя", _property_panel_rows(window))
            self.assertIn("Показывать на канвасе", _property_panel_rows(window))
            angle_id_label = window.properties_browser.itemWidget(
                window.properties_browser.property_item(f"angle:{angle_id}:id"),
                1,
            )
            self.assertEqual(angle_id_label.text(), angle_id)
            visibility_field = window._angle_visibility_fields[angle_id]
            self.assertTrue(visibility_field.isEnabled())
            self.assertTrue(visibility_field.isChecked())
            self.assertTrue(window.canvas.is_angle_measurement_visible(angle_id))

            name_field = window._angle_name_fields[angle_id]
            self.assertEqual(name_field.text(), "")
            name_field.setText("Контрольный угол")
            window._handle_angle_name_edit_finished(angle_id, name_field)
            self.assertEqual(record.measurements.angles[0].name, "Контрольный угол")
            self.assertIn("--- Контрольный угол", _property_panel_rows(window))
            self.assertEqual(
                window.canvas._angle_graphics[0].label.toPlainText(),
                "Контрольный угол: 90°",
            )

            note_field = window._angle_note_fields[angle_id]
            note_field.setText("контрольный угол")
            window._handle_angle_note_edit_finished(angle_id, note_field)
            self.assertEqual(record.measurements.angles[0].note, "контрольный угол")

            window.canvas._angle_graphics[0].handles[1].setSelected(True)
            self.assertTrue(window.delete_angle_action.isEnabled())

            visibility_field = window._angle_visibility_fields[angle_id]
            window._project_autosave_timer.stop()
            visibility_field.setChecked(False)

            self.assertFalse(visibility_field.isChecked())
            self.assertFalse(window.canvas.is_angle_measurement_visible(angle_id))
            self.assertFalse(window.canvas._angle_graphics[0].label.isVisible())
            self.assertFalse(window.canvas.has_selected_angle_vertex())
            self.assertFalse(window.delete_angle_action.isEnabled())
            self.assertEqual(len(record.measurements.angles), 1)
            self.assertFalse(window._project_autosave_timer.isActive())

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_angle = payload["images"][0]["measurements"]["angles"][0]
            self.assertNotIn("visible", saved_angle)

            visibility_field.setChecked(True)

            self.assertTrue(window.canvas.is_angle_measurement_visible(angle_id))
            self.assertTrue(window.canvas._angle_graphics[0].label.isVisible())

            window.canvas._angle_graphics[0].handles[1].setSelected(True)

            self.assertTrue(window.delete_angle_action.isEnabled())

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_angle = payload["images"][0]["measurements"]["angles"][0]
            self.assertEqual(saved_angle["id"], angle_id)
            self.assertEqual(saved_angle["first"], {"x": 2.0, "y": 10.0})
            self.assertEqual(saved_angle["vertex"], {"x": 2.0, "y": 2.0})
            self.assertEqual(saved_angle["second"], {"x": 10.0, "y": 2.0})
            self.assertEqual(saved_angle["name"], "Контрольный угол")
            self.assertEqual(saved_angle["note"], "контрольный угол")
            self.assertNotIn("visible", saved_angle)

            window.project_list.setCurrentRow(1)

            self.assertEqual(window.canvas.angle_measurements(), [])

            window.project_list.setCurrentRow(0)

            self.assertEqual(len(window.canvas.angle_measurements()), 1)
            self.assertEqual(window.canvas.angle_measurement_records()[0][0], angle_id)
            self.assertEqual(
                window.canvas._angle_graphics[0].label.toPlainText(),
                "Контрольный угол: 90°",
            )

            window.canvas._angle_graphics[0].handles[1].setSelected(True)
            window.delete_selected_angle()

            self.assertEqual(window.canvas.angle_measurements(), [])
            self.assertEqual(record.measurements.angles, [])
            self.assertFalse(window.delete_angle_action.isEnabled())

    def test_segment_measurements_are_project_entities_and_saved(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (40, 30), (120, 80, 40)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (40, 30), (10, 20, 30)).save(image_dir / "leaf_b.png")

            project = ProjectDocument(
                name="segments",
                images=[
                    ProjectImageRecord(
                        id="leaf-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        image_width=40,
                        image_height=30,
                        calibration=ImageCalibration(
                            start=Point(0, 0),
                            end=Point(10, 0),
                            length_mm=5,
                        ),
                    ),
                    ProjectImageRecord(
                        id="leaf-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                        image_width=40,
                        image_height=30,
                    ),
                ],
            )
            project_path = root / "segments.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.measure_segment_action.isEnabled())
            self.assertFalse(window.delete_segment_action.isEnabled())
            self.assertIn("--- Отрезки", _property_panel_rows(window))
            self.assertIn("Нет отрезков", _property_panel_rows(window))

            window.canvas.set_segment_measurement_records(
                [("segment-a", Point(2, 2), Point(12, 2), "A", "B")]
            )

            record = window._project_image_by_id("leaf-a")
            self.assertIsNotNone(record)
            self.assertEqual(len(record.measurements.segments), 1)
            segment_id = record.measurements.segments[0].id
            self.assertEqual(segment_id, "segment-a")
            self.assertIn("--- Отрезок 1", _property_panel_rows(window))
            self.assertIn("Имя", _property_panel_rows(window))
            self.assertIn("Показывать на канвасе", _property_panel_rows(window))
            self.assertIn("Подпись первой точки", _property_panel_rows(window))
            self.assertIn("Подпись второй точки", _property_panel_rows(window))
            self.assertIn("Примечание", _property_panel_rows(window))
            segment_length_label = window.properties_browser.itemWidget(
                window.properties_browser.property_item(f"segment:{segment_id}:length"),
                1,
            )
            self.assertEqual(segment_length_label.text(), "10 px / 5 мм")
            self.assertEqual(
                window.canvas._segment_graphics[0].length_label.toPlainText(),
                "Отрезок 1\n10 px / 5 мм",
            )
            visibility_field = window._segment_visibility_fields[segment_id]
            self.assertTrue(visibility_field.isEnabled())
            self.assertTrue(visibility_field.isChecked())
            self.assertTrue(window.canvas.is_segment_measurement_visible(segment_id))

            name_field = window._segment_name_fields[segment_id]
            self.assertEqual(name_field.text(), "")
            name_field.setText("Контрольный отрезок")
            window._handle_segment_name_edit_finished(segment_id, name_field)
            self.assertEqual(
                record.measurements.segments[0].name, "Контрольный отрезок"
            )
            self.assertIn("--- Контрольный отрезок", _property_panel_rows(window))
            self.assertEqual(
                window.canvas._segment_graphics[0].length_label.toPlainText(),
                "Контрольный отрезок\n10 px / 5 мм",
            )

            start_label_field = window._segment_start_label_fields[segment_id]
            start_label_field.setText("Начало")
            window._handle_segment_label_edit_finished(
                segment_id, start_label_field, "start"
            )
            end_label_field = window._segment_end_label_fields[segment_id]
            end_label_field.setText("Конец")
            window._handle_segment_label_edit_finished(
                segment_id, end_label_field, "end"
            )
            note_field = window._segment_note_fields[segment_id]
            note_field.setText("измерить повторно")
            window._handle_segment_note_edit_finished(segment_id, note_field)

            self.assertEqual(
                record.measurements.segments[0].name, "Контрольный отрезок"
            )
            self.assertEqual(record.measurements.segments[0].start_label, "Начало")
            self.assertEqual(record.measurements.segments[0].end_label, "Конец")
            self.assertEqual(record.measurements.segments[0].note, "измерить повторно")
            self.assertEqual(
                window.canvas._segment_graphics[0].start_label.toPlainText(), "Начало"
            )
            self.assertEqual(
                window.canvas._segment_graphics[0].end_label.toPlainText(), "Конец"
            )

            window.canvas._segment_graphics[0].handles[0].setSelected(True)

            self.assertTrue(window.delete_segment_action.isEnabled())

            visibility_field = window._segment_visibility_fields[segment_id]
            window._project_autosave_timer.stop()
            visibility_field.setChecked(False)

            self.assertFalse(visibility_field.isChecked())
            self.assertFalse(window.canvas.is_segment_measurement_visible(segment_id))
            self.assertFalse(window.canvas._segment_graphics[0].line.isVisible())
            self.assertFalse(
                window.canvas._segment_graphics[0].length_label.isVisible()
            )
            self.assertFalse(window.canvas.has_selected_segment_endpoint())
            self.assertFalse(window.delete_segment_action.isEnabled())
            self.assertEqual(len(record.measurements.segments), 1)
            self.assertFalse(window._project_autosave_timer.isActive())

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_segment = payload["images"][0]["measurements"]["segments"][0]
            self.assertNotIn("visible", saved_segment)

            visibility_field.setChecked(True)

            self.assertTrue(window.canvas.is_segment_measurement_visible(segment_id))
            self.assertTrue(window.canvas._segment_graphics[0].line.isVisible())
            self.assertTrue(window.canvas._segment_graphics[0].length_label.isVisible())

            window.canvas._segment_graphics[0].handles[0].setSelected(True)
            self.assertTrue(window.delete_segment_action.isEnabled())

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_segment = payload["images"][0]["measurements"]["segments"][0]
            self.assertEqual(saved_segment["id"], segment_id)
            self.assertEqual(saved_segment["name"], "Контрольный отрезок")
            self.assertEqual(saved_segment["start"], {"x": 2.0, "y": 2.0})
            self.assertEqual(saved_segment["end"], {"x": 12.0, "y": 2.0})
            self.assertEqual(saved_segment["start_label"], "Начало")
            self.assertEqual(saved_segment["end_label"], "Конец")
            self.assertEqual(saved_segment["note"], "измерить повторно")
            self.assertNotIn("visible", saved_segment)

            window.project_list.setCurrentRow(1)

            self.assertEqual(window.canvas.segment_measurements(), [])

            window.project_list.setCurrentRow(0)

            self.assertEqual(len(window.canvas.segment_measurements()), 1)
            self.assertEqual(
                window.canvas.segment_measurement_records()[0][0], segment_id
            )
            self.assertEqual(
                window.canvas._segment_graphics[0].length_label.toPlainText(),
                "Контрольный отрезок\n10 px / 5 мм",
            )
            self.assertEqual(
                window._segment_name_fields[segment_id].text(), "Контрольный отрезок"
            )
            self.assertEqual(
                window._segment_note_fields[segment_id].text(), "измерить повторно"
            )

            window.canvas._segment_graphics[0].handles[0].setSelected(True)
            window.delete_selected_segment()

            self.assertEqual(window.canvas.segment_measurements(), [])
            self.assertEqual(record.measurements.segments, [])
            self.assertFalse(window.delete_segment_action.isEnabled())

    def test_oiv_assessment_selection_filters_saves_and_exports(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, project_path = _measurement_export_window(root)
            record = window._selected_project_image()
            self.assertIsNotNone(record)

            angle_combo = window._angle_assessment_fields["angle-a"]
            angle_codes = _combo_item_data(angle_combo)
            self.assertIn("", angle_codes)
            self.assertIn("OIV 607", angle_codes)
            self.assertNotIn("OIV 601", angle_codes)

            angle_combo.setCurrentIndex(angle_combo.findData("OIV 607"))

            self.assertIsNotNone(record.measurements.angles[0].assessment)
            self.assertEqual(record.measurements.angles[0].assessment.code, "OIV 607")
            self.assertEqual(
                window._angle_assessment_result_fields["angle-a"].text(),
                "90°; оценка 9 - очень большой",
            )

            segment_combo = window._segment_assessment_fields["segment-a"]
            segment_codes = _combo_item_data(segment_combo)
            self.assertIn("", segment_codes)
            self.assertIn("OIV 601", segment_codes)
            self.assertNotIn("OIV 607", segment_codes)

            segment_combo.setCurrentIndex(segment_combo.findData("OIV 601"))

            self.assertIsNotNone(record.measurements.segments[0].assessment)
            self.assertEqual(record.measurements.segments[0].assessment.code, "OIV 601")
            self.assertEqual(
                window._segment_assessment_result_fields["segment-a"].text(),
                "5 мм; оценка 1 - очень короткая",
            )

            keys = window._image_property_export_leaf_keys(record)
            self.assertIn("angle:angle-a:oiv_code", keys)
            self.assertIn("segment:segment-a:oiv_code", keys)
            self.assertNotIn("angle:angle-b:oiv_code", keys)

            export_path = root / "oiv_measurement_properties.xlsx"
            window._write_image_properties_excel(
                export_path,
                [
                    "angle:angle-a:oiv_code",
                    "angle:angle-a:oiv_short_name",
                    "angle:angle-a:oiv_value",
                    "angle:angle-a:oiv_score",
                    "angle:angle-a:oiv_label",
                    "segment:segment-a:oiv_code",
                    "segment:segment-a:oiv_short_name",
                    "segment:segment-a:oiv_value",
                    "segment:segment-a:oiv_score",
                    "segment:segment-a:oiv_label",
                ],
            )

            self.assertEqual(
                _read_xlsx_dict_rows(export_path),
                [
                    {"Свойство": "Измерения", "Значение": ""},
                    {"Свойство": "Углы", "Значение": ""},
                    {"Свойство": "Контрольный угол", "Значение": ""},
                    {"Свойство": "OIV-код", "Значение": "OIV 607"},
                    {"Свойство": "OIV-признак", "Значение": "угол между N1 и N2"},
                    {"Свойство": "Фактическое значение", "Значение": "90°"},
                    {"Свойство": "Оценка OIV", "Значение": "9"},
                    {"Свойство": "Значение шкалы", "Значение": "очень большой"},
                    {"Свойство": "Отрезки", "Значение": ""},
                    {"Свойство": "Контрольный отрезок", "Значение": ""},
                    {"Свойство": "OIV-код", "Значение": "OIV 601"},
                    {"Свойство": "OIV-признак", "Значение": "длина жилки N1"},
                    {"Свойство": "Фактическое значение", "Значение": "5 мм"},
                    {"Свойство": "Оценка OIV", "Значение": "1"},
                    {"Свойство": "Значение шкалы", "Значение": "очень короткая"},
                ],
            )

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_angle = payload["images"][0]["measurements"]["angles"][0]
            saved_segment = payload["images"][0]["measurements"]["segments"][0]
            self.assertEqual(
                saved_angle["assessment"], {"system": "OIV", "code": "OIV 607"}
            )
            self.assertEqual(
                saved_segment["assessment"], {"system": "OIV", "code": "OIV 601"}
            )
            self.assertNotIn("score", saved_angle)
            self.assertNotIn("score", saved_segment)

    def test_oiv_assessment_selection_keeps_current_measurement_group_expanded(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            record = window._selected_project_image()
            self.assertIsNotNone(record)

            measurements_group = window.properties_browser.group_item("measurements")
            angles_group = window.properties_browser.group_item("measurements:angles")
            angle_group = window.properties_browser.group_item("angle:angle-a")
            other_angle_group = window.properties_browser.group_item("angle:angle-b")
            segments_group = window.properties_browser.group_item(
                "measurements:segments"
            )
            segment_group = window.properties_browser.group_item("segment:segment-a")

            measurements_group.setExpanded(True)
            angles_group.setExpanded(True)
            angle_group.setExpanded(True)
            segments_group.setExpanded(True)
            segment_group.setExpanded(True)
            other_angle_group.setExpanded(False)

            angle_combo = window._angle_assessment_fields["angle-a"]
            angle_combo.setCurrentIndex(angle_combo.findData("OIV 607"))

            self.assertEqual(record.measurements.angles[0].assessment.code, "OIV 607")
            self.assertTrue(
                window.properties_browser.group_item("measurements").isExpanded()
            )
            self.assertTrue(
                window.properties_browser.group_item("measurements:angles").isExpanded()
            )
            self.assertTrue(
                window.properties_browser.group_item("angle:angle-a").isExpanded()
            )
            self.assertFalse(
                window.properties_browser.group_item("angle:angle-b").isExpanded()
            )
            self.assertTrue(
                window.properties_browser.is_property_visible(
                    "angle:angle-a:assessment_result"
                )
            )
            self.assertEqual(
                window._angle_assessment_result_fields["angle-a"].text(),
                "90°; оценка 9 - очень большой",
            )

            segment_combo = window._segment_assessment_fields["segment-a"]
            segment_combo.setCurrentIndex(segment_combo.findData("OIV 601"))

            self.assertEqual(record.measurements.segments[0].assessment.code, "OIV 601")
            self.assertTrue(
                window.properties_browser.group_item("measurements").isExpanded()
            )
            self.assertTrue(
                window.properties_browser.group_item(
                    "measurements:segments"
                ).isExpanded()
            )
            self.assertTrue(
                window.properties_browser.group_item("segment:segment-a").isExpanded()
            )
            self.assertTrue(
                window.properties_browser.is_property_visible(
                    "segment:segment-a:assessment_result"
                )
            )
            self.assertEqual(
                window._segment_assessment_result_fields["segment-a"].text(),
                "5 мм; оценка 1 - очень короткая",
            )

    def test_segment_oiv_assessment_requires_calibration(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            record = window._selected_project_image()
            self.assertIsNotNone(record)

            record.calibration = None
            record.measurements.segments[0].assessment = ProjectMeasurementAssessment(
                system="OIV",
                code="OIV 601",
            )
            window._update_project_properties()

            self.assertEqual(
                window._segment_assessment_result_fields["segment-a"].text(),
                "Нужна калибровка",
            )

    def test_image_calibration_scale_round_trip_edit_reset_and_export(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (140, 30), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="calibration",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=140,
                        image_height=30,
                        calibration=ImageCalibration(
                            start=Point(1, 2),
                            end=Point(121, 2),
                            length_mm=10,
                        ),
                    )
                ],
            )
            project_path = root / "calibration.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.canvas.has_calibration())
            self.assertTrue(window.calibrate_scale_action.isEnabled())
            self.assertTrue(window.reset_calibration_action.isEnabled())
            self.assertEqual(
                window.property_calibration_scale.text(),
                "120 px = 10 мм; 12 px/мм; 0.0833 мм/px",
            )

            window.canvas.calibration_handle_moved(1, QPointF(61, 2))

            record = window.project_document.images[0]
            self.assertIsNotNone(record.calibration)
            self.assertEqual(record.calibration.end, Point(61, 2))
            self.assertEqual(
                window.property_calibration_scale.text(),
                "60 px = 10 мм; 6 px/мм; 0.1667 мм/px",
            )

            export_path = root / "calibration_properties.xlsx"
            window._write_image_properties_excel(export_path, ["calibration_scale"])
            self.assertEqual(
                _read_xlsx_dict_rows(export_path),
                [
                    {
                        "Свойство": "Калибровка",
                        "Значение": "",
                    },
                    {
                        "Свойство": "Масштаб",
                        "Значение": "60 px = 10 мм; 6 px/мм; 0.1667 мм/px",
                    },
                ],
            )

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["images"][0]["calibration"],
                {
                    "start": {"x": 1.0, "y": 2.0},
                    "end": {"x": 61.0, "y": 2.0},
                    "length_mm": 10.0,
                },
            )

            window.reset_current_calibration()

            self.assertFalse(window.canvas.has_calibration())
            self.assertIsNone(record.calibration)
            self.assertFalse(window.reset_calibration_action.isEnabled())
            self.assertEqual(window.property_calibration_scale.text(), "-")
            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn("calibration", payload["images"][0])

            with patch(
                "magicborder.main_window.QInputDialog.getDouble",
                return_value=(5.0, False),
            ):
                window._handle_calibration_segment_selected([Point(2, 2), Point(22, 2)])

            self.assertFalse(window.canvas.has_calibration())
            self.assertIsNone(record.calibration)

            with patch(
                "magicborder.main_window.QInputDialog.getDouble",
                return_value=(5.0, True),
            ):
                window._handle_calibration_segment_selected([Point(2, 2), Point(22, 2)])

            self.assertTrue(window.canvas.has_calibration())
            self.assertIsNotNone(record.calibration)
            self.assertEqual(record.calibration.length_mm, 5.0)
            self.assertEqual(
                window.property_calibration_scale.text(),
                "20 px = 5 мм; 4 px/мм; 0.25 мм/px",
            )

    def test_image_contour_area_mm2_updates_with_contour_and_calibration(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (30, 30), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="area_mm2",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=30,
                        image_height=30,
                        annotation=Annotation(
                            image_path="images/leaf.png",
                            image_width=30,
                            image_height=30,
                            points=[
                                Point(1, 1),
                                Point(18, 1),
                                Point(18, 18),
                                Point(1, 18),
                            ],
                        ),
                        calibration=ImageCalibration(
                            start=Point(1, 1),
                            end=Point(19, 1),
                            length_mm=6,
                        ),
                    )
                ],
            )
            project_path = root / "area_mm2.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            # Аналитика считается только по команде «Обновить».
            window.refresh_analysis()

            self.assertEqual(window.property_contour_pixels.text(), "324")
            self.assertEqual(window.property_contour_area_mm2.text(), "36 мм²")
            self.assertEqual(window.property_calibration_length.text(), "6 мм")
            self.assertTrue(window.property_calibration_length.isEnabled())
            self.assertEqual(window.canvas.calibration_label_text(), "6 мм")
            self.assertTrue(window.canvas._calibration_label_item.isVisible())

            window.canvas.set_contour(
                [
                    Point(1, 1),
                    Point(9, 1),
                    Point(9, 9),
                    Point(1, 9),
                ]
            )
            window.refresh_analysis()

            self.assertEqual(window.property_contour_pixels.text(), "81")
            self.assertEqual(window.property_contour_area_mm2.text(), "9 мм²")

            window.property_calibration_length.setText("3,0")
            window._handle_calibration_length_edit_finished()

            record = window.project_document.images[0]
            self.assertIsNotNone(record.calibration)
            self.assertEqual(record.calibration.start, Point(1, 1))
            self.assertEqual(record.calibration.end, Point(19, 1))
            self.assertEqual(record.calibration.length_mm, 3.0)
            self.assertEqual(window.property_calibration_length.text(), "3 мм")
            self.assertEqual(
                window.property_calibration_scale.text(),
                "18 px = 3 мм; 6 px/мм; 0.1667 мм/px",
            )
            self.assertEqual(window.property_contour_area_mm2.text(), "2.25 мм²")
            self.assertEqual(window.canvas.calibration_label_text(), "3 мм")

            with patch.object(window, "_show_warning") as show_warning:
                window.property_calibration_length.setText("0")
                window._handle_calibration_length_edit_finished()

            show_warning.assert_called_once_with(
                "Некорректная калибровка",
                "Длина калибровочного отрезка должна быть положительным числом.",
            )
            self.assertEqual(record.calibration.length_mm, 3.0)
            self.assertEqual(window.property_calibration_length.text(), "3 мм")

            label_position = window.canvas._calibration_label_item.pos()

            window.canvas.calibration_handle_moved(1, QPointF(10, 1))

            self.assertNotEqual(
                window.canvas._calibration_label_item.pos(), label_position
            )
            self.assertEqual(window.canvas.calibration_label_text(), "3 мм")
            self.assertEqual(window.property_contour_area_mm2.text(), "9 мм²")

            export_path = root / "area_properties.xlsx"
            window._write_image_properties_excel(
                export_path,
                ["calibration_length_mm", "contour_area_mm2"],
            )
            self.assertEqual(
                _read_xlsx_dict_rows(export_path),
                [
                    {
                        "Свойство": "Калибровка",
                        "Значение": "",
                    },
                    {
                        "Свойство": "Длина калибровки, мм",
                        "Значение": "3 мм",
                    },
                    {
                        "Свойство": "Информация о главном контуре",
                        "Значение": "",
                    },
                    {
                        "Свойство": "Площадь контура, мм²",
                        "Значение": "9 мм²",
                    },
                ],
            )

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn("contour_area_mm2", payload["images"][0])
            self.assertEqual(payload["images"][0]["calibration"]["length_mm"], 3.0)

            window.reset_current_calibration()

            self.assertEqual(
                window.property_contour_area_mm2.text(),
                "калибровка не произведена",
            )
            self.assertEqual(
                window.property_calibration_length.text(),
                "калибровка не произведена",
            )
            self.assertFalse(window.property_calibration_length.isEnabled())
            self.assertFalse(window.canvas._calibration_label_item.isVisible())

    def test_image_properties_lab_values_update_export_and_stay_out_of_json(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            image = Image.new("RGB", (20, 20), (10, 20, 30))
            for x in range(10):
                for y in range(20):
                    image.putpixel((x, y), (120, 80, 40))
            image.save(image_dir / "leaf.png")

            left_contour = [
                Point(1, 1),
                Point(8, 1),
                Point(8, 18),
                Point(1, 18),
            ]
            project = ProjectDocument(
                name="lab_values",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                        annotation=Annotation(
                            image_path="images/leaf.png",
                            image_width=20,
                            image_height=20,
                            points=left_contour,
                        ),
                    )
                ],
            )
            project_path = root / "lab_values.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            # Аналитика считается только по команде «Обновить».
            window.refresh_analysis()

            self.assertEqual(window.property_lab_l.text(), "38")
            self.assertEqual(window.property_lab_a.text(), "12")
            self.assertEqual(window.property_lab_b.text(), "30")
            self.assertEqual(window.property_hsv_h.text(), "30")
            self.assertEqual(window.property_hsv_s.text(), "170")
            self.assertEqual(window.property_hsv_v.text(), "120")
            self.assertEqual(window.property_yuv_y.text(), "87")
            self.assertEqual(window.property_yuv_u.text(), "105")
            self.assertEqual(window.property_yuv_v.text(), "157")
            self.assertEqual(window.property_lms_l.text(), "255")
            self.assertEqual(window.property_lms_m.text(), "255")
            self.assertEqual(window.property_lms_s.text(), "255")

            window.canvas.set_contour(
                [
                    Point(11, 1),
                    Point(18, 1),
                    Point(18, 18),
                    Point(11, 18),
                ]
            )
            window.refresh_analysis()

            self.assertEqual(window.property_lab_l.text(), "6")
            self.assertEqual(window.property_lab_a.text(), "0")
            self.assertEqual(window.property_lab_b.text(), "-8")
            self.assertEqual(window.property_hsv_h.text(), "210")
            self.assertEqual(window.property_hsv_s.text(), "170")
            self.assertEqual(window.property_hsv_v.text(), "30")
            self.assertEqual(window.property_yuv_y.text(), "18")
            self.assertEqual(window.property_yuv_u.text(), "134")
            self.assertEqual(window.property_yuv_v.text(), "121")
            self.assertEqual(window.property_lms_l.text(), "255")
            self.assertEqual(window.property_lms_m.text(), "255")
            self.assertEqual(window.property_lms_s.text(), "255")

            export_path = root / "lab_properties.xlsx"
            window._write_image_properties_excel(
                export_path,
                [
                    "lab_l",
                    "lab_a",
                    "lab_b",
                    "hsv_h",
                    "hsv_s",
                    "hsv_v",
                    "yuv_y",
                    "yuv_u",
                    "yuv_v",
                    "lms_l",
                    "lms_m",
                    "lms_s",
                ],
            )
            self.assertEqual(
                _read_xlsx_dict_rows(export_path),
                [
                    {"Свойство": "Цветовые пространства", "Значение": ""},
                    {"Свойство": "Цветовое пространство Lab", "Значение": ""},
                    {"Свойство": "L", "Значение": "6"},
                    {"Свойство": "a", "Значение": "0"},
                    {"Свойство": "b", "Значение": "-8"},
                    {"Свойство": "Цветовое пространство HSV", "Значение": ""},
                    {"Свойство": "H", "Значение": "210"},
                    {"Свойство": "S", "Значение": "170"},
                    {"Свойство": "V", "Значение": "30"},
                    {"Свойство": "Цветовое пространство YUV", "Значение": ""},
                    {"Свойство": "Y", "Значение": "18"},
                    {"Свойство": "U", "Значение": "134"},
                    {"Свойство": "V", "Значение": "121"},
                    {"Свойство": "Цветовое пространство LMS", "Значение": ""},
                    {"Свойство": "L", "Значение": "255"},
                    {"Свойство": "M", "Значение": "255"},
                    {"Свойство": "S", "Значение": "255"},
                ],
            )

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn("lab_l", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("lab_a", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("lab_b", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("hsv_h", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("yuv_y", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("lms_l", json.dumps(payload, ensure_ascii=False))

    def test_project_properties_save_general_info_and_count_images(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = ProjectDocument(
                name="project_info",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                    ),
                    ProjectImageRecord(
                        id="leaf-2",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                    ),
                ],
            )
            project_path = root / "project_info.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.project_image_count.text(), "2")
            window.project_general_info.setPlainText("Серия измерений 2026")
            window.save_project_file()

            loaded_project = load_project(project_path)
            self.assertEqual(
                loaded_project.project_info.general_info, "Серия измерений 2026"
            )

            reopened = MainWindow()
            reopened._set_project(project_path, load_project(project_path))
            self.assertEqual(
                reopened.project_general_info.toPlainText(), "Серия измерений 2026"
            )
            self.assertEqual(reopened.project_image_count.text(), "2")
            self.assertEqual(reopened.project_mean_lab_l.text(), "-")
            self.assertEqual(reopened.project_mean_lab_a.text(), "-")
            self.assertEqual(reopened.project_mean_lab_b.text(), "-")
            self.assertEqual(reopened.project_mean_hsv_h.text(), "-")
            self.assertEqual(reopened.project_mean_hsv_s.text(), "-")
            self.assertEqual(reopened.project_mean_hsv_v.text(), "-")
            self.assertEqual(reopened.project_mean_yuv_y.text(), "-")
            self.assertEqual(reopened.project_mean_yuv_u.text(), "-")
            self.assertEqual(reopened.project_mean_yuv_v.text(), "-")
            self.assertEqual(reopened.project_mean_lms_l.text(), "-")
            self.assertEqual(reopened.project_mean_lms_m.text(), "-")
            self.assertEqual(reopened.project_mean_lms_s.text(), "-")

    def test_project_properties_average_rgb_uses_all_available_contours(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (100, 50, 0)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "leaf_b.png")
            Image.new("RGB", (20, 20), (200, 200, 200)).save(
                image_dir / "no_contour.png"
            )

            contour = [
                Point(1, 1),
                Point(18, 1),
                Point(18, 18),
                Point(1, 18),
            ]
            project = ProjectDocument(
                name="project_rgb",
                images=[
                    ProjectImageRecord(
                        id="leaf-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        image_width=20,
                        image_height=20,
                        annotation=Annotation(
                            image_path="images/leaf_a.png",
                            image_width=20,
                            image_height=20,
                            points=contour,
                        ),
                    ),
                    ProjectImageRecord(
                        id="leaf-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                        image_width=20,
                        image_height=20,
                        annotation=Annotation(
                            image_path="images/leaf_b.png",
                            image_width=20,
                            image_height=20,
                            points=contour,
                        ),
                    ),
                    ProjectImageRecord(
                        id="no-contour",
                        relative_path="images/no_contour.png",
                        display_name="no_contour.png",
                    ),
                    ProjectImageRecord(
                        id="missing",
                        relative_path="images/missing.png",
                        display_name="missing.png",
                        annotation=Annotation(
                            image_path="images/missing.png",
                            image_width=20,
                            image_height=20,
                            points=contour,
                        ),
                    ),
                ],
            )
            project_path = root / "project_rgb.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            # Аналитика считается только по команде «Обновить».
            window.refresh_analysis()

            self.assertEqual(window.project_mean_red.text(), "55")
            self.assertEqual(window.project_mean_green.text(), "35")
            self.assertEqual(window.project_mean_blue.text(), "15")
            self.assertEqual(window.project_mean_lab_l.text(), "16")
            self.assertEqual(window.project_mean_lab_a.text(), "10")
            self.assertEqual(window.project_mean_lab_b.text(), "14")
            self.assertEqual(window.project_mean_hsv_h.text(), "120")
            self.assertEqual(window.project_mean_hsv_s.text(), "212")
            self.assertEqual(window.project_mean_hsv_v.text(), "65")
            self.assertEqual(window.project_mean_yuv_y.text(), "38")
            self.assertEqual(window.project_mean_yuv_u.text(), "116")
            self.assertEqual(window.project_mean_yuv_v.text(), "142")
            self.assertEqual(window.project_mean_lms_l.text(), "137")
            self.assertEqual(window.project_mean_lms_m.text(), "152")
            self.assertEqual(window.project_mean_lms_s.text(), "180")

            window.canvas.set_contour(
                [
                    Point(1, 1),
                    Point(4, 1),
                    Point(4, 4),
                    Point(1, 4),
                ]
            )
            window.refresh_analysis()

            self.assertEqual(window.project_mean_red.text(), "14")
            self.assertEqual(window.project_mean_green.text(), "21")
            self.assertEqual(window.project_mean_blue.text(), "29")
            self.assertEqual(window.project_mean_lab_l.text(), "7")
            self.assertEqual(window.project_mean_lab_a.text(), "1")
            self.assertEqual(window.project_mean_lab_b.text(), "-6")
            self.assertEqual(window.project_mean_hsv_h.text(), "202")
            self.assertEqual(window.project_mean_hsv_s.text(), "174")
            self.assertEqual(window.project_mean_hsv_v.text(), "33")
            self.assertEqual(window.project_mean_yuv_y.text(), "20")
            self.assertEqual(window.project_mean_yuv_u.text(), "132")
            self.assertEqual(window.project_mean_yuv_v.text(), "123")
            self.assertEqual(window.project_mean_lms_l.text(), "30")
            self.assertEqual(window.project_mean_lms_m.text(), "58")
            self.assertEqual(window.project_mean_lms_s.text(), "248")

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn(
                "project_mean_lab_l", json.dumps(payload, ensure_ascii=False)
            )
            self.assertNotIn(
                "project_mean_lab_a", json.dumps(payload, ensure_ascii=False)
            )
            self.assertNotIn(
                "project_mean_lab_b", json.dumps(payload, ensure_ascii=False)
            )
            self.assertNotIn(
                "project_mean_hsv_h", json.dumps(payload, ensure_ascii=False)
            )
            self.assertNotIn(
                "project_mean_yuv_y", json.dumps(payload, ensure_ascii=False)
            )
            self.assertNotIn(
                "project_mean_lms_l", json.dumps(payload, ensure_ascii=False)
            )

    def test_project_list_item_color_tracks_annotation_state(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="list_colors",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                    )
                ],
            )
            project_path = root / "list_colors.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertIn("без аннотации", window.project_list.item(0).text())
            self.assertEqual(_list_item_color_name(window), "#b42318")

            window.canvas.set_contour(
                [
                    Point(1, 1),
                    Point(18, 1),
                    Point(18, 18),
                    Point(1, 18),
                ]
            )

            self.assertIn("аннотация есть", window.project_list.item(0).text())
            self.assertEqual(_list_item_color_name(window), "#1f2937")

            window.delete_current_contour()

            self.assertIn("без аннотации", window.project_list.item(0).text())
            self.assertEqual(_list_item_color_name(window), "#b42318")

    def test_project_list_item_color_marks_annotation_errors_red(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project_path = root / "list_error.json"
            project_path.write_text(
                json.dumps(
                    {
                        "name": "list_error",
                        "images": [
                            {
                                "file": {
                                    "id": "leaf-1",
                                    "path": "images/leaf.png",
                                    "display_name": "leaf.png",
                                    "image_size": {"width": 20, "height": 20},
                                },
                                "contour": {
                                    "annotation": {
                                        "image_path": "images/leaf.png",
                                        "image_size": {"width": 20, "height": 20},
                                        "points": [],
                                    },
                                },
                                "location": {},
                                "details": {},
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertIn("ошибка аннотации", window.project_list.item(0).text())
            self.assertEqual(_list_item_color_name(window), "#b42318")

    def test_new_contour_command_creates_project_annotation(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="new_contour",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                    )
                ],
            )
            project_path = root / "new_contour.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.new_contour_action.isEnabled())
            with patch(
                "magicborder.main_window.QInputDialog.getInt", return_value=(5, True)
            ):
                window.create_new_contour()

            annotation = window.project_document.images[0].annotation
            self.assertIsNotNone(annotation)
            self.assertEqual(len(annotation.points), 5)
            self.assertEqual(window.property_points.text(), "5")

    def test_metadata_fields_save_and_file_name_renames_image(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="metadata",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                    )
                ],
            )
            project_path = root / "metadata.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            window.metadata_humidity.setText("65")
            window._handle_metadata_line_edit_finished(
                "humidity", window.metadata_humidity
            )
            window.metadata_latitude.setText("48.7")
            window._handle_metadata_line_edit_finished(
                "latitude", window.metadata_latitude
            )
            window.metadata_notes.setPlainText("Проверка метаданных")
            window.property_file_name.setText("renamed")
            window._handle_file_name_edit_finished()
            window.save_project_file()

            payload = json.loads(project_path.read_text(encoding="utf-8"))
            loaded_project = load_project(project_path)
            record = loaded_project.images[0]
            record_payload = payload["images"][0]

            self.assertNotIn("metadata", record_payload)
            self.assertEqual(record_payload["file"]["display_name"], "renamed.png")
            self.assertEqual(record_payload["file"]["path"], "images/renamed.png")
            self.assertEqual(record_payload["location"]["humidity"], "65")
            self.assertEqual(record_payload["location"]["latitude"], "48.7")
            self.assertEqual(record_payload["details"]["notes"], "Проверка метаданных")
            self.assertEqual(record.display_name, "renamed.png")
            self.assertEqual(record.relative_path, "images/renamed.png")
            self.assertTrue((image_dir / "renamed.png").exists())
            self.assertFalse((image_dir / "leaf.png").exists())
            self.assertEqual(record.metadata["humidity"], "65")
            self.assertEqual(record.metadata["latitude"], "48.7")
            self.assertEqual(record.metadata["notes"], "Проверка метаданных")

    def test_file_name_button_renames_image_to_record_id(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="rename_as_id",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    )
                ],
            )
            project_path = root / "rename_as_id.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(
                window.rename_file_as_id_button.toolTip(), "Переименовать файл как ID"
            )
            self.assertTrue(window.rename_file_as_id_button.isEnabled())

            window.image_id.setText("SAMPLE_001")
            window._rename_file_to_image_id()
            window.save_project_file()

            loaded_project = load_project(project_path)
            record = loaded_project.images[0]

            self.assertEqual(record.display_name, "SAMPLE_001.png")
            self.assertEqual(record.relative_path, "images/SAMPLE_001.png")
            self.assertEqual(record.id, "SAMPLE_001")
            self.assertNotIn("sample_id", record.metadata)
            self.assertTrue((image_dir / "SAMPLE_001.png").exists())
            self.assertFalse((image_dir / "leaf.png").exists())

    def test_generate_image_id_button_creates_new_uuid(self) -> None:
        _app()
        window = MainWindow()

        self.assertFalse(window.generate_image_id_button.isEnabled())
        self.assertEqual(window.generate_image_id_button.toolTip(), "Сгенерировать ID")
        self.assertEqual(
            window.generate_image_id_button.statusTip(),
            "Сгенерировать новый GUID изображения.",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "other.png")

            project = ProjectDocument(
                name="generate_id",
                images=[
                    ProjectImageRecord(
                        id="11111111-1111-4111-8111-111111111111",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    ),
                    ProjectImageRecord(
                        id="22222222-2222-4222-8222-222222222222",
                        relative_path="images/other.png",
                        display_name="other.png",
                    ),
                ],
            )
            project_path = root / "generate_id.json"
            save_project(project_path, project)

            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.generate_image_id_button.isEnabled())
            old_id = window.project_document.images[0].id

            window._generate_image_id()
            first_generated_id = window.project_document.images[0].id
            window._generate_image_id()
            second_generated_id = window.project_document.images[0].id
            window.save_project_file()

            loaded_project = load_project(project_path)
            self.assertNotEqual(first_generated_id, old_id)
            self.assertNotEqual(second_generated_id, first_generated_id)
            _assert_uuid4(self, first_generated_id)
            _assert_uuid4(self, second_generated_id)
            self.assertEqual(window.image_id.text(), second_generated_id)
            self.assertEqual(loaded_project.images[0].id, second_generated_id)
            self.assertEqual(
                loaded_project.images[1].id, "22222222-2222-4222-8222-222222222222"
            )
            self.assertNotIn("sample_id", loaded_project.images[0].metadata)

            output_path = root / "generated_ids"
            with (
                patch.object(
                    window,
                    "_select_project_export_columns",
                    return_value=PROJECT_EXPORT_FIELDNAMES,
                ),
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), ""),
                ),
            ):
                window.export_project_excel()

            xlsx_path = root / "generated_ids.xlsx"
            rows = _read_xlsx_dict_rows(xlsx_path)
            self.assertEqual(rows[0]["ID изображения"], second_generated_id)

    def test_record_id_is_editable_and_unique(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "leaf_b.png")

            project = ProjectDocument(
                name="ids",
                images=[
                    ProjectImageRecord(
                        id="A",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                    ),
                    ProjectImageRecord(
                        id="B",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                    ),
                ],
            )
            project_path = root / "ids.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.image_id.text(), "A")

            window.image_id.setText("B")
            with patch.object(window, "_show_warning"):
                window._handle_image_id_edit_finished()
            self.assertEqual(window.image_id.text(), "A")
            self.assertEqual(window.project_document.images[0].id, "A")

            window.image_id.setText("C")
            window._handle_image_id_edit_finished()
            window.save_project_file()

            loaded_project = load_project(project_path)
            self.assertEqual(loaded_project.images[0].id, "C")
            self.assertEqual(window.project_list.currentItem().data(Qt.UserRole), "C")

    def test_project_excel_export_contains_contour_statistics(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "leaf_b.png")

            annotation = Annotation(
                image_path="images/leaf_a.png",
                image_width=20,
                image_height=20,
                points=[
                    Point(1, 1),
                    Point(18, 1),
                    Point(18, 18),
                    Point(1, 18),
                ],
            )
            project = ProjectDocument(
                name="excel",
                images=[
                    ProjectImageRecord(
                        id="record-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        annotation=annotation,
                        metadata={"diagnosis": "class_a"},
                    ),
                    ProjectImageRecord(
                        id="record-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                    ),
                    ProjectImageRecord(
                        id="record-missing",
                        relative_path="images/missing.png",
                        display_name="missing.png",
                    ),
                ],
            )
            project_path = root / "excel.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            output_path = root / "report"

            self.assertEqual(
                window.export_project_excel_action.text(), "Экспорт списка в Excel..."
            )
            self.assertEqual(
                window.export_project_excel_action.toolTip(), "Экспорт списка в Excel"
            )
            self.assertEqual(
                window.export_project_excel_button.toolTip(), "Экспорт списка в Excel"
            )

            with (
                patch.object(
                    window,
                    "_select_project_export_columns",
                    return_value=PROJECT_EXPORT_FIELDNAMES,
                ),
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), ""),
                ) as get_save_file_name,
            ):
                window.export_project_excel()

            self.assertEqual(
                get_save_file_name.call_args.args[1], "Экспорт списка в Excel"
            )
            xlsx_path = root / "report.xlsx"
            raw_rows = _read_xlsx_rows(xlsx_path)
            rows = _read_xlsx_dict_rows(xlsx_path)

            self.assertEqual(
                raw_rows[0],
                [
                    "ID изображения",
                    "Имя файла",
                    "Относительный путь",
                    "Есть аннотация",
                    "Статус",
                    "Диагноз",
                    "Средний R",
                    "Средний G",
                    "Средний B",
                    "Количество пикселов контура",
                ],
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["ID изображения"], "record-a")
            self.assertEqual(rows[0]["Имя файла"], "leaf_a.png")
            self.assertEqual(rows[0]["Диагноз"], "class_a")
            self.assertEqual(rows[0]["Средний R"], "120")
            self.assertEqual(rows[0]["Средний G"], "80")
            self.assertEqual(rows[0]["Средний B"], "40")
            self.assertEqual(rows[0]["Количество пикселов контура"], "324")
            self.assertEqual(rows[1]["Статус"], "нет контура")
            self.assertEqual(rows[1]["Средний R"], "")
            self.assertEqual(rows[2]["Статус"], "файл не найден")

    def test_project_excel_export_uses_selected_columns_only(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)
            output_path = root / "selected"

            with (
                patch.object(
                    window,
                    "_select_project_export_columns",
                    return_value=["file_name", "diagnosis"],
                ),
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), ""),
                ),
            ):
                window.export_project_excel()

            xlsx_path = root / "selected.xlsx"
            raw_rows = _read_xlsx_rows(xlsx_path)
            rows = _read_xlsx_dict_rows(xlsx_path)

            self.assertEqual(raw_rows[0], ["Имя файла", "Диагноз"])
            self.assertEqual(rows, [{"Имя файла": "missing.png", "Диагноз": "class_x"}])

    def test_project_excel_export_cancel_column_selection_does_not_open_file_dialog(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)

            with (
                patch.object(
                    window, "_select_project_export_columns", return_value=None
                ),
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                ) as get_save_file_name,
            ):
                window.export_project_excel()

            get_save_file_name.assert_not_called()

    def test_project_excel_export_requires_at_least_one_column(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)

            with (
                patch.object(window, "_select_project_export_columns", return_value=[]),
                patch.object(
                    window,
                    "_show_warning",
                ) as show_warning,
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                ) as get_save_file_name,
            ):
                window.export_project_excel()

            show_warning.assert_called_once_with(
                "Нет выбранных столбцов",
                "Выберите хотя бы один столбец для экспорта.",
            )
            get_save_file_name.assert_not_called()

    def test_project_excel_column_dialog_selects_all_columns_by_default(self) -> None:
        _app()
        window = MainWindow()

        with patch(
            "magicborder.main_window.QDialog.exec_", return_value=QDialog.Accepted
        ):
            selected_fieldnames = window._select_project_export_columns()

        self.assertEqual(selected_fieldnames, PROJECT_EXPORT_FIELDNAMES)

    def test_image_properties_excel_export_uses_selected_properties_with_group_rows(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)
            output_path = root / "properties"

            self.assertTrue(window.export_image_properties_excel_action.isEnabled())
            self.assertTrue(window.export_image_properties_excel_button.isEnabled())
            self.assertEqual(
                window.export_image_properties_excel_action.text(),
                "Экспорт свойств изображения в Excel...",
            )
            self.assertEqual(
                window.export_image_properties_excel_button.toolTip(),
                "Экспорт свойств изображения в Excel",
            )

            with (
                patch.object(
                    window,
                    "_select_image_property_export_items",
                    return_value=["file_name", "diagnosis", "status"],
                ),
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                    return_value=(str(output_path), ""),
                ) as get_save_file_name,
            ):
                window.export_image_properties_excel()

            self.assertEqual(
                get_save_file_name.call_args.args[1],
                "Экспорт свойств изображения в Excel",
            )
            xlsx_path = root / "properties.xlsx"
            raw_rows = _read_xlsx_rows(xlsx_path)
            rows = _read_xlsx_dict_rows(xlsx_path)

            self.assertEqual(raw_rows[0], ["Свойство", "Значение"])
            self.assertEqual(
                rows,
                [
                    {"Свойство": "Общая информация о файле", "Значение": ""},
                    {"Свойство": "Файл", "Значение": "missing.png"},
                    {"Свойство": "Статус", "Значение": "отсутствует"},
                    {"Свойство": "Дополнительно", "Значение": ""},
                    {"Свойство": "Диагноз", "Значение": "class_x"},
                ],
            )

    def test_image_properties_excel_export_cancel_property_selection_does_not_open_file_dialog(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)

            with (
                patch.object(
                    window, "_select_image_property_export_items", return_value=None
                ),
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                ) as get_save_file_name,
            ):
                window.export_image_properties_excel()

            get_save_file_name.assert_not_called()

    def test_image_properties_excel_export_requires_at_least_one_property(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)

            with (
                patch.object(
                    window, "_select_image_property_export_items", return_value=[]
                ),
                patch.object(
                    window,
                    "_show_warning",
                ) as show_warning,
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                ) as get_save_file_name,
            ):
                window.export_image_properties_excel()

            show_warning.assert_called_once_with(
                "Нет выбранных свойств",
                "Выберите хотя бы одно свойство для экспорта.",
            )
            get_save_file_name.assert_not_called()

    def test_image_properties_excel_export_requires_selected_image(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_path = root / "empty.json"
            save_project(project_path, ProjectDocument(name="empty", images=[]))

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertFalse(window.export_image_properties_excel_action.isEnabled())
            self.assertFalse(window.export_image_properties_excel_button.isEnabled())

            with (
                patch.object(window, "_show_warning") as show_warning,
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                ) as get_save_file_name,
            ):
                window.export_image_properties_excel()

            show_warning.assert_called_once_with(
                "Нет выбранного изображения",
                "Выберите изображение в списке проекта.",
            )
            get_save_file_name.assert_not_called()

    def test_image_properties_export_dialog_selects_file_info_by_default(self) -> None:
        _app()
        window = MainWindow()

        def accept_and_check_defaults(dialog: QDialog) -> int:
            tree = dialog.findChild(QTreeWidget, "exportItemTree")
            self.assertIsNotNone(tree)
            self.assertEqual(
                _tree_item_by_text(tree, "Общая информация о файле").checkState(0),
                Qt.Checked,
            )
            self.assertEqual(
                _tree_item_by_text(tree, "Калибровка").checkState(0), Qt.Unchecked
            )
            self.assertEqual(
                _tree_item_by_text(tree, "Информация о главном контуре").checkState(0),
                Qt.Unchecked,
            )
            self.assertEqual(
                _tree_item_by_text(tree, "Цветовые пространства").checkState(0),
                Qt.Unchecked,
            )
            return QDialog.Accepted

        with patch(
            "magicborder.main_window.QDialog.exec_", new=accept_and_check_defaults
        ):
            selected_properties = window._select_image_property_export_items()

        self.assertEqual(
            selected_properties,
            [
                "id",
                "file_name",
                "relative_path",
                "size",
                "status",
                "added_at",
                "captured_at",
            ],
        )

    def test_image_properties_export_tree_omits_visibility_properties(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            record = window._selected_project_image()
            self.assertIsNotNone(record)

            keys = window._image_property_export_leaf_keys(record)
            labels = set(window._image_property_export_labels(record).values())

            self.assertNotIn("contour_visible", keys)
            self.assertNotIn("angle:angle-a:visible", keys)
            self.assertNotIn("segment:segment-a:visible", keys)
            self.assertNotIn("Показывать контур", labels)
            self.assertNotIn("Контрольный угол / Показывать на канвасе", labels)
            self.assertNotIn("Контрольный отрезок / Показывать на канвасе", labels)
            for key in (
                "red",
                "green",
                "blue",
                "average_color",
                "lab_l",
                "lab_a",
                "lab_b",
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "yuv_y",
                "yuv_u",
                "yuv_v",
                "lms_l",
                "lms_m",
                "lms_s",
            ):
                self.assertIn(key, keys)

    def test_image_properties_excel_export_contains_measurement_properties(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, project_path = _measurement_export_window(root)
            window.canvas.set_angle_measurement_visible("angle-a", False)
            window.canvas.set_segment_measurement_visible("segment-a", False)

            export_path = root / "measurement_properties.xlsx"
            window._write_image_properties_excel(
                export_path,
                [
                    "angle:angle-a:id",
                    "angle:angle-a:value",
                    "angle:angle-a:note",
                    "segment:segment-a:length",
                    "segment:segment-a:start_label",
                    "segment:segment-a:end_label",
                    "segment:segment-a:note",
                ],
            )

            self.assertEqual(
                _read_xlsx_dict_rows(export_path),
                [
                    {"Свойство": "Измерения", "Значение": ""},
                    {"Свойство": "Углы", "Значение": ""},
                    {"Свойство": "Контрольный угол", "Значение": ""},
                    {"Свойство": "ID", "Значение": "angle-a"},
                    {"Свойство": "Значение", "Значение": "90°"},
                    {"Свойство": "Примечание", "Значение": "контрольный угол"},
                    {"Свойство": "Отрезки", "Значение": ""},
                    {"Свойство": "Контрольный отрезок", "Значение": ""},
                    {"Свойство": "Длина", "Значение": "10 px / 5 мм"},
                    {"Свойство": "Подпись первой точки", "Значение": "Начало"},
                    {"Свойство": "Подпись второй точки", "Значение": "Конец"},
                    {"Свойство": "Примечание", "Значение": "измерить повторно"},
                ],
            )
            cell_styles = _read_xlsx_cell_styles(export_path)
            self.assertIn("<b/>", _read_xlsx_styles_xml(export_path))
            for cell_ref in ("A2", "B2", "A3", "A4", "A8", "A9"):
                self.assertEqual(cell_styles[cell_ref], "1")
            for cell_ref in ("A1", "A5", "A6", "A7", "A10", "A11"):
                self.assertEqual(cell_styles[cell_ref], "")

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn(
                "visible", payload["images"][0]["measurements"]["angles"][0]
            )
            self.assertNotIn(
                "visible", payload["images"][0]["measurements"]["segments"][0]
            )

    def test_image_properties_export_keys_skip_placeholder_and_action_rows(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            record = window._selected_project_image()
            self.assertIsNotNone(record)

            keys = window._image_property_export_leaf_keys(record)
            labels = set(window._image_property_export_labels(record).values())

            self.assertIn("angle:angle-a:id", keys)
            self.assertIn("segment:segment-a:length", keys)
            self.assertNotIn("Управление", labels)
            self.assertNotIn("Нет углов", labels)
            self.assertNotIn("Нет отрезков", labels)

    def test_image_properties_export_dialog_toggles_whole_groups(self) -> None:
        _app()
        window = MainWindow()
        observed_states: dict[str, Qt.CheckState] = {}

        def check_rgb_group(dialog: QDialog) -> int:
            tree = dialog.findChild(QTreeWidget, "exportItemTree")
            self.assertIsNotNone(tree)
            _tree_item_by_text(tree, "Цветовое пространство RGB").setCheckState(
                0, Qt.Checked
            )
            observed_states["color_spaces"] = _tree_item_by_text(
                tree, "Цветовые пространства"
            ).checkState(0)
            return QDialog.Accepted

        with patch("magicborder.main_window.QDialog.exec_", new=check_rgb_group):
            selected_properties = window._select_image_property_export_items()

        self.assertEqual(observed_states["color_spaces"], Qt.PartiallyChecked)
        self.assertIn("red", selected_properties)
        self.assertIn("green", selected_properties)
        self.assertIn("blue", selected_properties)
        self.assertIn("average_color", selected_properties)
        self.assertIn("file_name", selected_properties)
        self.assertNotIn("lab_l", selected_properties)
        self.assertNotIn("annotation", selected_properties)

    def test_image_properties_export_dialog_toggles_color_spaces_parent_group(
        self,
    ) -> None:
        _app()
        window = MainWindow()

        def check_color_spaces_group(dialog: QDialog) -> int:
            tree = dialog.findChild(QTreeWidget, "exportItemTree")
            self.assertIsNotNone(tree)
            _tree_item_by_text(tree, "Цветовые пространства").setCheckState(
                0, Qt.Checked
            )
            return QDialog.Accepted

        with patch(
            "magicborder.main_window.QDialog.exec_", new=check_color_spaces_group
        ):
            selected_properties = window._select_image_property_export_items()

        for key in (
            "red",
            "green",
            "blue",
            "average_color",
            "lab_l",
            "lab_a",
            "lab_b",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "yuv_y",
            "yuv_u",
            "yuv_v",
            "lms_l",
            "lms_m",
            "lms_s",
        ):
            self.assertIn(key, selected_properties)
        self.assertIn("file_name", selected_properties)
        self.assertNotIn("annotation", selected_properties)

    def test_image_properties_excel_export_groups_color_space_rows_under_parent(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)
            window.property_red.setText("10")
            window.property_green.setText("20")
            window.property_blue.setText("30")

            export_path = root / "rgb_properties.xlsx"
            window._write_image_properties_excel(
                export_path,
                ["red", "green", "blue", "average_color"],
            )

            self.assertEqual(
                _read_xlsx_dict_rows(export_path),
                [
                    {"Свойство": "Цветовые пространства", "Значение": ""},
                    {"Свойство": "Цветовое пространство RGB", "Значение": ""},
                    {"Свойство": "Красный", "Значение": "10"},
                    {"Свойство": "Зелёный", "Значение": "20"},
                    {"Свойство": "Синий", "Значение": "30"},
                    {"Свойство": "Средний цвет", "Значение": "RGB(10, 20, 30)"},
                ],
            )
            cell_styles = _read_xlsx_cell_styles(export_path)
            for cell_ref in ("A2", "B2", "A3", "B3"):
                self.assertEqual(cell_styles[cell_ref], "1")
            for cell_ref in ("A4", "B4", "A5", "B5", "A6", "B6", "A7", "B7"):
                self.assertEqual(cell_styles[cell_ref], "")

    def test_image_properties_excel_export_adds_average_color_fill_cell(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)
            window.property_red.setText("10")
            window.property_green.setText("20")
            window.property_blue.setText("30")
            window._current_project_image_id = "row-1"
            window._set_average_color_swatch((10, 20, 30))

            export_path = root / "rgb_fill_properties.xlsx"
            window._write_image_properties_excel(
                export_path,
                ["red", "green", "blue", "average_color"],
            )

            self.assertEqual(
                _read_xlsx_rows(export_path)[0], ["Свойство", "Значение", "Цвет"]
            )
            rows = _read_xlsx_dict_rows(export_path)
            self.assertEqual(
                rows[-1],
                {"Свойство": "Средний цвет", "Значение": "RGB(10, 20, 30)", "Цвет": ""},
            )
            cell_styles = _read_xlsx_cell_styles(export_path)
            self.assertNotEqual(cell_styles["C7"], "")
            styles_xml = _read_xlsx_styles_xml(export_path)
            self.assertIn('patternType="solid"', styles_xml)
            self.assertIn('rgb="FF0A141E"', styles_xml)
            for cell_ref in ("A2", "B2", "A3", "B3"):
                self.assertEqual(cell_styles[cell_ref], "1")

    def test_average_color_swatch_saves_png_sample(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            _add_test_contour(window)

            output_path = root / "average_sample"
            with patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
                return_value=(str(output_path), ""),
            ):
                window.save_average_color_sample()

            saved_path = root / "average_sample.png"
            with Image.open(saved_path) as image:
                self.assertEqual(image.size, (100, 100))
                self.assertEqual(image.getpixel((0, 0)), (120, 80, 40))
                self.assertEqual(image.getpixel((99, 99)), (120, 80, 40))

    def test_average_color_swatch_requires_calculated_color(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)

            with (
                patch.object(window, "_show_warning") as show_warning,
                patch(
                    "magicborder.main_window.QFileDialog.getSaveFileName",
                ) as get_save_file_name,
            ):
                window.save_average_color_sample()

            show_warning.assert_called_once_with(
                "Средний цвет недоступен",
                "Средний цвет ещё не рассчитан. Сначала создайте контур и дождитесь расчёта.",
            )
            get_save_file_name.assert_not_called()

    def test_flatten_background_confirms_and_saves_current_image_file(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            _add_test_contour(window)
            image_path = root / "images" / "leaf.png"

            with patch(
                "magicborder.main_window.QMessageBox.question",
                return_value=QMessageBox.Yes,
            ):
                window.flatten_background()

            with Image.open(image_path) as image:
                self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(image.convert("RGB").getpixel((10, 10)), (120, 80, 40))
            canvas_rgb = window.canvas.current_rgb_array()
            self.assertEqual(
                tuple(int(value) for value in canvas_rgb[0, 0]), (255, 255, 255)
            )
            self.assertEqual(
                tuple(int(value) for value in canvas_rgb[10, 10]), (120, 80, 40)
            )

    def test_flatten_background_survives_switching_images(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (40, 30), (120, 80, 40)).save(image_dir / "leaf-a.png")
            Image.new("RGB", (40, 30), (20, 60, 100)).save(image_dir / "leaf-b.png")

            project = ProjectDocument(
                name="switch_flatten",
                images=[
                    ProjectImageRecord(
                        id="leaf-a",
                        relative_path="images/leaf-a.png",
                        display_name="leaf-a.png",
                        image_width=40,
                        image_height=30,
                    ),
                    ProjectImageRecord(
                        id="leaf-b",
                        relative_path="images/leaf-b.png",
                        display_name="leaf-b.png",
                        image_width=40,
                        image_height=30,
                    ),
                ],
            )
            project_path = root / "switch_flatten.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            _add_test_contour(window)

            with patch(
                "magicborder.main_window.QMessageBox.question",
                return_value=QMessageBox.Yes,
            ):
                window.flatten_background()

            window.project_list.setCurrentRow(1)
            window.project_list.setCurrentRow(0)

            canvas_rgb = window.canvas.current_rgb_array()
            self.assertEqual(
                tuple(int(value) for value in canvas_rgb[0, 0]), (255, 255, 255)
            )
            self.assertEqual(
                tuple(int(value) for value in canvas_rgb[10, 10]), (120, 80, 40)
            )

    def test_flatten_background_cancel_leaves_canvas_and_file_unchanged(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            _add_test_contour(window)
            image_path = root / "images" / "leaf.png"

            with patch(
                "magicborder.main_window.QMessageBox.question",
                return_value=QMessageBox.No,
            ):
                window.flatten_background()

            with Image.open(image_path) as image:
                self.assertEqual(image.convert("RGB").getpixel((0, 0)), (120, 80, 40))
            canvas_rgb = window.canvas.current_rgb_array()
            self.assertEqual(
                tuple(int(value) for value in canvas_rgb[0, 0]), (120, 80, 40)
            )

    def test_image_properties_export_dialog_toggles_nested_measurement_groups(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)

            def uncheck_first_angle_group(dialog: QDialog) -> int:
                tree = dialog.findChild(QTreeWidget, "exportItemTree")
                self.assertIsNotNone(tree)
                _tree_item_by_text(tree, "Измерения").setCheckState(0, Qt.Checked)
                _tree_item_by_text(tree, "Контрольный угол").setCheckState(
                    0, Qt.Unchecked
                )
                return QDialog.Accepted

            with patch(
                "magicborder.main_window.QDialog.exec_", new=uncheck_first_angle_group
            ):
                selected_properties = window._select_image_property_export_items()

            self.assertNotIn("angle:angle-a:id", selected_properties)
            self.assertNotIn("angle:angle-a:value", selected_properties)
            self.assertIn("angle:angle-b:id", selected_properties)
            self.assertIn("segment:segment-a:id", selected_properties)

    def test_image_properties_export_dialog_marks_parent_groups_partially_checked(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window, _project_path = _measurement_export_window(root)
            observed_states: dict[str, Qt.CheckState] = {}

            def uncheck_single_angle_leaf(dialog: QDialog) -> int:
                tree = dialog.findChild(QTreeWidget, "exportItemTree")
                self.assertIsNotNone(tree)
                _tree_item_by_text(tree, "Измерения").setCheckState(0, Qt.Checked)
                _tree_item_by_key(tree, "angle:angle-a:id").setCheckState(
                    0, Qt.Unchecked
                )
                observed_states["angle"] = _tree_item_by_text(
                    tree, "Контрольный угол"
                ).checkState(0)
                observed_states["angles"] = _tree_item_by_text(tree, "Углы").checkState(
                    0
                )
                observed_states["measurements"] = _tree_item_by_text(
                    tree, "Измерения"
                ).checkState(0)
                return QDialog.Accepted

            with patch(
                "magicborder.main_window.QDialog.exec_", new=uncheck_single_angle_leaf
            ):
                selected_properties = window._select_image_property_export_items()

            self.assertNotIn("angle:angle-a:id", selected_properties)
            self.assertIn("angle:angle-a:value", selected_properties)
            self.assertEqual(observed_states["angle"], Qt.PartiallyChecked)
            self.assertEqual(observed_states["angles"], Qt.PartiallyChecked)
            self.assertEqual(observed_states["measurements"], Qt.PartiallyChecked)

    def test_datetime_metadata_fields_have_picker_buttons_and_validate_text(
        self,
    ) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="metadata",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    )
                ],
            )
            project_path = root / "metadata.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(
                window.metadata_added_at_button.toolTip(), "Выбрать дату и время"
            )
            self.assertTrue(_qdatetime_from_text("2026-05-03T10:20:30").isValid())

            window.metadata_captured_at.setText("2026-05-03 10:20:30")
            window._handle_metadata_line_edit_finished(
                "captured_at", window.metadata_captured_at
            )
            self.assertEqual(
                window.project_document.images[0].metadata["captured_at"],
                "2026-05-03 10:20:30",
            )

            window.metadata_captured_at.setText("не дата")
            with patch.object(window, "_show_warning"):
                window._handle_metadata_line_edit_finished(
                    "captured_at", window.metadata_captured_at
                )
            self.assertEqual(window.metadata_captured_at.text(), "2026-05-03 10:20:30")

    def test_added_image_gets_added_at_metadata(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            source_path = source_dir / "leaf.png"
            Image.new("RGB", (20, 20), (120, 80, 40)).save(source_path)

            project_dir = root / "project"
            project_dir.mkdir()
            project_path = project_dir / "project.json"
            save_project(project_path, ProjectDocument(name="project", images=[]))

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            with patch(
                "magicborder.main_window.QFileDialog.getOpenFileNames",
                return_value=([str(source_path)], ""),
            ):
                window.add_images_to_project()

            loaded_project = load_project(project_path)
            record = loaded_project.images[0]

            _assert_uuid4(self, record.id)
            self.assertTrue(record.metadata["added_at"])
            self.assertEqual(record.metadata["diagnosis"], "Не указано")
            self.assertNotIn("sample_id", record.metadata)
            self.assertTrue((project_dir / "images" / "leaf.png").exists())

    def test_sync_project_images_folder_adds_untracked_images_only(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "project"
            image_dir = project_dir / "images"
            nested_dir = image_dir / "nested"
            nested_dir.mkdir(parents=True)
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "existing.png")
            Image.new("RGB", (18, 12), (10, 20, 30)).save(image_dir / "new.png")
            Image.new("RGB", (16, 14), (40, 50, 60)).save(nested_dir / "nested.jpg")
            (image_dir / "broken.png").write_bytes(b"not an image")
            (image_dir / "ignored.txt").write_text("not an image", encoding="utf-8")

            project_path = project_dir / "project.json"
            project = ProjectDocument(
                name="project",
                images=[
                    ProjectImageRecord(
                        id="existing-id",
                        relative_path="images/existing.png",
                        display_name="existing.png",
                    )
                ],
            )
            save_project(project_path, project)

            window = MainWindow()

            self.assertFalse(window.sync_images_action.isEnabled())
            self.assertEqual(
                window.sync_images_action.toolTip(),
                "Синхронизировать папку изображений",
            )
            self.assertEqual(
                window.sync_images_action.statusTip(),
                "Добавить в проект изображения из папки проекта, которых ещё нет в JSON.",
            )

            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.sync_images_action.isEnabled())

            with patch.object(window, "_show_warning") as show_warning:
                window.sync_project_images_folder()
            self.assertTrue(show_warning.called)

            loaded_project = load_project(project_path)
            records_by_path = {
                record.relative_path: record for record in loaded_project.images
            }

            self.assertEqual(len(loaded_project.images), 3)
            self.assertIn("images/existing.png", records_by_path)
            self.assertIn("images/new.png", records_by_path)
            self.assertIn("images/nested/nested.jpg", records_by_path)
            self.assertNotIn("images/broken.png", records_by_path)
            self.assertNotIn("images/ignored.txt", records_by_path)

            new_record = records_by_path["images/new.png"]
            nested_record = records_by_path["images/nested/nested.jpg"]
            self.assertIsNone(new_record.annotation)
            self.assertIsNone(nested_record.annotation)
            self.assertEqual(new_record.display_name, "new.png")
            self.assertEqual(
                (new_record.image_width, new_record.image_height), (18, 12)
            )
            self.assertEqual(nested_record.display_name, "nested.jpg")
            self.assertEqual(
                (nested_record.image_width, nested_record.image_height), (16, 14)
            )
            self.assertTrue(new_record.metadata["added_at"])
            self.assertEqual(new_record.metadata["diagnosis"], "Не указано")
            self.assertNotIn("sample_id", new_record.metadata)
            _assert_uuid4(self, new_record.id)
            _assert_uuid4(self, nested_record.id)

            with patch.object(window, "_show_warning"):
                window.sync_project_images_folder()
            reloaded_project = load_project(project_path)
            self.assertEqual(len(reloaded_project.images), 3)

    def test_invalid_metadata_value_is_not_saved(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="metadata",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    )
                ],
            )
            project_path = root / "metadata.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            window.metadata_humidity.setText("150")
            with patch.object(window, "_show_warning"):
                window._handle_metadata_line_edit_finished(
                    "humidity", window.metadata_humidity
                )

            self.assertEqual(window.project_document.images[0].metadata["humidity"], "")
            self.assertEqual(window.metadata_humidity.text(), "")


if __name__ == "__main__":
    unittest.main()
