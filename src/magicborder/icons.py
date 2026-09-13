from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction

ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"
TOOLBAR_ICON_SIZE = QSize(30, 30)


@dataclass(frozen=True, slots=True)
class ActionVisual:
    icon_name: str
    tool_tip: str
    status_tip: str


ACTION_VISUALS: dict[str, ActionVisual] = {
    "new_project": ActionVisual(
        icon_name="new-project",
        tool_tip="Новый проект",
        status_tip="Создать папку проекта и новый JSON-файл проекта.",
    ),
    "open_project": ActionVisual(
        icon_name="open-project",
        tool_tip="Открыть проект",
        status_tip="Открыть JSON-файл проекта MagicBorder.",
    ),
    "save_project": ActionVisual(
        icon_name="save-project",
        tool_tip="Сохранить проект",
        status_tip="Сохранить список изображений и аннотации текущего проекта.",
    ),
    "close_project": ActionVisual(
        icon_name="close-project",
        tool_tip="Закрыть проект",
        status_tip="Сохранить и закрыть текущий проект.",
    ),
    "add_images": ActionVisual(
        icon_name="add-images",
        tool_tip="Добавить изображения",
        status_tip="Добавить одно или несколько изображений в текущий проект.",
    ),
    "sync_images": ActionVisual(
        icon_name="sync-images",
        tool_tip="Синхронизировать папку изображений",
        status_tip="Добавить в проект изображения из папки проекта, которых ещё нет в JSON.",
    ),
    "remove_image": ActionVisual(
        icon_name="remove-project-image",
        tool_tip="Удалить изображение из проекта",
        status_tip="Удалить выбранное изображение из списка проекта.",
    ),
    "export_project_excel": ActionVisual(
        icon_name="export-csv",
        tool_tip="Экспорт списка в Excel",
        status_tip="Экспорт списка в Excel: сохранить список изображений проекта в файл .xlsx.",
    ),
    "export_project_csv": ActionVisual(
        icon_name="export-csv",
        tool_tip="Экспорт списка в Excel",
        status_tip="Экспорт списка в Excel: сохранить список изображений проекта в файл .xlsx.",
    ),
    "export_image_properties_excel": ActionVisual(
        icon_name="export-csv",
        tool_tip="Экспорт свойств изображения в Excel",
        status_tip="Сохранить свойства выбранного изображения в файл .xlsx.",
    ),
    "exit": ActionVisual(
        icon_name="exit-app",
        tool_tip="Выход",
        status_tip="Закрыть приложение MagicBorder.",
    ),
    "zoom_in": ActionVisual(
        icon_name="zoom-in",
        tool_tip="Увеличить",
        status_tip="Увеличить масштаб изображения на канвасе.",
    ),
    "zoom_out": ActionVisual(
        icon_name="zoom-out",
        tool_tip="Уменьшить",
        status_tip="Уменьшить масштаб изображения на канвасе.",
    ),
    "fit_image": ActionVisual(
        icon_name="fit-image",
        tool_tip="Показать целиком",
        status_tip="Вписать изображение целиком в область просмотра.",
    ),
    "actual_size": ActionVisual(
        icon_name="actual-size",
        tool_tip="Масштаб 100%",
        status_tip="Вернуть отображение изображения к масштабу 100%.",
    ),
    "crop_image": ActionVisual(
        icon_name="crop-image",
        tool_tip="Обрезать",
        status_tip="Выбрать область фотографии и обрезать файл изображения.",
    ),
    "default_view": ActionVisual(
        icon_name="default-view",
        tool_tip="Вид по умолчанию",
        status_tip="Вернуть стандартные пропорции панелей и вписать изображение в область просмотра.",
    ),
    "show_all_canvas_elements": ActionVisual(
        icon_name="show-all-canvas-elements",
        tool_tip="Показать все элементы на канвасе",
        status_tip="Показать главный контур, углы и отрезки текущего изображения на канвасе.",
    ),
    "hide_all_canvas_elements": ActionVisual(
        icon_name="hide-all-canvas-elements",
        tool_tip="Скрыть все элементы на канвасе",
        status_tip="Скрыть главный контур, углы и отрезки текущего изображения на канвасе.",
    ),
    "new_contour": ActionVisual(
        icon_name="new-contour",
        tool_tip="Новый контур",
        status_tip="Создать окружный контур в центре текущего изображения.",
    ),
    "detect_contour": ActionVisual(
        icon_name="detect-contour",
        tool_tip="Определить контур",
        status_tip="Автоматически определить границу листа и построить контур.",
    ),
    "delete_contour": ActionVisual(
        icon_name="delete-contour",
        tool_tip="Удалить контур",
        status_tip="Удалить контур текущего выбранного изображения.",
    ),
    "flatten_background": ActionVisual(
        icon_name="flatten-background",
        tool_tip="Выровнять фон",
        status_tip="Сделать область вне текущего контура белой.",
    ),
    "calibrate_scale": ActionVisual(
        icon_name="calibrate-scale",
        tool_tip="Калибровать масштаб",
        status_tip="Задать калибровочный отрезок известной длины для выбранного изображения.",
    ),
    "reset_calibration": ActionVisual(
        icon_name="reset-calibration",
        tool_tip="Сбросить калибровку",
        status_tip="Удалить калибровочный отрезок и масштаб выбранного изображения.",
    ),
    "measure_angle": ActionVisual(
        icon_name="measure-angle",
        tool_tip="Угол",
        status_tip="Измерить угол тремя точками на выбранном изображении.",
    ),
    "delete_angle": ActionVisual(
        icon_name="delete-angle",
        tool_tip="Удалить угол",
        status_tip="Удалить выбранное временное измерение угла.",
    ),
    "measure_segment": ActionVisual(
        icon_name="measure-segment",
        tool_tip="Отрезок",
        status_tip="Измерить отрезок двумя точками на выбранном изображении.",
    ),
    "delete_segment": ActionVisual(
        icon_name="delete-segment",
        tool_tip="Удалить отрезок",
        status_tip="Удалить выбранное измерение отрезка.",
    ),
    "save_annotation": ActionVisual(
        icon_name="save-annotation",
        tool_tip="Сохранить аннотацию",
        status_tip="Сохранить текущий контур в JSON-файл аннотации.",
    ),
    "open_annotation": ActionVisual(
        icon_name="open-annotation",
        tool_tip="Открыть аннотацию",
        status_tip="Загрузить контур из JSON-файла аннотации.",
    ),
    "refresh_analysis": ActionVisual(
        icon_name="refresh-analysis",
        tool_tip="Обновить",
        status_tip="Пересчитать гистограммы и аналитику по загруженным изображениям.",
    ),
    "about": ActionVisual(
        icon_name="about",
        tool_tip="О программе",
        status_tip="Показать информацию о приложении MagicBorder.",
    ),
}


def load_icon(icon_name: str) -> QIcon:
    return QIcon(str(ICON_DIR / f"{icon_name}.svg"))


def apply_action_visual(action: QAction, visual: ActionVisual) -> None:
    action.setIcon(load_icon(visual.icon_name))
    action.setToolTip(visual.tool_tip)
    action.setStatusTip(visual.status_tip)
    action.setWhatsThis(visual.status_tip)
    action.setIconVisibleInMenu(True)
