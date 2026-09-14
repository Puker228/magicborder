from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

from .io_utils import LoadedImage, loaded_image_from_rgb_array
from .models import (
    ANGLE_LABEL_COLOR,
    ANGLE_LINE_COLOR,
    CONTOUR_LINE_COLOR,
    SEGMENT_LABEL_COLOR,
    SEGMENT_LINE_COLOR,
    Point,
    normalize_line_color,
)

ANGLE_ARC_COLOR = "#7c3aed"
CONTOUR_LINE_WIDTH = 2.0
CONTOUR_Z = 10
MEASUREMENT_LINE_WIDTH = 2.0
MEASUREMENT_HIGHLIGHT_LINE_WIDTH = 4.4
ANGLE_LINE_Z = 22
ANGLE_ARC_Z = 23
SEGMENT_LINE_Z = 24
MEASUREMENT_HIGHLIGHT_Z_OFFSET = 10


@dataclass
class AngleMeasurement:
    id: str
    first: QPointF
    vertex: QPointF
    second: QPointF
    name: str = ""
    visible: bool = True
    line_color: str = ANGLE_LINE_COLOR
    label_color: str = ANGLE_LABEL_COLOR


@dataclass
class AngleGraphics:
    first_line: QGraphicsLineItem
    second_line: QGraphicsLineItem
    arc: QGraphicsPathItem
    label: QGraphicsTextItem
    handles: list[AngleHandleItem]


@dataclass
class SegmentMeasurement:
    id: str
    start: QPointF
    end: QPointF
    name: str = ""
    start_label: str = ""
    end_label: str = ""
    visible: bool = True
    line_color: str = SEGMENT_LINE_COLOR
    label_color: str = SEGMENT_LABEL_COLOR


@dataclass
class SegmentGraphics:
    line: QGraphicsLineItem
    length_label: QGraphicsTextItem
    start_label: QGraphicsTextItem
    end_label: QGraphicsTextItem
    handles: list[SegmentHandleItem]


class NodeHandleItem(QGraphicsEllipseItem):
    def __init__(self, canvas: ImageCanvas, index: int, position: QPointF) -> None:
        radius = 5.5
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0)
        self.canvas = canvas
        self.index = index
        self._normal_pen_width = 1.4
        self._normal_z_value = 20
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setBrush(QColor("#ffb000"))
        self.setPen(QPen(QColor("white"), self._normal_pen_width))
        self.setZValue(self._normal_z_value)
        self.setToolTip(
            "Перетащите для редактирования. Правая кнопка мыши удаляет узел."
        )
        self.setPos(position)

    def set_highlighted(self, highlighted: bool) -> None:
        self.setPen(
            QPen(QColor("white"), 2.8 if highlighted else self._normal_pen_width)
        )
        self.setZValue(
            self._normal_z_value + MEASUREMENT_HIGHLIGHT_Z_OFFSET
            if highlighted
            else self._normal_z_value
        )

    def itemChange(
        self, change: QGraphicsItem.GraphicsItemChange, value: object
    ) -> object:
        if change == QGraphicsItem.ItemPositionChange and isinstance(value, QPointF):
            return self.canvas.constrain_point(value)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.canvas.handle_moved(self.index, self.pos())
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.canvas.contour_selection_changed(self.isSelected())
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self.canvas.remove_node(self.index)
            event.accept()
            return
        super().mousePressEvent(event)


class CalibrationHandleItem(QGraphicsEllipseItem):
    def __init__(self, canvas: ImageCanvas, index: int, position: QPointF) -> None:
        radius = 6.0
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0)
        self.canvas = canvas
        self.index = index
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setBrush(QColor("#2a9d8f"))
        self.setPen(QPen(QColor("white"), 1.5))
        self.setZValue(26)
        self.setToolTip("Перетащите для редактирования калибровочного отрезка.")
        self.setPos(position)

    def itemChange(
        self, change: QGraphicsItem.GraphicsItemChange, value: object
    ) -> object:
        if change == QGraphicsItem.ItemPositionChange and isinstance(value, QPointF):
            return self.canvas.constrain_point(value)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.canvas.calibration_handle_moved(self.index, self.pos())
        return super().itemChange(change, value)


class AngleHandleItem(QGraphicsEllipseItem):
    def __init__(
        self,
        canvas: ImageCanvas,
        angle_index: int,
        point_index: int,
        position: QPointF,
    ) -> None:
        is_vertex = point_index == 1
        radius = 6.5 if is_vertex else 5.3
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0)
        self.canvas = canvas
        self.angle_index = angle_index
        self.point_index = point_index
        self._normal_pen_width = 1.6
        self._normal_z_value = 34 if is_vertex else 32
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setBrush(QColor("#7c3aed") if is_vertex else QColor("#22c55e"))
        self.setPen(QPen(QColor("white"), self._normal_pen_width))
        self.setZValue(self._normal_z_value)
        self.setToolTip(
            "Вершина угла. Выберите её, чтобы удалить измерение."
            if is_vertex
            else "Перетащите точку луча, чтобы изменить угол."
        )
        self.setPos(position)

    def set_highlighted(self, highlighted: bool) -> None:
        self.setPen(
            QPen(QColor("white"), 2.9 if highlighted else self._normal_pen_width)
        )
        self.setZValue(
            self._normal_z_value + MEASUREMENT_HIGHLIGHT_Z_OFFSET
            if highlighted
            else self._normal_z_value
        )

    def itemChange(
        self, change: QGraphicsItem.GraphicsItemChange, value: object
    ) -> object:
        if change == QGraphicsItem.ItemPositionChange and isinstance(value, QPointF):
            return self.canvas.constrain_point(value)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.canvas.angle_handle_moved(
                self.angle_index, self.point_index, self.pos()
            )
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.canvas.angle_selection_changed(
                self.angle_index if self.isSelected() else None
            )
        return super().itemChange(change, value)


class SegmentHandleItem(QGraphicsEllipseItem):
    def __init__(
        self,
        canvas: ImageCanvas,
        segment_index: int,
        point_index: int,
        position: QPointF,
    ) -> None:
        radius = 5.8
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0)
        self.canvas = canvas
        self.segment_index = segment_index
        self.point_index = point_index
        self._normal_pen_width = 1.5
        self._normal_z_value = 31
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setBrush(QColor("#f97316"))
        self.setPen(QPen(QColor("white"), self._normal_pen_width))
        self.setZValue(self._normal_z_value)
        self.setToolTip("Перетащите точку отрезка. Выберите её, чтобы удалить отрезок.")
        self.setPos(position)

    def set_highlighted(self, highlighted: bool) -> None:
        self.setPen(
            QPen(QColor("white"), 2.8 if highlighted else self._normal_pen_width)
        )
        self.setZValue(
            self._normal_z_value + MEASUREMENT_HIGHLIGHT_Z_OFFSET
            if highlighted
            else self._normal_z_value
        )

    def itemChange(
        self, change: QGraphicsItem.GraphicsItemChange, value: object
    ) -> object:
        if change == QGraphicsItem.ItemPositionChange and isinstance(value, QPointF):
            return self.canvas.constrain_point(value)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.canvas.segment_handle_moved(
                self.segment_index, self.point_index, self.pos()
            )
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.canvas.segment_selection_changed(
                self.segment_index if self.isSelected() else None
            )
        return super().itemChange(change, value)


class ImageCanvas(QGraphicsView):
    message_changed = pyqtSignal(str)
    image_state_changed = pyqtSignal(bool)
    contour_state_changed = pyqtSignal(bool)
    contour_geometry_changed = pyqtSignal()
    calibration_segment_selected = pyqtSignal(object)
    calibration_geometry_changed = pyqtSignal()
    angle_state_changed = pyqtSignal()
    segment_state_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._image_item = QGraphicsPixmapItem()
        self._image_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._image_item)

        self._contour_line_color = CONTOUR_LINE_COLOR
        self._path_item = QGraphicsPathItem()
        contour_pen = QPen(QColor(self._contour_line_color), CONTOUR_LINE_WIDTH)
        contour_pen.setCosmetic(True)
        self._path_item.setPen(contour_pen)
        self._path_item.setBrush(QBrush(Qt.NoBrush))
        self._path_item.setZValue(CONTOUR_Z)
        self._path_item.setVisible(False)
        self._scene.addItem(self._path_item)

        self._calibration_line_item = QGraphicsLineItem()
        calibration_pen = QPen(QColor("#d9467f"), 2.0)
        calibration_pen.setCosmetic(True)
        calibration_pen.setStyle(Qt.DashLine)
        self._calibration_line_item.setPen(calibration_pen)
        self._calibration_line_item.setZValue(16)
        self._calibration_line_item.setVisible(False)
        self._scene.addItem(self._calibration_line_item)

        self._calibration_label_item = QGraphicsTextItem()
        self._calibration_label_item.setDefaultTextColor(QColor("#9d174d"))
        label_font = self._calibration_label_item.font()
        label_font.setBold(True)
        label_font.setPointSize(9)
        self._calibration_label_item.setFont(label_font)
        self._calibration_label_item.setFlag(
            QGraphicsItem.ItemIgnoresTransformations, True
        )
        self._calibration_label_item.setZValue(30)
        self._calibration_label_item.setVisible(False)
        self._scene.addItem(self._calibration_label_item)

        self._calibration_preview_line_item = QGraphicsLineItem()
        calibration_preview_pen = QPen(QColor(217, 70, 127, 170), 1.8)
        calibration_preview_pen.setCosmetic(True)
        calibration_preview_pen.setStyle(Qt.DotLine)
        self._calibration_preview_line_item.setPen(calibration_preview_pen)
        self._calibration_preview_line_item.setZValue(24)
        self._calibration_preview_line_item.setVisible(False)
        self._scene.addItem(self._calibration_preview_line_item)

        preview_radius = 5.5
        self._calibration_preview_start_item = QGraphicsEllipseItem(
            -preview_radius,
            -preview_radius,
            preview_radius * 2.0,
            preview_radius * 2.0,
        )
        self._calibration_preview_start_item.setFlag(
            QGraphicsItem.ItemIgnoresTransformations,
            True,
        )
        self._calibration_preview_start_item.setBrush(QColor("#d9467f"))
        self._calibration_preview_start_item.setPen(QPen(QColor("white"), 1.4))
        self._calibration_preview_start_item.setZValue(28)
        self._calibration_preview_start_item.setVisible(False)
        self._scene.addItem(self._calibration_preview_start_item)

        angle_preview_pen = QPen(QColor(34, 197, 94, 180), 1.7)
        angle_preview_pen.setCosmetic(True)
        angle_preview_pen.setStyle(Qt.DotLine)
        self._angle_preview_line_item = QGraphicsLineItem()
        self._angle_preview_line_item.setPen(angle_preview_pen)
        self._angle_preview_line_item.setZValue(25)
        self._angle_preview_line_item.setVisible(False)
        self._scene.addItem(self._angle_preview_line_item)

        segment_preview_pen = QPen(QColor(249, 115, 22, 180), 1.8)
        segment_preview_pen.setCosmetic(True)
        segment_preview_pen.setStyle(Qt.DotLine)
        self._segment_preview_line_item = QGraphicsLineItem()
        self._segment_preview_line_item.setPen(segment_preview_pen)
        self._segment_preview_line_item.setZValue(25)
        self._segment_preview_line_item.setVisible(False)
        self._scene.addItem(self._segment_preview_line_item)

        self._angle_preview_fixed_line_item = QGraphicsLineItem()
        fixed_angle_preview_pen = QPen(QColor("#22c55e"), 1.8)
        fixed_angle_preview_pen.setCosmetic(True)
        self._angle_preview_fixed_line_item.setPen(fixed_angle_preview_pen)
        self._angle_preview_fixed_line_item.setZValue(25)
        self._angle_preview_fixed_line_item.setVisible(False)
        self._scene.addItem(self._angle_preview_fixed_line_item)

        self._angle_preview_first_item = self._create_angle_preview_marker(
            "#22c55e", 29
        )
        self._angle_preview_vertex_item = self._create_angle_preview_marker(
            "#7c3aed", 30
        )
        self._segment_preview_start_item = self._create_angle_preview_marker(
            "#f97316", 29
        )

        self._loaded_image: LoadedImage | None = None
        self._contour_points: list[QPointF] = []
        self._handles: list[NodeHandleItem] = []
        self._suppress_handle_events = False
        self._suppress_contour_selection_highlight = False
        self._contour_visible = True
        self._contour_highlighted = False
        self._calibration_points: list[QPointF] = []
        self._calibration_handles: list[CalibrationHandleItem] = []
        self._calibration_label_text = ""
        self._calibration_length_mm: float | None = None
        self._suppress_calibration_handle_events = False
        self._calibration_capture_active = False
        self._calibration_capture_points: list[QPointF] = []
        self._angle_measurements: list[AngleMeasurement] = []
        self._angle_graphics: list[AngleGraphics] = []
        self._suppress_angle_handle_events = False
        self._angle_capture_active = False
        self._angle_capture_points: list[QPointF] = []
        self._highlighted_angle_id: str | None = None
        self._segment_measurements: list[SegmentMeasurement] = []
        self._segment_graphics: list[SegmentGraphics] = []
        self._suppress_segment_handle_events = False
        self._segment_capture_active = False
        self._segment_capture_points: list[QPointF] = []
        self._highlighted_segment_id: str | None = None
        self._suppress_measurement_selection_highlight = False

        self.setRenderHints(
            QPainter.Antialiasing
            | QPainter.SmoothPixmapTransform
            | QPainter.TextAntialiasing
        )
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#eef2f7"))
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def has_image(self) -> bool:
        return self._loaded_image is not None

    def has_contour(self) -> bool:
        return len(self._contour_points) >= 3

    def is_contour_visible(self) -> bool:
        return self.has_contour() and self._contour_visible

    def contour_line_color(self) -> str:
        return self._contour_line_color

    def set_contour_line_color(self, color_name: str) -> bool:
        normalized_color = normalize_line_color(color_name, CONTOUR_LINE_COLOR)
        if normalized_color == self._contour_line_color:
            return False
        self._contour_line_color = normalized_color
        self._apply_contour_highlight()
        return True

    def has_calibration(self) -> bool:
        return len(self._calibration_points) == 2

    def has_angle_measurements(self) -> bool:
        return bool(self._angle_measurements)

    def has_segment_measurements(self) -> bool:
        return bool(self._segment_measurements)

    def has_selected_angle_vertex(self) -> bool:
        return any(
            index < len(self._angle_measurements)
            and self._angle_measurements[index].visible
            and len(graphics.handles) > 1
            and graphics.handles[1].isSelected()
            for index, graphics in enumerate(self._angle_graphics)
        )

    def has_selected_segment_endpoint(self) -> bool:
        return any(
            index < len(self._segment_measurements)
            and self._segment_measurements[index].visible
            and any(handle.isSelected() for handle in graphics.handles)
            for index, graphics in enumerate(self._segment_graphics)
        )

    def is_angle_measurement_visible(self, angle_id: str) -> bool:
        for measurement in self._angle_measurements:
            if measurement.id == angle_id:
                return measurement.visible
        return False

    def set_angle_measurement_visible(self, angle_id: str, visible: bool) -> bool:
        normalized_visible = bool(visible)
        for index, measurement in enumerate(self._angle_measurements):
            if measurement.id != angle_id:
                continue
            if measurement.visible == normalized_visible:
                return False
            measurement.visible = normalized_visible
            self._apply_angle_visibility(index)
            self.angle_state_changed.emit()
            return True
        return False

    def angle_measurement_line_color(self, angle_id: str) -> str | None:
        for measurement in self._angle_measurements:
            if measurement.id == angle_id:
                return measurement.line_color
        return None

    def set_angle_measurement_line_color(self, angle_id: str, color_name: str) -> bool:
        normalized_color = normalize_line_color(color_name, ANGLE_LINE_COLOR)
        for index, measurement in enumerate(self._angle_measurements):
            if measurement.id != angle_id:
                continue
            if measurement.line_color == normalized_color:
                return False
            measurement.line_color = normalized_color
            self._apply_angle_highlight(index)
            return True
        return False

    def angle_measurement_label_color(self, angle_id: str) -> str | None:
        for measurement in self._angle_measurements:
            if measurement.id == angle_id:
                return measurement.label_color
        return None

    def set_angle_measurement_label_color(self, angle_id: str, color_name: str) -> bool:
        normalized_color = normalize_line_color(color_name, ANGLE_LABEL_COLOR)
        for index, measurement in enumerate(self._angle_measurements):
            if measurement.id != angle_id:
                continue
            if measurement.label_color == normalized_color:
                return False
            measurement.label_color = normalized_color
            self._apply_angle_label_color(index)
            return True
        return False

    def is_segment_measurement_visible(self, segment_id: str) -> bool:
        for measurement in self._segment_measurements:
            if measurement.id == segment_id:
                return measurement.visible
        return False

    def set_segment_measurement_visible(self, segment_id: str, visible: bool) -> bool:
        normalized_visible = bool(visible)
        for index, measurement in enumerate(self._segment_measurements):
            if measurement.id != segment_id:
                continue
            if measurement.visible == normalized_visible:
                return False
            measurement.visible = normalized_visible
            self._apply_segment_visibility(index)
            self.segment_state_changed.emit()
            return True
        return False

    def segment_measurement_line_color(self, segment_id: str) -> str | None:
        for measurement in self._segment_measurements:
            if measurement.id == segment_id:
                return measurement.line_color
        return None

    def set_segment_measurement_line_color(
        self, segment_id: str, color_name: str
    ) -> bool:
        normalized_color = normalize_line_color(color_name, SEGMENT_LINE_COLOR)
        for index, measurement in enumerate(self._segment_measurements):
            if measurement.id != segment_id:
                continue
            if measurement.line_color == normalized_color:
                return False
            measurement.line_color = normalized_color
            self._apply_segment_highlight(index)
            return True
        return False

    def segment_measurement_label_color(self, segment_id: str) -> str | None:
        for measurement in self._segment_measurements:
            if measurement.id == segment_id:
                return measurement.label_color
        return None

    def set_segment_measurement_label_color(
        self, segment_id: str, color_name: str
    ) -> bool:
        normalized_color = normalize_line_color(color_name, SEGMENT_LABEL_COLOR)
        for index, measurement in enumerate(self._segment_measurements):
            if measurement.id != segment_id:
                continue
            if measurement.label_color == normalized_color:
                return False
            measurement.label_color = normalized_color
            self._apply_segment_label_color(index)
            return True
        return False

    def selected_angle_id(self) -> str | None:
        selected_index = self._selected_angle_vertex_index()
        if selected_index is None or selected_index >= len(self._angle_measurements):
            return None
        return self._angle_measurements[selected_index].id

    def selected_segment_id(self) -> str | None:
        selected_index = self._selected_segment_index()
        if selected_index is None or selected_index >= len(self._segment_measurements):
            return None
        return self._segment_measurements[selected_index].id

    def highlighted_angle_id(self) -> str | None:
        return self._highlighted_angle_id

    def highlighted_segment_id(self) -> str | None:
        return self._highlighted_segment_id

    def is_contour_highlighted(self) -> bool:
        return self._contour_highlighted

    def highlight_contour(self) -> bool:
        if not self.has_contour():
            return self.clear_canvas_highlight()
        if (
            self._contour_highlighted
            and self._highlighted_angle_id is None
            and self._highlighted_segment_id is None
        ):
            return False

        self._contour_highlighted = True
        self._highlighted_angle_id = None
        self._highlighted_segment_id = None
        self._refresh_measurement_highlights()
        self._apply_contour_highlight()
        return True

    def clear_contour_highlight(self) -> bool:
        if not self._contour_highlighted:
            return False
        self._contour_highlighted = False
        self._apply_contour_highlight()
        return True

    def clear_canvas_highlight(self) -> bool:
        measurement_changed = self.clear_measurement_highlight()
        contour_changed = self.clear_contour_highlight()
        return measurement_changed or contour_changed

    def highlight_angle_measurement(self, angle_id: str | None) -> bool:
        normalized_id = str(angle_id or "").strip()
        if not normalized_id:
            return self.clear_canvas_highlight()
        if not any(
            measurement.id == normalized_id for measurement in self._angle_measurements
        ):
            return self.clear_canvas_highlight()
        if (
            self._highlighted_angle_id == normalized_id
            and self._highlighted_segment_id is None
            and not self._contour_highlighted
        ):
            return False
        self._highlighted_angle_id = normalized_id
        self._highlighted_segment_id = None
        self._contour_highlighted = False
        self._refresh_measurement_highlights()
        self._apply_contour_highlight()
        return True

    def highlight_segment_measurement(self, segment_id: str | None) -> bool:
        normalized_id = str(segment_id or "").strip()
        if not normalized_id:
            return self.clear_canvas_highlight()
        if not any(
            measurement.id == normalized_id
            for measurement in self._segment_measurements
        ):
            return self.clear_canvas_highlight()
        if (
            self._highlighted_segment_id == normalized_id
            and self._highlighted_angle_id is None
            and not self._contour_highlighted
        ):
            return False
        self._highlighted_angle_id = None
        self._highlighted_segment_id = normalized_id
        self._contour_highlighted = False
        self._refresh_measurement_highlights()
        self._apply_contour_highlight()
        return True

    def clear_measurement_highlight(self) -> bool:
        if self._highlighted_angle_id is None and self._highlighted_segment_id is None:
            return False
        self._highlighted_angle_id = None
        self._highlighted_segment_id = None
        self._refresh_measurement_highlights()
        return True

    def current_image_path(self) -> Path | None:
        return self._loaded_image.path if self._loaded_image else None

    def current_pixmap(self) -> QPixmap | None:
        return self._loaded_image.pixmap if self._loaded_image else None

    def image_size(self) -> tuple[int, int] | None:
        if not self._loaded_image:
            return None
        return self._loaded_image.width, self._loaded_image.height

    def current_rgb_array(self) -> np.ndarray:
        if not self._loaded_image:
            raise ValueError("Сначала загрузите изображение проекта.")
        return self._loaded_image.rgb_array.copy()

    def current_rgb_array_view(self) -> np.ndarray:
        """Пиксели текущего кадра без копирования (массив только для чтения).

        Канвас не меняет массив на месте, а заменяет LoadedImage целиком,
        поэтому view можно безопасно отдавать в рабочие потоки.
        """
        if not self._loaded_image:
            raise ValueError("Сначала загрузите изображение проекта.")
        view = self._loaded_image.rgb_array.view()
        view.flags.writeable = False
        return view

    def contour_points(self) -> list[Point]:
        return [Point(point.x(), point.y()) for point in self._contour_points]

    def calibration_points(self) -> list[Point]:
        return [Point(point.x(), point.y()) for point in self._calibration_points]

    def calibration_label_text(self) -> str:
        return self._calibration_label_text

    def angle_measurements(self) -> list[tuple[Point, Point, Point]]:
        return [
            (
                Point(measurement.first.x(), measurement.first.y()),
                Point(measurement.vertex.x(), measurement.vertex.y()),
                Point(measurement.second.x(), measurement.second.y()),
            )
            for measurement in self._angle_measurements
        ]

    def angle_measurement_records(
        self,
        *,
        include_names: bool = False,
    ) -> (
        list[tuple[str, Point, Point, Point]]
        | list[tuple[str, Point, Point, Point, str]]
    ):
        records: (
            list[tuple[str, Point, Point, Point]]
            | list[tuple[str, Point, Point, Point, str]]
        ) = []
        for measurement in self._angle_measurements:
            first = Point(measurement.first.x(), measurement.first.y())
            vertex = Point(measurement.vertex.x(), measurement.vertex.y())
            second = Point(measurement.second.x(), measurement.second.y())
            if include_names:
                records.append(
                    (measurement.id, first, vertex, second, measurement.name)
                )
            else:
                records.append((measurement.id, first, vertex, second))
        return records

    def segment_measurements(self) -> list[tuple[Point, Point]]:
        return [
            (
                Point(measurement.start.x(), measurement.start.y()),
                Point(measurement.end.x(), measurement.end.y()),
            )
            for measurement in self._segment_measurements
        ]

    def segment_measurement_records(
        self,
        *,
        include_names: bool = False,
    ) -> (
        list[tuple[str, Point, Point, str, str]]
        | list[tuple[str, Point, Point, str, str, str]]
    ):
        records: (
            list[tuple[str, Point, Point, str, str]]
            | list[tuple[str, Point, Point, str, str, str]]
        ) = []
        for measurement in self._segment_measurements:
            start = Point(measurement.start.x(), measurement.start.y())
            end = Point(measurement.end.x(), measurement.end.y())
            if include_names:
                records.append(
                    (
                        measurement.id,
                        start,
                        end,
                        measurement.start_label,
                        measurement.end_label,
                        measurement.name,
                    )
                )
            else:
                records.append(
                    (
                        measurement.id,
                        start,
                        end,
                        measurement.start_label,
                        measurement.end_label,
                    )
                )
        return records

    def contour_rgb_pixels(self) -> np.ndarray:
        if not self._loaded_image or len(self._contour_points) < 3:
            return np.empty((0, 3), dtype=np.uint8)

        mask = self._contour_mask()
        pixels = self._loaded_image.rgb_array[mask > 0]
        return np.ascontiguousarray(pixels.reshape((-1, 3)))

    def set_loaded_image(self, image: LoadedImage) -> None:
        self._loaded_image = image
        self._image_item.setPixmap(image.pixmap)
        self._image_item.setOffset(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, image.width, image.height))
        self.clear_contour()
        self.clear_calibration()
        self.clear_angles()
        self.clear_segments()
        self.fit_to_image()
        self.image_state_changed.emit(True)
        self.message_changed.emit(f"Открыто изображение: {image.path.name}")

    def clear_image(self) -> None:
        self._loaded_image = None
        self._image_item.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF())
        self.clear_contour()
        self.clear_calibration()
        self.clear_angles()
        self.clear_segments()
        self.resetTransform()
        self.image_state_changed.emit(False)
        self.message_changed.emit("Изображение очищено.")

    def replace_current_rgb_array(self, rgb_array: np.ndarray) -> None:
        if not self._loaded_image:
            raise ValueError("Сначала загрузите изображение проекта.")
        updated_image = loaded_image_from_rgb_array(self._loaded_image.path, rgb_array)
        self._loaded_image = updated_image
        self._image_item.setPixmap(updated_image.pixmap)
        self._image_item.setOffset(0, 0)
        self._scene.setSceneRect(
            QRectF(0, 0, updated_image.width, updated_image.height)
        )
        self._refresh_path()
        self._rebuild_handles()
        self.image_state_changed.emit(True)

    def flatten_background_to_white(self) -> None:
        if not self._loaded_image:
            raise ValueError("Сначала загрузите изображение проекта.")
        if len(self._contour_points) < 3:
            raise ValueError("Сначала постройте или загрузите контур.")

        rgb_array = self.current_rgb_array()
        mask = self._contour_mask()
        rgb_array[mask == 0] = 255
        self.replace_current_rgb_array(rgb_array)
        self.message_changed.emit("Фон за пределами контура выровнен до белого.")

    def clear_contour(self) -> None:
        self._contour_points.clear()
        self._contour_line_color = CONTOUR_LINE_COLOR
        self._contour_visible = True
        self._contour_highlighted = False
        self._path_item.setPath(QPainterPath())
        self._apply_contour_highlight()
        self._apply_contour_visibility()
        self._clear_handles()
        self.contour_state_changed.emit(False)
        self.contour_geometry_changed.emit()

    def set_contour(self, points: Sequence[Point]) -> None:
        if not self._loaded_image:
            raise ValueError("Нельзя задать контур без загруженного изображения.")
        if len(points) < 3:
            raise ValueError("Контур должен содержать минимум 3 точки.")

        self._contour_points = [
            self.constrain_point(QPointF(float(point.x), float(point.y)))
            for point in points
        ]
        self._contour_visible = True
        self._contour_highlighted = False
        self._refresh_path()
        self._rebuild_handles()
        self.contour_state_changed.emit(True)
        self.contour_geometry_changed.emit()
        self.message_changed.emit(
            f"Контур загружен: {len(self._contour_points)} узлов."
        )

    def set_contour_visible(self, visible: bool) -> None:
        self._contour_visible = bool(visible)
        self._apply_contour_visibility()

    def begin_calibration(self) -> None:
        if not self.has_image():
            self.message_changed.emit("Сначала выберите изображение проекта.")
            return
        if self._angle_capture_active:
            self.cancel_angle_measurement(show_message=False)
        if self._segment_capture_active:
            self.cancel_segment_measurement(show_message=False)
        self._calibration_capture_active = True
        self._calibration_capture_points.clear()
        self._clear_calibration_preview()
        self.setDragMode(QGraphicsView.NoDrag)
        self.message_changed.emit("Калибровка: укажите начало отрезка.")

    def cancel_calibration(self) -> None:
        if not self._calibration_capture_active:
            return
        self._calibration_capture_active = False
        self._calibration_capture_points.clear()
        self._clear_calibration_preview()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.message_changed.emit("Калибровка отменена.")

    def set_calibration(self, start: Point, end: Point, label_text: str = "") -> None:
        if not self._loaded_image:
            raise ValueError("Нельзя задать калибровку без загруженного изображения.")
        self._calibration_capture_active = False
        self._calibration_capture_points.clear()
        self._clear_calibration_preview()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._calibration_points = [
            self.constrain_point(QPointF(float(start.x), float(start.y))),
            self.constrain_point(QPointF(float(end.x), float(end.y))),
        ]
        self._calibration_label_text = label_text
        self._calibration_length_mm = _calibration_length_mm_from_text(label_text)
        self._refresh_calibration_line()
        self._rebuild_calibration_handles()
        self._refresh_segment_labels()
        self.message_changed.emit("Калибровочный отрезок задан.")

    def set_calibration_label_text(self, label_text: str) -> None:
        self._calibration_label_text = label_text
        self._calibration_length_mm = _calibration_length_mm_from_text(label_text)
        self._refresh_calibration_label()
        self._refresh_segment_labels()

    def clear_calibration(self) -> None:
        self._calibration_capture_active = False
        self._calibration_capture_points.clear()
        self._clear_calibration_preview()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._calibration_points.clear()
        self._calibration_label_text = ""
        self._calibration_length_mm = None
        self._refresh_calibration_line()
        self._clear_calibration_handles()
        self._refresh_segment_labels()

    def begin_angle_measurement(self) -> None:
        if not self.has_image():
            self.message_changed.emit("Сначала выберите изображение проекта.")
            return
        if self._calibration_capture_active:
            self.cancel_calibration()
        if self._segment_capture_active:
            self.cancel_segment_measurement(show_message=False)
        self._angle_capture_active = True
        self._angle_capture_points.clear()
        self._clear_angle_preview()
        self.setDragMode(QGraphicsView.NoDrag)
        self.message_changed.emit("Угол: укажите точку первого луча.")
        self.angle_state_changed.emit()

    def cancel_angle_measurement(self, *, show_message: bool = True) -> None:
        if not self._angle_capture_active:
            return
        self._angle_capture_active = False
        self._angle_capture_points.clear()
        self._clear_angle_preview()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        if show_message:
            self.message_changed.emit("Измерение угла отменено.")
        self.angle_state_changed.emit()

    def set_angle_measurements(self, measurements: Sequence[Sequence[Point]]) -> None:
        records: list[tuple[str, Point, Point, Point, str]] = []
        for measurement in measurements:
            points = list(measurement)
            if len(points) != 3:
                continue
            records.append((_new_angle_id(), points[0], points[1], points[2], ""))
        self.set_angle_measurement_records(records)

    def set_angle_measurement_records(
        self,
        measurements: Sequence[Sequence[object]],
    ) -> None:
        if not self._loaded_image:
            self.clear_angles()
            return

        self.cancel_angle_measurement(show_message=False)
        self._angle_measurements = []
        for measurement in measurements:
            line_color = ANGLE_LINE_COLOR
            label_color = ANGLE_LABEL_COLOR
            if len(measurement) == 4:
                angle_id, first_point, vertex_point, second_point = measurement
                name = ""
            elif len(measurement) == 5:
                angle_id = measurement[0]
                if hasattr(measurement[1], "x"):
                    first_point, vertex_point, second_point, name = measurement[1:]
                else:
                    name, first_point, vertex_point, second_point = measurement[1:]
            elif len(measurement) == 6:
                angle_id = measurement[0]
                if hasattr(measurement[1], "x"):
                    first_point, vertex_point, second_point, name, line_color = (
                        measurement[1:]
                    )
                else:
                    name, first_point, vertex_point, second_point, line_color = (
                        measurement[1:]
                    )
            elif len(measurement) == 7:
                angle_id = measurement[0]
                if hasattr(measurement[1], "x"):
                    (
                        first_point,
                        vertex_point,
                        second_point,
                        name,
                        line_color,
                        label_color,
                    ) = measurement[1:]
                else:
                    (
                        name,
                        first_point,
                        vertex_point,
                        second_point,
                        line_color,
                        label_color,
                    ) = measurement[1:]
            else:
                continue
            try:
                first = self.constrain_point(
                    QPointF(float(first_point.x), float(first_point.y))
                )
                vertex = self.constrain_point(
                    QPointF(float(vertex_point.x), float(vertex_point.y))
                )
                second = self.constrain_point(
                    QPointF(float(second_point.x), float(second_point.y))
                )
            except (AttributeError, TypeError, ValueError):
                continue
            if _angle_degrees(first, vertex, second) is None:
                continue
            self._angle_measurements.append(
                AngleMeasurement(
                    id=str(angle_id or _new_angle_id()),
                    first=first,
                    vertex=vertex,
                    second=second,
                    name=str(name or "").strip(),
                    line_color=normalize_line_color(line_color, ANGLE_LINE_COLOR),
                    label_color=normalize_line_color(label_color, ANGLE_LABEL_COLOR),
                )
            )

        self._drop_missing_measurement_highlight()
        self._rebuild_angle_graphics()
        self.angle_state_changed.emit()

    def set_angle_measurement_name(self, angle_id: str, name: str) -> bool:
        normalized_name = str(name or "").strip()
        for index, measurement in enumerate(self._angle_measurements):
            if measurement.id != angle_id:
                continue
            if measurement.name == normalized_name:
                return False
            measurement.name = normalized_name
            self._refresh_angle_graphic(index)
            return True
        return False

    def clear_angles(self) -> None:
        was_active = self._angle_capture_active
        self._angle_capture_active = False
        self._angle_capture_points.clear()
        self._clear_angle_preview()
        if was_active:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._angle_measurements.clear()
        self._highlighted_angle_id = None
        self._clear_angle_graphics()
        self.angle_state_changed.emit()

    def begin_segment_measurement(self) -> None:
        if not self.has_image():
            self.message_changed.emit("Сначала выберите изображение проекта.")
            return
        if self._calibration_capture_active:
            self.cancel_calibration()
        if self._angle_capture_active:
            self.cancel_angle_measurement(show_message=False)
        self._segment_capture_active = True
        self._segment_capture_points.clear()
        self._clear_segment_preview()
        self.setDragMode(QGraphicsView.NoDrag)
        self.message_changed.emit("Отрезок: укажите первую точку.")
        self.segment_state_changed.emit()

    def set_segment_measurements(self, measurements: Sequence[Sequence[Point]]) -> None:
        records: list[tuple[str, Point, Point, str, str]] = []
        for measurement in measurements:
            points = list(measurement)
            if len(points) != 2:
                continue
            records.append((_new_segment_id(), points[0], points[1], "", ""))
        self.set_segment_measurement_records(records)

    def cancel_segment_measurement(self, *, show_message: bool = True) -> None:
        if not self._segment_capture_active:
            return
        self._segment_capture_active = False
        self._segment_capture_points.clear()
        self._clear_segment_preview()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        if show_message:
            self.message_changed.emit("Измерение отрезка отменено.")
        self.segment_state_changed.emit()

    def set_segment_measurement_records(
        self,
        measurements: Sequence[Sequence[object]],
    ) -> None:
        if not self._loaded_image:
            self.clear_segments()
            return

        self.cancel_segment_measurement(show_message=False)
        self._segment_measurements = []
        for measurement in measurements:
            line_color = SEGMENT_LINE_COLOR
            label_color = SEGMENT_LABEL_COLOR
            if len(measurement) == 5:
                segment_id, start_point, end_point, start_label, end_label = measurement
                name = ""
            elif len(measurement) == 6:
                segment_id = measurement[0]
                if hasattr(measurement[1], "x"):
                    start_point, end_point, start_label, end_label, name = measurement[
                        1:
                    ]
                else:
                    name, start_point, end_point, start_label, end_label = measurement[
                        1:
                    ]
            elif len(measurement) == 7:
                segment_id = measurement[0]
                if hasattr(measurement[1], "x"):
                    (
                        start_point,
                        end_point,
                        start_label,
                        end_label,
                        name,
                        line_color,
                    ) = measurement[1:]
                else:
                    (
                        name,
                        start_point,
                        end_point,
                        start_label,
                        end_label,
                        line_color,
                    ) = measurement[1:]
            elif len(measurement) == 8:
                segment_id = measurement[0]
                if hasattr(measurement[1], "x"):
                    (
                        start_point,
                        end_point,
                        start_label,
                        end_label,
                        name,
                        line_color,
                        label_color,
                    ) = measurement[1:]
                else:
                    (
                        name,
                        start_point,
                        end_point,
                        start_label,
                        end_label,
                        line_color,
                        label_color,
                    ) = measurement[1:]
            else:
                continue
            try:
                start = self.constrain_point(
                    QPointF(float(start_point.x), float(start_point.y))
                )
                end = self.constrain_point(
                    QPointF(float(end_point.x), float(end_point.y))
                )
            except (AttributeError, TypeError, ValueError):
                continue
            if _point_distance(start, end) <= 1e-6:
                continue
            self._segment_measurements.append(
                SegmentMeasurement(
                    id=str(segment_id or _new_segment_id()),
                    start=start,
                    end=end,
                    name=str(name or "").strip(),
                    start_label=str(start_label or "").strip(),
                    end_label=str(end_label or "").strip(),
                    line_color=normalize_line_color(line_color, SEGMENT_LINE_COLOR),
                    label_color=normalize_line_color(label_color, SEGMENT_LABEL_COLOR),
                )
            )

        self._drop_missing_measurement_highlight()
        self._rebuild_segment_graphics()
        self.segment_state_changed.emit()

    def set_segment_measurement_name(self, segment_id: str, name: str) -> bool:
        normalized_name = str(name or "").strip()
        for index, measurement in enumerate(self._segment_measurements):
            if measurement.id != segment_id:
                continue
            if measurement.name == normalized_name:
                return False
            measurement.name = normalized_name
            self._refresh_segment_graphic(index)
            return True
        return False

    def set_segment_labels(
        self, segment_id: str, start_label: str, end_label: str
    ) -> bool:
        normalized_start = str(start_label or "").strip()
        normalized_end = str(end_label or "").strip()
        for index, measurement in enumerate(self._segment_measurements):
            if measurement.id != segment_id:
                continue
            if (
                measurement.start_label == normalized_start
                and measurement.end_label == normalized_end
            ):
                return False
            measurement.start_label = normalized_start
            measurement.end_label = normalized_end
            self._refresh_segment_graphic(index)
            return True
        return False

    def clear_segments(self) -> None:
        was_active = self._segment_capture_active
        self._segment_capture_active = False
        self._segment_capture_points.clear()
        self._clear_segment_preview()
        if was_active:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._segment_measurements.clear()
        self._highlighted_segment_id = None
        self._clear_segment_graphics()
        self.segment_state_changed.emit()

    def delete_selected_segment(self) -> bool:
        selected_index = self._selected_segment_index()
        if selected_index is None:
            self.message_changed.emit(
                "Выберите точку отрезка, чтобы удалить измерение."
            )
            return False

        deleted_id = self._segment_measurements[selected_index].id
        del self._segment_measurements[selected_index]
        if self._highlighted_segment_id == deleted_id:
            self._highlighted_segment_id = None
        self._rebuild_segment_graphics()
        self.message_changed.emit("Измерение отрезка удалено.")
        self.segment_state_changed.emit()
        return True

    def delete_selected_angle(self) -> bool:
        selected_index = self._selected_angle_vertex_index()
        if selected_index is None:
            self.message_changed.emit("Выберите вершину угла, чтобы удалить измерение.")
            return False

        deleted_id = self._angle_measurements[selected_index].id
        del self._angle_measurements[selected_index]
        if self._highlighted_angle_id == deleted_id:
            self._highlighted_angle_id = None
        self._rebuild_angle_graphics()
        self.message_changed.emit("Измерение угла удалено.")
        self.angle_state_changed.emit()
        return True

    def constrain_point(self, point: QPointF) -> QPointF:
        if not self._loaded_image:
            return QPointF(point)
        rect = self._image_item.boundingRect()
        clamped_x = min(max(point.x(), rect.left()), rect.right())
        clamped_y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(clamped_x, clamped_y)

    def handle_moved(self, index: int, position: QPointF) -> None:
        if self._suppress_handle_events or index >= len(self._contour_points):
            return
        self.highlight_contour()
        self._contour_points[index] = self.constrain_point(position)
        self._refresh_path()
        self.contour_geometry_changed.emit()

    def contour_selection_changed(self, selected: bool | None = None) -> None:
        if self._suppress_contour_selection_highlight:
            return
        self._sync_canvas_highlight_from_selection(preferred_contour_selected=selected)

    def calibration_handle_moved(self, index: int, position: QPointF) -> None:
        if self._suppress_calibration_handle_events or index >= len(
            self._calibration_points
        ):
            return
        self._calibration_points[index] = self.constrain_point(position)
        self._refresh_calibration_line()
        self._refresh_segment_labels()
        self.calibration_geometry_changed.emit()

    def angle_handle_moved(
        self, angle_index: int, point_index: int, position: QPointF
    ) -> None:
        if (
            self._suppress_angle_handle_events
            or angle_index >= len(self._angle_measurements)
            or point_index not in (0, 1, 2)
        ):
            return

        measurement = self._angle_measurements[angle_index]
        self._highlight_angle_index(angle_index)
        point = self.constrain_point(position)
        if point_index == 0:
            measurement.first = point
        elif point_index == 1:
            measurement.vertex = point
        else:
            measurement.second = point
        self._refresh_angle_graphic(angle_index)
        self.angle_state_changed.emit()

    def angle_selection_changed(self, angle_index: int | None = None) -> None:
        self._sync_canvas_highlight_from_selection(preferred_angle_index=angle_index)
        self.angle_state_changed.emit()

    def segment_handle_moved(
        self, segment_index: int, point_index: int, position: QPointF
    ) -> None:
        if (
            self._suppress_segment_handle_events
            or segment_index >= len(self._segment_measurements)
            or point_index not in (0, 1)
        ):
            return

        measurement = self._segment_measurements[segment_index]
        self._highlight_segment_index(segment_index)
        point = self.constrain_point(position)
        if point_index == 0:
            measurement.start = point
        else:
            measurement.end = point
        self._refresh_segment_graphic(segment_index)
        self.segment_state_changed.emit()

    def segment_selection_changed(self, segment_index: int | None = None) -> None:
        self._sync_canvas_highlight_from_selection(
            preferred_segment_index=segment_index
        )
        self.segment_state_changed.emit()

    def remove_node(self, index: int) -> bool:
        if len(self._contour_points) <= 3:
            self.message_changed.emit("Контур должен содержать минимум 3 узла.")
            return False
        if not (0 <= index < len(self._contour_points)):
            return False

        del self._contour_points[index]
        self._refresh_path()
        self._rebuild_handles()
        self.highlight_contour()
        self.contour_state_changed.emit(True)
        self.contour_geometry_changed.emit()
        self.message_changed.emit("Узел удалён.")
        return True

    def delete_selected_nodes(self) -> bool:
        selected_indexes = sorted(
            {handle.index for handle in self._handles if handle.isSelected()},
            reverse=True,
        )
        if not selected_indexes:
            return False

        removable_count = max(0, len(self._contour_points) - 3)
        if removable_count == 0:
            self.message_changed.emit(
                "Нельзя удалить больше узлов: контур должен остаться замкнутым."
            )
            return False

        removed = 0
        for index in selected_indexes[:removable_count]:
            del self._contour_points[index]
            removed += 1

        if removed == 0:
            return False

        self._refresh_path()
        self._rebuild_handles()
        self.highlight_contour()
        self.contour_state_changed.emit(True)
        self.contour_geometry_changed.emit()
        self.message_changed.emit(f"Удалено узлов: {removed}.")
        return True

    def insert_node_near(self, scene_pos: QPointF) -> bool:
        if not self.has_contour():
            return False

        best_distance = float("inf")
        best_projection = None
        best_insert_index = None

        for index, start in enumerate(self._contour_points):
            end = self._contour_points[(index + 1) % len(self._contour_points)]
            distance, projection = _distance_to_segment(scene_pos, start, end)
            if distance < best_distance:
                best_distance = distance
                best_projection = projection
                best_insert_index = index + 1

        if best_projection is None or best_insert_index is None:
            return False

        if best_distance > self._segment_pick_tolerance():
            self.message_changed.emit("Щёлкните ближе к сегменту, чтобы добавить узел.")
            return False

        self._contour_points.insert(
            best_insert_index, self.constrain_point(best_projection)
        )
        self._refresh_path()
        self._rebuild_handles()
        self.highlight_contour()
        self.contour_state_changed.emit(True)
        self.contour_geometry_changed.emit()
        self.message_changed.emit("Новый узел добавлен.")
        return True

    def zoom_in(self) -> None:
        self.scale(1.2, 1.2)
        self._refresh_calibration_label()
        self._refresh_angle_labels()
        self._refresh_segment_labels()
        self.message_changed.emit("Масштаб увеличен.")

    def zoom_out(self) -> None:
        self.scale(1 / 1.2, 1 / 1.2)
        self._refresh_calibration_label()
        self._refresh_angle_labels()
        self._refresh_segment_labels()
        self.message_changed.emit("Масштаб уменьшен.")

    def reset_zoom(self) -> None:
        self.resetTransform()
        if self.has_image():
            self.centerOn(self._image_item)
        self._refresh_calibration_label()
        self._refresh_angle_labels()
        self._refresh_segment_labels()
        self.message_changed.emit("Масштаб: 100%.")

    def fit_to_image(self) -> None:
        if not self.has_image():
            return
        self.fitInView(self._image_item, Qt.KeepAspectRatio)
        self._refresh_calibration_label()
        self._refresh_angle_labels()
        self._refresh_segment_labels()
        self.message_changed.emit("Изображение вписано в окно.")

    def mousePressEvent(self, event) -> None:
        if self._segment_capture_active and event.button() == Qt.RightButton:
            self.cancel_segment_measurement()
            event.accept()
            return

        if self._segment_capture_active and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if not self._image_item.boundingRect().contains(scene_pos):
                self.message_changed.emit("Отрезок: укажите точку внутри изображения.")
                event.accept()
                return

            point = self.constrain_point(scene_pos)
            self._segment_capture_points.append(point)
            if len(self._segment_capture_points) == 1:
                self._refresh_segment_preview(point)
                self.message_changed.emit("Отрезок: укажите вторую точку.")
            else:
                start, end = self._segment_capture_points[:2]
                if _point_distance(start, end) <= 1e-6:
                    self._segment_capture_points = self._segment_capture_points[:1]
                    self._refresh_segment_preview(start)
                    self.message_changed.emit(
                        "Отрезок некорректен: точки должны различаться."
                    )
                    event.accept()
                    return
                self._segment_measurements.append(
                    SegmentMeasurement(_new_segment_id(), start, end)
                )
                self._segment_capture_points.clear()
                self._clear_segment_preview()
                self._rebuild_segment_graphics()
                self.setDragMode(QGraphicsView.NoDrag)
                self.message_changed.emit(
                    "Отрезок задан. Укажите первую точку для следующего отрезка."
                )
                self.segment_state_changed.emit()
            event.accept()
            return

        if self._angle_capture_active and event.button() == Qt.RightButton:
            self.cancel_angle_measurement()
            event.accept()
            return

        if self._angle_capture_active and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if not self._image_item.boundingRect().contains(scene_pos):
                self.message_changed.emit("Угол: укажите точку внутри изображения.")
                event.accept()
                return

            self._angle_capture_points.append(self.constrain_point(scene_pos))
            if len(self._angle_capture_points) == 1:
                self._refresh_angle_preview(self._angle_capture_points[0])
                self.message_changed.emit("Угол: укажите вершину угла.")
            elif len(self._angle_capture_points) == 2:
                self._refresh_angle_preview(self._angle_capture_points[1])
                self.message_changed.emit("Угол: укажите точку второго луча.")
            else:
                first, vertex, second = self._angle_capture_points[:3]
                if _angle_degrees(first, vertex, second) is None:
                    self._angle_capture_points = self._angle_capture_points[:2]
                    self._refresh_angle_preview(vertex)
                    self.message_changed.emit(
                        "Угол некорректен: точки лучей должны отличаться от вершины."
                    )
                    event.accept()
                    return

                self._angle_measurements.append(
                    AngleMeasurement(_new_angle_id(), first, vertex, second)
                )
                self._angle_capture_points.clear()
                self._clear_angle_preview()
                self._rebuild_angle_graphics()
                self.setDragMode(QGraphicsView.NoDrag)
                self.message_changed.emit(
                    "Угол задан. Укажите точку первого луча для следующего угла."
                )
                self.angle_state_changed.emit()
            event.accept()
            return

        if self._calibration_capture_active and event.button() == Qt.RightButton:
            self.cancel_calibration()
            event.accept()
            return

        if self._calibration_capture_active and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if not self._image_item.boundingRect().contains(scene_pos):
                self.message_changed.emit(
                    "Калибровка: укажите точку внутри изображения."
                )
                event.accept()
                return

            self._calibration_capture_points.append(self.constrain_point(scene_pos))
            if len(self._calibration_capture_points) == 1:
                self._refresh_calibration_preview(self._calibration_capture_points[0])
                self.message_changed.emit("Калибровка: укажите конец отрезка.")
            else:
                points = [
                    Point(point.x(), point.y())
                    for point in self._calibration_capture_points[:2]
                ]
                self._calibration_capture_active = False
                self._calibration_capture_points.clear()
                self._clear_calibration_preview()
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                self.calibration_segment_selected.emit(points)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._segment_capture_active and self._segment_capture_points:
            self._refresh_segment_preview(self.mapToScene(event.pos()))
            event.accept()
            return
        if self._angle_capture_active and self._angle_capture_points:
            self._refresh_angle_preview(self.mapToScene(event.pos()))
            event.accept()
            return
        if self._calibration_capture_active and self._calibration_capture_points:
            self._refresh_calibration_preview(self.mapToScene(event.pos()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if (
            self._angle_capture_active
            or self._calibration_capture_active
            or self._segment_capture_active
        ):
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.is_contour_visible():
            item = self.itemAt(event.pos())
            if not isinstance(item, NodeHandleItem):
                scene_pos = self.mapToScene(event.pos())
                if self._image_item.boundingRect().contains(scene_pos):
                    if self.insert_node_near(scene_pos):
                        event.accept()
                        return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape and self._angle_capture_active:
            self.cancel_angle_measurement()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._segment_capture_active:
            self.cancel_segment_measurement()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._calibration_capture_active:
            self.cancel_calibration()
            event.accept()
            return
        if (
            event.key() in (Qt.Key_Delete, Qt.Key_Backspace)
            and self.has_selected_angle_vertex()
            and self.delete_selected_angle()
        ):
            event.accept()
            return
        if (
            event.key() in (Qt.Key_Delete, Qt.Key_Backspace)
            and self.has_selected_segment_endpoint()
            and self.delete_selected_segment()
        ):
            event.accept()
            return
        if (
            event.key() in (Qt.Key_Delete, Qt.Key_Backspace)
            and self.delete_selected_nodes()
        ):
            event.accept()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
            event.accept()
            return
        if event.key() == Qt.Key_Minus:
            self.zoom_out()
            event.accept()
            return
        if event.key() == Qt.Key_0:
            self.reset_zoom()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def _segment_pick_tolerance(self) -> float:
        scale = abs(self.transform().m11()) or 1.0
        return 14.0 / scale

    def _refresh_path(self) -> None:
        path = QPainterPath()
        if self._contour_points:
            path.moveTo(self._contour_points[0])
            for point in self._contour_points[1:]:
                path.lineTo(point)
            path.closeSubpath()
        self._path_item.setPath(path)
        self._apply_contour_visibility()

    def _refresh_calibration_line(self) -> None:
        if len(self._calibration_points) != 2:
            self._calibration_line_item.setVisible(False)
            self._calibration_line_item.setLine(0.0, 0.0, 0.0, 0.0)
            self._refresh_calibration_label()
            return

        start, end = self._calibration_points
        self._calibration_line_item.setLine(start.x(), start.y(), end.x(), end.y())
        self._calibration_line_item.setVisible(True)
        self._refresh_calibration_label()

    def _refresh_calibration_label(self) -> None:
        if (
            len(self._calibration_points) != 2
            or not self._calibration_label_text
            or self._calibration_capture_active
        ):
            self._calibration_label_item.setVisible(False)
            return

        start, end = self._calibration_points
        self._calibration_label_item.setPlainText(self._calibration_label_text)
        midpoint = QPointF((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0)
        text_rect = self._calibration_label_item.boundingRect()
        scale_x = abs(self.transform().m11()) or 1.0
        scale_y = abs(self.transform().m22()) or scale_x
        self._calibration_label_item.setPos(
            midpoint.x() - text_rect.center().x() / scale_x,
            midpoint.y() - text_rect.center().y() / scale_y,
        )
        self._calibration_label_item.setVisible(True)

    def _refresh_calibration_preview(self, end: QPointF) -> None:
        if not self._loaded_image or not self._calibration_capture_points:
            self._clear_calibration_preview()
            return

        start = self._calibration_capture_points[0]
        preview_end = self.constrain_point(end)
        self._calibration_preview_start_item.setPos(start)
        self._calibration_preview_start_item.setVisible(True)
        self._calibration_preview_line_item.setLine(
            start.x(),
            start.y(),
            preview_end.x(),
            preview_end.y(),
        )
        self._calibration_preview_line_item.setVisible(True)

    def _clear_calibration_preview(self) -> None:
        self._calibration_preview_start_item.setVisible(False)
        self._calibration_preview_line_item.setVisible(False)
        self._calibration_preview_line_item.setLine(0.0, 0.0, 0.0, 0.0)

    def _create_angle_preview_marker(
        self, color_name: str, z_value: float
    ) -> QGraphicsEllipseItem:
        radius = 5.4
        marker = QGraphicsEllipseItem(-radius, -radius, radius * 2.0, radius * 2.0)
        marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        marker.setBrush(QColor(color_name))
        marker.setPen(QPen(QColor("white"), 1.4))
        marker.setZValue(z_value)
        marker.setVisible(False)
        self._scene.addItem(marker)
        return marker

    def _refresh_angle_preview(self, end: QPointF) -> None:
        if not self._loaded_image or not self._angle_capture_points:
            self._clear_angle_preview()
            return

        preview_end = self.constrain_point(end)
        first = self._angle_capture_points[0]
        self._angle_preview_first_item.setPos(first)
        self._angle_preview_first_item.setVisible(True)

        if len(self._angle_capture_points) == 1:
            self._angle_preview_vertex_item.setVisible(False)
            self._angle_preview_fixed_line_item.setVisible(False)
            self._angle_preview_fixed_line_item.setLine(0.0, 0.0, 0.0, 0.0)
            self._angle_preview_line_item.setLine(
                first.x(),
                first.y(),
                preview_end.x(),
                preview_end.y(),
            )
            self._angle_preview_line_item.setVisible(True)
            return

        vertex = self._angle_capture_points[1]
        self._angle_preview_vertex_item.setPos(vertex)
        self._angle_preview_vertex_item.setVisible(True)
        self._angle_preview_fixed_line_item.setLine(
            vertex.x(),
            vertex.y(),
            first.x(),
            first.y(),
        )
        self._angle_preview_fixed_line_item.setVisible(True)
        self._angle_preview_line_item.setLine(
            vertex.x(),
            vertex.y(),
            preview_end.x(),
            preview_end.y(),
        )
        self._angle_preview_line_item.setVisible(True)

    def _clear_angle_preview(self) -> None:
        self._angle_preview_first_item.setVisible(False)
        self._angle_preview_vertex_item.setVisible(False)
        self._angle_preview_line_item.setVisible(False)
        self._angle_preview_line_item.setLine(0.0, 0.0, 0.0, 0.0)
        self._angle_preview_fixed_line_item.setVisible(False)
        self._angle_preview_fixed_line_item.setLine(0.0, 0.0, 0.0, 0.0)

    def _refresh_segment_preview(self, end: QPointF) -> None:
        if not self._loaded_image or not self._segment_capture_points:
            self._clear_segment_preview()
            return

        start = self._segment_capture_points[0]
        preview_end = self.constrain_point(end)
        self._segment_preview_start_item.setPos(start)
        self._segment_preview_start_item.setVisible(True)
        self._segment_preview_line_item.setLine(
            start.x(),
            start.y(),
            preview_end.x(),
            preview_end.y(),
        )
        self._segment_preview_line_item.setVisible(True)

    def _clear_segment_preview(self) -> None:
        self._segment_preview_start_item.setVisible(False)
        self._segment_preview_line_item.setVisible(False)
        self._segment_preview_line_item.setLine(0.0, 0.0, 0.0, 0.0)

    def _clear_angle_graphics(self) -> None:
        for graphics in self._angle_graphics:
            for handle in graphics.handles:
                self._scene.removeItem(handle)
            self._scene.removeItem(graphics.first_line)
            self._scene.removeItem(graphics.second_line)
            self._scene.removeItem(graphics.arc)
            self._scene.removeItem(graphics.label)
        self._angle_graphics.clear()

    def _clear_segment_graphics(self) -> None:
        for graphics in self._segment_graphics:
            for handle in graphics.handles:
                self._scene.removeItem(handle)
            self._scene.removeItem(graphics.line)
            self._scene.removeItem(graphics.length_label)
            self._scene.removeItem(graphics.start_label)
            self._scene.removeItem(graphics.end_label)
        self._segment_graphics.clear()

    def _rebuild_angle_graphics(self) -> None:
        highlighted_angle_id = self._highlighted_angle_id
        highlighted_segment_id = self._highlighted_segment_id
        self._clear_angle_graphics()
        self._suppress_angle_handle_events = True
        try:
            for index, measurement in enumerate(self._angle_measurements):
                self._angle_graphics.append(
                    self._create_angle_graphic(index, measurement)
                )
                self._refresh_angle_graphic(index)
        finally:
            self._suppress_angle_handle_events = False
        self._highlighted_angle_id = highlighted_angle_id
        self._highlighted_segment_id = highlighted_segment_id
        self._drop_missing_measurement_highlight()
        self._refresh_measurement_highlights()

    def _create_angle_graphic(
        self,
        angle_index: int,
        measurement: AngleMeasurement,
    ) -> AngleGraphics:
        line_pen = self._measurement_pen(measurement.line_color, MEASUREMENT_LINE_WIDTH)

        first_line = QGraphicsLineItem()
        first_line.setPen(line_pen)
        first_line.setZValue(ANGLE_LINE_Z)
        self._scene.addItem(first_line)

        second_line = QGraphicsLineItem()
        second_line.setPen(line_pen)
        second_line.setZValue(ANGLE_LINE_Z)
        self._scene.addItem(second_line)

        arc = QGraphicsPathItem()
        arc.setPen(self._measurement_pen(ANGLE_ARC_COLOR, MEASUREMENT_LINE_WIDTH))
        arc.setBrush(QBrush(Qt.NoBrush))
        arc.setZValue(ANGLE_ARC_Z)
        arc.setVisible(False)
        self._scene.addItem(arc)

        label = QGraphicsTextItem()
        label.setDefaultTextColor(QColor(measurement.label_color))
        label_font = label.font()
        label_font.setBold(True)
        label_font.setPointSize(9)
        label.setFont(label_font)
        label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        label.setZValue(33)
        self._scene.addItem(label)

        handles = [
            AngleHandleItem(self, angle_index, 0, measurement.first),
            AngleHandleItem(self, angle_index, 1, measurement.vertex),
            AngleHandleItem(self, angle_index, 2, measurement.second),
        ]
        for handle in handles:
            self._scene.addItem(handle)

        return AngleGraphics(first_line, second_line, arc, label, handles)

    def _refresh_angle_graphic(self, angle_index: int) -> None:
        if angle_index >= len(self._angle_measurements) or angle_index >= len(
            self._angle_graphics
        ):
            return

        measurement = self._angle_measurements[angle_index]
        graphics = self._angle_graphics[angle_index]
        graphics.first_line.setLine(
            measurement.vertex.x(),
            measurement.vertex.y(),
            measurement.first.x(),
            measurement.first.y(),
        )
        graphics.second_line.setLine(
            measurement.vertex.x(),
            measurement.vertex.y(),
            measurement.second.x(),
            measurement.second.y(),
        )
        angle_degrees = _angle_degrees(
            measurement.first, measurement.vertex, measurement.second
        )
        arc_path = _angle_arc_path(
            measurement.first, measurement.vertex, measurement.second
        )
        graphics.arc.setPath(arc_path)
        display_name = _angle_display_name(measurement.name, angle_index)
        graphics.label.setPlainText(_format_angle_label(display_name, angle_degrees))
        self._apply_angle_label_color(angle_index)
        self._position_angle_label(graphics.label, measurement.vertex)

        self._suppress_angle_handle_events = True
        try:
            points = (measurement.first, measurement.vertex, measurement.second)
            for handle, point in zip(graphics.handles, points, strict=True):
                handle.setPos(point)
        finally:
            self._suppress_angle_handle_events = False
        self._apply_angle_highlight(angle_index)
        self._apply_angle_visibility(angle_index)

    def _refresh_angle_labels(self) -> None:
        for index in range(len(self._angle_measurements)):
            self._refresh_angle_graphic(index)

    def _apply_angle_visibility(self, angle_index: int) -> None:
        if angle_index >= len(self._angle_measurements) or angle_index >= len(
            self._angle_graphics
        ):
            return

        measurement = self._angle_measurements[angle_index]
        graphics = self._angle_graphics[angle_index]
        visible = measurement.visible
        graphics.first_line.setVisible(visible)
        graphics.second_line.setVisible(visible)
        graphics.arc.setVisible(visible and not graphics.arc.path().isEmpty())
        graphics.label.setVisible(visible)
        self._set_measurement_handles_visible(graphics.handles, visible)

    def _apply_angle_label_color(self, angle_index: int) -> None:
        if angle_index >= len(self._angle_measurements) or angle_index >= len(
            self._angle_graphics
        ):
            return

        measurement = self._angle_measurements[angle_index]
        graphics = self._angle_graphics[angle_index]
        graphics.label.setDefaultTextColor(QColor(measurement.label_color))

    def _rebuild_segment_graphics(self) -> None:
        highlighted_angle_id = self._highlighted_angle_id
        highlighted_segment_id = self._highlighted_segment_id
        self._clear_segment_graphics()
        self._suppress_segment_handle_events = True
        try:
            for index, measurement in enumerate(self._segment_measurements):
                self._segment_graphics.append(
                    self._create_segment_graphic(index, measurement)
                )
                self._refresh_segment_graphic(index)
        finally:
            self._suppress_segment_handle_events = False
        self._highlighted_angle_id = highlighted_angle_id
        self._highlighted_segment_id = highlighted_segment_id
        self._drop_missing_measurement_highlight()
        self._refresh_measurement_highlights()

    def _create_segment_graphic(
        self,
        segment_index: int,
        measurement: SegmentMeasurement,
    ) -> SegmentGraphics:
        line = QGraphicsLineItem()
        line.setPen(
            self._measurement_pen(measurement.line_color, MEASUREMENT_LINE_WIDTH)
        )
        line.setZValue(SEGMENT_LINE_Z)
        self._scene.addItem(line)

        length_label = self._create_segment_text_item(measurement.label_color, 34)
        start_label = self._create_segment_text_item(measurement.label_color, 34)
        end_label = self._create_segment_text_item(measurement.label_color, 34)

        handles = [
            SegmentHandleItem(self, segment_index, 0, measurement.start),
            SegmentHandleItem(self, segment_index, 1, measurement.end),
        ]
        for handle in handles:
            self._scene.addItem(handle)

        return SegmentGraphics(line, length_label, start_label, end_label, handles)

    def _create_segment_text_item(
        self, color_name: str, z_value: float
    ) -> QGraphicsTextItem:
        label = QGraphicsTextItem()
        label.setDefaultTextColor(QColor(color_name))
        label_font = label.font()
        label_font.setBold(True)
        label_font.setPointSize(9)
        label.setFont(label_font)
        label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        label.setZValue(z_value)
        label.setVisible(False)
        self._scene.addItem(label)
        return label

    def _refresh_segment_graphic(self, segment_index: int) -> None:
        if segment_index >= len(self._segment_measurements) or segment_index >= len(
            self._segment_graphics
        ):
            return

        measurement = self._segment_measurements[segment_index]
        graphics = self._segment_graphics[segment_index]
        graphics.line.setLine(
            measurement.start.x(),
            measurement.start.y(),
            measurement.end.x(),
            measurement.end.y(),
        )

        length_text = _format_segment_length_label(
            measurement.start,
            measurement.end,
            self._calibration_points,
            self._calibration_length_mm,
        )
        graphics.length_label.setPlainText(
            _format_segment_label(
                _segment_display_name(measurement.name, segment_index + 1),
                length_text,
            )
        )
        self._apply_segment_label_color(segment_index)
        midpoint = QPointF(
            (measurement.start.x() + measurement.end.x()) / 2.0,
            (measurement.start.y() + measurement.end.y()) / 2.0,
        )
        self._position_segment_text(graphics.length_label, midpoint, 0.0, -16.0)

        self._set_endpoint_label(
            graphics.start_label,
            measurement.start_label,
            measurement.start,
            -12.0,
        )
        self._set_endpoint_label(
            graphics.end_label,
            measurement.end_label,
            measurement.end,
            12.0,
        )

        self._suppress_segment_handle_events = True
        try:
            for handle, point in zip(
                graphics.handles, (measurement.start, measurement.end), strict=True
            ):
                handle.setPos(point)
        finally:
            self._suppress_segment_handle_events = False
        self._apply_segment_highlight(segment_index)
        self._apply_segment_visibility(segment_index)

    def _refresh_segment_labels(self) -> None:
        for index in range(len(self._segment_measurements)):
            self._refresh_segment_graphic(index)

    def _apply_segment_visibility(self, segment_index: int) -> None:
        if segment_index >= len(self._segment_measurements) or segment_index >= len(
            self._segment_graphics
        ):
            return

        measurement = self._segment_measurements[segment_index]
        graphics = self._segment_graphics[segment_index]
        visible = measurement.visible
        graphics.line.setVisible(visible)
        graphics.length_label.setVisible(visible)
        graphics.start_label.setVisible(visible and bool(measurement.start_label))
        graphics.end_label.setVisible(visible and bool(measurement.end_label))
        self._set_measurement_handles_visible(graphics.handles, visible)

    def _apply_segment_label_color(self, segment_index: int) -> None:
        if segment_index >= len(self._segment_measurements) or segment_index >= len(
            self._segment_graphics
        ):
            return

        measurement = self._segment_measurements[segment_index]
        graphics = self._segment_graphics[segment_index]
        color = QColor(measurement.label_color)
        graphics.length_label.setDefaultTextColor(color)
        graphics.start_label.setDefaultTextColor(color)
        graphics.end_label.setDefaultTextColor(color)

    def _set_measurement_handles_visible(
        self,
        handles: list[AngleHandleItem] | list[SegmentHandleItem],
        visible: bool,
    ) -> None:
        for handle in handles:
            handle.setVisible(visible)
            if not visible:
                self._suppress_measurement_selection_highlight = True
                try:
                    handle.setSelected(False)
                finally:
                    self._suppress_measurement_selection_highlight = False

    def _measurement_pen(self, color_name: str, width: float) -> QPen:
        pen = QPen(QColor(color_name), width)
        pen.setCosmetic(True)
        return pen

    def _refresh_measurement_highlights(self) -> None:
        for index in range(len(self._angle_measurements)):
            self._apply_angle_highlight(index)
        for index in range(len(self._segment_measurements)):
            self._apply_segment_highlight(index)

    def _apply_contour_highlight(self) -> None:
        highlighted = self._contour_highlighted and self.has_contour()
        line_width = (
            MEASUREMENT_HIGHLIGHT_LINE_WIDTH if highlighted else CONTOUR_LINE_WIDTH
        )
        line_z = (
            CONTOUR_Z + MEASUREMENT_HIGHLIGHT_Z_OFFSET if highlighted else CONTOUR_Z
        )

        pen = QPen(QColor(self._contour_line_color), line_width)
        pen.setCosmetic(True)
        self._path_item.setPen(pen)
        self._path_item.setZValue(line_z)
        for handle in self._handles:
            handle.set_highlighted(highlighted)

    def _drop_missing_measurement_highlight(self) -> None:
        if self._highlighted_angle_id is not None and not any(
            measurement.id == self._highlighted_angle_id
            for measurement in self._angle_measurements
        ):
            self._highlighted_angle_id = None
        if self._highlighted_segment_id is not None and not any(
            measurement.id == self._highlighted_segment_id
            for measurement in self._segment_measurements
        ):
            self._highlighted_segment_id = None

    def _highlight_angle_index(self, angle_index: int | None) -> bool:
        if (
            angle_index is None
            or angle_index < 0
            or angle_index >= len(self._angle_measurements)
        ):
            return self.clear_measurement_highlight()
        return self.highlight_angle_measurement(
            self._angle_measurements[angle_index].id
        )

    def _highlight_segment_index(self, segment_index: int | None) -> bool:
        if (
            segment_index is None
            or segment_index < 0
            or segment_index >= len(self._segment_measurements)
        ):
            return self.clear_measurement_highlight()
        return self.highlight_segment_measurement(
            self._segment_measurements[segment_index].id
        )

    def _sync_canvas_highlight_from_selection(
        self,
        *,
        preferred_contour_selected: bool | None = None,
        preferred_angle_index: int | None = None,
        preferred_segment_index: int | None = None,
    ) -> None:
        if self._suppress_measurement_selection_highlight:
            return
        if self._is_selected_visible_angle(preferred_angle_index):
            self._highlight_angle_index(preferred_angle_index)
            return
        if self._is_selected_visible_segment(preferred_segment_index):
            self._highlight_segment_index(preferred_segment_index)
            return
        if preferred_contour_selected and self._has_selected_contour_handle():
            self.highlight_contour()
            return

        selected_angle_index = self._selected_angle_handle_index()
        if selected_angle_index is not None:
            self._highlight_angle_index(selected_angle_index)
            return

        selected_segment_index = self._selected_segment_handle_index()
        if selected_segment_index is not None:
            self._highlight_segment_index(selected_segment_index)
            return

        if self._has_selected_contour_handle():
            self.highlight_contour()
            return

        self.clear_canvas_highlight()

    def _sync_measurement_highlight_from_selection(
        self,
        *,
        preferred_angle_index: int | None = None,
        preferred_segment_index: int | None = None,
    ) -> None:
        self._sync_canvas_highlight_from_selection(
            preferred_angle_index=preferred_angle_index,
            preferred_segment_index=preferred_segment_index,
        )

    def _selected_angle_handle_index(self) -> int | None:
        for index, graphics in enumerate(self._angle_graphics):
            if self._is_selected_visible_angle(index, graphics):
                return index
        return None

    def _selected_segment_handle_index(self) -> int | None:
        for index, graphics in enumerate(self._segment_graphics):
            if self._is_selected_visible_segment(index, graphics):
                return index
        return None

    def _has_selected_contour_handle(self) -> bool:
        return self.is_contour_visible() and any(
            handle.isSelected() for handle in self._handles
        )

    def _is_selected_visible_angle(
        self,
        angle_index: int | None,
        graphics: AngleGraphics | None = None,
    ) -> bool:
        if (
            angle_index is None
            or angle_index < 0
            or angle_index >= len(self._angle_measurements)
        ):
            return False
        if graphics is None:
            if angle_index >= len(self._angle_graphics):
                return False
            graphics = self._angle_graphics[angle_index]
        return self._angle_measurements[angle_index].visible and any(
            handle.isSelected() for handle in graphics.handles
        )

    def _is_selected_visible_segment(
        self,
        segment_index: int | None,
        graphics: SegmentGraphics | None = None,
    ) -> bool:
        if (
            segment_index is None
            or segment_index < 0
            or segment_index >= len(self._segment_measurements)
        ):
            return False
        if graphics is None:
            if segment_index >= len(self._segment_graphics):
                return False
            graphics = self._segment_graphics[segment_index]
        return self._segment_measurements[segment_index].visible and any(
            handle.isSelected() for handle in graphics.handles
        )

    def _apply_angle_highlight(self, angle_index: int) -> None:
        if angle_index >= len(self._angle_measurements) or angle_index >= len(
            self._angle_graphics
        ):
            return

        measurement = self._angle_measurements[angle_index]
        graphics = self._angle_graphics[angle_index]
        highlighted = measurement.id == self._highlighted_angle_id
        line_width = (
            MEASUREMENT_HIGHLIGHT_LINE_WIDTH if highlighted else MEASUREMENT_LINE_WIDTH
        )
        line_z = (
            ANGLE_LINE_Z + MEASUREMENT_HIGHLIGHT_Z_OFFSET
            if highlighted
            else ANGLE_LINE_Z
        )
        arc_z = (
            ANGLE_ARC_Z + MEASUREMENT_HIGHLIGHT_Z_OFFSET if highlighted else ANGLE_ARC_Z
        )

        graphics.first_line.setPen(
            self._measurement_pen(measurement.line_color, line_width)
        )
        graphics.second_line.setPen(
            self._measurement_pen(measurement.line_color, line_width)
        )
        graphics.arc.setPen(self._measurement_pen(ANGLE_ARC_COLOR, line_width))
        graphics.first_line.setZValue(line_z)
        graphics.second_line.setZValue(line_z)
        graphics.arc.setZValue(arc_z)
        for handle in graphics.handles:
            handle.set_highlighted(highlighted)

    def _apply_segment_highlight(self, segment_index: int) -> None:
        if segment_index >= len(self._segment_measurements) or segment_index >= len(
            self._segment_graphics
        ):
            return

        measurement = self._segment_measurements[segment_index]
        graphics = self._segment_graphics[segment_index]
        highlighted = measurement.id == self._highlighted_segment_id
        line_width = (
            MEASUREMENT_HIGHLIGHT_LINE_WIDTH if highlighted else MEASUREMENT_LINE_WIDTH
        )
        line_z = (
            SEGMENT_LINE_Z + MEASUREMENT_HIGHLIGHT_Z_OFFSET
            if highlighted
            else SEGMENT_LINE_Z
        )

        graphics.line.setPen(self._measurement_pen(measurement.line_color, line_width))
        graphics.line.setZValue(line_z)
        for handle in graphics.handles:
            handle.set_highlighted(highlighted)

    def _set_endpoint_label(
        self,
        label: QGraphicsTextItem,
        text: str,
        point: QPointF,
        horizontal_offset: float,
    ) -> None:
        if not text:
            label.setVisible(False)
            return
        label.setPlainText(text)
        self._position_segment_text(label, point, horizontal_offset, -18.0)

    def _position_segment_text(
        self,
        label: QGraphicsTextItem,
        anchor: QPointF,
        horizontal_offset_px: float,
        vertical_offset_px: float,
    ) -> None:
        text_rect = label.boundingRect()
        scale_x = abs(self.transform().m11()) or 1.0
        scale_y = abs(self.transform().m22()) or scale_x
        x = (
            anchor.x()
            + horizontal_offset_px / scale_x
            - text_rect.center().x() / scale_x
        )
        y = (
            anchor.y()
            + vertical_offset_px / scale_y
            - text_rect.height() / (2.0 * scale_y)
        )
        if self._loaded_image is not None:
            image_rect = self._image_item.boundingRect()
            label_width = text_rect.width() / scale_x
            label_height = text_rect.height() / scale_y
            max_x = max(image_rect.left(), image_rect.right() - label_width)
            max_y = max(image_rect.top(), image_rect.bottom() - label_height)
            x = min(max(x, image_rect.left()), max_x)
            y = min(max(y, image_rect.top()), max_y)
        label.setPos(x, y)
        label.setVisible(True)

    def _position_angle_label(self, label: QGraphicsTextItem, vertex: QPointF) -> None:
        text_rect = label.boundingRect()
        scale_x = abs(self.transform().m11()) or 1.0
        scale_y = abs(self.transform().m22()) or scale_x
        vertical_offset = 18.0 / scale_y
        x = vertex.x() - text_rect.center().x() / scale_x
        y = vertex.y() - vertical_offset - text_rect.height() / scale_y
        if self._loaded_image is not None:
            image_rect = self._image_item.boundingRect()
            label_width = text_rect.width() / scale_x
            label_height = text_rect.height() / scale_y
            max_x = max(image_rect.left(), image_rect.right() - label_width)
            max_y = max(image_rect.top(), image_rect.bottom() - label_height)
            x = min(max(x, image_rect.left()), max_x)
            y = min(max(y, image_rect.top()), max_y)
        label.setPos(x, y)
        label.setVisible(True)

    def _selected_angle_vertex_index(self) -> int | None:
        for index, graphics in enumerate(self._angle_graphics):
            if (
                index < len(self._angle_measurements)
                and self._angle_measurements[index].visible
                and len(graphics.handles) > 1
                and graphics.handles[1].isSelected()
            ):
                return index
        return None

    def _selected_segment_index(self) -> int | None:
        for index, graphics in enumerate(self._segment_graphics):
            if (
                index < len(self._segment_measurements)
                and self._segment_measurements[index].visible
                and any(handle.isSelected() for handle in graphics.handles)
            ):
                return index
        return None

    def _clear_handles(self) -> None:
        for handle in self._handles:
            self._scene.removeItem(handle)
        self._handles.clear()

    def _rebuild_handles(self) -> None:
        self._clear_handles()
        self._suppress_handle_events = True
        try:
            for index, point in enumerate(self._contour_points):
                handle = NodeHandleItem(self, index, point)
                self._scene.addItem(handle)
                self._handles.append(handle)
            self._apply_contour_visibility()
            self._apply_contour_highlight()
        finally:
            self._suppress_handle_events = False

    def _apply_contour_visibility(self) -> None:
        visible = self.has_contour() and self._contour_visible
        self._path_item.setVisible(visible)
        for handle in self._handles:
            handle.setVisible(visible)
            if not visible:
                self._suppress_contour_selection_highlight = True
                try:
                    handle.setSelected(False)
                finally:
                    self._suppress_contour_selection_highlight = False

    def _clear_calibration_handles(self) -> None:
        for handle in self._calibration_handles:
            self._scene.removeItem(handle)
        self._calibration_handles.clear()

    def _rebuild_calibration_handles(self) -> None:
        self._clear_calibration_handles()
        self._suppress_calibration_handle_events = True
        try:
            for index, point in enumerate(self._calibration_points):
                handle = CalibrationHandleItem(self, index, point)
                self._scene.addItem(handle)
                self._calibration_handles.append(handle)
        finally:
            self._suppress_calibration_handle_events = False

    def _contour_mask(self) -> np.ndarray:
        if not self._loaded_image:
            raise ValueError("Сначала загрузите изображение проекта.")

        mask = np.zeros(self._loaded_image.rgb_array.shape[:2], dtype=np.uint8)
        if len(self._contour_points) < 3:
            return mask

        polygon = np.array(
            [
                [int(round(point.x())), int(round(point.y()))]
                for point in self._contour_points
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 255)
        return mask


def _distance_to_segment(
    point: QPointF, start: QPointF, end: QPointF
) -> tuple[float, QPointF]:
    start_x, start_y = start.x(), start.y()
    end_x, end_y = end.x(), end.y()
    point_x, point_y = point.x(), point.y()

    delta_x = end_x - start_x
    delta_y = end_y - start_y
    segment_length_squared = delta_x * delta_x + delta_y * delta_y
    if segment_length_squared <= 1e-9:
        projection = QPointF(start_x, start_y)
        distance = ((point_x - start_x) ** 2 + (point_y - start_y) ** 2) ** 0.5
        return distance, projection

    projection_factor = (
        (point_x - start_x) * delta_x + (point_y - start_y) * delta_y
    ) / segment_length_squared
    projection_factor = max(0.0, min(1.0, projection_factor))

    projected_x = start_x + projection_factor * delta_x
    projected_y = start_y + projection_factor * delta_y
    distance = ((point_x - projected_x) ** 2 + (point_y - projected_y) ** 2) ** 0.5
    return distance, QPointF(projected_x, projected_y)


def _new_angle_id() -> str:
    return f"angle-{uuid4()}"


def _new_segment_id() -> str:
    return f"segment-{uuid4()}"


def _point_distance(start: QPointF, end: QPointF) -> float:
    return math.hypot(end.x() - start.x(), end.y() - start.y())


def _calibration_length_mm_from_text(text: str) -> float | None:
    normalized = str(text or "").strip().lower().replace(",", ".")
    for suffix in ("мм", "mm"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
            break
    try:
        value = float(normalized)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _format_segment_length_label(
    start: QPointF,
    end: QPointF,
    calibration_points: list[QPointF],
    calibration_length_mm: float | None,
) -> str:
    pixel_length = _point_distance(start, end)
    pixel_text = f"{_compact_float(pixel_length, 1)} px"
    if len(calibration_points) != 2 or calibration_length_mm is None:
        return pixel_text

    calibration_pixel_length = _point_distance(
        calibration_points[0], calibration_points[1]
    )
    if calibration_pixel_length <= 1e-6:
        return pixel_text
    length_mm = pixel_length * calibration_length_mm / calibration_pixel_length
    return f"{pixel_text} / {_compact_float(length_mm, 2)} мм"


def _segment_display_name(name: str, segment_index: int) -> str:
    return str(name or "").strip() or f"Отрезок {segment_index}"


def _format_segment_label(display_name: str, length_text: str) -> str:
    return f"{display_name}\n{length_text}"


def _compact_float(value: float, decimals: int) -> str:
    text = f"{float(value):.{decimals}f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _angle_arc_path(first: QPointF, vertex: QPointF, second: QPointF) -> QPainterPath:
    path = QPainterPath()
    first_x = first.x() - vertex.x()
    first_y = first.y() - vertex.y()
    second_x = second.x() - vertex.x()
    second_y = second.y() - vertex.y()
    first_length = math.hypot(first_x, first_y)
    second_length = math.hypot(second_x, second_y)
    shortest_ray = min(first_length, second_length)
    if shortest_ray <= 1e-6:
        return path

    start_angle = math.atan2(first_y, first_x)
    end_angle = math.atan2(second_y, second_x)
    span_angle = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    if abs(span_angle) <= 1e-6:
        return path

    radius = min(28.0, max(shortest_ray * 0.4, min(10.0, shortest_ray * 0.75)))
    if radius <= 1e-6:
        return path

    step_count = max(8, int(math.ceil(abs(span_angle) / (math.pi / 24.0))))
    for step in range(step_count + 1):
        angle = start_angle + span_angle * (step / step_count)
        point = QPointF(
            vertex.x() + math.cos(angle) * radius,
            vertex.y() + math.sin(angle) * radius,
        )
        if step == 0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    return path


def _angle_degrees(first: QPointF, vertex: QPointF, second: QPointF) -> float | None:
    first_x = first.x() - vertex.x()
    first_y = first.y() - vertex.y()
    second_x = second.x() - vertex.x()
    second_y = second.y() - vertex.y()
    first_length = math.hypot(first_x, first_y)
    second_length = math.hypot(second_x, second_y)
    if first_length <= 1e-6 or second_length <= 1e-6:
        return None

    cosine = (first_x * second_x + first_y * second_y) / (first_length * second_length)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _format_angle_degrees(value: float) -> str:
    rounded = round(value, 1)
    if math.isclose(rounded, round(rounded), abs_tol=1e-9):
        return f"{int(round(rounded))}°"
    return f"{rounded:.1f}°"


def _angle_display_name(name: str, angle_index: int) -> str:
    return str(name or "").strip() or f"Угол {angle_index + 1}"


def _format_angle_label(display_name: str, value: float | None) -> str:
    angle_value = "-" if value is None else _format_angle_degrees(value)
    return f"{display_name}: {angle_value}"
