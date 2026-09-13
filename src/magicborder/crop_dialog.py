from __future__ import annotations

from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .image_crop import CropBox, full_crop_box, normalize_crop_box

CROP_DIALOG_TITLE = "Обрезка изображения"
CROP_BUTTON_TEXT = "Обрезать"
KEEP_AS_IS_BUTTON_TEXT = "Оставить как есть"
RESET_SELECTION_BUTTON_TEXT = "Сбросить рамку"
CROP_HINT_TEXT = (
    "Потяните за углы или края рамки, перетащите её целиком "
    "либо выделите новую область мышью. Изображение открыто в исходном "
    "разрешении; понизить его можно на следующем шаге."
)
CROP_FRAME_COLOR = "#2a9d8f"
CROP_SHADE_COLOR = QColor(15, 23, 42, 130)
HANDLE_RADIUS = 5.5
HANDLE_HIT_PX = 10.0
VIEW_MARGIN_PX = 12

_HANDLE_CURSORS = {
    "tl": Qt.SizeFDiagCursor,
    "br": Qt.SizeFDiagCursor,
    "tr": Qt.SizeBDiagCursor,
    "bl": Qt.SizeBDiagCursor,
    "l": Qt.SizeHorCursor,
    "r": Qt.SizeHorCursor,
    "t": Qt.SizeVerCursor,
    "b": Qt.SizeVerCursor,
    "move": Qt.SizeAllCursor,
}


class CropSelectionView(QGraphicsView):
    """Просмотр фотографии с рамкой выбора области обрезки."""

    selection_changed = pyqtSignal(object)

    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_size = (pixmap.width(), pixmap.height())
        self._box = full_crop_box(self._image_size)
        self._drag_mode: str | None = None
        self._drag_origin_box = self._box
        self._drag_origin_pos = QPointF()

        scene = QGraphicsScene(self)
        self.setScene(scene)
        image_item = QGraphicsPixmapItem(pixmap)
        image_item.setTransformationMode(Qt.SmoothTransformation)
        scene.addItem(image_item)
        scene.setSceneRect(QRectF(0, 0, *self._image_size))

        self._shade_item = QGraphicsPathItem()
        self._shade_item.setBrush(QBrush(CROP_SHADE_COLOR))
        self._shade_item.setPen(QPen(Qt.NoPen))
        self._shade_item.setZValue(1)
        scene.addItem(self._shade_item)

        thirds_pen = QPen(QColor(255, 255, 255, 120), 1.0)
        thirds_pen.setCosmetic(True)
        self._thirds_item = QGraphicsPathItem()
        self._thirds_item.setPen(thirds_pen)
        self._thirds_item.setZValue(2)
        scene.addItem(self._thirds_item)

        frame_pen = QPen(QColor(CROP_FRAME_COLOR), 2.0)
        frame_pen.setCosmetic(True)
        self._frame_item = QGraphicsRectItem()
        self._frame_item.setPen(frame_pen)
        self._frame_item.setBrush(QBrush(Qt.NoBrush))
        self._frame_item.setZValue(3)
        scene.addItem(self._frame_item)

        self._handle_items: dict[str, QGraphicsEllipseItem] = {}
        for key in ("tl", "t", "tr", "r", "br", "b", "bl", "l"):
            handle = QGraphicsEllipseItem(
                -HANDLE_RADIUS, -HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2
            )
            handle.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            handle.setBrush(QColor(CROP_FRAME_COLOR))
            handle.setPen(QPen(QColor("white"), 1.5))
            handle.setZValue(4)
            scene.addItem(handle)
            self._handle_items[key] = handle

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#eef2f7"))
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setMinimumSize(360, 260)
        self._refresh_selection_items()

    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def crop_box(self) -> CropBox:
        return self._box

    def set_crop_box(self, box: CropBox) -> None:
        normalized = normalize_crop_box(
            box.left, box.top, box.right, box.bottom, self._image_size
        )
        if normalized == self._box:
            return
        self._box = normalized
        self._refresh_selection_items()
        self.selection_changed.emit(normalized)

    def reset_selection(self) -> None:
        self.set_crop_box(full_crop_box(self._image_size))

    def fit_image(self) -> None:
        width, height = self._image_size
        margin_x = VIEW_MARGIN_PX * width / max(1, self.viewport().width())
        margin_y = VIEW_MARGIN_PX * height / max(1, self.viewport().height())
        self.fitInView(
            QRectF(-margin_x, -margin_y, width + 2 * margin_x, height + 2 * margin_y),
            Qt.KeepAspectRatio,
        )

    def handle_at(self, view_pos: QPoint) -> str | None:
        """Возвращает маркер под курсором, 'move' внутри рамки или None."""
        box = self._box
        top_left = self.mapFromScene(QPointF(box.left, box.top))
        bottom_right = self.mapFromScene(QPointF(box.right, box.bottom))
        x, y = float(view_pos.x()), float(view_pos.y())
        left, top = float(top_left.x()), float(top_left.y())
        right, bottom = float(bottom_right.x()), float(bottom_right.y())

        near_left = abs(x - left) <= HANDLE_HIT_PX
        near_right = abs(x - right) <= HANDLE_HIT_PX
        near_top = abs(y - top) <= HANDLE_HIT_PX
        near_bottom = abs(y - bottom) <= HANDLE_HIT_PX
        within_x = left - HANDLE_HIT_PX <= x <= right + HANDLE_HIT_PX
        within_y = top - HANDLE_HIT_PX <= y <= bottom + HANDLE_HIT_PX

        vertical = "t" if near_top else "b" if near_bottom else ""
        horizontal = "l" if near_left else "r" if near_right else ""
        if vertical and horizontal:
            return vertical + horizontal
        if horizontal and within_y:
            return horizontal
        if vertical and within_x:
            return vertical
        if left < x < right and top < y < bottom:
            return "move"
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        scene_pos = self.mapToScene(event.pos())
        mode = self.handle_at(event.pos())
        if mode is None:
            if not self.sceneRect().contains(scene_pos):
                event.accept()
                return
            mode = "new"
        self._drag_mode = mode
        self._drag_origin_box = self._box
        self._drag_origin_pos = self._clamped_scene_pos(scene_pos)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_mode is None:
            self._update_hover_cursor(event.pos())
            super().mouseMoveEvent(event)
            return
        self._drag_to(self._clamped_scene_pos(self.mapToScene(event.pos())))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drag_mode is not None:
            self._drag_to(self._clamped_scene_pos(self.mapToScene(event.pos())))
            self._drag_mode = None
            self._update_hover_cursor(event.pos())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit_image()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.fit_image()

    def _drag_to(self, pos: QPointF) -> None:
        origin = self._drag_origin_box
        mode = self._drag_mode
        if mode == "new":
            start = self._drag_origin_pos
            box = normalize_crop_box(
                start.x(), start.y(), pos.x(), pos.y(), self._image_size
            )
        elif mode == "move":
            image_width, image_height = self._image_size
            dx = pos.x() - self._drag_origin_pos.x()
            dy = pos.y() - self._drag_origin_pos.y()
            left = min(max(round(origin.left + dx), 0), image_width - origin.width)
            top = min(max(round(origin.top + dy), 0), image_height - origin.height)
            box = CropBox(left, top, origin.width, origin.height)
        elif mode:
            left, top = float(origin.left), float(origin.top)
            right, bottom = float(origin.right), float(origin.bottom)
            if "l" in mode:
                left = pos.x()
            if "r" in mode:
                right = pos.x()
            if "t" in mode:
                top = pos.y()
            if "b" in mode:
                bottom = pos.y()
            box = normalize_crop_box(left, top, right, bottom, self._image_size)
        else:
            return
        self.set_crop_box(box)

    def _clamped_scene_pos(self, pos: QPointF) -> QPointF:
        width, height = self._image_size
        return QPointF(min(max(pos.x(), 0.0), width), min(max(pos.y(), 0.0), height))

    def _update_hover_cursor(self, view_pos: QPoint) -> None:
        mode = self.handle_at(view_pos)
        if mode is not None:
            self.viewport().setCursor(_HANDLE_CURSORS[mode])
        elif self.sceneRect().contains(self.mapToScene(view_pos)):
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().unsetCursor()

    def _refresh_selection_items(self) -> None:
        box = self._box
        rect = QRectF(box.left, box.top, box.width, box.height)

        shade = QPainterPath()
        shade.setFillRule(Qt.OddEvenFill)
        shade.addRect(QRectF(0, 0, *self._image_size))
        shade.addRect(rect)
        self._shade_item.setPath(shade)
        self._frame_item.setRect(rect)

        thirds = QPainterPath()
        for fraction in (1 / 3, 2 / 3):
            x = rect.left() + rect.width() * fraction
            y = rect.top() + rect.height() * fraction
            thirds.moveTo(x, rect.top())
            thirds.lineTo(x, rect.bottom())
            thirds.moveTo(rect.left(), y)
            thirds.lineTo(rect.right(), y)
        self._thirds_item.setPath(thirds)

        center_x, center_y = rect.center().x(), rect.center().y()
        positions = {
            "tl": (rect.left(), rect.top()),
            "t": (center_x, rect.top()),
            "tr": (rect.right(), rect.top()),
            "r": (rect.right(), center_y),
            "br": (rect.right(), rect.bottom()),
            "b": (center_x, rect.bottom()),
            "bl": (rect.left(), rect.bottom()),
            "l": (rect.left(), center_y),
        }
        for key, (x, y) in positions.items():
            self._handle_items[key].setPos(x, y)


class CropDialog(QDialog):
    """Диалог выбора области обрезки в исходном разрешении изображения.

    Разрешение здесь не меняется: уменьшение предлагается отдельным шагом
    после подтверждения обрезки.
    """

    def __init__(
        self,
        pixmap: QPixmap,
        *,
        display_name: str,
        geometry_note: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(CROP_DIALOG_TITLE)
        self.setObjectName("cropDialog")
        self.setStyleSheet(
            "QLabel#cropDialogTitle { color: #1f2937; font-size: 13px; font-weight: 600; }"
            "QLabel#cropDialogHint { color: #5f6b7a; }"
            "QLabel#cropDialogWarning { color: #b45309; }"
            "QLabel#cropDialogArea { color: #1f2937; font-weight: 600; }"
        )

        self.selection_view = CropSelectionView(pixmap, self)
        image_size = self.selection_view.image_size()

        title_label = QLabel(
            f"{display_name} — {image_size[0]}×{image_size[1]} px", self
        )
        title_label.setObjectName("cropDialogTitle")
        title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        hint_label = QLabel(CROP_HINT_TEXT, self)
        hint_label.setObjectName("cropDialogHint")
        hint_label.setWordWrap(True)

        self.area_label = QLabel(self)
        self.area_label.setObjectName("cropDialogArea")

        warning_lines = ["Файл изображения в папке проекта будет перезаписан."]
        if geometry_note:
            warning_lines.append(
                "Контур, калибровка и измерения будут пересчитаны; "
                "точки за пределами области прижмутся к её краю."
            )
        warning_label = QLabel("\n".join(warning_lines), self)
        warning_label.setObjectName("cropDialogWarning")
        warning_label.setWordWrap(True)

        self.button_box = QDialogButtonBox(self)
        self.reset_button = self.button_box.addButton(
            RESET_SELECTION_BUTTON_TEXT, QDialogButtonBox.ResetRole
        )
        self.keep_button = self.button_box.addButton(
            KEEP_AS_IS_BUTTON_TEXT, QDialogButtonBox.RejectRole
        )
        self.keep_button.setToolTip(
            "Не обрезать изображение. Затем можно понизить его разрешение."
        )
        self.crop_button = self.button_box.addButton(
            CROP_BUTTON_TEXT, QDialogButtonBox.AcceptRole
        )
        self.crop_button.setDefault(True)
        self.reset_button.clicked.connect(self.selection_view.reset_selection)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        area_row = QHBoxLayout()
        area_row.addWidget(self.area_label)
        area_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(title_label)
        layout.addWidget(hint_label)
        layout.addWidget(self.selection_view, 1)
        layout.addLayout(area_row)
        layout.addWidget(warning_label)
        layout.addWidget(self.button_box)

        self.selection_view.selection_changed.connect(self._update_summary)
        self._update_summary(self.selection_view.crop_box())

        if parent is not None:
            parent_size = parent.size()
            self.resize(
                max(640, int(parent_size.width() * 0.75)),
                max(520, int(parent_size.height() * 0.8)),
            )
        else:
            self.resize(900, 700)

    def crop_box(self) -> CropBox:
        return self.selection_view.crop_box()

    def _update_summary(self, box: CropBox) -> None:
        self.area_label.setText(
            f"Область: {box.left}, {box.top} — {box.width}×{box.height} px"
        )
        # Отказ от обрезки — кнопка «Оставить как есть», поэтому «Обрезать»
        # доступна только для рамки меньше всего кадра.
        self.crop_button.setEnabled(not box.is_full(self.selection_view.image_size()))
