from __future__ import annotations

import math
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from PyQt5.QtCore import (
    QDateTime,
    QEventLoop,
    QObject,
    QRunnable,
    QSignalBlocker,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .canvas import ImageCanvas
from .contour_analysis import (
    ContourAnalysis,
    ContourColorSums,
    ContourSignature,
    build_contour_analysis,
    color_stats_from_sums,
    combine_color_sums,
    contour_color_sums,
    contour_rgb_pixels_from_points,
    contour_signature,
    flatten_background_outside_contour,
    hsv_values_from_rgb_pixels,
    lab_values_from_rgb_pixels,
    mean_lms_values_from_total,
    yuv_values_from_rgb_pixels,
)
from .detector import detect_leaf_contour
from .histograms import (
    HistogramPanel,
    rgb_to_lms,
)
from .icons import ACTION_VISUALS, TOOLBAR_ICON_SIZE, apply_action_visual, load_icon
from .image_downscale import (
    DOWNSCALE_PRESETS,
    LARGE_IMAGE_THRESHOLD,
    DownscalePreset,
    downscale_image_file,
    fit_size,
    is_large_image,
    read_image_header,
    verify_raster_image,
)
from .io_utils import (
    SUPPORTED_RASTER_SUFFIXES,
    image_open_filter,
    load_annotation,
    load_project,
    load_raster_image,
    load_rgb_array,
    save_annotation,
    save_project,
    save_rgb_array_atomically,
    write_xlsx_table,
)
from .models import (
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
    ProjectImageRecord,
    ProjectMeasurementAssessment,
    ProjectSegmentMeasurement,
    default_project_image_metadata,
    normalize_line_color,
)
from .oiv import (
    OIV_SYSTEM_NAME,
    OivClassificationResult,
    OivScaleCatalog,
    OivScaleError,
    OivTrait,
    default_oiv_catalog,
    make_status_result,
    normalize_oiv_code,
)
from .path_utils import portable_path_reference
from .property_browser import PropertyBrowser, PropertyValueLabel

APP_TITLE = "MagicBorder"
WORKSPACE_DEFAULT_SIZES = [260, 860, 280]
PROJECT_PANEL_DEFAULT_SIZES = [380, 220, 320]
HISTOGRAM_DEFAULT_SIZES = [170, 170, 170, 170, 170]
CONTOUR_ANALYSIS_SYNC_PIXEL_LIMIT = 300_000
CONTOUR_ANALYSIS_PENDING_TEXT = "расчёт..."
CONTOUR_ANALYSIS_OUTDATED_TEXT = "нажмите «Обновить»"
HISTOGRAM_MANUAL_REFRESH_TEXT = (
    "Нажмите «Обновить», чтобы построить гистограммы по текущему контуру."
)
ANALYSIS_OUTDATED_STATUS_TEXT = "Данные устарели"
DOWNSCALE_CANCELLED = "cancel"
LARGE_IMAGE_LIST_LIMIT = 8
IMAGE_PREPARE_CANCELLED_TEXT = "Отменено пользователем."
IMAGE_PREPARE_PROGRESS_DELAY_MS = 300
BACKGROUND_TASK_SHUTDOWN_TIMEOUT_MS = 2000
ContourAnalysisCacheKey = tuple[str | None, str | None, ContourSignature]
# Путь, mtime_ns и размер файла + сигнатура контура: если меняется файл
# (например, после выравнивания фона) или контур, запись пересчитывается.
ProjectColorSumsCacheKey = tuple[str, int, int, ContourSignature]
ProjectMeanColorStats = tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]


@dataclass(frozen=True, slots=True)
class ExportTreeItem:
    label: str
    key: str = ""
    export_label: str = ""
    children: tuple[ExportTreeItem, ...] = ()


@dataclass(frozen=True, slots=True)
class ContourAnalysisWorkResult:
    request_id: int
    record_id: str | None
    image_path: str | None
    signature: ContourSignature
    analysis: ContourAnalysis | None


class _ContourAnalysisWorkerSignals(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(int, str)


class _ContourAnalysisWorker(QRunnable):
    def __init__(
        self,
        request_id: int,
        record_id: str | None,
        image_path: str | None,
        signature: ContourSignature,
        rgb_array: np.ndarray,
        points: list[Point],
    ) -> None:
        super().__init__()
        self.signals = _ContourAnalysisWorkerSignals()
        self._request_id = request_id
        self._record_id = record_id
        self._image_path = image_path
        self._signature = signature
        self._rgb_array = rgb_array
        self._points = points

    def run(self) -> None:
        try:
            analysis = build_contour_analysis(self._rgb_array, self._points)
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive boundary for worker failures
            self.signals.failed.emit(self._request_id, str(exc))
            return

        self.signals.finished.emit(
            ContourAnalysisWorkResult(
                request_id=self._request_id,
                record_id=self._record_id,
                image_path=self._image_path,
                signature=self._signature,
                analysis=analysis,
            )
        )


@dataclass(frozen=True, slots=True)
class ImageImportCandidate:
    source: Path
    label: str
    size: tuple[int, int]
    captured_at: str


@dataclass(frozen=True, slots=True)
class ImagePrepareJob:
    source: Path
    destination: Path
    preset: DownscalePreset | None = None


@dataclass(frozen=True, slots=True)
class ImagePrepareResult:
    width: int
    height: int
    downscaled: bool


def prepare_image_file(job: ImagePrepareJob) -> ImagePrepareResult:
    if job.preset is not None:
        width, height = downscale_image_file(job.source, job.destination, job.preset)
        return ImagePrepareResult(width, height, downscaled=True)

    width, height = verify_raster_image(job.source)
    if job.source.resolve() != job.destination.resolve():
        shutil.copy2(job.source, job.destination)
    return ImagePrepareResult(width, height, downscaled=False)


class _ImagePrepareWorkerSignals(QObject):
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class _ImagePrepareWorker(QRunnable):
    def __init__(
        self, index: int, job: ImagePrepareJob, cancel_event: threading.Event
    ) -> None:
        super().__init__()
        self.signals = _ImagePrepareWorkerSignals()
        self._index = index
        self._job = job
        self._cancel_event = cancel_event

    def run(self) -> None:
        if self._cancel_event.is_set():
            self.signals.failed.emit(self._index, IMAGE_PREPARE_CANCELLED_TEXT)
            return
        try:
            result = prepare_image_file(self._job)
        except Exception as exc:  # defensive boundary for worker failures
            self.signals.failed.emit(self._index, str(exc))
            return
        self.signals.finished.emit(self._index, result)


class _ContourDetectionWorkerSignals(QObject):
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class _ContourDetectionWorker(QRunnable):
    def __init__(self, request_id: int, rgb_array: np.ndarray) -> None:
        super().__init__()
        self.signals = _ContourDetectionWorkerSignals()
        self._request_id = request_id
        # View только для чтения, без копии: канвас не меняет массив на месте.
        self._rgb_array = rgb_array

    def run(self) -> None:
        try:
            # cv2.grabCut и остальные функции OpenCV отпускают GIL, поэтому
            # GUI-поток продолжает обрабатывать события, пока идёт расчёт.
            points = detect_leaf_contour(self._rgb_array)
        except Exception as exc:  # defensive boundary for worker failures
            self.signals.failed.emit(self._request_id, str(exc))
            return
        self.signals.finished.emit(self._request_id, points)


@dataclass(frozen=True, slots=True)
class PendingContourDetection:
    request_id: int
    record_id: str | None
    image_path: Path | None


@dataclass(frozen=True, slots=True)
class ProjectColorStatsEntry:
    cache_key: ProjectColorSumsCacheKey
    image_path: Path
    points: list[Point]
    expected_size: tuple[int, int]


def compute_project_color_sums(
    entry: ProjectColorStatsEntry,
) -> ContourColorSums | None:
    """Суммы цветов одного изображения проекта. Безопасно вызывать вне GUI-потока."""
    try:
        # Только RGB-массив: без QImage/QPixmap, которые для статистики не нужны.
        rgb_array = load_rgb_array(entry.image_path)
    except (OSError, ValueError):
        return None
    height, width = rgb_array.shape[:2]
    if (width, height) != entry.expected_size:
        return None
    return contour_color_sums(rgb_array, entry.points)


class _ProjectColorStatsWorkerSignals(QObject):
    progress = pyqtSignal(int, int, int)
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class _ProjectColorStatsWorker(QRunnable):
    def __init__(
        self,
        request_id: int,
        entries: list[ProjectColorStatsEntry],
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.signals = _ProjectColorStatsWorkerSignals()
        self._request_id = request_id
        self._entries = entries
        self._cancel_event = cancel_event

    def run(self) -> None:
        results: dict[ProjectColorSumsCacheKey, ContourColorSums | None] = {}
        total = len(self._entries)
        try:
            for index, entry in enumerate(self._entries, start=1):
                if self._cancel_event.is_set():
                    self.signals.failed.emit(
                        self._request_id, IMAGE_PREPARE_CANCELLED_TEXT
                    )
                    return
                # Изображения обрабатываются по одному: в памяти не больше
                # одного декодированного кадра, а не весь проект сразу.
                results[entry.cache_key] = compute_project_color_sums(entry)
                self.signals.progress.emit(self._request_id, index, total)
        except Exception as exc:  # defensive boundary for worker failures
            self.signals.failed.emit(self._request_id, str(exc))
            return
        self.signals.finished.emit(self._request_id, results)


@dataclass(slots=True)
class PendingProjectColorStats:
    request_id: int
    entries: list[ProjectColorStatsEntry]
    cancel_event: threading.Event


@dataclass(frozen=True, slots=True)
class FlattenBackgroundJob:
    request_id: int
    record_id: str
    image_path: Path
    rgb_array: np.ndarray
    points: list[Point]


def flatten_background_file(job: FlattenBackgroundJob) -> np.ndarray:
    flattened = flatten_background_outside_contour(job.rgb_array, job.points)
    save_rgb_array_atomically(flattened, job.image_path)
    return flattened


class _FlattenBackgroundWorkerSignals(QObject):
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class _FlattenBackgroundWorker(QRunnable):
    def __init__(self, job: FlattenBackgroundJob) -> None:
        super().__init__()
        self.signals = _FlattenBackgroundWorkerSignals()
        self._job = job

    def run(self) -> None:
        try:
            # Маска, заливка и кодирование PNG/JPEG/TIFF идут вне GUI-потока.
            flattened = flatten_background_file(self._job)
        except Exception as exc:  # defensive boundary for worker failures
            self.signals.failed.emit(self._job.request_id, str(exc))
            return
        self.signals.finished.emit(self._job.request_id, flattened)


class AverageColorSwatch(QFrame):
    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


PROJECT_EXPORT_COLUMNS = [
    ("id", "ID изображения"),
    ("file_name", "Имя файла"),
    ("relative_path", "Относительный путь"),
    ("has_annotation", "Есть аннотация"),
    ("status", "Статус"),
    ("diagnosis", "Диагноз"),
    ("r", "Средний R"),
    ("g", "Средний G"),
    ("b", "Средний B"),
    ("contour_pixel_count", "Количество пикселов контура"),
]
PROJECT_EXPORT_FIELDNAMES = [
    field_name for field_name, _label in PROJECT_EXPORT_COLUMNS
]
PROJECT_EXPORT_COLUMN_LABELS = dict(PROJECT_EXPORT_COLUMNS)
COLOR_SPACES_GROUP_TITLE = "Цветовые пространства"
IMAGE_FILE_PROPERTY_GROUPS = [
    (
        "Общая информация о файле",
        [
            ("id", "ID"),
            ("file_name", "Файл"),
            ("relative_path", "Путь"),
            ("size", "Размер"),
            ("status", "Статус"),
            ("added_at", "Дата добавления"),
            ("captured_at", "Дата съёмки"),
        ],
    ),
    (
        "Калибровка",
        [
            ("calibration_length_mm", "Длина калибровки, мм"),
            ("calibration_scale", "Масштаб"),
        ],
    ),
    (
        "Информация о главном контуре",
        [
            ("annotation", "Аннотация"),
            ("point_count", "Количество узлов контура"),
            ("contour_pixel_count", "Количество пикселов контура"),
            ("contour_area_mm2", "Площадь контура, мм²"),
        ],
    ),
]
IMAGE_COLOR_SPACE_PROPERTY_GROUPS = [
    (
        "Цветовое пространство RGB",
        [
            ("red", "Красный"),
            ("green", "Зелёный"),
            ("blue", "Синий"),
            ("average_color", "Средний цвет"),
        ],
    ),
    (
        "Цветовое пространство Lab",
        [
            ("lab_l", "L"),
            ("lab_a", "a"),
            ("lab_b", "b"),
        ],
    ),
    (
        "Цветовое пространство HSV",
        [
            ("hsv_h", "H"),
            ("hsv_s", "S"),
            ("hsv_v", "V"),
        ],
    ),
    (
        "Цветовое пространство YUV",
        [
            ("yuv_y", "Y"),
            ("yuv_u", "U"),
            ("yuv_v", "V"),
        ],
    ),
    (
        "Цветовое пространство LMS",
        [
            ("lms_l", "L"),
            ("lms_m", "M"),
            ("lms_s", "S"),
        ],
    ),
]
IMAGE_TRAILING_PROPERTY_GROUPS = [
    (
        "Локация",
        [
            ("illumination", "Освещённость"),
            ("humidity", "Влажность, %"),
            ("wind_speed", "Скорость ветра"),
            ("wind_direction", "Направление ветра"),
            ("latitude", "Широта"),
            ("longitude", "Долгота"),
        ],
    ),
    (
        "Дополнительно",
        [
            ("diagnosis", "Диагноз"),
            ("notes", "Дополнительные сведения"),
        ],
    ),
]
IMAGE_PROPERTY_GROUPS = [
    *IMAGE_FILE_PROPERTY_GROUPS,
    *IMAGE_COLOR_SPACE_PROPERTY_GROUPS,
    *IMAGE_TRAILING_PROPERTY_GROUPS,
]
IMAGE_PROPERTY_EXPORT_ITEMS = [
    item for _group_title, group_items in IMAGE_PROPERTY_GROUPS for item in group_items
]
IMAGE_PROPERTY_EXPORT_KEYS = [
    field_name for field_name, _label in IMAGE_PROPERTY_EXPORT_ITEMS
]
IMAGE_PROPERTY_EXPORT_LABELS = dict(IMAGE_PROPERTY_EXPORT_ITEMS)


def _export_leaf(
    key: str, label: str, *, export_label: str | None = None
) -> ExportTreeItem:
    return ExportTreeItem(label=label, key=key, export_label=export_label or label)


def _export_group(
    group_title: str, group_items: list[tuple[str, str]]
) -> ExportTreeItem:
    return ExportTreeItem(
        label=group_title,
        children=tuple(
            _export_leaf(field_name, label) for field_name, label in group_items
        ),
    )


def _static_image_property_export_groups() -> list[ExportTreeItem]:
    return [
        *(
            _export_group(group_title, group_items)
            for group_title, group_items in IMAGE_FILE_PROPERTY_GROUPS
        ),
        ExportTreeItem(
            label=COLOR_SPACES_GROUP_TITLE,
            children=tuple(
                _export_group(group_title, group_items)
                for group_title, group_items in IMAGE_COLOR_SPACE_PROPERTY_GROUPS
            ),
        ),
        *(
            _export_group(group_title, group_items)
            for group_title, group_items in IMAGE_TRAILING_PROPERTY_GROUPS
        ),
    ]


def _export_tree_leaf_keys(
    items: list[ExportTreeItem] | tuple[ExportTreeItem, ...],
) -> list[str]:
    keys: list[str] = []
    for item in items:
        if item.key:
            keys.append(item.key)
        if item.children:
            keys.extend(_export_tree_leaf_keys(item.children))
    return keys


def _export_tree_labels(
    items: list[ExportTreeItem] | tuple[ExportTreeItem, ...],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in items:
        if item.key:
            labels[item.key] = item.export_label or item.label
        if item.children:
            labels.update(_export_tree_labels(item.children))
    return labels


def _default_image_property_export_keys() -> set[str]:
    for group_title, group_items in IMAGE_PROPERTY_GROUPS:
        if group_title == "Общая информация о файле":
            return {field_name for field_name, _label in group_items}
    return set()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 920)

        self.canvas = ImageCanvas(self)
        self.rgb_histogram_panel = HistogramPanel(
            "RGB-гистограмма",
            "rgb_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("rgb"),
        )
        self.lab_histogram_panel = HistogramPanel(
            "Lab-гистограмма",
            "lab_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("lab"),
        )
        self.hsv_histogram_panel = HistogramPanel(
            "HSV-гистограмма",
            "hsv_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("hsv"),
        )
        self.yuv_histogram_panel = HistogramPanel(
            "YUV-гистограмма",
            "yuv_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("yuv"),
        )
        self.lms_histogram_panel = HistogramPanel(
            "LMS-гистограмма",
            "lms_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("lms"),
        )
        self._analysis_refresh_allowed = False
        self._project_summary_refresh_timer = QTimer(self)
        self._project_summary_refresh_timer.setSingleShot(True)
        self._project_summary_refresh_timer.timeout.connect(
            self._update_project_summary_properties
        )
        self._contour_analysis_thread_pool = QThreadPool.globalInstance()
        self._image_prepare_thread_pool = QThreadPool(self)
        self._contour_analysis_request_id = 0
        self._contour_analysis_cache: ContourAnalysisWorkResult | None = None
        self._project_mean_color_stats_cache: ProjectMeanColorStats | None = None
        self._pending_contour_analysis_key: ContourAnalysisCacheKey | None = None
        self._pending_contour_analysis_request_id: int | None = None
        self._contour_analysis_workers: set[_ContourAnalysisWorker] = set()
        # Тяжёлые команды над изображениями выполняются в пуле потоков, чтобы
        # цикл событий Qt не останавливался и Windows не помечала окно «Не отвечает».
        self._contour_detection_thread_pool = QThreadPool.globalInstance()
        self._contour_detection_request_id = 0
        self._pending_contour_detection: PendingContourDetection | None = None
        # Ссылки на воркеры по request_id: без них сборщик мусора может удалить
        # объект сигналов раньше, чем поток их отправит.
        self._contour_detection_workers: dict[int, _ContourDetectionWorker] = {}
        self._project_color_stats_thread_pool = QThreadPool.globalInstance()
        self._project_color_stats_request_id = 0
        self._pending_project_color_stats: PendingProjectColorStats | None = None
        self._project_color_stats_workers: dict[int, _ProjectColorStatsWorker] = {}
        self._project_color_sums_cache: dict[
            ProjectColorSumsCacheKey, ContourColorSums | None
        ] = {}
        self._flatten_background_thread_pool = QThreadPool.globalInstance()
        self._flatten_background_request_id = 0
        self._pending_flatten_background: FlattenBackgroundJob | None = None
        self._flatten_background_workers: dict[int, _FlattenBackgroundWorker] = {}
        self._busy_task_count = 0
        self._current_average_color_rgb: tuple[int, int, int] | None = None

        self.project_document: ProjectDocument | None = None
        self.project_path: Path | None = None
        self._current_project_image_id: str | None = None
        self._loading_project_image = False
        self._updating_project_list = False
        self._updating_project_identity_fields = False
        self._updating_project_info_fields = False
        self._updating_metadata_fields = False
        self._updating_calibration_length_field = False
        self._project_autosave_timer = QTimer(self)
        self._project_autosave_timer.setSingleShot(True)
        self._project_autosave_timer.timeout.connect(self._save_project_silently)
        try:
            self._oiv_catalog: OivScaleCatalog | None = default_oiv_catalog()
            self._oiv_catalog_error = ""
        except OivScaleError as exc:
            self._oiv_catalog = None
            self._oiv_catalog_error = str(exc)

        self.setCentralWidget(self._create_workspace())

        self._current_annotation_path: Path | None = None

        self._create_actions()
        self._apply_action_visuals()
        self._create_menus()
        self._create_toolbar()

        self.statusBar().showMessage("Откройте фотографию листа растения.")
        # Неблокирующий индикатор фоновой работы в статус-баре: окно остаётся
        # интерактивным, а пользователь видит, что расчёт идёт.
        self.busy_progress = QProgressBar(self)
        self.busy_progress.setObjectName("busyProgress")
        self.busy_progress.setMaximumWidth(180)
        self.busy_progress.setTextVisible(False)
        self.busy_progress.hide()
        self.statusBar().addPermanentWidget(self.busy_progress)
        self.canvas.message_changed.connect(self.statusBar().showMessage)
        self.canvas.image_state_changed.connect(self._update_action_states)
        self.canvas.image_state_changed.connect(self._update_project_properties)
        self.canvas.contour_state_changed.connect(self._update_action_states)
        self.canvas.contour_geometry_changed.connect(
            self._handle_contour_geometry_changed
        )
        self.canvas.calibration_segment_selected.connect(
            self._handle_calibration_segment_selected
        )
        self.canvas.calibration_geometry_changed.connect(
            self._handle_calibration_geometry_changed
        )
        self.canvas.angle_state_changed.connect(self._handle_angle_state_changed)
        self.canvas.segment_state_changed.connect(self._handle_segment_state_changed)
        self._update_action_states()
        self._refresh_histograms()
        self._update_project_panel()

    def _create_workspace(self) -> QSplitter:
        project_panel = self._create_project_panel()
        analysis_panel = self._create_analysis_panel()

        self.workspace_splitter = QSplitter(Qt.Horizontal, self)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(project_panel)
        self.workspace_splitter.addWidget(self.canvas)
        self.workspace_splitter.addWidget(analysis_panel)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 4)
        self.workspace_splitter.setStretchFactor(2, 1)
        self.workspace_splitter.setSizes(WORKSPACE_DEFAULT_SIZES)
        self.workspace_splitter.setStyleSheet(
            "QSplitter::handle { background: #d5dbe5; width: 8px; }"
            "QSplitter::handle:hover { background: #bdc7d5; }"
        )
        return self.workspace_splitter

    def _create_project_panel(self) -> QWidget:
        project_panel = QWidget(self)
        project_panel.setObjectName("projectPanel")
        project_panel.setMinimumWidth(220)

        self.project_list = QListWidget(project_panel)
        self.project_list.setObjectName("projectImageList")
        self.project_list.setAlternatingRowColors(True)
        self.project_list.currentItemChanged.connect(
            self._handle_project_selection_changed
        )

        images_title = QLabel("Изображения проекта")
        images_title.setObjectName("projectPanelTitle")

        self.export_project_excel_button = QToolButton(project_panel)
        self.export_project_excel_button.setText("Excel")
        self.export_project_excel_button.setIcon(load_icon("export-csv"))
        self.export_project_excel_button.setToolTip("Экспорт списка в Excel")
        self.export_project_excel_button.setStatusTip(
            "Экспорт списка в Excel: сохранить список изображений проекта в файл .xlsx."
        )
        self.export_project_excel_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.export_project_excel_button.clicked.connect(self.export_project_excel)
        self.export_project_csv_button = self.export_project_excel_button

        images_header_layout = QHBoxLayout()
        images_header_layout.setContentsMargins(0, 0, 0, 0)
        images_header_layout.setSpacing(6)
        images_header_layout.addWidget(images_title)
        images_header_layout.addStretch(1)
        images_header_layout.addWidget(self.export_project_excel_button)

        images_widget = QWidget(project_panel)
        images_widget.setObjectName("projectImagesPanel")
        images_layout = QVBoxLayout(images_widget)
        images_layout.setContentsMargins(8, 8, 8, 8)
        images_layout.setSpacing(6)
        images_layout.addLayout(images_header_layout)
        images_layout.addWidget(self.project_list, 1)

        project_properties_widget = self._create_project_properties_panel(project_panel)
        properties_widget = self._create_image_properties_panel(project_panel)

        self.project_splitter = QSplitter(Qt.Vertical, project_panel)
        self.project_splitter.setChildrenCollapsible(False)
        self.project_splitter.addWidget(images_widget)
        self.project_splitter.addWidget(project_properties_widget)
        self.project_splitter.addWidget(properties_widget)
        self.project_splitter.setSizes(PROJECT_PANEL_DEFAULT_SIZES)
        self.project_splitter.setStyleSheet(
            "QSplitter::handle { background: #d5dbe5; height: 7px; }"
            "QSplitter::handle:hover { background: #bdc7d5; }"
        )

        layout = QVBoxLayout(project_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.project_splitter)

        project_panel.setStyleSheet(
            "QWidget#projectPanel { background: #f5f7fb; border-right: 1px solid #d6dde8; }"
            "QWidget#projectImagesPanel { background: #f7fbff; }"
            "QWidget#projectPropertiesPanel { background: #f8fcf6; }"
            "QWidget#imagePropertiesPanel { background: #fffaf4; }"
            "QLabel#projectPanelTitle { color: #1f2937; font-size: 13px; font-weight: 600; }"
            "QListWidget#projectImageList { background: #ffffff; color: #1f2937; border: 1px solid #ccd6e1; border-radius: 5px; outline: none; }"
            "QListWidget#projectImageList::item { padding: 7px 8px; border-bottom: 1px solid #e5eaf2; }"
            "QListWidget#projectImageList::item:selected { background: #d8edf3; color: #102a32; }"
            "QListWidget#projectImageList::item:alternate { background: #f7f9fc; }"
            "QLabel#propertyEmpty { color: #7a8696; padding: 12px; }"
            "QLineEdit, QTextEdit, QComboBox { background: #ffffff; color: #1f2937; border: 1px solid #ccd6e1; border-radius: 4px; padding: 4px; selection-background-color: #cdeaf2; selection-color: #102a32; }"
            "QLineEdit:focus, QTextEdit:focus, QComboBox:focus { border-color: #2A9D8F; }"
            "QLineEdit:read-only { color: #5f6b7a; background: #f1f4f8; }"
            "QFrame#averageColorSwatch { border: 1px solid #95a3b8; border-radius: 4px; background: transparent; }"
            "QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 4px; color: #1f2937; }"
            "QToolButton:hover { background: #e9eef5; border-color: #c6d1de; }"
            "QToolButton:pressed { background: #dce4ee; }"
        )
        return project_panel

    def _create_project_properties_panel(self, parent: QWidget) -> QWidget:
        properties_widget = QWidget(parent)
        properties_widget.setObjectName("projectPropertiesPanel")
        self.project_properties_panel = properties_widget

        title = QLabel("Свойства проекта")
        title.setObjectName("projectPanelTitle")

        self.project_properties_empty_label = QLabel("Проект не открыт.")
        self.project_properties_empty_label.setObjectName("propertyEmpty")
        self.project_properties_empty_label.setWordWrap(True)
        self.project_properties_empty_label.setAlignment(Qt.AlignCenter)

        self.project_name = self._metadata_line_edit()
        self.project_name.setToolTip("Имя проекта без расширения .json.")
        self.project_name.editingFinished.connect(
            self._handle_project_name_edit_finished
        )
        self.project_path_field = self._metadata_line_edit()
        self.project_path_field.setReadOnly(True)
        self.project_path_field.setClearButtonEnabled(False)
        self.project_general_info = QTextEdit(properties_widget)
        self.project_general_info.setAcceptRichText(False)
        self.project_general_info.setMinimumHeight(58)
        self.project_general_info.textChanged.connect(
            self._handle_project_general_info_changed
        )
        self.project_image_count = self._property_value_label()
        self.project_mean_red = self._property_value_label()
        self.project_mean_green = self._property_value_label()
        self.project_mean_blue = self._property_value_label()
        self.project_mean_lab_l = self._property_value_label()
        self.project_mean_lab_a = self._property_value_label()
        self.project_mean_lab_b = self._property_value_label()
        self.project_mean_hsv_h = self._property_value_label()
        self.project_mean_hsv_s = self._property_value_label()
        self.project_mean_hsv_v = self._property_value_label()
        self.project_mean_yuv_y = self._property_value_label()
        self.project_mean_yuv_u = self._property_value_label()
        self.project_mean_yuv_v = self._property_value_label()
        self.project_mean_lms_l = self._property_value_label()
        self.project_mean_lms_m = self._property_value_label()
        self.project_mean_lms_s = self._property_value_label()

        self.project_properties_browser = PropertyBrowser(properties_widget)
        self._add_property_browser_group(
            self.project_properties_browser,
            "Общие свойства",
            [
                ("Имя проекта", self.project_name, "project_name"),
                ("Путь", self.project_path_field, "project_path"),
                ("Количество файлов", self.project_image_count),
                ("Общая информация", self.project_general_info),
            ],
        )
        project_color_spaces_group = self.project_properties_browser.add_group(
            COLOR_SPACES_GROUP_TITLE,
            expanded=False,
        )
        self._add_property_browser_group(
            self.project_properties_browser,
            "Цветовое пространство RGB",
            [
                ("Средний R", self.project_mean_red),
                ("Средний G", self.project_mean_green),
                ("Средний B", self.project_mean_blue),
            ],
            parent=project_color_spaces_group,
        )
        self._add_property_browser_group(
            self.project_properties_browser,
            "Цветовое пространство Lab",
            [
                ("Средний L", self.project_mean_lab_l),
                ("Средний a", self.project_mean_lab_a),
                ("Средний b", self.project_mean_lab_b),
            ],
            parent=project_color_spaces_group,
        )
        self._add_property_browser_group(
            self.project_properties_browser,
            "Цветовое пространство HSV",
            [
                ("Средний H", self.project_mean_hsv_h),
                ("Средний S", self.project_mean_hsv_s),
                ("Средний V", self.project_mean_hsv_v),
            ],
            parent=project_color_spaces_group,
        )
        self._add_property_browser_group(
            self.project_properties_browser,
            "Цветовое пространство YUV",
            [
                ("Средний Y", self.project_mean_yuv_y),
                ("Средний U", self.project_mean_yuv_u),
                ("Средний V", self.project_mean_yuv_v),
            ],
            parent=project_color_spaces_group,
        )
        self._add_property_browser_group(
            self.project_properties_browser,
            "Цветовое пространство LMS",
            [
                ("Средний L", self.project_mean_lms_l),
                ("Средний M", self.project_mean_lms_m),
                ("Средний S", self.project_mean_lms_s),
            ],
            parent=project_color_spaces_group,
        )

        layout = QVBoxLayout(properties_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(self.project_properties_empty_label, 1)
        layout.addWidget(self.project_properties_browser, 1)
        return properties_widget

    def _create_image_properties_panel(self, parent: QWidget) -> QWidget:
        properties_widget = QWidget(parent)
        properties_widget.setObjectName("imagePropertiesPanel")
        self.image_properties_panel = properties_widget

        title = QLabel("Свойства изображения")
        title.setObjectName("projectPanelTitle")

        self.export_image_properties_excel_button = QToolButton(properties_widget)
        self.export_image_properties_excel_button.setText("Excel")
        self.export_image_properties_excel_button.setIcon(load_icon("export-csv"))
        self.export_image_properties_excel_button.setToolTip(
            "Экспорт свойств изображения в Excel"
        )
        self.export_image_properties_excel_button.setStatusTip(
            "Сохранить свойства выбранного изображения в файл .xlsx."
        )
        self.export_image_properties_excel_button.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.export_image_properties_excel_button.clicked.connect(
            self.export_image_properties_excel
        )

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.export_image_properties_excel_button)

        self.properties_empty_label = QLabel("Изображение не выбрано.")
        self.properties_empty_label.setObjectName("propertyEmpty")
        self.properties_empty_label.setWordWrap(True)
        self.properties_empty_label.setAlignment(Qt.AlignCenter)

        self.property_file_name = self._metadata_line_edit()
        self.property_file_name.editingFinished.connect(
            self._handle_file_name_edit_finished
        )
        self.property_file_name_widget, self.rename_file_as_id_button = (
            self._file_name_widget(properties_widget)
        )
        self.property_path = self._property_value_label()
        self.property_size = self._property_value_label()
        self.property_annotation = self._property_value_label()
        self.property_contour_visible = QCheckBox(properties_widget)
        self.property_contour_visible.setEnabled(False)
        self.property_contour_visible.setToolTip(
            "Показать или скрыть главный контур на канвасе."
        )
        self.property_contour_visible.toggled.connect(
            self._handle_contour_visibility_changed
        )
        self.property_contour_line_color = self._line_color_button(
            properties_widget,
            CONTOUR_LINE_COLOR,
            "Выбрать цвет линии главного контура",
        )
        self.property_contour_line_color.clicked.connect(
            self._open_contour_line_color_dialog
        )
        self.property_points = self._property_value_label()
        self.property_contour_pixels = self._property_value_label()
        self.property_contour_area_mm2 = self._property_value_label()
        self.property_calibration_length = self._calibration_length_field()
        self.property_calibration_scale = self._property_value_label()
        self.property_red = self._property_value_label()
        self.property_green = self._property_value_label()
        self.property_blue = self._property_value_label()
        self.property_lab_l = self._property_value_label()
        self.property_lab_a = self._property_value_label()
        self.property_lab_b = self._property_value_label()
        self.property_hsv_h = self._property_value_label()
        self.property_hsv_s = self._property_value_label()
        self.property_hsv_v = self._property_value_label()
        self.property_yuv_y = self._property_value_label()
        self.property_yuv_u = self._property_value_label()
        self.property_yuv_v = self._property_value_label()
        self.property_lms_l = self._property_value_label()
        self.property_lms_m = self._property_value_label()
        self.property_lms_s = self._property_value_label()
        self.property_status = self._property_value_label()
        self.average_color_swatch = AverageColorSwatch(properties_widget)
        self.average_color_swatch.setObjectName("averageColorSwatch")
        self.average_color_swatch.setFixedSize(58, 24)
        self.average_color_swatch.clicked.connect(self.save_average_color_sample)

        self.image_id = self._metadata_line_edit()
        self.image_id.editingFinished.connect(self._handle_image_id_edit_finished)
        self.image_id_widget, self.generate_image_id_button = self._image_id_widget(
            self.image_id,
            properties_widget,
        )
        self.metadata_added_at = self._metadata_line_edit()
        self.metadata_captured_at = self._metadata_line_edit()
        self.metadata_added_at_widget, self.metadata_added_at_button = (
            self._metadata_datetime_widget(
                self.metadata_added_at,
                "added_at",
                properties_widget,
            )
        )
        self.metadata_captured_at_widget, self.metadata_captured_at_button = (
            self._metadata_datetime_widget(
                self.metadata_captured_at,
                "captured_at",
                properties_widget,
            )
        )
        self.metadata_illumination = self._metadata_line_edit()
        self.metadata_humidity = self._metadata_line_edit()
        self.metadata_wind_speed = self._metadata_line_edit()
        self.metadata_wind_direction = self._metadata_line_edit()
        self.metadata_latitude = self._metadata_line_edit()
        self.metadata_longitude = self._metadata_line_edit()
        self.metadata_diagnosis = self._metadata_line_edit()
        self.metadata_notes = QTextEdit(properties_widget)
        self.metadata_notes.setAcceptRichText(False)
        self.metadata_notes.setMinimumHeight(70)
        self.metadata_notes.textChanged.connect(self._handle_metadata_notes_changed)

        self._metadata_fields = {
            "added_at": self.metadata_added_at,
            "captured_at": self.metadata_captured_at,
            "illumination": self.metadata_illumination,
            "humidity": self.metadata_humidity,
            "wind_speed": self.metadata_wind_speed,
            "wind_direction": self.metadata_wind_direction,
            "latitude": self.metadata_latitude,
            "longitude": self.metadata_longitude,
            "diagnosis": self.metadata_diagnosis,
        }
        for metadata_key, field in self._metadata_fields.items():
            field.editingFinished.connect(
                lambda key=metadata_key, editor=field: (
                    self._handle_metadata_line_edit_finished(key, editor)
                )
            )
        self._angle_name_fields: dict[str, QLineEdit] = {}
        self._angle_visibility_fields: dict[str, QCheckBox] = {}
        self._angle_line_color_fields: dict[str, QToolButton] = {}
        self._angle_label_color_fields: dict[str, QToolButton] = {}
        self._angle_note_fields: dict[str, QLineEdit] = {}
        self._angle_assessment_fields: dict[str, QComboBox] = {}
        self._angle_assessment_result_fields: dict[str, QLabel] = {}
        self._segment_name_fields: dict[str, QLineEdit] = {}
        self._segment_visibility_fields: dict[str, QCheckBox] = {}
        self._segment_line_color_fields: dict[str, QToolButton] = {}
        self._segment_label_color_fields: dict[str, QToolButton] = {}
        self._segment_start_label_fields: dict[str, QLineEdit] = {}
        self._segment_end_label_fields: dict[str, QLineEdit] = {}
        self._segment_note_fields: dict[str, QLineEdit] = {}
        self._segment_assessment_fields: dict[str, QComboBox] = {}
        self._segment_assessment_result_fields: dict[str, QLabel] = {}

        self.properties_browser = PropertyBrowser(properties_widget)
        self.properties_browser.itemClicked.connect(
            self._handle_image_property_item_clicked
        )
        self.properties_browser.empty_area_clicked.connect(
            self._clear_image_property_canvas_highlight
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Общая информация о файле",
            [
                ("ID", self.image_id_widget),
                ("Файл", self.property_file_name_widget),
                ("Путь", self.property_path),
                ("Размер", self.property_size),
                ("Статус", self.property_status),
                ("Дата добавления", self.metadata_added_at_widget),
                ("Дата съёмки", self.metadata_captured_at_widget),
            ],
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Калибровка",
            [
                ("Длина калибровки, мм", self.property_calibration_length),
                ("Масштаб", self.property_calibration_scale),
            ],
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Информация о главном контуре",
            [
                ("Аннотация", self.property_annotation, "contour:annotation"),
                ("Показывать контур", self.property_contour_visible, "contour:visible"),
                (
                    "Цвет линии",
                    self.property_contour_line_color,
                    "contour:line_color",
                ),
                (
                    "Количество узлов контура",
                    self.property_points,
                    "contour:point_count",
                ),
                (
                    "Количество пикселов контура",
                    self.property_contour_pixels,
                    "contour:pixel_count",
                ),
                (
                    "Площадь контура, мм²",
                    self.property_contour_area_mm2,
                    "contour:area_mm2",
                ),
            ],
            key="contour",
        )
        self.measurements_group_item = self.properties_browser.add_group(
            "Измерения",
            expanded=False,
            key="measurements",
        )
        self.angles_group_item = self.properties_browser.add_group(
            "Углы",
            expanded=False,
            parent=self.measurements_group_item,
            key="measurements:angles",
        )
        self.segments_group_item = self.properties_browser.add_group(
            "Отрезки",
            expanded=False,
            parent=self.measurements_group_item,
            key="measurements:segments",
        )
        image_color_spaces_group = self.properties_browser.add_group(
            COLOR_SPACES_GROUP_TITLE,
            expanded=False,
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Цветовое пространство RGB",
            [
                ("Красный", self.property_red),
                ("Зелёный", self.property_green),
                ("Синий", self.property_blue),
                ("Средний цвет", self.average_color_swatch),
            ],
            parent=image_color_spaces_group,
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Цветовое пространство Lab",
            [
                ("L", self.property_lab_l),
                ("a", self.property_lab_a),
                ("b", self.property_lab_b),
            ],
            parent=image_color_spaces_group,
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Цветовое пространство HSV",
            [
                ("H", self.property_hsv_h),
                ("S", self.property_hsv_s),
                ("V", self.property_hsv_v),
            ],
            parent=image_color_spaces_group,
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Цветовое пространство YUV",
            [
                ("Y", self.property_yuv_y),
                ("U", self.property_yuv_u),
                ("V", self.property_yuv_v),
            ],
            parent=image_color_spaces_group,
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Цветовое пространство LMS",
            [
                ("L", self.property_lms_l),
                ("M", self.property_lms_m),
                ("S", self.property_lms_s),
            ],
            parent=image_color_spaces_group,
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Локация",
            [
                ("Освещённость", self.metadata_illumination),
                ("Влажность, %", self.metadata_humidity),
                ("Скорость ветра", self.metadata_wind_speed),
                ("Направление ветра", self.metadata_wind_direction),
                ("Широта", self.metadata_latitude),
                ("Долгота", self.metadata_longitude),
            ],
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Дополнительно",
            [
                ("Диагноз", self.metadata_diagnosis),
                ("Дополнительные сведения", self.metadata_notes),
            ],
        )

        layout = QVBoxLayout(properties_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header_layout)
        layout.addWidget(self.properties_empty_label, 1)
        layout.addWidget(self.properties_browser, 1)
        return properties_widget

    def _add_property_browser_group(
        self,
        browser: PropertyBrowser,
        title: str,
        rows: list[tuple[str, QWidget] | tuple[str, QWidget, str]],
        *,
        key: str | None = None,
        parent: QTreeWidgetItem | None = None,
    ) -> QTreeWidgetItem:
        group_item = browser.add_group(title, expanded=False, parent=parent)
        if key is not None:
            group_item.setData(0, Qt.UserRole + 1, key)
        for row in rows:
            label, widget = row[0], row[1]
            row_key = row[2] if len(row) > 2 else None
            aliases = [label] if row_key is not None else None
            if parent is None:
                browser.add_property(title, label, widget, key=row_key, aliases=aliases)
            else:
                browser.add_property_to_item(
                    group_item, label, widget, key=row_key, aliases=aliases
                )
        return group_item

    def _rebuild_angle_measurement_properties(
        self, record: ProjectImageRecord | None
    ) -> None:
        self._angle_name_fields.clear()
        self._angle_visibility_fields.clear()
        self._angle_line_color_fields.clear()
        self._angle_label_color_fields.clear()
        self._angle_note_fields.clear()
        self._angle_assessment_fields.clear()
        self._angle_assessment_result_fields.clear()
        self.properties_browser.clear_children(self.angles_group_item)
        if record is None or not record.measurements.angles:
            self.properties_browser.add_property_to_item(
                self.angles_group_item,
                "Нет углов",
                self._property_value_label(),
                key="measurements:angles:empty",
            )
            return

        for index, angle in enumerate(record.measurements.angles, start=1):
            angle_group = self.properties_browser.add_group(
                _angle_display_name(angle, index),
                expanded=False,
                parent=self.angles_group_item,
                key=f"angle:{angle.id}",
            )
            name_field = QLineEdit(self.properties_browser)
            name_field.setClearButtonEnabled(True)
            name_field.setPlaceholderText(f"Угол {index}")
            name_field.setText(angle.name)
            name_field.editingFinished.connect(
                lambda angle_id=angle.id, editor=name_field: (
                    self._handle_angle_name_edit_finished(
                        angle_id,
                        editor,
                    )
                )
            )
            self._angle_name_fields[angle.id] = name_field
            self.properties_browser.add_property_to_item(
                angle_group,
                "Имя",
                name_field,
                key=f"angle:{angle.id}:name",
            )
            visible_checkbox = QCheckBox(self.properties_browser)
            visible_checkbox.setToolTip("Показать или скрыть этот угол на канвасе.")
            visible_checkbox.setEnabled(
                record.id == self._current_project_image_id and self.canvas.has_image()
            )
            with QSignalBlocker(visible_checkbox):
                visible_checkbox.setChecked(
                    self.canvas.is_angle_measurement_visible(angle.id)
                    if visible_checkbox.isEnabled()
                    else True
                )
            visible_checkbox.toggled.connect(
                lambda checked, angle_id=angle.id: (
                    self._handle_angle_visibility_changed(
                        angle_id,
                        checked,
                    )
                )
            )
            self._angle_visibility_fields[angle.id] = visible_checkbox
            self.properties_browser.add_property_to_item(
                angle_group,
                "Показывать на канвасе",
                visible_checkbox,
                key=f"angle:{angle.id}:visible",
            )
            line_color_button = self._line_color_button(
                self.properties_browser,
                angle.line_color,
                "Выбрать цвет линии угла",
            )
            line_color_button.clicked.connect(
                lambda _checked=False, angle_id=angle.id: (
                    self._open_angle_line_color_dialog(angle_id)
                )
            )
            self._angle_line_color_fields[angle.id] = line_color_button
            self.properties_browser.add_property_to_item(
                angle_group,
                "Цвет линии",
                line_color_button,
                key=f"angle:{angle.id}:line_color",
            )
            label_color_button = self._line_color_button(
                self.properties_browser,
                angle.label_color,
                "Выбрать цвет подписей угла",
            )
            label_color_button.clicked.connect(
                lambda _checked=False, angle_id=angle.id: (
                    self._open_angle_label_color_dialog(angle_id)
                )
            )
            self._angle_label_color_fields[angle.id] = label_color_button
            self.properties_browser.add_property_to_item(
                angle_group,
                "Цвет подписей",
                label_color_button,
                key=f"angle:{angle.id}:label_color",
            )
            self.properties_browser.add_property_to_item(
                angle_group,
                "ID",
                self._property_value_label(angle.id),
                key=f"angle:{angle.id}:id",
            )
            self.properties_browser.add_property_to_item(
                angle_group,
                "Значение",
                self._property_value_label(_angle_measurement_value_text(angle)),
                key=f"angle:{angle.id}:value",
            )
            assessment_combo = self._oiv_assessment_combo(
                "angle",
                _measurement_oiv_code(angle.assessment),
            )
            assessment_combo.currentIndexChanged.connect(
                lambda _index, angle_id=angle.id, editor=assessment_combo: (
                    self._handle_angle_assessment_changed(
                        angle_id,
                        editor,
                    )
                )
            )
            self._angle_assessment_fields[angle.id] = assessment_combo
            self.properties_browser.add_property_to_item(
                angle_group,
                "Оценка",
                assessment_combo,
                key=f"angle:{angle.id}:assessment",
            )
            assessment_result = self._property_value_label(
                _oiv_result_text(self._angle_oiv_result(angle))
            )
            self._angle_assessment_result_fields[angle.id] = assessment_result
            self.properties_browser.add_property_to_item(
                angle_group,
                "Результат оценки",
                assessment_result,
                key=f"angle:{angle.id}:assessment_result",
            )
            self.properties_browser.add_property_to_item(
                angle_group,
                "Первая точка",
                self._property_value_label(_point_text(angle.first)),
                key=f"angle:{angle.id}:first",
            )
            self.properties_browser.add_property_to_item(
                angle_group,
                "Вершина",
                self._property_value_label(_point_text(angle.vertex)),
                key=f"angle:{angle.id}:vertex",
            )
            self.properties_browser.add_property_to_item(
                angle_group,
                "Вторая точка",
                self._property_value_label(_point_text(angle.second)),
                key=f"angle:{angle.id}:second",
            )

            note_field = QLineEdit(self.properties_browser)
            note_field.setText(angle.note)
            note_field.editingFinished.connect(
                lambda angle_id=angle.id, editor=note_field: (
                    self._handle_angle_note_edit_finished(
                        angle_id,
                        editor,
                    )
                )
            )
            self._angle_note_fields[angle.id] = note_field
            self.properties_browser.add_property_to_item(
                angle_group,
                "Примечание",
                note_field,
                key=f"angle:{angle.id}:note",
            )

            delete_button = QToolButton(self.properties_browser)
            delete_button.setText("Удалить")
            delete_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            delete_button.setIcon(load_icon("delete-angle"))
            delete_button.clicked.connect(
                lambda _checked=False, angle_id=angle.id: self._delete_angle_by_id(
                    angle_id
                )
            )
            self.properties_browser.add_property_to_item(
                angle_group,
                "Управление",
                delete_button,
                key=f"angle:{angle.id}:actions",
            )

    def _rebuild_segment_measurement_properties(
        self, record: ProjectImageRecord | None
    ) -> None:
        self._segment_name_fields.clear()
        self._segment_visibility_fields.clear()
        self._segment_line_color_fields.clear()
        self._segment_label_color_fields.clear()
        self._segment_start_label_fields.clear()
        self._segment_end_label_fields.clear()
        self._segment_note_fields.clear()
        self._segment_assessment_fields.clear()
        self._segment_assessment_result_fields.clear()
        self.properties_browser.clear_children(self.segments_group_item)
        if record is None or not record.measurements.segments:
            self.properties_browser.add_property_to_item(
                self.segments_group_item,
                "Нет отрезков",
                self._property_value_label(),
                key="measurements:segments:empty",
            )
            return

        for index, segment in enumerate(record.measurements.segments, start=1):
            segment_group = self.properties_browser.add_group(
                _segment_display_name(segment, index),
                expanded=False,
                parent=self.segments_group_item,
                key=f"segment:{segment.id}",
            )
            name_field = QLineEdit(self.properties_browser)
            name_field.setClearButtonEnabled(True)
            name_field.setPlaceholderText(f"Отрезок {index}")
            name_field.setText(segment.name)
            name_field.editingFinished.connect(
                lambda segment_id=segment.id, editor=name_field: (
                    self._handle_segment_name_edit_finished(
                        segment_id,
                        editor,
                    )
                )
            )
            self._segment_name_fields[segment.id] = name_field
            self.properties_browser.add_property_to_item(
                segment_group,
                "Имя",
                name_field,
                key=f"segment:{segment.id}:name",
            )
            visible_checkbox = QCheckBox(self.properties_browser)
            visible_checkbox.setToolTip("Показать или скрыть этот отрезок на канвасе.")
            visible_checkbox.setEnabled(
                record.id == self._current_project_image_id and self.canvas.has_image()
            )
            with QSignalBlocker(visible_checkbox):
                visible_checkbox.setChecked(
                    self.canvas.is_segment_measurement_visible(segment.id)
                    if visible_checkbox.isEnabled()
                    else True
                )
            visible_checkbox.toggled.connect(
                lambda checked, segment_id=segment.id: (
                    self._handle_segment_visibility_changed(
                        segment_id,
                        checked,
                    )
                )
            )
            self._segment_visibility_fields[segment.id] = visible_checkbox
            self.properties_browser.add_property_to_item(
                segment_group,
                "Показывать на канвасе",
                visible_checkbox,
                key=f"segment:{segment.id}:visible",
            )
            line_color_button = self._line_color_button(
                self.properties_browser,
                segment.line_color,
                "Выбрать цвет линии отрезка",
            )
            line_color_button.clicked.connect(
                lambda _checked=False, segment_id=segment.id: (
                    self._open_segment_line_color_dialog(segment_id)
                )
            )
            self._segment_line_color_fields[segment.id] = line_color_button
            self.properties_browser.add_property_to_item(
                segment_group,
                "Цвет линии",
                line_color_button,
                key=f"segment:{segment.id}:line_color",
            )
            label_color_button = self._line_color_button(
                self.properties_browser,
                segment.label_color,
                "Выбрать цвет подписей отрезка",
            )
            label_color_button.clicked.connect(
                lambda _checked=False, segment_id=segment.id: (
                    self._open_segment_label_color_dialog(segment_id)
                )
            )
            self._segment_label_color_fields[segment.id] = label_color_button
            self.properties_browser.add_property_to_item(
                segment_group,
                "Цвет подписей",
                label_color_button,
                key=f"segment:{segment.id}:label_color",
            )
            self.properties_browser.add_property_to_item(
                segment_group,
                "ID",
                self._property_value_label(segment.id),
                key=f"segment:{segment.id}:id",
            )
            self.properties_browser.add_property_to_item(
                segment_group,
                "Длина",
                self._property_value_label(
                    _segment_measurement_length_text(segment, record.calibration)
                ),
                key=f"segment:{segment.id}:length",
            )
            assessment_combo = self._oiv_assessment_combo(
                "segment",
                _measurement_oiv_code(segment.assessment),
            )
            assessment_combo.currentIndexChanged.connect(
                lambda _index, segment_id=segment.id, editor=assessment_combo: (
                    self._handle_segment_assessment_changed(
                        segment_id,
                        editor,
                    )
                )
            )
            self._segment_assessment_fields[segment.id] = assessment_combo
            self.properties_browser.add_property_to_item(
                segment_group,
                "Оценка",
                assessment_combo,
                key=f"segment:{segment.id}:assessment",
            )
            assessment_result = self._property_value_label(
                _oiv_result_text(self._segment_oiv_result(segment, record))
            )
            self._segment_assessment_result_fields[segment.id] = assessment_result
            self.properties_browser.add_property_to_item(
                segment_group,
                "Результат оценки",
                assessment_result,
                key=f"segment:{segment.id}:assessment_result",
            )
            self.properties_browser.add_property_to_item(
                segment_group,
                "Первая точка",
                self._property_value_label(_point_text(segment.start)),
                key=f"segment:{segment.id}:start",
            )
            self.properties_browser.add_property_to_item(
                segment_group,
                "Вторая точка",
                self._property_value_label(_point_text(segment.end)),
                key=f"segment:{segment.id}:end",
            )

            start_label_field = QLineEdit(self.properties_browser)
            start_label_field.setClearButtonEnabled(True)
            start_label_field.setText(segment.start_label)
            start_label_field.editingFinished.connect(
                lambda segment_id=segment.id, editor=start_label_field: (
                    self._handle_segment_label_edit_finished(
                        segment_id,
                        editor,
                        "start",
                    )
                )
            )
            self._segment_start_label_fields[segment.id] = start_label_field
            self.properties_browser.add_property_to_item(
                segment_group,
                "Подпись первой точки",
                start_label_field,
                key=f"segment:{segment.id}:start_label",
            )

            end_label_field = QLineEdit(self.properties_browser)
            end_label_field.setClearButtonEnabled(True)
            end_label_field.setText(segment.end_label)
            end_label_field.editingFinished.connect(
                lambda segment_id=segment.id, editor=end_label_field: (
                    self._handle_segment_label_edit_finished(
                        segment_id,
                        editor,
                        "end",
                    )
                )
            )
            self._segment_end_label_fields[segment.id] = end_label_field
            self.properties_browser.add_property_to_item(
                segment_group,
                "Подпись второй точки",
                end_label_field,
                key=f"segment:{segment.id}:end_label",
            )

            note_field = QLineEdit(self.properties_browser)
            note_field.setText(segment.note)
            note_field.editingFinished.connect(
                lambda segment_id=segment.id, editor=note_field: (
                    self._handle_segment_note_edit_finished(
                        segment_id,
                        editor,
                    )
                )
            )
            self._segment_note_fields[segment.id] = note_field
            self.properties_browser.add_property_to_item(
                segment_group,
                "Примечание",
                note_field,
                key=f"segment:{segment.id}:note",
            )

            delete_button = QToolButton(self.properties_browser)
            delete_button.setText("Удалить")
            delete_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            delete_button.setIcon(load_icon("delete-segment"))
            delete_button.clicked.connect(
                lambda _checked=False, segment_id=segment.id: (
                    self._delete_segment_by_id(segment_id)
                )
            )
            self.properties_browser.add_property_to_item(
                segment_group,
                "Управление",
                delete_button,
                key=f"segment:{segment.id}:actions",
            )

    def _property_value_label(self, text: str = "-") -> QLabel:
        label = PropertyValueLabel(text)
        label.setObjectName("propertyValue")
        return label

    def _line_color_button(
        self,
        parent: QWidget,
        color_name: str,
        tooltip: str,
    ) -> QToolButton:
        button = QToolButton(parent)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setIconSize(QSize(18, 18))
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.setToolTip(tooltip)
        self._set_line_color_button_value(button, color_name)
        return button

    def _set_line_color_button_value(
        self,
        button: QToolButton,
        color_name: str,
    ) -> None:
        normalized_color = normalize_line_color(color_name, "#000000")
        swatch = QPixmap(18, 18)
        swatch.fill(QColor(normalized_color))
        button.setIcon(QIcon(swatch))
        button.setText(normalized_color)
        button.setProperty("lineColor", normalized_color)

    def _choose_line_color(self, current_color: str, title: str) -> str | None:
        selected_color = QColorDialog.getColor(QColor(current_color), self, title)
        if not selected_color.isValid():
            return None
        return selected_color.name().lower()

    def _oiv_assessment_combo(self, tool_kind: str, selected_code: str) -> QComboBox:
        combo = QComboBox(self.properties_browser)
        combo.setMinimumContentsLength(18)
        combo.addItem("Не выбрано", "")

        normalized_selected_code = normalize_oiv_code(selected_code)
        available_codes: set[str] = set()
        if self._oiv_catalog is None:
            combo.setEnabled(False)
            combo.setToolTip(self._oiv_catalog_error or "Шкалы OIV недоступны.")
        else:
            for trait in self._oiv_catalog.traits_for_tool(tool_kind):
                combo.addItem(_oiv_trait_option_text(trait), trait.code)
                available_codes.add(trait.code)

            if (
                normalized_selected_code
                and normalized_selected_code not in available_codes
            ):
                selected_trait = self._oiv_catalog.trait_by_code(
                    normalized_selected_code
                )
                if selected_trait is None:
                    label = f"{normalized_selected_code} - неизвестный признак"
                else:
                    label = _oiv_trait_option_text(selected_trait)
                combo.addItem(label, normalized_selected_code)

        selected_index = combo.findData(normalized_selected_code)
        combo.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        return combo

    def _angle_oiv_result(
        self,
        angle: ProjectAngleMeasurement,
    ) -> OivClassificationResult | None:
        unsupported_result = self._unsupported_assessment_result(angle.assessment)
        if unsupported_result is not None:
            return unsupported_result

        code = _measurement_oiv_code(angle.assessment)
        if not code:
            return None

        selection_error = self._oiv_selection_error(code, "angle")
        if selection_error is not None:
            return selection_error

        value = _angle_degrees(angle.first, angle.vertex, angle.second)
        trait = self._oiv_trait_by_code(code)
        if value is None:
            return make_status_result(
                status="invalid_geometry",
                message="-",
                code=code,
                unit="deg",
                trait=trait,
            )
        return self._classify_oiv_value(code, value, "angle")

    def _segment_oiv_result(
        self,
        segment: ProjectSegmentMeasurement,
        record: ProjectImageRecord,
    ) -> OivClassificationResult | None:
        unsupported_result = self._unsupported_assessment_result(segment.assessment)
        if unsupported_result is not None:
            return unsupported_result

        code = _measurement_oiv_code(segment.assessment)
        if not code:
            return None

        selection_error = self._oiv_selection_error(code, "segment")
        if selection_error is not None:
            return selection_error

        pixel_length = math.hypot(
            segment.end.x - segment.start.x, segment.end.y - segment.start.y
        )
        trait = self._oiv_trait_by_code(code)
        if pixel_length <= 1e-6:
            return make_status_result(
                status="invalid_geometry",
                message="-",
                code=code,
                unit="mm",
                trait=trait,
            )
        if record.calibration_error:
            return make_status_result(
                status="calibration_error",
                message="Ошибка калибровки",
                code=code,
                unit="mm",
                trait=trait,
            )
        if record.calibration is None:
            return make_status_result(
                status="missing_calibration",
                message="Нужна калибровка",
                code=code,
                unit="mm",
                trait=trait,
            )

        value_mm = pixel_length * record.calibration.mm_per_pixel()
        return self._classify_oiv_value(code, value_mm, "segment")

    def _unsupported_assessment_result(
        self,
        assessment: ProjectMeasurementAssessment | None,
    ) -> OivClassificationResult | None:
        if assessment is None:
            return None
        if assessment.system.strip().upper() == OIV_SYSTEM_NAME:
            return None
        return make_status_result(
            status="unsupported_system",
            message="Неподдерживаемая система оценки",
        )

    def _oiv_selection_error(
        self,
        code: str,
        tool_kind: str,
    ) -> OivClassificationResult | None:
        if self._oiv_catalog is None:
            return make_status_result(
                status="catalog_error",
                message="Шкалы OIV недоступны",
                code=code,
            )

        trait = self._oiv_catalog.trait_by_code(code)
        if trait is None:
            return make_status_result(
                status="unknown_trait",
                message="Неизвестный признак",
                code=code,
            )
        if tool_kind not in trait.tool_kinds:
            return make_status_result(
                status="incompatible_tool",
                message="Несовместимый признак",
                code=trait.code,
                unit=trait.unit,
                trait=trait,
            )
        return None

    def _oiv_trait_by_code(self, code: str) -> OivTrait | None:
        if self._oiv_catalog is None:
            return None
        return self._oiv_catalog.trait_by_code(code)

    def _classify_oiv_value(
        self,
        code: str,
        value: float,
        tool_kind: str,
    ) -> OivClassificationResult:
        if self._oiv_catalog is None:
            return make_status_result(
                status="catalog_error",
                message="Шкалы OIV недоступны",
                code=code,
                value=value,
            )
        return self._oiv_catalog.classify(code, value, tool_kind=tool_kind)

    def _metadata_line_edit(self) -> QLineEdit:
        field = QLineEdit()
        field.setClearButtonEnabled(True)
        return field

    def _calibration_length_field(self) -> QLineEdit:
        field = QLineEdit()
        field.setClearButtonEnabled(False)
        field.setToolTip("Реальная длина калибровочного отрезка в миллиметрах.")
        field.editingFinished.connect(self._handle_calibration_length_edit_finished)
        return field

    def _file_name_widget(self, parent: QWidget) -> tuple[QWidget, QToolButton]:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        button = QToolButton(container)
        button.setText("ID")
        button.setToolTip("Переименовать файл как ID")
        button.setStatusTip(
            "Переименовать файл изображения по текущему ID с сохранением расширения."
        )
        button.setAutoRaise(True)
        button.clicked.connect(self._rename_file_to_image_id)

        layout.addWidget(self.property_file_name, 1)
        layout.addWidget(button)
        return container, button

    def _image_id_widget(
        self, field: QLineEdit, parent: QWidget
    ) -> tuple[QWidget, QToolButton]:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        button = QToolButton(container)
        button.setIcon(load_icon("generate-id"))
        button.setToolTip("Сгенерировать ID")
        button.setStatusTip("Сгенерировать новый GUID изображения.")
        button.setAutoRaise(True)
        button.clicked.connect(self._generate_image_id)

        layout.addWidget(field, 1)
        layout.addWidget(button)
        return container, button

    def _metadata_datetime_widget(
        self,
        field: QLineEdit,
        metadata_key: str,
        parent: QWidget,
    ) -> tuple[QWidget, QToolButton]:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        button = QToolButton(container)
        button.setIcon(load_icon("calendar-time"))
        button.setToolTip("Выбрать дату и время")
        button.setStatusTip("Открыть диалог выбора даты и времени.")
        button.setAutoRaise(True)
        button.clicked.connect(
            lambda: self._open_metadata_datetime_dialog(metadata_key, field)
        )

        layout.addWidget(field, 1)
        layout.addWidget(button)
        return container, button

    def _create_analysis_panel(self) -> QWidget:
        analysis_panel = QWidget(self)
        analysis_panel.setObjectName("analysisPanel")

        analysis_title = QLabel("Свойства и аналитика")
        analysis_title.setObjectName("analysisTitle")

        self.analysis_status_label = QLabel("")
        self.analysis_status_label.setObjectName("analysisStatus")
        self.analysis_status_label.setWordWrap(True)

        self.refresh_analysis_button = QToolButton(analysis_panel)
        self.refresh_analysis_button.setObjectName("refreshAnalysisButton")
        self.refresh_analysis_button.setText("Обновить")
        self.refresh_analysis_button.setIcon(load_icon("refresh-analysis"))
        self.refresh_analysis_button.setToolTip("Обновить графики (F9)")
        self.refresh_analysis_button.setStatusTip(
            "Пересчитать гистограммы и аналитику по загруженным изображениям."
        )
        self.refresh_analysis_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.refresh_analysis_button.clicked.connect(self.refresh_analysis)

        analysis_header_layout = QHBoxLayout()
        analysis_header_layout.setContentsMargins(0, 0, 0, 0)
        analysis_header_layout.setSpacing(6)
        analysis_header_layout.addWidget(analysis_title)
        analysis_header_layout.addWidget(self.analysis_status_label, 1)
        analysis_header_layout.addWidget(self.refresh_analysis_button)

        self.histogram_splitter = QSplitter(Qt.Vertical, analysis_panel)
        self.histogram_splitter.setChildrenCollapsible(False)
        self.histogram_splitter.addWidget(self.rgb_histogram_panel)
        self.histogram_splitter.addWidget(self.lab_histogram_panel)
        self.histogram_splitter.addWidget(self.hsv_histogram_panel)
        self.histogram_splitter.addWidget(self.yuv_histogram_panel)
        self.histogram_splitter.addWidget(self.lms_histogram_panel)
        self.histogram_splitter.setSizes(HISTOGRAM_DEFAULT_SIZES)
        self.histogram_splitter.setStyleSheet(
            "QSplitter::handle { background: #d5dbe5; height: 7px; }"
            "QSplitter::handle:hover { background: #bdc7d5; }"
        )

        analysis_layout = QVBoxLayout(analysis_panel)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        analysis_layout.setSpacing(8)
        analysis_layout.addLayout(analysis_header_layout)
        analysis_layout.addWidget(self.histogram_splitter, 1)

        analysis_panel.setStyleSheet(
            "QWidget#analysisPanel { background: #f7f9fc; border-left: 1px solid #d6dde8; }"
            "QLabel#analysisTitle { color: #1f2937; font-size: 13px; font-weight: 600; }"
            "QLabel#analysisStatus { color: #b45309; font-size: 11px; }"
            "QToolButton#refreshAnalysisButton { padding: 5px 10px; font-weight: 600; }"
            "QFrame#histogramPanel { background: #ffffff; border: 1px solid #ccd6e1; border-radius: 6px; }"
            "QLabel#histogramTitle { color: #1f2937; font-size: 12px; font-weight: 600; }"
            "QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 4px; }"
            "QToolButton:hover { background: #e9eef5; border-color: #c6d1de; }"
            "QToolButton:pressed { background: #dce4ee; }"
        )
        return analysis_panel

    def _create_actions(self) -> None:
        self.new_project_action = QAction("Новый проект...", self)
        self.new_project_action.setShortcut("Ctrl+N")
        self.new_project_action.triggered.connect(self.new_project)

        self.open_project_action = QAction("Открыть проект...", self)
        self.open_project_action.setShortcut("Ctrl+Alt+O")
        self.open_project_action.triggered.connect(self.open_project)

        self.save_project_action = QAction("Сохранить проект", self)
        self.save_project_action.setShortcut("Ctrl+Shift+S")
        self.save_project_action.triggered.connect(self.save_project_file)

        self.close_project_action = QAction("Закрыть проект", self)
        self.close_project_action.setShortcut("Ctrl+W")
        self.close_project_action.triggered.connect(self.close_project)

        self.add_images_action = QAction("Добавить изображения...", self)
        self.add_images_action.setShortcut("Ctrl+I")
        self.add_images_action.triggered.connect(self.add_images_to_project)

        self.sync_images_action = QAction("Синхронизировать папку изображений", self)
        self.sync_images_action.triggered.connect(self.sync_project_images_folder)

        self.remove_image_action = QAction("Удалить изображение из проекта", self)
        self.remove_image_action.triggered.connect(self.remove_selected_project_image)

        self.export_project_excel_action = QAction("Экспорт списка в Excel...", self)
        self.export_project_excel_action.triggered.connect(self.export_project_excel)
        self.export_project_csv_action = self.export_project_excel_action

        self.export_image_properties_excel_action = QAction(
            "Экспорт свойств изображения в Excel...", self
        )
        self.export_image_properties_excel_action.triggered.connect(
            self.export_image_properties_excel
        )

        self.exit_action = QAction("Выход", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)

        self.zoom_in_action = QAction("Увеличить", self)
        self.zoom_in_action.setShortcut("Ctrl++")
        self.zoom_in_action.triggered.connect(self.canvas.zoom_in)

        self.zoom_out_action = QAction("Уменьшить", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.triggered.connect(self.canvas.zoom_out)

        self.fit_image_action = QAction("Показать целиком", self)
        self.fit_image_action.setShortcut("Ctrl+9")
        self.fit_image_action.triggered.connect(self.canvas.fit_to_image)

        self.actual_size_action = QAction("100%", self)
        self.actual_size_action.setShortcut("Ctrl+0")
        self.actual_size_action.triggered.connect(self.canvas.reset_zoom)

        self.default_view_action = QAction("Вид по умолчанию", self)
        self.default_view_action.setShortcut("Ctrl+R")
        self.default_view_action.triggered.connect(self.restore_default_view)

        self.show_all_canvas_elements_action = QAction("Показать все элементы", self)
        self.show_all_canvas_elements_action.triggered.connect(
            self.show_all_canvas_elements
        )

        self.hide_all_canvas_elements_action = QAction("Скрыть все элементы", self)
        self.hide_all_canvas_elements_action.triggered.connect(
            self.hide_all_canvas_elements
        )

        self.new_contour_action = QAction("Новый контур", self)
        self.new_contour_action.setShortcut("Ctrl+Shift+N")
        self.new_contour_action.triggered.connect(self.create_new_contour)

        self.detect_contour_action = QAction("Определить контур", self)
        self.detect_contour_action.setShortcut("F5")
        self.detect_contour_action.triggered.connect(self.detect_contour)

        self.delete_contour_action = QAction("Удалить контур", self)
        self.delete_contour_action.setShortcut("Ctrl+Delete")
        self.delete_contour_action.triggered.connect(self.delete_current_contour)

        self.flatten_background_action = QAction("Выровнять фон", self)
        self.flatten_background_action.setShortcut("F6")
        self.flatten_background_action.triggered.connect(self.flatten_background)

        self.calibrate_scale_action = QAction("Калибровать масштаб", self)
        self.calibrate_scale_action.triggered.connect(self.start_scale_calibration)

        self.reset_calibration_action = QAction("Сбросить калибровку", self)
        self.reset_calibration_action.triggered.connect(self.reset_current_calibration)

        self.measure_angle_action = QAction("Угол", self)
        self.measure_angle_action.triggered.connect(self.start_angle_measurement)

        self.delete_angle_action = QAction("Удалить угол", self)
        self.delete_angle_action.triggered.connect(self.delete_selected_angle)

        self.measure_segment_action = QAction("Отрезок", self)
        self.measure_segment_action.triggered.connect(self.start_segment_measurement)

        self.delete_segment_action = QAction("Удалить отрезок", self)
        self.delete_segment_action.triggered.connect(self.delete_selected_segment)

        self.save_annotation_action = QAction("Сохранить аннотацию...", self)
        self.save_annotation_action.setShortcut("Ctrl+Alt+S")
        self.save_annotation_action.triggered.connect(self.save_annotation_file)

        self.open_annotation_action = QAction("Открыть аннотацию...", self)
        self.open_annotation_action.setShortcut("Ctrl+Shift+O")
        self.open_annotation_action.triggered.connect(self.open_annotation_file)

        self.refresh_analysis_action = QAction("Обновить", self)
        self.refresh_analysis_action.setShortcut("F9")
        self.refresh_analysis_action.triggered.connect(self.refresh_analysis)

        self.about_action = QAction("О программе", self)
        self.about_action.triggered.connect(self.show_about_dialog)

    def _apply_action_visuals(self) -> None:
        action_map = {
            "new_project": self.new_project_action,
            "open_project": self.open_project_action,
            "save_project": self.save_project_action,
            "close_project": self.close_project_action,
            "add_images": self.add_images_action,
            "sync_images": self.sync_images_action,
            "remove_image": self.remove_image_action,
            "export_project_excel": self.export_project_excel_action,
            "export_image_properties_excel": self.export_image_properties_excel_action,
            "exit": self.exit_action,
            "zoom_in": self.zoom_in_action,
            "zoom_out": self.zoom_out_action,
            "fit_image": self.fit_image_action,
            "actual_size": self.actual_size_action,
            "default_view": self.default_view_action,
            "show_all_canvas_elements": self.show_all_canvas_elements_action,
            "hide_all_canvas_elements": self.hide_all_canvas_elements_action,
            "new_contour": self.new_contour_action,
            "detect_contour": self.detect_contour_action,
            "delete_contour": self.delete_contour_action,
            "flatten_background": self.flatten_background_action,
            "calibrate_scale": self.calibrate_scale_action,
            "reset_calibration": self.reset_calibration_action,
            "measure_angle": self.measure_angle_action,
            "delete_angle": self.delete_angle_action,
            "measure_segment": self.measure_segment_action,
            "delete_segment": self.delete_segment_action,
            "save_annotation": self.save_annotation_action,
            "open_annotation": self.open_annotation_action,
            "refresh_analysis": self.refresh_analysis_action,
            "about": self.about_action,
        }

        for action_name, action in action_map.items():
            apply_action_visual(action, ACTION_VISUALS[action_name])

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.close_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.add_images_action)
        file_menu.addAction(self.sync_images_action)
        file_menu.addAction(self.remove_image_action)
        file_menu.addAction(self.export_project_excel_action)
        file_menu.addAction(self.export_image_properties_excel_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("Вид")
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)
        view_menu.addAction(self.fit_image_action)
        view_menu.addAction(self.actual_size_action)
        view_menu.addSeparator()
        view_menu.addAction(self.default_view_action)
        view_menu.addSeparator()
        view_menu.addAction(self.show_all_canvas_elements_action)
        view_menu.addAction(self.hide_all_canvas_elements_action)
        view_menu.addSeparator()
        view_menu.addAction(self.refresh_analysis_action)

        tools_menu = self.menuBar().addMenu("Инструменты")
        tools_menu.addAction(self.new_contour_action)
        tools_menu.addAction(self.detect_contour_action)
        tools_menu.addAction(self.delete_contour_action)
        tools_menu.addAction(self.flatten_background_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.calibrate_scale_action)
        tools_menu.addAction(self.reset_calibration_action)
        tools_menu.addAction(self.measure_angle_action)
        tools_menu.addAction(self.delete_angle_action)
        tools_menu.addAction(self.measure_segment_action)
        tools_menu.addAction(self.delete_segment_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.save_annotation_action)
        tools_menu.addAction(self.open_annotation_action)

        help_menu = self.menuBar().addMenu("Помощь")
        help_menu.addAction(self.about_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Основная панель", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        toolbar.setIconSize(TOOLBAR_ICON_SIZE)
        toolbar.setStyleSheet(
            "QToolBar { spacing: 4px; }"
            "QToolButton { padding: 6px; min-width: 38px; min-height: 38px; }"
        )
        self.addToolBar(toolbar)

        toolbar.addAction(self.new_project_action)
        toolbar.addAction(self.open_project_action)
        toolbar.addAction(self.save_project_action)
        toolbar.addAction(self.close_project_action)
        toolbar.addSeparator()
        toolbar.addAction(self.add_images_action)
        toolbar.addAction(self.sync_images_action)
        toolbar.addAction(self.remove_image_action)
        toolbar.addAction(self.export_project_excel_action)
        toolbar.addAction(self.export_image_properties_excel_action)
        toolbar.addSeparator()
        toolbar.addAction(self.zoom_in_action)
        toolbar.addAction(self.zoom_out_action)
        toolbar.addAction(self.fit_image_action)
        toolbar.addAction(self.actual_size_action)
        toolbar.addAction(self.default_view_action)
        toolbar.addSeparator()
        toolbar.addAction(self.show_all_canvas_elements_action)
        toolbar.addAction(self.hide_all_canvas_elements_action)
        toolbar.addSeparator()
        toolbar.addAction(self.refresh_analysis_action)
        toolbar.addSeparator()
        toolbar.addAction(self.new_contour_action)
        toolbar.addAction(self.detect_contour_action)
        toolbar.addAction(self.delete_contour_action)
        toolbar.addAction(self.flatten_background_action)
        toolbar.addAction(self.calibrate_scale_action)
        toolbar.addAction(self.reset_calibration_action)
        toolbar.addAction(self.measure_angle_action)
        toolbar.addAction(self.delete_angle_action)
        toolbar.addAction(self.measure_segment_action)
        toolbar.addAction(self.delete_segment_action)
        toolbar.addSeparator()
        toolbar.addAction(self.save_annotation_action)
        toolbar.addAction(self.open_annotation_action)
        toolbar.addSeparator()
        toolbar.addAction(self.about_action)
        toolbar_spacer = QWidget(toolbar)
        toolbar_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(toolbar_spacer)
        toolbar.addAction(self.exit_action)

    def _update_action_states(self, *_args) -> None:
        self._update_analysis_status()
        has_image = self.canvas.has_image()
        has_contour = self.canvas.has_contour()
        has_project = self.project_document is not None
        selected_project_image = self._selected_project_image()
        has_project_image = has_project and selected_project_image is not None
        has_project_contour = selected_project_image is not None and (
            selected_project_image.annotation is not None
            or selected_project_image.raw_annotation is not None
        )
        has_calibration = selected_project_image is not None and (
            selected_project_image.calibration is not None
            or selected_project_image.raw_calibration is not None
        )

        self.save_project_action.setEnabled(has_project)
        self.close_project_action.setEnabled(has_project)
        self.add_images_action.setEnabled(has_project)
        self.sync_images_action.setEnabled(has_project)
        self.remove_image_action.setEnabled(has_project_image)
        self.export_project_excel_action.setEnabled(has_project)
        self.export_project_excel_button.setEnabled(has_project)
        self.export_image_properties_excel_action.setEnabled(has_project_image)
        self.export_image_properties_excel_button.setEnabled(has_project_image)
        self.zoom_in_action.setEnabled(has_image)
        self.zoom_out_action.setEnabled(has_image)
        self.fit_image_action.setEnabled(has_image)
        self.actual_size_action.setEnabled(has_image)
        self.show_all_canvas_elements_action.setEnabled(has_project_image and has_image)
        self.hide_all_canvas_elements_action.setEnabled(has_project_image and has_image)
        # Пока фоновая задача меняет контур или пиксели текущего кадра,
        # конфликтующие команды выключены: их результат перезаписал бы друг друга.
        image_job_pending = (
            self._pending_contour_detection is not None
            or self._pending_flatten_background is not None
        )
        self.new_contour_action.setEnabled(
            has_project_image and has_image and not image_job_pending
        )
        self.detect_contour_action.setEnabled(
            has_project_image and has_image and not image_job_pending
        )
        self.delete_contour_action.setEnabled(
            has_project_image and (has_contour or has_project_contour)
        )
        self.flatten_background_action.setEnabled(
            has_project_image and has_image and has_contour and not image_job_pending
        )
        self.calibrate_scale_action.setEnabled(has_project_image and has_image)
        self.reset_calibration_action.setEnabled(has_project_image and has_calibration)
        self.measure_angle_action.setEnabled(has_project_image and has_image)
        self.delete_angle_action.setEnabled(
            has_project_image and has_image and self.canvas.has_selected_angle_vertex()
        )
        self.measure_segment_action.setEnabled(has_project_image and has_image)
        self.delete_segment_action.setEnabled(
            has_project_image
            and has_image
            and self.canvas.has_selected_segment_endpoint()
        )
        self.save_annotation_action.setEnabled(has_project_image and has_contour)
        self.open_annotation_action.setEnabled(
            has_project_image and not image_job_pending
        )
        self.refresh_analysis_action.setEnabled(has_project)
        self.refresh_analysis_button.setEnabled(has_project)

    def restore_default_view(self) -> None:
        self._restore_splitter_defaults()
        if self.canvas.has_image():
            self.canvas.fit_to_image()
        self.statusBar().showMessage("Вид по умолчанию восстановлен.")

    def show_all_canvas_elements(self) -> None:
        self._set_all_canvas_elements_visible(True)

    def hide_all_canvas_elements(self) -> None:
        self._set_all_canvas_elements_visible(False)

    def _set_all_canvas_elements_visible(self, visible: bool) -> None:
        record = self._selected_project_image()
        if record is None or not self.canvas.has_image():
            return

        if self.canvas.has_contour():
            self.canvas.set_contour_visible(visible)
        for angle in record.measurements.angles:
            self.canvas.set_angle_measurement_visible(angle.id, visible)
        for segment in record.measurements.segments:
            self.canvas.set_segment_measurement_visible(segment.id, visible)

        self._sync_canvas_visibility_fields()
        self._update_action_states()
        self.statusBar().showMessage(
            "Показаны все элементы на канвасе."
            if visible
            else "Скрыты все элементы на канвасе."
        )

    def _sync_canvas_visibility_fields(self) -> None:
        self._update_contour_visibility_field()
        for angle_id, field in self._angle_visibility_fields.items():
            with QSignalBlocker(field):
                field.setChecked(self.canvas.is_angle_measurement_visible(angle_id))
        for segment_id, field in self._segment_visibility_fields.items():
            with QSignalBlocker(field):
                field.setChecked(self.canvas.is_segment_measurement_visible(segment_id))

    def _handle_image_property_item_clicked(
        self, item: QTreeWidgetItem, _column: int
    ) -> None:
        reference = self._canvas_reference_from_property_item(item)
        if reference is None:
            self._clear_image_property_canvas_highlight()
            return

        reference_type, reference_id = reference
        if reference_type == "contour":
            self.canvas.highlight_contour()
        elif reference_type == "angle" and reference_id is not None:
            self.canvas.highlight_angle_measurement(reference_id)
        elif reference_type == "segment" and reference_id is not None:
            self.canvas.highlight_segment_measurement(reference_id)
        else:
            self.canvas.clear_canvas_highlight()
        self._update_action_states()

    def _clear_image_property_canvas_highlight(self) -> None:
        self.canvas.clear_canvas_highlight()
        self._update_action_states()

    def _clear_image_property_measurement_highlight(self) -> None:
        self._clear_image_property_canvas_highlight()

    def _canvas_reference_from_property_item(
        self,
        item: QTreeWidgetItem | None,
    ) -> tuple[str, str | None] | None:
        current = item
        while current is not None:
            key = str(current.data(0, Qt.UserRole + 1) or "")
            if key == "contour" or key.startswith("contour:"):
                return "contour", None
            parts = key.split(":")
            if len(parts) >= 2 and parts[0] in {"angle", "segment"} and parts[1]:
                return parts[0], parts[1]
            current = current.parent()
        return None

    def _measurement_reference_from_property_item(
        self,
        item: QTreeWidgetItem | None,
    ) -> tuple[str, str] | None:
        reference = self._canvas_reference_from_property_item(item)
        if reference is None:
            return None
        measurement_type, measurement_id = reference
        if measurement_type == "angle" and measurement_id is not None:
            return measurement_type, measurement_id
        if measurement_type == "segment" and measurement_id is not None:
            return measurement_type, measurement_id
        return None

    def refresh_analysis(self) -> None:
        self._analysis_refresh_allowed = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._refresh_histograms()
            self._update_project_summary_properties()
            self._update_project_properties()
        finally:
            QApplication.restoreOverrideCursor()
            self._analysis_refresh_allowed = False

        self._update_action_states()
        if self._is_current_contour_analysis_pending():
            self.statusBar().showMessage("Идёт расчёт графиков...")
        elif self.canvas.has_image() and self.canvas.has_contour():
            self.statusBar().showMessage("Графики и аналитика обновлены.")
        else:
            self.statusBar().showMessage(
                "Нечего пересчитывать: выберите изображение с контуром."
            )

    def _refresh_histograms(self) -> None:
        previously_allowed = self._analysis_refresh_allowed
        self._analysis_refresh_allowed = True
        try:
            self._apply_refreshed_histograms()
        finally:
            self._analysis_refresh_allowed = previously_allowed

    def _apply_refreshed_histograms(self) -> None:
        result = self._ensure_current_contour_analysis(defer_large_async=False)
        if result is None:
            if self._is_current_contour_analysis_pending():
                self._clear_histograms(CONTOUR_ANALYSIS_PENDING_TEXT)
                return
            message = (
                "Откройте изображение и создайте контур, чтобы увидеть гистограмму."
            )
            if self.canvas.has_image() and not self.canvas.has_contour():
                message = "Создайте основной контур, чтобы увидеть гистограмму."
            elif self.canvas.has_image() and self.canvas.has_contour():
                message = "Внутри контура нет пикселей для анализа."
            self._clear_histograms(message)
            return

        self._apply_contour_analysis_histograms(result)

    def _apply_contour_analysis_histograms(
        self, result: ContourAnalysisWorkResult
    ) -> None:
        if result.analysis is None:
            self._clear_histograms("Внутри контура нет пикселей для анализа.")
            return

        histogram_specs = (
            (self.rgb_histogram_panel, result.analysis.histograms.rgb, "RGB"),
            (self.lab_histogram_panel, result.analysis.histograms.lab, "Lab"),
            (self.hsv_histogram_panel, result.analysis.histograms.hsv, "HSV"),
            (self.yuv_histogram_panel, result.analysis.histograms.yuv, "YUV"),
            (self.lms_histogram_panel, result.analysis.histograms.lms, "LMS"),
        )
        for panel, histogram, name in histogram_specs:
            if histogram is None:
                panel.clear_histogram(
                    f"Внутри контура нет пикселей для {name}-гистограммы."
                )
            else:
                panel.set_histogram(histogram)
        self._update_analysis_status()

    def _clear_histograms(self, message: str) -> None:
        for panel in (
            self.rgb_histogram_panel,
            self.lab_histogram_panel,
            self.hsv_histogram_panel,
            self.yuv_histogram_panel,
            self.lms_histogram_panel,
        ):
            panel.clear_histogram(message)
        self._update_analysis_status()

    def _ensure_current_contour_analysis(
        self,
        record: ProjectImageRecord | None = None,
        *,
        defer_large_async: bool = True,
    ) -> ContourAnalysisWorkResult | None:
        inputs = self._current_contour_analysis_inputs(record)
        if inputs is None:
            return None
        key, record_id, image_path, points = inputs

        if self._contour_analysis_cache_matches(key):
            return self._contour_analysis_cache
        if self._pending_contour_analysis_key == key:
            return None
        if not self._analysis_refresh_allowed:
            return None

        image_size = self.canvas.image_size()
        if image_size is None:
            return None
        width, height = image_size
        is_large_image = width * height > CONTOUR_ANALYSIS_SYNC_PIXEL_LIMIT
        if is_large_image and defer_large_async:
            return None

        try:
            # Без копии: массив только для чтения, канвас его не меняет.
            rgb_array = self.canvas.current_rgb_array_view()
        except ValueError as exc:
            self._clear_histograms(str(exc))
            return None

        self._contour_analysis_request_id += 1
        request_id = self._contour_analysis_request_id
        if not is_large_image:
            result = ContourAnalysisWorkResult(
                request_id=request_id,
                record_id=record_id,
                image_path=image_path,
                signature=key[2],
                analysis=build_contour_analysis(rgb_array, points),
            )
            self._contour_analysis_cache = result
            self._pending_contour_analysis_key = None
            self._pending_contour_analysis_request_id = None
            return result

        self._pending_contour_analysis_key = key
        self._pending_contour_analysis_request_id = request_id
        worker = _ContourAnalysisWorker(
            request_id=request_id,
            record_id=record_id,
            image_path=image_path,
            signature=key[2],
            rgb_array=rgb_array,
            points=points,
        )
        worker.signals.finished.connect(self._handle_contour_analysis_finished)
        worker.signals.failed.connect(self._handle_contour_analysis_failed)
        worker.signals.finished.connect(
            lambda _result, current_worker=worker: (
                self._contour_analysis_workers.discard(current_worker)
            )
        )
        worker.signals.failed.connect(
            lambda _request_id, _message, current_worker=worker: (
                self._contour_analysis_workers.discard(current_worker)
            )
        )
        self._contour_analysis_workers.add(worker)
        self._contour_analysis_thread_pool.start(worker)
        return None

    def _current_contour_analysis_inputs(
        self,
        record: ProjectImageRecord | None = None,
    ) -> tuple[ContourAnalysisCacheKey, str | None, str | None, list[Point]] | None:
        if not self.canvas.has_image() or not self.canvas.has_contour():
            return None

        record_id = record.id if record is not None else self._current_project_image_id
        image_path = self.canvas.current_image_path()
        image_path_text = str(image_path) if image_path is not None else None
        points = self.canvas.contour_points()
        signature = contour_signature(points)
        return (
            (record_id, image_path_text, signature),
            record_id,
            image_path_text,
            points,
        )

    def _contour_analysis_cache_matches(self, key: ContourAnalysisCacheKey) -> bool:
        cached = self._contour_analysis_cache
        return (
            cached is not None
            and cached.record_id == key[0]
            and cached.image_path == key[1]
            and cached.signature == key[2]
        )

    def _is_current_contour_analysis_pending(
        self,
        record: ProjectImageRecord | None = None,
    ) -> bool:
        inputs = self._current_contour_analysis_inputs(record)
        if inputs is None:
            return False
        return self._pending_contour_analysis_key == inputs[0]

    def _is_current_contour_analysis_outdated(
        self,
        record: ProjectImageRecord | None = None,
    ) -> bool:
        inputs = self._current_contour_analysis_inputs(record)
        if inputs is None:
            return False
        if self._contour_analysis_cache_matches(inputs[0]):
            return False
        return not self._is_current_contour_analysis_pending(record)

    def _update_analysis_status(self) -> None:
        if self._is_current_contour_analysis_pending():
            self.analysis_status_label.setText(CONTOUR_ANALYSIS_PENDING_TEXT)
        elif self._is_current_contour_analysis_outdated():
            self.analysis_status_label.setText(ANALYSIS_OUTDATED_STATUS_TEXT)
        else:
            self.analysis_status_label.setText("")

    def _is_contour_analysis_result_current(
        self, result: ContourAnalysisWorkResult
    ) -> bool:
        inputs = self._current_contour_analysis_inputs()
        if inputs is None:
            return False
        key = inputs[0]
        return (
            result.record_id == key[0]
            and result.image_path == key[1]
            and result.signature == key[2]
        )

    def _handle_contour_analysis_finished(
        self, result: ContourAnalysisWorkResult
    ) -> None:
        if result.request_id != self._contour_analysis_request_id:
            return
        if not self._is_contour_analysis_result_current(result):
            return

        self._pending_contour_analysis_key = None
        self._pending_contour_analysis_request_id = None
        self._contour_analysis_cache = result
        self._update_project_properties()
        self._apply_contour_analysis_histograms(result)

    def _handle_contour_analysis_failed(self, request_id: int, message: str) -> None:
        if request_id != self._pending_contour_analysis_request_id:
            return

        self._pending_contour_analysis_key = None
        self._pending_contour_analysis_request_id = None
        self._clear_histograms(message or "Не удалось рассчитать гистограммы.")
        self._update_project_properties()

    def _invalidate_current_contour_analysis(self) -> None:
        self._contour_analysis_request_id += 1
        self._contour_analysis_cache = None
        self._pending_contour_analysis_key = None
        self._pending_contour_analysis_request_id = None

    def _restore_splitter_defaults(self) -> None:
        self.workspace_splitter.setSizes(WORKSPACE_DEFAULT_SIZES)
        self.project_splitter.setSizes(PROJECT_PANEL_DEFAULT_SIZES)
        self.histogram_splitter.setSizes(HISTOGRAM_DEFAULT_SIZES)

    def new_project(self) -> None:
        project_name, accepted = QInputDialog.getText(
            self,
            "Новый проект",
            "Имя проекта:",
        )
        if not accepted:
            return

        safe_name = _safe_project_name(project_name)
        if not safe_name:
            self._show_warning("Некорректное имя", "Введите имя проекта.")
            return

        parent_dir_name = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку, где будет создан проект",
            "",
        )
        if not parent_dir_name:
            return

        project_dir = Path(parent_dir_name) / safe_name
        if project_dir.exists() and any(project_dir.iterdir()):
            self._show_warning(
                "Папка уже существует",
                "Выберите другое имя проекта или пустую папку.",
            )
            return

        self._save_current_project_annotation()
        if not self._save_project_silently(show_error=True):
            return

        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "images").mkdir(exist_ok=True)
        except OSError as exc:
            self._show_error(
                "Ошибка создания проекта", f"Не удалось создать папку проекта: {exc}"
            )
            return

        project_path = project_dir / f"{safe_name}.json"
        document = ProjectDocument(name=safe_name, images=[])
        try:
            save_project(project_path, document)
        except OSError as exc:
            self._show_error(
                "Ошибка создания проекта", f"Не удалось сохранить файл проекта: {exc}"
            )
            return

        self._set_project(project_path, document)
        self.statusBar().showMessage(f"Создан проект: {safe_name}")

    def open_project(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть проект",
            "",
            "MagicBorder project (*.json);;JSON (*.json);;All files (*)",
        )
        if not file_name:
            return

        self._save_current_project_annotation()
        if not self._save_project_silently(show_error=True):
            return

        project_path = Path(file_name)
        try:
            document = load_project(project_path)
        except (OSError, ValueError) as exc:
            self._show_error("Ошибка открытия проекта", str(exc))
            return

        self._set_project(project_path, document)
        self.statusBar().showMessage(f"Открыт проект: {document.name}")

    def save_project_file(self) -> None:
        if self.project_document is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return
        self._store_current_angle_measurements()
        self._store_current_segment_measurements()
        self._save_current_project_annotation()
        if self._save_project_silently(show_error=True):
            self.statusBar().showMessage("Проект сохранён.")

    def close_project(self) -> None:
        if self.project_document is None:
            return
        self._store_current_angle_measurements()
        self._store_current_segment_measurements()
        self._save_current_project_annotation()
        if not self._save_project_silently(show_error=True):
            return
        self._clear_project_state()
        self.statusBar().showMessage("Проект закрыт.")

    def add_images_to_project(self) -> None:
        if self.project_document is None or self.project_path is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return

        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Добавить изображения в проект",
            "",
            image_open_filter(),
        )
        if not file_names:
            return

        project_dir = self.project_path.parent
        image_dir = project_dir / self.project_document.images_dir
        errors: list[str] = []
        added_ids: list[str] = []

        try:
            image_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._show_error(
                "Ошибка добавления", f"Не удалось создать папку изображений: {exc}"
            )
            return

        candidates = self._probe_image_import_candidates(
            [(Path(file_name), Path(file_name).name) for file_name in file_names],
            errors,
        )
        preset = self._choose_large_image_downscale(candidates, overwrite=False)
        if preset == DOWNSCALE_CANCELLED:
            return

        reserved_paths: set[Path] = set()
        jobs = [
            ImagePrepareJob(
                source=candidate.source,
                destination=_unique_destination_path(
                    image_dir, candidate.source.name, reserved=reserved_paths
                ),
                preset=preset if is_large_image(candidate.size) else None,
            )
            for candidate in candidates
        ]
        results = self._run_image_prepare_jobs(jobs)

        downscaled_count = 0
        for candidate, job, result in zip(candidates, jobs, results, strict=True):
            if isinstance(result, str):
                errors.append(f"{candidate.label}: {result}")
                continue

            record_id = self._new_project_image_id()
            record = ProjectImageRecord(
                id=record_id,
                relative_path=portable_path_reference(job.destination, project_dir),
                display_name=job.destination.name,
                image_width=result.width,
                image_height=result.height,
                metadata=default_project_image_metadata(
                    added_at=_current_timestamp(),
                    captured_at=candidate.captured_at,
                ),
            )
            self.project_document.images.append(record)
            added_ids.append(record.id)
            downscaled_count += int(result.downscaled)

        if added_ids:
            self._refresh_project_list()
            self._select_project_image(added_ids[0])
            self._update_project_summary_properties()
            self._save_project_silently(show_error=True)
            self.statusBar().showMessage(
                f"Добавлено изображений: {len(added_ids)}"
                + _downscaled_count_suffix(downscaled_count)
            )

        if errors:
            self._show_warning("Не все изображения добавлены", "\n".join(errors[:8]))

    def sync_project_images_folder(self) -> None:
        if self.project_document is None or self.project_path is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return

        self._save_current_project_annotation()

        project_dir = self.project_path.parent
        image_dir = project_dir / self.project_document.images_dir
        errors: list[str] = []
        added_ids: list[str] = []

        try:
            image_dir.mkdir(parents=True, exist_ok=True)
            candidate_paths = sorted(
                path
                for path in image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_RASTER_SUFFIXES
            )
        except OSError as exc:
            self._show_error(
                "Ошибка синхронизации",
                f"Не удалось прочитать папку изображений проекта: {exc}",
            )
            return

        existing_paths = {
            _project_relative_path_key(record.relative_path)
            for record in self.project_document.images
        }
        untracked_paths: list[tuple[Path, str]] = []
        for image_path in candidate_paths:
            relative_path = portable_path_reference(image_path, project_dir)
            relative_path_key = _project_relative_path_key(relative_path)
            if relative_path_key in existing_paths:
                continue
            existing_paths.add(relative_path_key)
            untracked_paths.append((image_path, relative_path))

        candidates = self._probe_image_import_candidates(untracked_paths, errors)
        preset = self._choose_large_image_downscale(candidates, overwrite=True)
        if preset == DOWNSCALE_CANCELLED:
            return

        jobs = [
            ImagePrepareJob(
                source=candidate.source,
                destination=candidate.source,
                preset=preset if is_large_image(candidate.size) else None,
            )
            for candidate in candidates
        ]
        results = self._run_image_prepare_jobs(jobs)

        downscaled_count = 0
        for candidate, result in zip(candidates, results, strict=True):
            if isinstance(result, str):
                errors.append(f"{candidate.label}: {result}")
                continue

            try:
                record_id = self._new_project_image_id()
            except ValueError as exc:
                self._show_error("Ошибка синхронизации", str(exc))
                return

            record = ProjectImageRecord(
                id=record_id,
                relative_path=candidate.label,
                display_name=candidate.source.name,
                image_width=result.width,
                image_height=result.height,
                metadata=default_project_image_metadata(
                    added_at=_current_timestamp(),
                    captured_at=candidate.captured_at,
                ),
            )
            self.project_document.images.append(record)
            added_ids.append(record.id)
            downscaled_count += int(result.downscaled)

        if added_ids:
            self._refresh_project_list()
            self._select_project_image(added_ids[0])
            self._update_project_summary_properties()
            self._save_project_silently(show_error=True)
            self.statusBar().showMessage(
                f"Синхронизировано изображений: {len(added_ids)}"
                + _downscaled_count_suffix(downscaled_count)
            )
        else:
            self.statusBar().showMessage("Новых изображений не найдено.")

        if errors:
            self._show_warning(
                "Не все изображения синхронизированы", "\n".join(errors[:8])
            )

    def _probe_image_import_candidates(
        self, sources: list[tuple[Path, str]], errors: list[str]
    ) -> list[ImageImportCandidate]:
        candidates: list[ImageImportCandidate] = []
        for source, label in sources:
            try:
                # Размер и дата съёмки за одно открытие файла вместо двух.
                size, captured_at = read_image_header(source)
            except (OSError, ValueError) as exc:
                errors.append(f"{label}: {exc}")
                continue
            candidates.append(
                ImageImportCandidate(
                    source=source,
                    label=label,
                    size=size,
                    captured_at=captured_at,
                )
            )
        return candidates

    def _choose_large_image_downscale(
        self, candidates: list[ImageImportCandidate], *, overwrite: bool
    ) -> DownscalePreset | str | None:
        large_images = [
            (candidate.label, candidate.size)
            for candidate in candidates
            if is_large_image(candidate.size)
        ]
        if not large_images:
            return None
        return self._ask_large_image_downscale(
            large_images, len(candidates), overwrite=overwrite
        )

    def _ask_large_image_downscale(
        self,
        large_images: list[tuple[str, tuple[int, int]]],
        total_count: int,
        *,
        overwrite: bool,
    ) -> DownscalePreset | str | None:
        """Возвращает выбранный пресет, None для оригинала или DOWNSCALE_CANCELLED."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Большие изображения")

        threshold_width, threshold_height = LARGE_IMAGE_THRESHOLD
        message_lines = [
            f"{len(large_images)} из {total_count} изображений имеют разрешение "
            f"от {threshold_width}×{threshold_height}. "
            "Обработка больших файлов может быть медленной.",
            "Уменьшить разрешение? Пропорции изображения сохраняются.",
        ]
        if overwrite:
            message_lines.append(
                "Файлы в папке проекта будут перезаписаны уменьшенными копиями."
            )
        message_label = QLabel("\n\n".join(message_lines), dialog)
        message_label.setWordWrap(True)

        listed_images = [
            f"{label} — {width}×{height}"
            for label, (width, height) in large_images[:LARGE_IMAGE_LIST_LIMIT]
        ]
        hidden_count = len(large_images) - len(listed_images)
        if hidden_count > 0:
            listed_images.append(f"…и ещё {hidden_count}")
        images_label = QLabel("\n".join(listed_images), dialog)
        images_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        example_label, example_size = large_images[0]
        preset_buttons: list[tuple[QRadioButton, DownscalePreset | None]] = []
        for index, preset in enumerate(DOWNSCALE_PRESETS):
            text = preset.label + (" — рекомендуется" if index == 0 else "")
            button = QRadioButton(text, dialog)
            target_width, target_height = fit_size(example_size, preset)
            button.setToolTip(
                f"{example_label}: {example_size[0]}×{example_size[1]} → "
                f"{target_width}×{target_height}"
            )
            preset_buttons.append((button, preset))
        original_button = QRadioButton("Оставить оригинальное разрешение", dialog)
        preset_buttons.append((original_button, None))
        preset_buttons[0][0].setChecked(True)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(message_label)
        layout.addWidget(images_label)
        for button, _preset in preset_buttons:
            layout.addWidget(button)
        layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return DOWNSCALE_CANCELLED
        for button, preset in preset_buttons:
            if button.isChecked():
                return preset
        return None

    def _run_image_prepare_jobs(
        self, jobs: list[ImagePrepareJob]
    ) -> list[ImagePrepareResult | str]:
        """Выполняет задачи в пуле потоков, не блокируя цикл событий Qt."""
        if not jobs:
            return []

        results: list[ImagePrepareResult | str | None] = [None] * len(jobs)
        remaining = len(jobs)
        cancel_event = threading.Event()
        loop = QEventLoop(self)

        progress = QProgressDialog(
            "Подготовка изображений…", "Отмена", 0, len(jobs), self
        )
        progress.setWindowTitle("Добавление изображений")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(IMAGE_PREPARE_PROGRESS_DELAY_MS)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.canceled.connect(cancel_event.set)

        def store_result(index: int, result: ImagePrepareResult | str) -> None:
            nonlocal remaining
            if results[index] is not None:
                return
            results[index] = result
            remaining -= 1
            progress.setValue(len(jobs) - remaining)
            if remaining == 0:
                loop.quit()

        workers: list[_ImagePrepareWorker] = []
        for index, job in enumerate(jobs):
            worker = _ImagePrepareWorker(index, job, cancel_event)
            worker.signals.finished.connect(store_result)
            worker.signals.failed.connect(store_result)
            workers.append(worker)
            self._image_prepare_thread_pool.start(worker)

        QApplication.setOverrideCursor(Qt.BusyCursor)
        try:
            if remaining:
                loop.exec_()
        finally:
            QApplication.restoreOverrideCursor()
            progress.canceled.disconnect()
            progress.close()
            progress.deleteLater()

        return [
            result if result is not None else IMAGE_PREPARE_CANCELLED_TEXT
            for result in results
        ]

    def remove_selected_project_image(self) -> None:
        if self.project_document is None or self.project_path is None:
            return

        record = self._selected_project_image()
        if record is None:
            self._show_warning(
                "Изображение не выбрано", "Выберите изображение в списке проекта."
            )
            return

        row_to_remove = self.project_list.currentRow()
        self._save_current_project_annotation()

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Удалить изображение")
        dialog.setIcon(QMessageBox.Question)
        dialog.setText(f"Удалить '{record.display_name}' из проекта?")
        delete_file_button = dialog.addButton(
            "Удалить файл", QMessageBox.DestructiveRole
        )
        remove_only_button = dialog.addButton(
            "Только убрать из проекта", QMessageBox.AcceptRole
        )
        cancel_button = dialog.addButton("Отмена", QMessageBox.RejectRole)
        dialog.setDefaultButton(remove_only_button)
        dialog.exec_()

        clicked_button = dialog.clickedButton()
        if clicked_button is cancel_button:
            return

        image_path = self._project_image_path(record)
        if clicked_button is delete_file_button and image_path.exists():
            try:
                image_path.unlink()
            except OSError as exc:
                self._show_error(
                    "Ошибка удаления", f"Не удалось удалить файл изображения: {exc}"
                )
                return

        self.project_document.images = [
            item for item in self.project_document.images if item.id != record.id
        ]
        if self._current_project_image_id == record.id:
            self._current_project_image_id = None

        self._refresh_project_list()
        if self.project_document.images:
            next_row = min(max(row_to_remove, 0), len(self.project_document.images) - 1)
            self.project_list.setCurrentRow(next_row)
        else:
            self._clear_current_image_display()
        self._update_project_summary_properties()
        self._save_project_silently(show_error=True)
        self.statusBar().showMessage("Изображение удалено из проекта.")

    def export_project_excel(self) -> None:
        if self.project_document is None or self.project_path is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return

        self._save_current_project_annotation()
        selected_fieldnames = self._select_project_export_columns()
        if selected_fieldnames is None:
            return
        if not selected_fieldnames:
            self._show_warning(
                "Нет выбранных столбцов", "Выберите хотя бы один столбец для экспорта."
            )
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт списка в Excel",
            str(self.project_path.with_suffix(".xlsx")),
            "Excel (*.xlsx);;All files (*)",
        )
        if not file_name:
            return

        output_path = _ensure_xlsx_suffix(Path(file_name))
        try:
            self._write_project_excel(output_path, selected_fieldnames)
        except OSError as exc:
            self._show_error(
                "Ошибка экспорта списка в Excel",
                f"Не удалось сохранить Excel-файл: {exc}",
            )
            return

        self.statusBar().showMessage(
            f"Экспорт списка в Excel выполнен: {output_path.name}"
        )

    def export_project_csv(self) -> None:
        self.export_project_excel()

    def _select_project_export_columns(self) -> list[str] | None:
        return self._select_checked_export_items(
            title="Столбцы экспорта",
            items=PROJECT_EXPORT_COLUMNS,
            empty_title="Нет выбранных столбцов",
            empty_message="Выберите хотя бы один столбец для экспорта.",
        )

    def _select_checked_export_items(
        self,
        *,
        title: str,
        items: list[tuple[str, str]] | None = None,
        grouped_items: list[tuple[str, list[tuple[str, str]]]] | None = None,
        empty_title: str,
        empty_message: str,
    ) -> list[str] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)

        item_list = QListWidget(dialog)
        item_list.setAlternatingRowColors(True)

        def add_checkable_item(field_name: str, label: str) -> None:
            item = QListWidgetItem(label)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, field_name)
            item_list.addItem(item)

        if grouped_items is not None:
            for group_title, group_items in grouped_items:
                group_item = QListWidgetItem(group_title)
                group_item.setFlags(Qt.ItemIsEnabled)
                group_item.setData(Qt.UserRole, "")
                item_list.addItem(group_item)
                for field_name, label in group_items:
                    add_checkable_item(field_name, label)
        else:
            for field_name, label in items or []:
                add_checkable_item(field_name, label)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        button_box.rejected.connect(dialog.reject)

        def selected_items() -> list[str]:
            return [
                str(item_list.item(index).data(Qt.UserRole))
                for index in range(item_list.count())
                if item_list.item(index).data(Qt.UserRole)
                and item_list.item(index).checkState() == Qt.Checked
            ]

        def accept_if_any_selected() -> None:
            if not selected_items():
                self._show_warning(empty_title, empty_message)
                return
            dialog.accept()

        ok_button = button_box.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.clicked.connect(accept_if_any_selected)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(item_list)
        layout.addWidget(button_box)
        dialog.resize(360, 360)

        if dialog.exec_() != QDialog.Accepted:
            return None
        return selected_items()

    def _select_checked_export_tree_items(
        self,
        *,
        title: str,
        groups: list[ExportTreeItem],
        empty_title: str,
        empty_message: str,
        default_checked_keys: set[str] | None = None,
    ) -> list[str] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)

        item_tree = QTreeWidget(dialog)
        item_tree.setObjectName("exportItemTree")
        item_tree.setHeaderHidden(True)
        item_tree.setAlternatingRowColors(True)

        def item_check_state_from_children(item: QTreeWidgetItem) -> Qt.CheckState:
            child_states = [
                item.child(index).checkState(0) for index in range(item.childCount())
            ]
            if all(state == Qt.Checked for state in child_states):
                return Qt.Checked
            if all(state == Qt.Unchecked for state in child_states):
                return Qt.Unchecked
            return Qt.PartiallyChecked

        def add_tree_item(
            node: ExportTreeItem, parent: QTreeWidgetItem | None = None
        ) -> QTreeWidgetItem:
            item = QTreeWidgetItem([node.label])
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setData(0, Qt.UserRole, node.key)
            item.setData(0, Qt.UserRole + 1, node.export_label or node.label)
            if parent is None:
                item_tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for child in node.children:
                add_tree_item(child, item)
            if node.children:
                item.setCheckState(0, item_check_state_from_children(item))
            else:
                checked = (
                    default_checked_keys is None or node.key in default_checked_keys
                )
                item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
            item.setExpanded(True)
            return item

        for group in groups:
            add_tree_item(group)

        updating_check_states = False

        def set_children_check_state(
            item: QTreeWidgetItem, state: Qt.CheckState
        ) -> None:
            for index in range(item.childCount()):
                child = item.child(index)
                child.setCheckState(0, state)
                set_children_check_state(child, state)

        def update_parent_check_states(item: QTreeWidgetItem) -> None:
            parent = item.parent()
            while parent is not None:
                parent.setCheckState(0, item_check_state_from_children(parent))
                parent = parent.parent()

        def handle_item_changed(item: QTreeWidgetItem, column: int) -> None:
            nonlocal updating_check_states
            if column != 0 or updating_check_states:
                return
            updating_check_states = True
            try:
                state = item.checkState(0)
                if item.childCount() and state != Qt.PartiallyChecked:
                    set_children_check_state(item, state)
                update_parent_check_states(item)
            finally:
                updating_check_states = False

        item_tree.itemChanged.connect(handle_item_changed)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        button_box.rejected.connect(dialog.reject)

        def selected_items() -> list[str]:
            selected: list[str] = []

            def collect_checked_leaves(item: QTreeWidgetItem) -> None:
                if item.childCount():
                    for child_index in range(item.childCount()):
                        collect_checked_leaves(item.child(child_index))
                    return
                key = str(item.data(0, Qt.UserRole) or "")
                if key and item.checkState(0) == Qt.Checked:
                    selected.append(key)

            for index in range(item_tree.topLevelItemCount()):
                collect_checked_leaves(item_tree.topLevelItem(index))
            return selected

        def accept_if_any_selected() -> None:
            if not selected_items():
                self._show_warning(empty_title, empty_message)
                return
            dialog.accept()

        ok_button = button_box.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.clicked.connect(accept_if_any_selected)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(item_tree)
        layout.addWidget(button_box)
        dialog.resize(420, 520)

        if dialog.exec_() != QDialog.Accepted:
            return None
        return selected_items()

    def _write_project_excel(
        self, output_path: Path, selected_fieldnames: list[str]
    ) -> None:
        if self.project_document is None:
            raise ValueError("Нет проекта для экспорта.")

        rows = [
            self._project_export_row(record) for record in self.project_document.images
        ]
        selected_headers = [
            PROJECT_EXPORT_COLUMN_LABELS[field_name]
            for field_name in selected_fieldnames
        ]
        localized_rows = [
            {
                PROJECT_EXPORT_COLUMN_LABELS[field_name]: row.get(field_name, "")
                for field_name in selected_fieldnames
            }
            for row in rows
        ]
        write_xlsx_table(
            output_path, selected_headers, localized_rows, sheet_name="Сводка"
        )

    def export_image_properties_excel(self) -> None:
        if self.project_document is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return

        record = self._selected_project_image()
        if record is None:
            self._show_warning(
                "Нет выбранного изображения", "Выберите изображение в списке проекта."
            )
            return

        self._save_current_project_annotation()
        self._update_project_properties()
        selected_properties = self._select_image_property_export_items()
        if selected_properties is None:
            return
        if not selected_properties:
            self._show_warning(
                "Нет выбранных свойств", "Выберите хотя бы одно свойство для экспорта."
            )
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт свойств изображения в Excel",
            str(self._default_image_properties_export_path(record)),
            "Excel (*.xlsx);;All files (*)",
        )
        if not file_name:
            return

        output_path = _ensure_xlsx_suffix(Path(file_name))
        try:
            self._write_image_properties_excel(output_path, selected_properties)
        except OSError as exc:
            self._show_error(
                "Ошибка экспорта свойств изображения в Excel",
                f"Не удалось сохранить Excel-файл: {exc}",
            )
            return

        self.statusBar().showMessage(
            f"Экспорт свойств изображения в Excel выполнен: {output_path.name}"
        )

    def _select_image_property_export_items(self) -> list[str] | None:
        record = self._selected_project_image()
        return self._select_checked_export_tree_items(
            title="Свойства экспорта",
            groups=self._image_property_export_groups(record),
            empty_title="Нет выбранных свойств",
            empty_message="Выберите хотя бы одно свойство для экспорта.",
            default_checked_keys=_default_image_property_export_keys(),
        )

    def _image_property_export_groups(
        self, record: ProjectImageRecord | None
    ) -> list[ExportTreeItem]:
        groups: list[ExportTreeItem] = []
        for group in _static_image_property_export_groups():
            groups.append(group)
            if group.label == "Информация о главном контуре" and record is not None:
                measurement_group = self._measurement_property_export_group(record)
                if measurement_group is not None:
                    groups.append(measurement_group)
        return groups

    def _image_property_export_labels(
        self, record: ProjectImageRecord | None
    ) -> dict[str, str]:
        return _export_tree_labels(self._image_property_export_groups(record))

    def _image_property_export_leaf_keys(
        self, record: ProjectImageRecord | None
    ) -> list[str]:
        return _export_tree_leaf_keys(self._image_property_export_groups(record))

    def _measurement_property_export_group(
        self, record: ProjectImageRecord
    ) -> ExportTreeItem | None:
        measurement_children: list[ExportTreeItem] = []
        angle_groups: list[ExportTreeItem] = []
        for index, angle in enumerate(record.measurements.angles, start=1):
            angle_title = _angle_display_name(angle, index)
            angle_leaves = [
                (f"angle:{angle.id}:name", "Имя"),
                (f"angle:{angle.id}:id", "ID"),
                (f"angle:{angle.id}:value", "Значение"),
                (f"angle:{angle.id}:first", "Первая точка"),
                (f"angle:{angle.id}:vertex", "Вершина"),
                (f"angle:{angle.id}:second", "Вторая точка"),
                (f"angle:{angle.id}:note", "Примечание"),
            ]
            if _measurement_oiv_code(angle.assessment):
                angle_leaves.extend(
                    [
                        (f"angle:{angle.id}:oiv_code", "OIV-код"),
                        (f"angle:{angle.id}:oiv_short_name", "OIV-признак"),
                        (f"angle:{angle.id}:oiv_value", "Фактическое значение"),
                        (f"angle:{angle.id}:oiv_score", "Оценка OIV"),
                        (f"angle:{angle.id}:oiv_label", "Значение шкалы"),
                    ]
                )
            angle_groups.append(
                ExportTreeItem(
                    label=angle_title,
                    children=tuple(
                        _export_leaf(key, label) for key, label in angle_leaves
                    ),
                )
            )
        if angle_groups:
            measurement_children.append(
                ExportTreeItem(label="Углы", children=tuple(angle_groups))
            )

        segment_groups: list[ExportTreeItem] = []
        for index, segment in enumerate(record.measurements.segments, start=1):
            segment_title = _segment_display_name(segment, index)
            segment_leaves = [
                (f"segment:{segment.id}:name", "Имя"),
                (f"segment:{segment.id}:id", "ID"),
                (f"segment:{segment.id}:length", "Длина"),
                (f"segment:{segment.id}:start", "Первая точка"),
                (f"segment:{segment.id}:end", "Вторая точка"),
                (f"segment:{segment.id}:start_label", "Подпись первой точки"),
                (f"segment:{segment.id}:end_label", "Подпись второй точки"),
                (f"segment:{segment.id}:note", "Примечание"),
            ]
            if _measurement_oiv_code(segment.assessment):
                segment_leaves.extend(
                    [
                        (f"segment:{segment.id}:oiv_code", "OIV-код"),
                        (f"segment:{segment.id}:oiv_short_name", "OIV-признак"),
                        (f"segment:{segment.id}:oiv_value", "Фактическое значение"),
                        (f"segment:{segment.id}:oiv_score", "Оценка OIV"),
                        (f"segment:{segment.id}:oiv_label", "Значение шкалы"),
                    ]
                )
            segment_groups.append(
                ExportTreeItem(
                    label=segment_title,
                    children=tuple(
                        _export_leaf(key, label) for key, label in segment_leaves
                    ),
                )
            )
        if segment_groups:
            measurement_children.append(
                ExportTreeItem(label="Отрезки", children=tuple(segment_groups))
            )

        if not measurement_children:
            return None
        return ExportTreeItem(label="Измерения", children=tuple(measurement_children))

    def _default_image_properties_export_path(self, record: ProjectImageRecord) -> Path:
        stem = Path(record.display_name).stem or "image"
        file_name = f"{stem}_properties.xlsx"
        if self.project_path is None:
            return Path(file_name)
        return self.project_path.parent / file_name

    def save_average_color_sample(self) -> None:
        record = self._selected_project_image()
        if record is None or not self.canvas.has_image():
            self._show_warning(
                "Нет изображения", "Сначала выберите изображение проекта."
            )
            return

        average_rgb = self._current_average_color_rgb
        if average_rgb is None:
            self._show_warning(
                "Средний цвет недоступен",
                "Средний цвет ещё не рассчитан. Сначала создайте контур и дождитесь расчёта.",
            )
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить средний цвет",
            str(self._default_average_color_sample_path(record)),
            "PNG (*.png);;All files (*)",
        )
        if not file_name:
            return

        output_path = _ensure_png_suffix(Path(file_name))
        try:
            Image.new("RGB", (100, 100), average_rgb).save(output_path, format="PNG")
        except OSError as exc:
            self._show_error(
                "Ошибка сохранения", f"Не удалось сохранить PNG-файл: {exc}"
            )
            return

        self.statusBar().showMessage(f"Средний цвет сохранён: {output_path.name}")

    def _default_average_color_sample_path(self, record: ProjectImageRecord) -> Path:
        stem = Path(record.display_name).stem or "image"
        file_name = f"{stem}_average_color.png"
        if self.project_path is None:
            return Path(file_name)
        return self.project_path.parent / file_name

    def _write_image_properties_excel(
        self, output_path: Path, selected_properties: list[str]
    ) -> None:
        record = self._selected_project_image()
        if record is None:
            raise ValueError("Нет выбранного изображения для экспорта.")

        rows, bold_rows = self._image_property_export_rows(record, selected_properties)
        fieldnames, cell_fills = self._image_property_export_table_styles(
            record,
            selected_properties,
            rows,
        )
        write_xlsx_table(
            output_path,
            fieldnames,
            rows,
            sheet_name="Свойства",
            bold_rows=bold_rows,
            cell_fills=cell_fills,
        )

    def _image_property_export_table_styles(
        self,
        record: ProjectImageRecord,
        selected_properties: list[str],
        rows: list[dict[str, str]],
    ) -> tuple[list[str], dict[tuple[int, str], str]]:
        fieldnames = ["Свойство", "Значение"]
        if "average_color" not in selected_properties:
            return fieldnames, {}

        average_rgb = self._average_color_rgb_for_export(record)
        if average_rgb is None:
            return fieldnames, {}

        average_text = _rgb_text(average_rgb)
        average_label = self._image_property_export_labels(record).get(
            "average_color", "Средний цвет"
        )
        fill_color = _rgb_hex(average_rgb)
        cell_fills: dict[tuple[int, str], str] = {}
        for row_index, row in enumerate(rows):
            if (
                row.get("Свойство") == average_label
                and row.get("Значение") == average_text
            ):
                row["Цвет"] = ""
                cell_fills[(row_index, "Цвет")] = fill_color
                fieldnames.append("Цвет")
                break
        return fieldnames, cell_fills

    def _average_color_rgb_for_export(
        self,
        record: ProjectImageRecord,
    ) -> tuple[int, int, int] | None:
        if record.id != self._current_project_image_id:
            return None
        return self._current_average_color_rgb

    def _image_property_export_rows(
        self,
        record: ProjectImageRecord,
        selected_properties: list[str],
    ) -> tuple[list[dict[str, str]], set[int]]:
        groups = self._image_property_export_groups(record)
        values = self._image_property_export_values(record)
        selected_keys = set(selected_properties)
        emitted_keys: set[str] = set()
        bold_rows: set[int] = set()

        def node_rows(node: ExportTreeItem) -> list[dict[str, str]]:
            if not node.children:
                if node.key not in selected_keys:
                    return []
                emitted_keys.add(node.key)
                return [{"Свойство": node.label, "Значение": values.get(node.key, "")}]

            rows: list[dict[str, str]] = []
            for child in node.children:
                rows.extend(node_rows(child))
            if not rows:
                return []
            return [{"Свойство": node.label, "Значение": "", "_xlsx_bold": "1"}, *rows]

        rows: list[dict[str, str]] = []
        for group in groups:
            rows.extend(node_rows(group))

        for row_index, row in enumerate(rows):
            if row.pop("_xlsx_bold", "") == "1":
                bold_rows.add(row_index)

        labels = self._image_property_export_labels(record)
        for property_key in selected_properties:
            if property_key in emitted_keys:
                continue
            rows.append(
                {
                    "Свойство": labels.get(
                        property_key,
                        IMAGE_PROPERTY_EXPORT_LABELS.get(property_key, property_key),
                    ),
                    "Значение": values.get(property_key, ""),
                }
            )
        return rows, bold_rows

    def _image_property_export_values(
        self, record: ProjectImageRecord
    ) -> dict[str, str]:
        self._normalize_record_metadata(record)
        red = self.property_red.text()
        green = self.property_green.text()
        blue = self.property_blue.text()
        average_color = "-"
        if all(value not in ("", "-") for value in (red, green, blue)):
            average_color = f"RGB({red}, {green}, {blue})"
        values = {
            "id": self.image_id.text().strip(),
            "file_name": self.property_file_name.text().strip(),
            "relative_path": self.property_path.text(),
            "size": self.property_size.text(),
            "annotation": self.property_annotation.text(),
            "point_count": self.property_points.text(),
            "contour_pixel_count": self.property_contour_pixels.text(),
            "contour_area_mm2": self.property_contour_area_mm2.text(),
            "calibration_length_mm": self.property_calibration_length.text().strip(),
            "calibration_scale": self.property_calibration_scale.text(),
            "red": red,
            "green": green,
            "blue": blue,
            "average_color": average_color,
            "lab_l": self.property_lab_l.text(),
            "lab_a": self.property_lab_a.text(),
            "lab_b": self.property_lab_b.text(),
            "hsv_h": self.property_hsv_h.text(),
            "hsv_s": self.property_hsv_s.text(),
            "hsv_v": self.property_hsv_v.text(),
            "yuv_y": self.property_yuv_y.text(),
            "yuv_u": self.property_yuv_u.text(),
            "yuv_v": self.property_yuv_v.text(),
            "lms_l": self.property_lms_l.text(),
            "lms_m": self.property_lms_m.text(),
            "lms_s": self.property_lms_s.text(),
            "status": self.property_status.text(),
            "added_at": self.metadata_added_at.text().strip(),
            "captured_at": self.metadata_captured_at.text().strip(),
            "illumination": self.metadata_illumination.text().strip(),
            "humidity": self.metadata_humidity.text().strip(),
            "wind_speed": self.metadata_wind_speed.text().strip(),
            "wind_direction": self.metadata_wind_direction.text().strip(),
            "latitude": self.metadata_latitude.text().strip(),
            "longitude": self.metadata_longitude.text().strip(),
            "diagnosis": self.metadata_diagnosis.text().strip(),
            "notes": self.metadata_notes.toPlainText(),
        }
        values.update(self._measurement_property_export_values(record))
        return values

    def _measurement_property_export_values(
        self, record: ProjectImageRecord
    ) -> dict[str, str]:
        values: dict[str, str] = {}

        for angle in record.measurements.angles:
            prefix = f"angle:{angle.id}"
            values.update(
                {
                    f"{prefix}:name": angle.name,
                    f"{prefix}:id": angle.id,
                    f"{prefix}:value": _angle_measurement_value_text(angle),
                    f"{prefix}:first": _point_text(angle.first),
                    f"{prefix}:vertex": _point_text(angle.vertex),
                    f"{prefix}:second": _point_text(angle.second),
                    f"{prefix}:note": angle.note,
                }
            )
            values.update(
                self._oiv_export_values(
                    prefix, angle.assessment, self._angle_oiv_result(angle)
                )
            )

        for segment in record.measurements.segments:
            prefix = f"segment:{segment.id}"
            values.update(
                {
                    f"{prefix}:name": segment.name,
                    f"{prefix}:id": segment.id,
                    f"{prefix}:length": _segment_measurement_length_text(
                        segment, record.calibration
                    ),
                    f"{prefix}:start": _point_text(segment.start),
                    f"{prefix}:end": _point_text(segment.end),
                    f"{prefix}:start_label": segment.start_label,
                    f"{prefix}:end_label": segment.end_label,
                    f"{prefix}:note": segment.note,
                }
            )
            values.update(
                self._oiv_export_values(
                    prefix,
                    segment.assessment,
                    self._segment_oiv_result(segment, record),
                )
            )
        return values

    def _oiv_export_values(
        self,
        prefix: str,
        assessment: ProjectMeasurementAssessment | None,
        result: OivClassificationResult | None,
    ) -> dict[str, str]:
        code = _measurement_oiv_code(assessment)
        if not code:
            return {}

        trait = result.trait if result is not None else self._oiv_trait_by_code(code)
        value_text = ""
        score_text = ""
        scale_label = ""
        if result is not None:
            value_text = _oiv_value_text(result.value, result.unit)
            if result.ok and result.score is not None:
                score_text = str(result.score)
                scale_label = result.label
            elif result.message and result.message != "ok":
                scale_label = result.message

        return {
            f"{prefix}:oiv_code": code,
            f"{prefix}:oiv_short_name": trait.short_name if trait is not None else "",
            f"{prefix}:oiv_value": value_text,
            f"{prefix}:oiv_score": score_text,
            f"{prefix}:oiv_label": scale_label,
        }

    def _project_export_row(self, record: ProjectImageRecord) -> dict[str, str]:
        self._normalize_record_metadata(record)
        row = {
            "id": record.id,
            "file_name": record.display_name,
            "relative_path": record.relative_path,
            "has_annotation": "1" if record.annotation is not None else "0",
            "status": "ok",
            "diagnosis": str(record.metadata.get("diagnosis", "")),
            "r": "",
            "g": "",
            "b": "",
            "contour_pixel_count": "",
        }

        image_path = self._project_image_path(record)
        if not image_path.exists():
            row["status"] = "файл не найден"
            return row
        if record.annotation is None:
            row["status"] = "нет контура"
            return row

        try:
            # Для экспорта нужны только пиксели: QImage/QPixmap не создаются.
            rgb_pixels = _annotation_rgb_pixels(
                load_rgb_array(image_path), record.annotation
            )
        except (OSError, ValueError) as exc:
            row["status"] = f"ошибка: {exc}"
            return row

        if rgb_pixels.size == 0:
            row["status"] = "внутри контура нет пикселов"
            return row

        mean_values = np.rint(rgb_pixels.mean(axis=0)).astype(int)
        row["r"] = str(int(mean_values[0]))
        row["g"] = str(int(mean_values[1]))
        row["b"] = str(int(mean_values[2]))
        row["contour_pixel_count"] = str(int(rgb_pixels.shape[0]))
        return row

    def create_new_contour(self) -> None:
        if self._selected_project_image() is None or not self.canvas.has_image():
            self._show_warning(
                "Нет изображения", "Сначала выберите изображение проекта."
            )
            return
        self.canvas.cancel_angle_measurement(show_message=False)
        self.canvas.cancel_segment_measurement(show_message=False)

        node_count, accepted = QInputDialog.getInt(
            self,
            "Новый контур",
            "Количество узлов контура:",
            5,
            3,
            128,
            1,
        )
        if not accepted:
            return

        record = self._selected_project_image()
        has_saved_project_contour = record is not None and (
            record.annotation is not None or record.raw_annotation is not None
        )
        if self.canvas.has_contour() or has_saved_project_contour:
            answer = QMessageBox.question(
                self,
                "Заменить контур?",
                "Текущий контур будет заменён новым окружным контуром. Продолжить?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        image_size = self.canvas.image_size()
        if image_size is None:
            self._show_warning(
                "Нет изображения", "Сначала выберите изображение проекта."
            )
            return

        width, height = image_size
        points = _circle_contour_points(width, height, node_count)
        try:
            self.canvas.set_contour(points)
        except ValueError as exc:
            self._show_error("Не удалось создать контур", str(exc))
            return

        self._save_current_project_annotation()
        self._clear_histograms(HISTOGRAM_MANUAL_REFRESH_TEXT)
        self._update_project_properties()
        self._update_action_states()
        self.statusBar().showMessage(f"Создан новый контур: {len(points)} узлов.")

    def detect_contour(self) -> None:
        if self._selected_project_image() is None or not self.canvas.has_image():
            self._show_warning(
                "Нет изображения", "Сначала выберите изображение проекта."
            )
            return
        if self._pending_contour_detection is not None:
            return
        self.canvas.cancel_angle_measurement(show_message=False)
        self.canvas.cancel_segment_measurement(show_message=False)

        rgb_array = self.canvas.current_rgb_array_view()
        height, width = rgb_array.shape[:2]
        if width * height > CONTOUR_ANALYSIS_SYNC_PIXEL_LIMIT:
            self._start_contour_detection(rgb_array)
            return

        # Маленький кадр посчитать сразу быстрее, чем передавать его в поток.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            points = detect_leaf_contour(rgb_array)
            self.canvas.set_contour(points)
        except ValueError as exc:
            self._show_error("Не удалось определить контур", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._apply_detected_contour_status(points)

    def _start_contour_detection(self, rgb_array: np.ndarray) -> None:
        self._contour_detection_request_id += 1
        request_id = self._contour_detection_request_id
        # Запоминаем изображение: пока идёт расчёт, пользователь может выбрать
        # другое фото, и тогда результат применять нельзя.
        self._pending_contour_detection = PendingContourDetection(
            request_id=request_id,
            record_id=self._current_project_image_id,
            image_path=self.canvas.current_image_path(),
        )
        worker = _ContourDetectionWorker(request_id, rgb_array)
        worker.signals.finished.connect(self._handle_contour_detection_finished)
        worker.signals.failed.connect(self._handle_contour_detection_failed)
        self._contour_detection_workers[request_id] = worker
        self._begin_busy("Поиск контура листа…")
        self._update_action_states()
        self._contour_detection_thread_pool.start(worker)

    def _finish_contour_detection(self, request_id: int) -> bool:
        """Снимает ожидание детекции; True, если результат ещё относится к текущему кадру."""
        self._contour_detection_workers.pop(request_id, None)
        pending = self._pending_contour_detection
        if pending is None or pending.request_id != request_id:
            return False
        self._pending_contour_detection = None
        self._end_busy()
        self._update_action_states()
        return (
            pending.record_id == self._current_project_image_id
            and pending.image_path == self.canvas.current_image_path()
            and self.canvas.has_image()
        )

    def _handle_contour_detection_finished(
        self, request_id: int, points: list[Point]
    ) -> None:
        if not self._finish_contour_detection(request_id):
            return
        try:
            self.canvas.set_contour(points)
        except ValueError as exc:
            self._show_error("Не удалось определить контур", str(exc))
            return
        self._apply_detected_contour_status(points)

    def _handle_contour_detection_failed(self, request_id: int, message: str) -> None:
        if not self._finish_contour_detection(request_id):
            return
        self._show_error("Не удалось определить контур", message)

    def _cancel_contour_detection(self) -> None:
        # grabCut нельзя прервать; отмена означает, что его результат будет проигнорирован.
        if self._pending_contour_detection is None:
            return
        self._contour_detection_request_id += 1
        self._pending_contour_detection = None
        self._end_busy()

    def _apply_detected_contour_status(self, points: list[Point]) -> None:
        self._clear_histograms(HISTOGRAM_MANUAL_REFRESH_TEXT)
        self.statusBar().showMessage(f"Контур построен: {len(points)} узлов.")

    def delete_current_contour(self) -> None:
        record = self._selected_project_image()
        has_project_contour = record is not None and (
            record.annotation is not None or record.raw_annotation is not None
        )
        if not self.canvas.has_contour() and not has_project_contour:
            self._show_warning("Нет контура", "Для текущего изображения нет контура.")
            return

        if record is None:
            self._show_warning(
                "Изображение не выбрано", "Выберите изображение в списке проекта."
            )
            return

        record.annotation = None
        record.raw_annotation = None
        record.annotation_error = None

        if self._current_project_image_id == record.id and self.canvas.has_contour():
            self.canvas.clear_contour()
        else:
            self._update_current_project_list_item(record)
            self._update_project_summary_properties()
            self._update_project_properties()
            self._update_action_states()
            self._schedule_project_save()

        self._clear_histograms("Создайте основной контур, чтобы увидеть гистограмму.")
        self._save_project_silently(show_error=True)
        self.statusBar().showMessage(f"Контур удалён: {record.display_name}")

    def flatten_background(self) -> None:
        record = self._selected_project_image()
        if record is None or not self.canvas.has_image():
            self._show_warning(
                "Нет изображения", "Сначала выберите изображение проекта."
            )
            return
        if not self.canvas.has_contour():
            self._show_warning("Нет контура", "Сначала постройте или загрузите контур.")
            return

        image_path = self._project_image_path(record)
        if not image_path.exists():
            self._show_error(
                "Не удалось сохранить изображение",
                f"Файл изображения не найден: {image_path}",
            )
            return

        answer = QMessageBox.question(
            self,
            "Выровнять фон?",
            (
                "Исходное изображение будет изменено и сохранено на диск.\n\n"
                f"{image_path}\n\n"
                "Продолжить?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        rgb_array = self.canvas.current_rgb_array_view()
        points = self.canvas.contour_points()
        height, width = rgb_array.shape[:2]
        if width * height > CONTOUR_ANALYSIS_SYNC_PIXEL_LIMIT:
            self._start_flatten_background(record, image_path, rgb_array, points)
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            flattened = flatten_background_file(
                FlattenBackgroundJob(
                    request_id=self._flatten_background_request_id,
                    record_id=record.id,
                    image_path=image_path,
                    rgb_array=rgb_array,
                    points=points,
                )
            )
        except (OSError, ValueError) as exc:
            # Канвас ещё не менялся: файл и экран остаются согласованными, откат не нужен.
            self._show_error("Не удалось выровнять фон", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._apply_flattened_background(flattened)

    def _start_flatten_background(
        self,
        record: ProjectImageRecord,
        image_path: Path,
        rgb_array: np.ndarray,
        points: list[Point],
    ) -> None:
        self._flatten_background_request_id += 1
        job = FlattenBackgroundJob(
            request_id=self._flatten_background_request_id,
            record_id=record.id,
            image_path=image_path,
            rgb_array=rgb_array,
            points=points,
        )
        self._pending_flatten_background = job
        worker = _FlattenBackgroundWorker(job)
        worker.signals.finished.connect(self._handle_flatten_background_finished)
        worker.signals.failed.connect(self._handle_flatten_background_failed)
        self._flatten_background_workers[job.request_id] = worker
        self._begin_busy("Выравнивание фона и сохранение изображения…")
        self._update_action_states()
        self._flatten_background_thread_pool.start(worker)

    def _finish_flatten_background(
        self, request_id: int
    ) -> FlattenBackgroundJob | None:
        self._flatten_background_workers.pop(request_id, None)
        job = self._pending_flatten_background
        if job is None or job.request_id != request_id:
            return None
        self._pending_flatten_background = None
        self._end_busy()
        self._update_action_states()
        return job

    def _handle_flatten_background_finished(
        self, request_id: int, flattened: np.ndarray
    ) -> None:
        job = self._finish_flatten_background(request_id)
        if job is None:
            return
        if (
            job.record_id != self._current_project_image_id
            or self.canvas.current_image_path() != job.image_path.resolve()
        ):
            # Файл уже сохранён; другое фото на экране не трогаем. При возврате
            # к этому изображению оно загрузится с диска уже с белым фоном.
            self._project_mean_color_stats_cache = None
            self.statusBar().showMessage(
                f"Фон выровнен и сохранён: {job.image_path.name}"
            )
            return
        self._apply_flattened_background(flattened)

    def _handle_flatten_background_failed(self, request_id: int, message: str) -> None:
        if self._finish_flatten_background(request_id) is None:
            return
        self._show_error("Не удалось выровнять фон", message)

    def _apply_flattened_background(self, flattened: np.ndarray) -> None:
        # Сохранённые пиксели уже есть в памяти: повторно декодировать только что
        # записанный файл (как раньше через load_raster_image) не нужно.
        self._invalidate_current_contour_analysis()
        try:
            self.canvas.replace_current_rgb_array(flattened)
        except ValueError as exc:
            self._show_error("Не удалось выровнять фон", str(exc))
            return

        if self._should_defer_project_summary_refresh():
            self._schedule_project_summary_refresh()
        else:
            self._update_project_summary_properties()
        self._update_project_properties()
        self.statusBar().showMessage(
            "Фон за пределами контура выровнен до белого, изображение сохранено."
        )

    def start_scale_calibration(self) -> None:
        if self._selected_project_image() is None or not self.canvas.has_image():
            self._show_warning(
                "Нет изображения", "Сначала выберите изображение проекта."
            )
            return
        self.canvas.begin_calibration()

    def reset_current_calibration(self) -> None:
        record = self._selected_project_image()
        if record is None:
            self._show_warning(
                "Изображение не выбрано", "Выберите изображение в списке проекта."
            )
            return

        if (
            record.calibration is None
            and record.raw_calibration is None
            and record.calibration_error is None
        ):
            return

        record.calibration = None
        record.raw_calibration = None
        record.calibration_error = None
        if self._current_project_image_id == record.id and self.canvas.has_image():
            self.canvas.clear_calibration()
        self._update_project_properties()
        self._update_action_states()
        self._schedule_project_save()
        self.statusBar().showMessage(f"Калибровка сброшена: {record.display_name}")

    def start_angle_measurement(self) -> None:
        if self._selected_project_image() is None or not self.canvas.has_image():
            self._show_warning(
                "Нет изображения", "Сначала выберите изображение проекта."
            )
            return
        self.canvas.begin_angle_measurement()
        self._update_action_states()

    def delete_selected_angle(self) -> None:
        if self.canvas.delete_selected_angle():
            self._store_current_angle_measurements()
        self._update_action_states()

    def start_segment_measurement(self) -> None:
        if self._selected_project_image() is None or not self.canvas.has_image():
            self._show_warning(
                "Нет изображения", "Сначала выберите изображение проекта."
            )
            return
        self.canvas.begin_segment_measurement()
        self._update_action_states()

    def delete_selected_segment(self) -> None:
        if self.canvas.delete_selected_segment():
            self._store_current_segment_measurements()
        self._update_action_states()

    def _handle_angle_state_changed(self) -> None:
        if not self._loading_project_image:
            self._store_current_angle_measurements()
        self._update_action_states()

    def _handle_segment_state_changed(self) -> None:
        if not self._loading_project_image:
            self._store_current_segment_measurements()
        self._update_action_states()

    def _store_current_angle_measurements(self) -> None:
        record = self._current_project_image()
        if record is None or not self.canvas.has_image():
            return

        existing_names = {angle.id: angle.name for angle in record.measurements.angles}
        existing_notes = {angle.id: angle.note for angle in record.measurements.angles}
        existing_assessments = {
            angle.id: angle.assessment for angle in record.measurements.angles
        }
        existing_line_colors = {
            angle.id: angle.line_color for angle in record.measurements.angles
        }
        existing_label_colors = {
            angle.id: angle.label_color for angle in record.measurements.angles
        }
        new_angles = [
            ProjectAngleMeasurement(
                id=angle_id,
                first=first,
                vertex=vertex,
                second=second,
                name=existing_names.get(angle_id, canvas_name),
                line_color=(
                    self.canvas.angle_measurement_line_color(angle_id)
                    or existing_line_colors.get(angle_id, ANGLE_LINE_COLOR)
                ),
                label_color=(
                    self.canvas.angle_measurement_label_color(angle_id)
                    or existing_label_colors.get(angle_id, ANGLE_LABEL_COLOR)
                ),
                note=existing_notes.get(angle_id, ""),
                assessment=existing_assessments.get(angle_id),
            )
            for angle_id, first, vertex, second, canvas_name in self.canvas.angle_measurement_records(
                include_names=True
            )
        ]
        if _angle_measurements_equal(record.measurements.angles, new_angles):
            return

        record.measurements.angles = new_angles
        self._update_project_properties()
        self._schedule_project_save()

    def _store_current_segment_measurements(self) -> None:
        record = self._current_project_image()
        if record is None or not self.canvas.has_image():
            return

        existing_start_labels = {
            segment.id: segment.start_label for segment in record.measurements.segments
        }
        existing_end_labels = {
            segment.id: segment.end_label for segment in record.measurements.segments
        }
        existing_names = {
            segment.id: segment.name for segment in record.measurements.segments
        }
        existing_notes = {
            segment.id: segment.note for segment in record.measurements.segments
        }
        existing_assessments = {
            segment.id: segment.assessment for segment in record.measurements.segments
        }
        existing_line_colors = {
            segment.id: segment.line_color for segment in record.measurements.segments
        }
        existing_label_colors = {
            segment.id: segment.label_color for segment in record.measurements.segments
        }
        new_segments = [
            ProjectSegmentMeasurement(
                id=segment_id,
                start=start,
                end=end,
                name=existing_names.get(segment_id, canvas_name),
                line_color=(
                    self.canvas.segment_measurement_line_color(segment_id)
                    or existing_line_colors.get(segment_id, SEGMENT_LINE_COLOR)
                ),
                label_color=(
                    self.canvas.segment_measurement_label_color(segment_id)
                    or existing_label_colors.get(segment_id, SEGMENT_LABEL_COLOR)
                ),
                start_label=existing_start_labels.get(segment_id, canvas_start_label),
                end_label=existing_end_labels.get(segment_id, canvas_end_label),
                note=existing_notes.get(segment_id, ""),
                assessment=existing_assessments.get(segment_id),
            )
            for (
                segment_id,
                start,
                end,
                canvas_start_label,
                canvas_end_label,
                canvas_name,
            ) in self.canvas.segment_measurement_records(include_names=True)
        ]
        if _segment_measurements_equal(record.measurements.segments, new_segments):
            return

        record.measurements.segments = new_segments
        self._update_project_properties()
        self._schedule_project_save()

    def _handle_angle_name_edit_finished(self, angle_id: str, field: QLineEdit) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        angle = _angle_by_id(record, angle_id)
        if angle is None:
            return
        raw_name = field.text()
        name = raw_name.strip()
        if raw_name != name:
            with QSignalBlocker(field):
                field.setText(name)
        if angle.name == name:
            return
        angle.name = name
        if record.id == self._current_project_image_id:
            self.canvas.set_angle_measurement_name(angle_id, name)
        self._update_project_properties()
        self._schedule_project_save()

    def _handle_angle_note_edit_finished(self, angle_id: str, field: QLineEdit) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        angle = _angle_by_id(record, angle_id)
        if angle is None:
            return
        note = field.text()
        if angle.note == note:
            return
        angle.note = note
        self._schedule_project_save()

    def _handle_angle_assessment_changed(self, angle_id: str, field: QComboBox) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        angle = _angle_by_id(record, angle_id)
        if angle is None:
            return

        new_assessment = _assessment_from_combo(field)
        if _measurement_assessments_equal(angle.assessment, new_assessment):
            return

        angle.assessment = new_assessment
        self._update_angle_assessment_result_field(angle)
        self._schedule_project_save()

    def _handle_angle_visibility_changed(self, angle_id: str, checked: bool) -> None:
        field = self._angle_visibility_fields.get(angle_id)
        if field is None:
            return
        changed = self.canvas.set_angle_measurement_visible(angle_id, checked)
        actual_visible = self.canvas.is_angle_measurement_visible(angle_id)
        if not changed and actual_visible != checked:
            with QSignalBlocker(field):
                field.setChecked(actual_visible)
        self._update_action_states()

    def _open_angle_line_color_dialog(self, angle_id: str) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        angle = _angle_by_id(record, angle_id)
        field = self._angle_line_color_fields.get(angle_id)
        if angle is None or field is None:
            return

        line_color = self._choose_line_color(angle.line_color, "Цвет линии угла")
        if line_color is None or line_color == angle.line_color:
            return

        angle.line_color = line_color
        if record.id == self._current_project_image_id:
            self.canvas.set_angle_measurement_line_color(angle_id, line_color)
        self._set_line_color_button_value(field, line_color)
        self._schedule_project_save()

    def _open_angle_label_color_dialog(self, angle_id: str) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        angle = _angle_by_id(record, angle_id)
        field = self._angle_label_color_fields.get(angle_id)
        if angle is None or field is None:
            return

        label_color = self._choose_line_color(angle.label_color, "Цвет подписей угла")
        if label_color is None or label_color == angle.label_color:
            return

        angle.label_color = label_color
        if record.id == self._current_project_image_id:
            self.canvas.set_angle_measurement_label_color(angle_id, label_color)
        self._set_line_color_button_value(field, label_color)
        self._schedule_project_save()

    def _handle_segment_name_edit_finished(
        self, segment_id: str, field: QLineEdit
    ) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        segment = _segment_by_id(record, segment_id)
        if segment is None:
            return
        raw_name = field.text()
        name = raw_name.strip()
        if raw_name != name:
            with QSignalBlocker(field):
                field.setText(name)
        if segment.name == name:
            return
        segment.name = name
        if record.id == self._current_project_image_id:
            self.canvas.set_segment_measurement_name(segment_id, name)
        self._update_project_properties()
        self._schedule_project_save()

    def _handle_segment_note_edit_finished(
        self, segment_id: str, field: QLineEdit
    ) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        segment = _segment_by_id(record, segment_id)
        if segment is None:
            return
        note = field.text()
        if segment.note == note:
            return
        segment.note = note
        self._schedule_project_save()

    def _handle_segment_assessment_changed(
        self, segment_id: str, field: QComboBox
    ) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        segment = _segment_by_id(record, segment_id)
        if segment is None:
            return

        new_assessment = _assessment_from_combo(field)
        if _measurement_assessments_equal(segment.assessment, new_assessment):
            return

        segment.assessment = new_assessment
        self._update_segment_assessment_result_field(segment, record)
        self._schedule_project_save()

    def _update_angle_assessment_result_field(
        self, angle: ProjectAngleMeasurement
    ) -> None:
        field = self._angle_assessment_result_fields.get(angle.id)
        if field is None:
            return
        field.setText(_oiv_result_text(self._angle_oiv_result(angle)))

    def _update_segment_assessment_result_field(
        self,
        segment: ProjectSegmentMeasurement,
        record: ProjectImageRecord,
    ) -> None:
        field = self._segment_assessment_result_fields.get(segment.id)
        if field is None:
            return
        field.setText(_oiv_result_text(self._segment_oiv_result(segment, record)))

    def _handle_segment_visibility_changed(
        self, segment_id: str, checked: bool
    ) -> None:
        field = self._segment_visibility_fields.get(segment_id)
        if field is None:
            return
        changed = self.canvas.set_segment_measurement_visible(segment_id, checked)
        actual_visible = self.canvas.is_segment_measurement_visible(segment_id)
        if not changed and actual_visible != checked:
            with QSignalBlocker(field):
                field.setChecked(actual_visible)
        self._update_action_states()

    def _open_segment_line_color_dialog(self, segment_id: str) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        segment = _segment_by_id(record, segment_id)
        field = self._segment_line_color_fields.get(segment_id)
        if segment is None or field is None:
            return

        line_color = self._choose_line_color(segment.line_color, "Цвет линии отрезка")
        if line_color is None or line_color == segment.line_color:
            return

        segment.line_color = line_color
        if record.id == self._current_project_image_id:
            self.canvas.set_segment_measurement_line_color(segment_id, line_color)
        self._set_line_color_button_value(field, line_color)
        self._schedule_project_save()

    def _open_segment_label_color_dialog(self, segment_id: str) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        segment = _segment_by_id(record, segment_id)
        field = self._segment_label_color_fields.get(segment_id)
        if segment is None or field is None:
            return

        label_color = self._choose_line_color(
            segment.label_color, "Цвет подписей отрезка"
        )
        if label_color is None or label_color == segment.label_color:
            return

        segment.label_color = label_color
        if record.id == self._current_project_image_id:
            self.canvas.set_segment_measurement_label_color(segment_id, label_color)
        self._set_line_color_button_value(field, label_color)
        self._schedule_project_save()

    def _handle_segment_label_edit_finished(
        self,
        segment_id: str,
        field: QLineEdit,
        endpoint: str,
    ) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        segment = _segment_by_id(record, segment_id)
        if segment is None:
            return
        raw_label = field.text()
        label = raw_label.strip()
        if raw_label != label:
            with QSignalBlocker(field):
                field.setText(label)

        if endpoint == "start":
            if segment.start_label == label:
                return
            segment.start_label = label
        else:
            if segment.end_label == label:
                return
            segment.end_label = label

        if record.id == self._current_project_image_id:
            self.canvas.set_segment_labels(
                segment_id, segment.start_label, segment.end_label
            )
        self._schedule_project_save()

    def _delete_angle_by_id(self, angle_id: str) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        original_count = len(record.measurements.angles)
        record.measurements.angles = [
            angle for angle in record.measurements.angles if angle.id != angle_id
        ]
        if len(record.measurements.angles) == original_count:
            return
        if record.id == self._current_project_image_id:
            self.canvas.set_angle_measurement_records(_angle_canvas_records(record))
        self._update_project_properties()
        self._update_action_states()
        self._schedule_project_save()
        self.statusBar().showMessage("Угол удалён.")

    def _delete_segment_by_id(self, segment_id: str) -> None:
        record = self._selected_project_image()
        if record is None:
            return
        original_count = len(record.measurements.segments)
        record.measurements.segments = [
            segment
            for segment in record.measurements.segments
            if segment.id != segment_id
        ]
        if len(record.measurements.segments) == original_count:
            return
        if record.id == self._current_project_image_id:
            self.canvas.set_segment_measurement_records(_segment_canvas_records(record))
        self._update_project_properties()
        self._update_action_states()
        self._schedule_project_save()
        self.statusBar().showMessage("Отрезок удалён.")

    def _handle_calibration_segment_selected(self, raw_points: object) -> None:
        record = self._selected_project_image()
        if record is None or not self.canvas.has_image():
            return

        points = _calibration_points_from_signal(raw_points)
        if points is None:
            return

        try:
            ImageCalibration(points[0], points[1], 1.0)
        except ValueError as exc:
            self._show_warning("Некорректная калибровка", str(exc))
            return

        length_mm, accepted = QInputDialog.getDouble(
            self,
            "Калибровка масштаба",
            "Длина отрезка, мм:",
            10.0,
            0.0001,
            1_000_000.0,
            4,
        )
        if not accepted:
            return

        try:
            calibration = ImageCalibration(points[0], points[1], length_mm)
        except ValueError as exc:
            self._show_warning("Некорректная калибровка", str(exc))
            return

        record.calibration = calibration
        record.calibration_error = None
        record.raw_calibration = None
        self.canvas.set_calibration(
            calibration.start,
            calibration.end,
            _calibration_length_text(calibration.length_mm),
        )
        self._update_project_properties()
        self._update_action_states()
        self._schedule_project_save()
        self.statusBar().showMessage(
            f"Калибровка задана: {_calibration_scale_text(calibration)}"
        )

    def _handle_calibration_geometry_changed(self) -> None:
        if self._loading_project_image:
            return

        record = self._current_project_image()
        if (
            record is None
            or record.calibration is None
            or not self.canvas.has_calibration()
        ):
            return

        points = self.canvas.calibration_points()
        if len(points) != 2:
            return

        try:
            calibration = ImageCalibration(
                points[0], points[1], record.calibration.length_mm
            )
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))
            return

        if (
            calibration.start == record.calibration.start
            and calibration.end == record.calibration.end
        ):
            return

        record.calibration = calibration
        record.calibration_error = None
        record.raw_calibration = None
        self._update_project_properties()
        self._update_action_states()
        self._schedule_project_save()

    def save_annotation_file(self) -> None:
        if not self.canvas.has_contour():
            self._show_warning("Нет контура", "Сначала постройте или загрузите контур.")
            return
        self._save_current_project_annotation()

        initial_name = self._default_annotation_file_name()
        if self._current_annotation_path is not None:
            initial_name = str(self._current_annotation_path)

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить аннотацию",
            initial_name,
            "JSON (*.json);;All files (*)",
        )
        if not file_name:
            return

        annotation_path = _ensure_json_suffix(Path(file_name))
        annotation = self._build_annotation(annotation_path)

        try:
            save_annotation(annotation_path, annotation)
        except OSError as exc:
            self._show_error(
                "Ошибка сохранения", f"Не удалось сохранить аннотацию: {exc}"
            )
            return

        self._current_annotation_path = annotation_path
        self.statusBar().showMessage(f"Аннотация сохранена: {annotation_path.name}")

    def open_annotation_file(self) -> None:
        if self.project_document is None or self._selected_project_image() is None:
            self._show_warning(
                "Нет выбранного изображения", "Сначала выберите изображение проекта."
            )
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть аннотацию",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not file_name:
            return

        annotation_path = Path(file_name)
        try:
            annotation = load_annotation(annotation_path)
        except (OSError, ValueError) as exc:
            self._show_error("Ошибка загрузки", str(exc))
            return

        if not self._prepare_image_for_annotation(annotation, annotation_path):
            return

        if self.canvas.image_size() != (
            annotation.image_width,
            annotation.image_height,
        ):
            self._show_error(
                "Несовпадение размеров",
                "Размеры изображения не совпадают с данными аннотации.",
            )
            return

        try:
            self.canvas.set_contour_line_color(annotation.line_color)
            self.canvas.set_contour(annotation.points)
        except ValueError as exc:
            self._show_error("Ошибка загрузки контура", str(exc))
            return

        self._current_annotation_path = annotation_path
        self._clear_histograms(HISTOGRAM_MANUAL_REFRESH_TEXT)
        self.statusBar().showMessage(f"Аннотация открыта: {annotation_path.name}")

    def _set_project(self, project_path: Path, document: ProjectDocument) -> None:
        self._cancel_background_image_tasks()
        self.project_path = project_path.resolve()
        self.project_document = document
        self._project_mean_color_stats_cache = None
        self._project_color_sums_cache = {}
        project_name_was_synced = self._sync_project_document_name_with_path()
        self._current_project_image_id = None
        self._current_annotation_path = None
        self._refresh_project_list()
        if document.images:
            self._select_project_image(document.images[0].id)
        else:
            self._clear_current_image_display()
        self._update_project_panel()
        self._update_window_title()
        if project_name_was_synced:
            self._schedule_project_save()

    def _sync_project_document_name_with_path(self) -> bool:
        if self.project_document is None or self.project_path is None:
            return False

        project_name = self.project_path.stem
        if not project_name or self.project_document.name == project_name:
            return False

        self.project_document.name = project_name
        return True

    def _clear_project_state(self) -> None:
        self._project_autosave_timer.stop()
        self._cancel_background_image_tasks()
        self._project_mean_color_stats_cache = None
        self._project_color_sums_cache = {}
        self.project_document = None
        self.project_path = None
        self._current_project_image_id = None
        self._current_annotation_path = None
        self._refresh_project_list()
        self._clear_current_image_display()
        self._update_project_panel()
        self._update_window_title()

    def _clear_current_image_display(self) -> None:
        self._cancel_contour_detection()
        self._loading_project_image = True
        try:
            self.canvas.clear_image()
        finally:
            self._loading_project_image = False
        self._clear_histograms(
            "Откройте изображение и создайте контур, чтобы увидеть гистограмму."
        )
        self._update_project_properties()
        self._update_action_states()

    def _update_project_panel(self) -> None:
        has_project = self.project_document is not None
        self.project_list.setEnabled(has_project)
        self.export_image_properties_excel_button.setEnabled(
            has_project and self._selected_project_image() is not None
        )
        self._update_project_summary_properties()
        self._update_project_properties()
        self._update_action_states()

    def _refresh_project_list(self) -> None:
        self._updating_project_list = True
        try:
            selected_id = self._current_project_image_id
            self.project_list.clear()
            if self.project_document is None:
                return
            for record in self.project_document.images:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, record.id)
                self._populate_project_list_item(item, record)
                self.project_list.addItem(item)
            if selected_id is not None:
                self._select_project_image(selected_id, emit_signal=False)
        finally:
            self._updating_project_list = False

    def _populate_project_list_item(
        self, item: QListWidgetItem, record: ProjectImageRecord
    ) -> None:
        details: list[str] = []
        if record.annotation is not None:
            details.append("аннотация есть")
        elif record.annotation_error:
            details.append("ошибка аннотации")
        else:
            details.append("без аннотации")

        image_path = self._project_image_path(record)
        file_is_missing = self.project_path is not None and not image_path.exists()
        has_valid_annotation = record.annotation is not None
        if file_is_missing:
            details.append("файл не найден")

        if file_is_missing or not has_valid_annotation:
            item.setForeground(QColor("#b42318"))
        else:
            item.setForeground(QColor("#1f2937"))

        item.setText(f"{record.display_name}\n{'; '.join(details)}")
        item.setToolTip(
            f"{record.display_name}\n{record.relative_path}\n{'; '.join(details)}"
        )

    def _update_current_project_list_item(self, record: ProjectImageRecord) -> None:
        current_item = self.project_list.currentItem()
        if current_item is None or current_item.data(Qt.UserRole) != record.id:
            return
        self._populate_project_list_item(current_item, record)

    def _select_project_image(
        self, record_id: str, *, emit_signal: bool = True
    ) -> None:
        previous_state = self._updating_project_list
        if not emit_signal:
            self._updating_project_list = True
        try:
            for row in range(self.project_list.count()):
                item = self.project_list.item(row)
                if item.data(Qt.UserRole) == record_id:
                    self.project_list.setCurrentRow(row)
                    if emit_signal:
                        return
                    self._current_project_image_id = record_id
                    return
            self.project_list.clearSelection()
            self._current_project_image_id = None
        finally:
            self._updating_project_list = previous_state

    def _handle_project_selection_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if self._updating_project_list:
            return

        selected_id = current.data(Qt.UserRole) if current is not None else None
        if selected_id == self._current_project_image_id:
            return

        self._store_current_angle_measurements()
        self._store_current_segment_measurements()
        self._save_current_project_annotation()
        self._current_project_image_id = str(selected_id) if selected_id else None
        self._current_annotation_path = None

        record = self._selected_project_image()
        if record is None:
            self._clear_current_image_display()
            return

        self._load_project_image(record)

    def _load_project_image(self, record: ProjectImageRecord) -> None:
        image_path = self._project_image_path(record)
        self._invalidate_current_contour_analysis()
        # Контур, найденный для предыдущего фото, к новому не применяется.
        self._cancel_contour_detection()
        self._loading_project_image = True
        size_changed = False
        loaded_ok = False
        try:
            if not image_path.exists():
                self.canvas.clear_image()
                self._clear_histograms("Файл изображения из проекта не найден.")
                self.statusBar().showMessage(
                    f"Файл изображения не найден: {record.display_name}"
                )
            else:
                try:
                    loaded_image = load_raster_image(image_path)
                except (OSError, ValueError) as exc:
                    self.canvas.clear_image()
                    self._clear_histograms(str(exc))
                    self.statusBar().showMessage(
                        f"Не удалось открыть изображение: {record.display_name}"
                    )
                else:
                    loaded_ok = True
                    self.canvas.set_loaded_image(loaded_image)
                    if (record.image_width, record.image_height) != (
                        loaded_image.width,
                        loaded_image.height,
                    ):
                        record.image_width = loaded_image.width
                        record.image_height = loaded_image.height
                        size_changed = True

                    if record.annotation is not None:
                        if (
                            record.annotation.image_width,
                            record.annotation.image_height,
                        ) != (
                            loaded_image.width,
                            loaded_image.height,
                        ):
                            self.statusBar().showMessage(
                                "Аннотация не загружена: размеры изображения не совпадают."
                            )
                        else:
                            try:
                                self.canvas.set_contour_line_color(
                                    record.annotation.line_color
                                )
                                self.canvas.set_contour(record.annotation.points)
                            except ValueError as exc:
                                record.annotation_error = str(exc)
                                self.statusBar().showMessage(
                                    "Аннотация повреждена и не загружена."
                                )
                    elif record.annotation_error:
                        self.statusBar().showMessage(
                            "У выбранного изображения повреждена аннотация."
                        )

                    if record.calibration is not None:
                        try:
                            self.canvas.set_calibration(
                                record.calibration.start,
                                record.calibration.end,
                                _calibration_length_text(record.calibration.length_mm),
                            )
                        except ValueError as exc:
                            record.calibration = None
                            record.calibration_error = str(exc)
                            self.statusBar().showMessage(
                                "Калибровка повреждена и не загружена."
                            )
                    elif record.calibration_error:
                        self.statusBar().showMessage(
                            "У выбранного изображения повреждена калибровка."
                        )

                    self.canvas.set_angle_measurement_records(
                        _angle_canvas_records(record)
                    )
                    self.canvas.set_segment_measurement_records(
                        _segment_canvas_records(record)
                    )
        finally:
            self._loading_project_image = False

        if size_changed:
            self._schedule_project_save()
        if loaded_ok:
            self._clear_histograms(HISTOGRAM_MANUAL_REFRESH_TEXT)
        self._update_current_project_list_item(record)
        self._update_project_summary_properties()
        self._update_project_properties()
        self._update_action_states()
        self._update_window_title()

    def _handle_contour_geometry_changed(self) -> None:
        if self._loading_project_image:
            return
        self._update_analysis_status()
        self._save_current_project_annotation()
        if self._should_defer_project_summary_refresh():
            self._schedule_project_summary_refresh()
        else:
            self._update_project_summary_properties()
        self._update_project_properties()

    def _handle_contour_visibility_changed(self, checked: bool) -> None:
        if not self.canvas.has_contour():
            self._update_contour_visibility_field()
            return
        self.canvas.set_contour_visible(checked)
        self._update_contour_visibility_field()

    def _open_contour_line_color_dialog(self) -> None:
        record = self._selected_project_image()
        if record is None or record.annotation is None or not self.canvas.has_contour():
            return

        line_color = self._choose_line_color(
            record.annotation.line_color,
            "Цвет линии главного контура",
        )
        if line_color is None or line_color == record.annotation.line_color:
            return

        record.annotation.line_color = line_color
        self.canvas.set_contour_line_color(line_color)
        self._set_line_color_button_value(self.property_contour_line_color, line_color)
        self._schedule_project_save()

    def _update_contour_line_color_field(
        self,
        record: ProjectImageRecord | None,
    ) -> None:
        line_color = (
            record.annotation.line_color
            if record is not None and record.annotation is not None
            else CONTOUR_LINE_COLOR
        )
        self._set_line_color_button_value(self.property_contour_line_color, line_color)
        self.property_contour_line_color.setEnabled(
            record is not None
            and record.annotation is not None
            and record.id == self._current_project_image_id
            and self.canvas.has_contour()
        )

    def _update_contour_visibility_field(self) -> None:
        has_visible_contour = self.canvas.has_image() and self.canvas.has_contour()
        with QSignalBlocker(self.property_contour_visible):
            self.property_contour_visible.setEnabled(has_visible_contour)
            self.property_contour_visible.setChecked(
                self.canvas.is_contour_visible() if has_visible_contour else False
            )

    def _save_current_project_annotation(self) -> None:
        if self._loading_project_image:
            return
        if self.project_document is None:
            return

        record = self._current_project_image()
        if record is None or not self.canvas.has_image():
            return

        image_size = self.canvas.image_size()
        if image_size is None:
            return

        width, height = image_size
        record.image_width = width
        record.image_height = height
        if self.canvas.has_contour():
            record.annotation = Annotation(
                image_path=record.relative_path,
                image_width=width,
                image_height=height,
                points=self.canvas.contour_points(),
                line_color=self.canvas.contour_line_color(),
                closed=True,
            )
            record.annotation_error = None
            record.raw_annotation = None
        else:
            record.annotation = None
            record.raw_annotation = None

        self._update_current_project_list_item(record)
        self._schedule_project_save()

    def _schedule_project_save(self) -> None:
        if self.project_document is None or self.project_path is None:
            return
        self._project_autosave_timer.start(350)

    def _save_project_silently(self, *, show_error: bool = False) -> bool:
        if self.project_document is None or self.project_path is None:
            return True
        try:
            save_project(self.project_path, self.project_document)
        except OSError as exc:
            if show_error:
                self._show_error(
                    "Ошибка сохранения проекта", f"Не удалось сохранить проект: {exc}"
                )
            return False
        return True

    def _selected_project_image_id(self) -> str | None:
        item = self.project_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.UserRole)
        return str(value) if value else None

    def _selected_project_image(self) -> ProjectImageRecord | None:
        selected_id = self._selected_project_image_id()
        if selected_id is None:
            return None
        return self._project_image_by_id(selected_id)

    def _current_project_image(self) -> ProjectImageRecord | None:
        if self._current_project_image_id is None:
            return None
        return self._project_image_by_id(self._current_project_image_id)

    def _project_image_by_id(self, record_id: str) -> ProjectImageRecord | None:
        if self.project_document is None:
            return None
        for record in self.project_document.images:
            if record.id == record_id:
                return record
        return None

    def _project_image_path(self, record: ProjectImageRecord) -> Path:
        if self.project_path is None:
            return Path(record.relative_path)
        return (self.project_path.parent / record.relative_path).resolve()

    def _update_project_summary_properties(self) -> None:
        has_project = self.project_document is not None
        self.project_properties_empty_label.setVisible(not has_project)
        self.project_properties_browser.setVisible(has_project)
        self.project_name.setEnabled(has_project)
        self.project_path_field.setEnabled(has_project)
        self.project_general_info.setEnabled(has_project)
        self._load_project_identity_fields()

        if self.project_document is None:
            self.project_properties_empty_label.setText("Создайте или откройте проект.")
            self.project_image_count.setText("-")
            self.project_mean_red.setText("-")
            self.project_mean_green.setText("-")
            self.project_mean_blue.setText("-")
            self.project_mean_lab_l.setText("-")
            self.project_mean_lab_a.setText("-")
            self.project_mean_lab_b.setText("-")
            self.project_mean_hsv_h.setText("-")
            self.project_mean_hsv_s.setText("-")
            self.project_mean_hsv_v.setText("-")
            self.project_mean_yuv_y.setText("-")
            self.project_mean_yuv_u.setText("-")
            self.project_mean_yuv_v.setText("-")
            self.project_mean_lms_l.setText("-")
            self.project_mean_lms_m.setText("-")
            self.project_mean_lms_s.setText("-")
            self._updating_project_info_fields = True
            try:
                with QSignalBlocker(self.project_general_info):
                    self.project_general_info.setPlainText("")
            finally:
                self._updating_project_info_fields = False
            return

        self._updating_project_info_fields = True
        try:
            with QSignalBlocker(self.project_general_info):
                self.project_general_info.setPlainText(
                    self.project_document.project_info.general_info
                )
        finally:
            self._updating_project_info_fields = False

        self.project_image_count.setText(str(len(self.project_document.images)))
        color_stats = self._project_contours_mean_color_stats()
        if color_stats is None:
            placeholder = (
                CONTOUR_ANALYSIS_PENDING_TEXT
                if self._pending_project_color_stats is not None
                else "-"
            )
            red_text = green_text = blue_text = placeholder
            lab_l_text = lab_a_text = lab_b_text = placeholder
            hsv_h_text = hsv_s_text = hsv_v_text = placeholder
            yuv_y_text = yuv_u_text = yuv_v_text = placeholder
            lms_l_text = lms_m_text = lms_s_text = placeholder
        else:
            mean_rgb, mean_lab, mean_hsv, mean_yuv, mean_lms = color_stats
            red, green, blue = mean_rgb
            lab_l, lab_a, lab_b = mean_lab
            hsv_h, hsv_s, hsv_v = mean_hsv
            yuv_y, yuv_u, yuv_v = mean_yuv
            lms_l, lms_m, lms_s = mean_lms
            red_text = str(red)
            green_text = str(green)
            blue_text = str(blue)
            lab_l_text = str(lab_l)
            lab_a_text = str(lab_a)
            lab_b_text = str(lab_b)
            hsv_h_text = str(hsv_h)
            hsv_s_text = str(hsv_s)
            hsv_v_text = str(hsv_v)
            yuv_y_text = str(yuv_y)
            yuv_u_text = str(yuv_u)
            yuv_v_text = str(yuv_v)
            lms_l_text = str(lms_l)
            lms_m_text = str(lms_m)
            lms_s_text = str(lms_s)
        self.project_mean_red.setText(red_text)
        self.project_mean_green.setText(green_text)
        self.project_mean_blue.setText(blue_text)
        self.project_mean_lab_l.setText(lab_l_text)
        self.project_mean_lab_a.setText(lab_a_text)
        self.project_mean_lab_b.setText(lab_b_text)
        self.project_mean_hsv_h.setText(hsv_h_text)
        self.project_mean_hsv_s.setText(hsv_s_text)
        self.project_mean_hsv_v.setText(hsv_v_text)
        self.project_mean_yuv_y.setText(yuv_y_text)
        self.project_mean_yuv_u.setText(yuv_u_text)
        self.project_mean_yuv_v.setText(yuv_v_text)
        self.project_mean_lms_l.setText(lms_l_text)
        self.project_mean_lms_m.setText(lms_m_text)
        self.project_mean_lms_s.setText(lms_s_text)

    def _schedule_project_summary_refresh(self) -> None:
        self._project_summary_refresh_timer.start(220)

    def _should_defer_project_summary_refresh(self) -> bool:
        if self.project_document is None:
            return False

        total_pixels = 0
        unknown_annotated_images = 0
        for record in self.project_document.images:
            if record.annotation is None or record.annotation_error:
                continue
            if record.image_width and record.image_height:
                total_pixels += int(record.image_width) * int(record.image_height)
            else:
                unknown_annotated_images += 1

        return (
            total_pixels > CONTOUR_ANALYSIS_SYNC_PIXEL_LIMIT
            or unknown_annotated_images > 2
        )

    def _load_project_identity_fields(self) -> None:
        project_name_text = "-"
        project_path_text = "-"
        if self.project_document is not None and self.project_path is not None:
            project_name_text = _project_name_from_path(
                self.project_path, self.project_document.name
            )
            project_path_text = str(self.project_path)

        self._updating_project_identity_fields = True
        try:
            with (
                QSignalBlocker(self.project_name),
                QSignalBlocker(self.project_path_field),
            ):
                self.project_name.setText(project_name_text)
                self.project_path_field.setText(project_path_text)
                self.project_path_field.setToolTip(
                    "" if project_path_text == "-" else project_path_text
                )
        finally:
            self._updating_project_identity_fields = False

    def _project_contours_mean_color_stats(self) -> ProjectMeanColorStats | None:
        if self.project_document is None:
            return None
        if not self._analysis_refresh_allowed:
            return self._project_mean_color_stats_cache

        entries = self._project_color_stats_entries()
        missing = [
            entry
            for entry in entries
            if entry.cache_key not in self._project_color_sums_cache
        ]
        missing_pixels = sum(
            entry.expected_size[0] * entry.expected_size[1] for entry in missing
        )
        if missing_pixels > CONTOUR_ANALYSIS_SYNC_PIXEL_LIMIT:
            # Большой объём: считаем в фоне с прогрессом, а в полях проекта
            # пока показываем «расчёт...».
            self._start_project_color_stats(entries, missing)
            return None

        # Небольшой объём (или всё уже в кэше) быстрее досчитать сразу.
        self._cancel_project_color_stats()
        for entry in missing:
            self._project_color_sums_cache[entry.cache_key] = (
                compute_project_color_sums(entry)
            )
        return self._store_project_mean_color_stats(entries)

    def _project_color_stats_entries(self) -> list[ProjectColorStatsEntry]:
        if self.project_document is None:
            return []
        entries: list[ProjectColorStatsEntry] = []
        for record in self.project_document.images:
            if record.annotation is None or record.annotation_error:
                continue
            image_path = self._project_image_path(record)
            try:
                file_stat = image_path.stat()
            except OSError:
                continue
            points = list(record.annotation.points)
            entries.append(
                ProjectColorStatsEntry(
                    cache_key=(
                        str(image_path),
                        file_stat.st_mtime_ns,
                        file_stat.st_size,
                        contour_signature(points),
                    ),
                    image_path=image_path,
                    points=points,
                    expected_size=(
                        int(record.annotation.image_width),
                        int(record.annotation.image_height),
                    ),
                )
            )
        return entries

    def _store_project_mean_color_stats(
        self, entries: list[ProjectColorStatsEntry]
    ) -> ProjectMeanColorStats | None:
        # Кэш хранит только записи текущих изображений и контуров, чтобы
        # он не рос при каждом редактировании контура.
        cache = self._project_color_sums_cache
        self._project_color_sums_cache = {
            entry.cache_key: cache[entry.cache_key]
            for entry in entries
            if entry.cache_key in cache
        }
        combined = combine_color_sums(
            sums for sums in self._project_color_sums_cache.values() if sums is not None
        )
        if combined is None:
            self._project_mean_color_stats_cache = None
            return None
        stats = color_stats_from_sums(combined)
        self._project_mean_color_stats_cache = (
            stats.mean_rgb,
            stats.mean_lab,
            stats.mean_hsv,
            stats.mean_yuv,
            stats.mean_lms,
        )
        return self._project_mean_color_stats_cache

    def _start_project_color_stats(
        self,
        entries: list[ProjectColorStatsEntry],
        missing: list[ProjectColorStatsEntry],
    ) -> None:
        self._cancel_project_color_stats()
        self._project_color_stats_request_id += 1
        request_id = self._project_color_stats_request_id
        cancel_event = threading.Event()
        self._pending_project_color_stats = PendingProjectColorStats(
            request_id=request_id,
            entries=entries,
            cancel_event=cancel_event,
        )
        # В поток уходят только изображения без записи в кэше.
        worker = _ProjectColorStatsWorker(request_id, missing, cancel_event)
        worker.signals.progress.connect(self._handle_project_color_stats_progress)
        worker.signals.finished.connect(self._handle_project_color_stats_finished)
        worker.signals.failed.connect(self._handle_project_color_stats_failed)
        self._project_color_stats_workers[request_id] = worker
        self._begin_busy("Расчёт статистики проекта…", maximum=len(missing))
        self._project_color_stats_thread_pool.start(worker)

    def _handle_project_color_stats_progress(
        self, request_id: int, done: int, total: int
    ) -> None:
        pending = self._pending_project_color_stats
        if pending is None or pending.request_id != request_id:
            return
        self._set_busy_progress(done, total)
        self.statusBar().showMessage(f"Расчёт статистики проекта: {done} из {total}")

    def _handle_project_color_stats_finished(
        self,
        request_id: int,
        results: dict[ProjectColorSumsCacheKey, ContourColorSums | None],
    ) -> None:
        self._project_color_stats_workers.pop(request_id, None)
        pending = self._pending_project_color_stats
        if pending is None or pending.request_id != request_id:
            return
        self._pending_project_color_stats = None
        self._end_busy()
        self._project_color_sums_cache.update(results)
        # Статистика соответствует моменту нажатия «Обновить», как и гистограммы.
        self._store_project_mean_color_stats(pending.entries)
        self._update_project_summary_properties()
        self.statusBar().showMessage("Статистика проекта обновлена.")

    def _handle_project_color_stats_failed(self, request_id: int, message: str) -> None:
        self._project_color_stats_workers.pop(request_id, None)
        pending = self._pending_project_color_stats
        if pending is None or pending.request_id != request_id:
            return
        self._pending_project_color_stats = None
        self._end_busy()
        self._update_project_summary_properties()
        if message and message != IMAGE_PREPARE_CANCELLED_TEXT:
            self.statusBar().showMessage(
                f"Не удалось рассчитать статистику проекта: {message}"
            )

    def _cancel_project_color_stats(self) -> None:
        pending = self._pending_project_color_stats
        if pending is None:
            return
        pending.cancel_event.set()
        self._project_color_stats_request_id += 1
        self._pending_project_color_stats = None
        self._end_busy()

    def _begin_busy(self, message: str, maximum: int = 0) -> None:
        # maximum=0 включает «бегущую» полосу для задач без измеримого прогресса.
        self._busy_task_count += 1
        self.busy_progress.setRange(0, maximum)
        self.busy_progress.setValue(0)
        self.busy_progress.show()
        self.statusBar().showMessage(message)

    def _set_busy_progress(self, value: int, maximum: int) -> None:
        self.busy_progress.setRange(0, maximum)
        self.busy_progress.setValue(value)

    def _end_busy(self) -> None:
        self._busy_task_count = max(0, self._busy_task_count - 1)
        if self._busy_task_count == 0:
            self.busy_progress.hide()

    def _cancel_background_image_tasks(self) -> None:
        self._cancel_contour_detection()
        self._cancel_project_color_stats()
        if self._pending_flatten_background is not None:
            # Запись файла атомарная; результат для закрытого проекта просто не применяется.
            self._flatten_background_request_id += 1
            self._pending_flatten_background = None
            self._end_busy()

    def _update_project_properties(self, *_args) -> None:
        record = self._selected_project_image()
        has_record = record is not None
        self.properties_empty_label.setVisible(not has_record)
        self.properties_browser.setVisible(has_record)
        self.export_image_properties_excel_button.setEnabled(has_record)
        self.rename_file_as_id_button.setEnabled(has_record)
        self.generate_image_id_button.setEnabled(has_record)

        if record is None:
            self.properties_empty_label.setText(
                "Создайте или откройте проект, затем выберите изображение."
                if self.project_document is None
                else "Изображение не выбрано."
            )
            self._update_calibration_length_field(None)
            self._update_contour_visibility_field()
            self._update_contour_line_color_field(None)
            self._rebuild_angle_measurement_properties(None)
            self._rebuild_segment_measurement_properties(None)
            self._set_average_color_swatch(None)
            return

        self._normalize_record_metadata(record)
        image_path = self._project_image_path(record)
        file_exists = image_path.exists()
        size_text = "-"
        if record.image_width and record.image_height:
            size_text = f"{record.image_width} x {record.image_height}"

        annotation_text = "нет"
        point_count_text = "-"
        if record.annotation is not None:
            annotation_text = "есть"
            point_count_text = str(record.point_count())
        elif record.annotation_error:
            annotation_text = f"ошибка: {record.annotation_error}"

        self._update_calibration_length_field(record)
        if record.calibration_error:
            calibration_scale_text = f"ошибка: {record.calibration_error}"
        else:
            calibration_scale_text = _calibration_scale_text(record.calibration)

        contour_stats = self._current_contour_stats(record)
        analysis_is_current_record = (
            contour_stats is None
            and record.id == self._current_project_image_id
            and self.canvas.has_image()
            and self.canvas.has_contour()
        )
        contour_analysis_pending = (
            analysis_is_current_record
            and self._is_current_contour_analysis_pending(record)
        )
        contour_analysis_outdated = (
            analysis_is_current_record
            and self._is_current_contour_analysis_outdated(record)
        )
        if contour_stats is None:
            red_text = green_text = blue_text = "-"
            lab_l_text = lab_a_text = lab_b_text = "-"
            hsv_h_text = hsv_s_text = hsv_v_text = "-"
            yuv_y_text = yuv_u_text = yuv_v_text = "-"
            lms_l_text = lms_m_text = lms_s_text = "-"
            contour_pixels_text = "-"
            contour_area_mm2_text = "-"
            placeholder = ""
            if contour_analysis_pending:
                placeholder = CONTOUR_ANALYSIS_PENDING_TEXT
            elif contour_analysis_outdated:
                placeholder = CONTOUR_ANALYSIS_OUTDATED_TEXT
            if placeholder:
                red_text = green_text = blue_text = placeholder
                lab_l_text = lab_a_text = lab_b_text = placeholder
                hsv_h_text = hsv_s_text = hsv_v_text = placeholder
                yuv_y_text = yuv_u_text = yuv_v_text = placeholder
                lms_l_text = lms_m_text = lms_s_text = placeholder
                contour_pixels_text = placeholder
                contour_area_mm2_text = placeholder
            mean_rgb = None
        else:
            mean_rgb, mean_lab, mean_hsv, mean_yuv, mean_lms, contour_pixel_count = (
                contour_stats
            )
            red, green, blue = mean_rgb
            lab_l, lab_a, lab_b = mean_lab
            hsv_h, hsv_s, hsv_v = mean_hsv
            yuv_y, yuv_u, yuv_v = mean_yuv
            lms_l, lms_m, lms_s = mean_lms
            red_text = str(red)
            green_text = str(green)
            blue_text = str(blue)
            lab_l_text = str(lab_l)
            lab_a_text = str(lab_a)
            lab_b_text = str(lab_b)
            hsv_h_text = str(hsv_h)
            hsv_s_text = str(hsv_s)
            hsv_v_text = str(hsv_v)
            yuv_y_text = str(yuv_y)
            yuv_u_text = str(yuv_u)
            yuv_v_text = str(yuv_v)
            lms_l_text = str(lms_l)
            lms_m_text = str(lms_m)
            lms_s_text = str(lms_s)
            contour_pixels_text = str(contour_pixel_count)
            contour_area_mm2_text = _contour_area_mm2_text(
                contour_pixel_count,
                record.calibration,
                record.calibration_error,
            )

        self.property_file_name.setText(record.display_name)
        self.property_path.setText(record.relative_path)
        self.property_size.setText(size_text)
        self.property_annotation.setText(annotation_text)
        self._update_contour_visibility_field()
        self._update_contour_line_color_field(record)
        self.property_points.setText(point_count_text)
        self.property_contour_pixels.setText(contour_pixels_text)
        self.property_contour_area_mm2.setText(contour_area_mm2_text)
        self.property_calibration_scale.setText(calibration_scale_text)
        self.property_red.setText(red_text)
        self.property_green.setText(green_text)
        self.property_blue.setText(blue_text)
        self.property_lab_l.setText(lab_l_text)
        self.property_lab_a.setText(lab_a_text)
        self.property_lab_b.setText(lab_b_text)
        self.property_hsv_h.setText(hsv_h_text)
        self.property_hsv_s.setText(hsv_s_text)
        self.property_hsv_v.setText(hsv_v_text)
        self.property_yuv_y.setText(yuv_y_text)
        self.property_yuv_u.setText(yuv_u_text)
        self.property_yuv_v.setText(yuv_v_text)
        self.property_lms_l.setText(lms_l_text)
        self.property_lms_m.setText(lms_m_text)
        self.property_lms_s.setText(lms_s_text)
        self.property_status.setText("найден" if file_exists else "отсутствует")
        self._set_average_color_swatch(mean_rgb)
        self._load_metadata_fields(record)
        self._rebuild_angle_measurement_properties(record)
        self._rebuild_segment_measurement_properties(record)

    def _update_calibration_length_field(
        self, record: ProjectImageRecord | None
    ) -> None:
        enabled = False
        if record is None:
            text = "-"
            tooltip = "Выберите изображение проекта."
        elif record.calibration_error:
            text = "ошибка калибровки"
            tooltip = record.calibration_error
        elif record.calibration is None:
            text = "калибровка не произведена"
            tooltip = "Сначала задайте калибровочный отрезок."
        else:
            text = _calibration_length_text(record.calibration.length_mm)
            tooltip = "Реальная длина калибровочного отрезка в миллиметрах."
            enabled = True

        self._updating_calibration_length_field = True
        try:
            with QSignalBlocker(self.property_calibration_length):
                self.property_calibration_length.setText(text)
                self.property_calibration_length.setEnabled(enabled)
                self.property_calibration_length.setToolTip(tooltip)
        finally:
            self._updating_calibration_length_field = False

    def _normalize_record_metadata(self, record: ProjectImageRecord) -> None:
        normalized_metadata = default_project_image_metadata()
        normalized_metadata.update(record.metadata)
        record.metadata = normalized_metadata

    def _load_metadata_fields(self, record: ProjectImageRecord) -> None:
        self._updating_metadata_fields = True
        try:
            with QSignalBlocker(self.property_file_name):
                self.property_file_name.setText(record.display_name)
            with QSignalBlocker(self.image_id):
                self.image_id.setText(record.id)
            for metadata_key, field in self._metadata_fields.items():
                with QSignalBlocker(field):
                    field.setText(str(record.metadata.get(metadata_key, "")))
            with QSignalBlocker(self.metadata_notes):
                self.metadata_notes.setPlainText(str(record.metadata.get("notes", "")))
        finally:
            self._updating_metadata_fields = False

    def _handle_project_name_edit_finished(self) -> None:
        if self._updating_project_identity_fields:
            return
        if self.project_document is None or self.project_path is None:
            return

        raw_name = self.project_name.text()
        normalized_name = _project_name_from_field_text(raw_name)
        if normalized_name is None:
            self._show_warning(
                "Некорректное имя",
                "Введите имя проекта без пути и без расширения .json.",
            )
            self._load_project_identity_fields()
            return

        self._rename_project_file(normalized_name)

    def _rename_project_file(self, new_name: str) -> bool:
        if self.project_document is None or self.project_path is None:
            return False

        old_path = self.project_path
        old_dir = old_path.parent
        old_document_name = self.project_document.name
        new_dir = old_dir.with_name(new_name)
        new_file_name = f"{new_name}.json"
        renamed_file_path = old_dir / new_file_name
        new_path = new_dir / new_file_name
        needs_file_rename = old_path.name != new_file_name
        needs_dir_rename = old_dir != new_dir
        if not needs_file_rename and not needs_dir_rename:
            if self.project_document.name != new_name:
                self.project_document.name = new_name
                self._schedule_project_save()
                self._update_window_title()
            self._load_project_identity_fields()
            return True

        if needs_dir_rename and new_dir.exists():
            self._show_warning(
                "Папка уже существует",
                f"Папка проекта уже существует: {new_dir.name}",
            )
            self._load_project_identity_fields()
            return False

        if needs_file_rename and renamed_file_path.exists():
            self._show_warning(
                "Файл уже существует",
                f"Файл проекта уже существует: {renamed_file_path.name}",
            )
            self._load_project_identity_fields()
            return False

        self._project_autosave_timer.stop()
        try:
            if needs_file_rename:
                old_path.rename(renamed_file_path)
            if needs_dir_rename:
                old_dir.rename(new_dir)
        except OSError as exc:
            if (
                needs_file_rename
                and renamed_file_path.exists()
                and not old_path.exists()
            ):
                try:
                    renamed_file_path.rename(old_path)
                except OSError:
                    pass
            self.project_path = old_path
            self.project_document.name = old_document_name
            self._show_error(
                "Ошибка переименования проекта",
                f"Не удалось переименовать файл или папку проекта: {exc}",
            )
            self._load_project_identity_fields()
            self._update_window_title()
            return False

        self.project_path = new_path.resolve()
        self.project_document.name = new_name
        self._load_project_identity_fields()
        self._update_window_title()
        if not self._save_project_silently(show_error=True):
            return False
        self.statusBar().showMessage(f"Проект переименован: {new_name}")
        return True

    def _handle_project_general_info_changed(self) -> None:
        if self._updating_project_info_fields:
            return
        if self.project_document is None:
            return

        value = self.project_general_info.toPlainText()
        if self.project_document.project_info.general_info == value:
            return

        self.project_document.project_info.general_info = value
        self._schedule_project_save()

    def _handle_metadata_line_edit_finished(
        self, metadata_key: str, field: QLineEdit
    ) -> None:
        if self._updating_metadata_fields:
            return

        self._store_metadata_text_value(metadata_key, field.text().strip(), field)

    def _handle_image_id_edit_finished(self) -> None:
        if self._updating_metadata_fields:
            return

        self._store_record_id_value(self.image_id.text().strip(), self.image_id)

    def _handle_calibration_length_edit_finished(self) -> None:
        if self._updating_calibration_length_field:
            return

        record = self._selected_project_image()
        if record is None or record.calibration is None or record.calibration_error:
            self._update_calibration_length_field(record)
            return

        length_mm = _parse_calibration_length_mm(
            self.property_calibration_length.text()
        )
        if length_mm is None:
            self._show_warning(
                "Некорректная калибровка",
                "Длина калибровочного отрезка должна быть положительным числом.",
            )
            self._update_calibration_length_field(record)
            return

        try:
            calibration = ImageCalibration(
                record.calibration.start,
                record.calibration.end,
                length_mm,
            )
        except ValueError as exc:
            self._show_warning("Некорректная калибровка", str(exc))
            self._update_calibration_length_field(record)
            return

        if math.isclose(
            calibration.length_mm,
            record.calibration.length_mm,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            self._update_calibration_length_field(record)
            return

        record.calibration = calibration
        record.calibration_error = None
        record.raw_calibration = None
        if (
            self._current_project_image_id == record.id
            and self.canvas.has_calibration()
        ):
            self.canvas.set_calibration_label_text(
                _calibration_length_text(calibration.length_mm)
            )
        self._update_project_properties()
        self._update_action_states()
        self._schedule_project_save()
        self.statusBar().showMessage(
            f"Длина калибровки изменена: {_calibration_length_text(calibration.length_mm)}"
        )

    def _store_metadata_text_value(
        self,
        metadata_key: str,
        value: str,
        field: QLineEdit | None = None,
    ) -> bool:
        record = self._selected_project_image()
        if record is None:
            return False

        validation_error = _metadata_validation_error(metadata_key, value)
        if validation_error:
            self._show_warning("Некорректное значение", validation_error)
            if field is not None:
                with QSignalBlocker(field):
                    field.setText(str(record.metadata.get(metadata_key, "")))
            return False

        self._normalize_record_metadata(record)
        if str(record.metadata.get(metadata_key, "")) == value:
            return True

        record.set_metadata_value(metadata_key, value)
        if field is not None:
            with QSignalBlocker(field):
                field.setText(value)
        self._schedule_project_save()
        return True

    def _store_record_id_value(
        self, value: str, field: QLineEdit | None = None
    ) -> bool:
        record = self._selected_project_image()
        if record is None:
            return False

        normalized_value = value.strip()
        validation_error = self._record_id_validation_error(record, normalized_value)
        if validation_error:
            self._show_warning("Некорректный ID", validation_error)
            if field is not None:
                with QSignalBlocker(field):
                    field.setText(record.id)
            return False

        if record.id == normalized_value:
            if field is not None:
                with QSignalBlocker(field):
                    field.setText(normalized_value)
            return True

        old_id = record.id
        record.id = normalized_value
        if self._current_project_image_id == old_id:
            self._current_project_image_id = normalized_value

        current_item = self.project_list.currentItem()
        if current_item is not None and current_item.data(Qt.UserRole) == old_id:
            current_item.setData(Qt.UserRole, normalized_value)
            self._populate_project_list_item(current_item, record)

        if field is not None:
            with QSignalBlocker(field):
                field.setText(normalized_value)
        self._schedule_project_save()
        return True

    def _record_id_validation_error(
        self, current_record: ProjectImageRecord, value: str
    ) -> str:
        if not value:
            return "ID изображения не должен быть пустым."
        if self.project_document is None:
            return ""

        normalized_value = value.strip()
        for record in self.project_document.images:
            if record.id == current_record.id:
                continue
            if record.id == normalized_value:
                return "Такой ID уже используется другим изображением проекта."
        return ""

    def _new_project_image_id(self) -> str:
        used_ids = set()
        if self.project_document is not None:
            used_ids = {record.id for record in self.project_document.images}

        for _attempt in range(10):
            candidate = str(uuid4())
            if candidate not in used_ids:
                return candidate
        raise ValueError("Не удалось сгенерировать уникальный ID изображения.")

    def _generate_image_id(self) -> None:
        record = self._selected_project_image()
        if record is None:
            return

        try:
            generated_id = self._new_project_image_id()
        except ValueError as exc:
            self._show_error("Ошибка генерации ID", str(exc))
            return

        if not self._store_record_id_value(generated_id, self.image_id):
            return

        self.statusBar().showMessage(
            f"Сгенерирован новый ID изображения: {generated_id}"
        )

    def _open_metadata_datetime_dialog(
        self, metadata_key: str, field: QLineEdit
    ) -> None:
        if self._updating_metadata_fields:
            return
        if self._selected_project_image() is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Выбор даты и времени")

        editor = QDateTimeEdit(_qdatetime_from_text(field.text()), dialog)
        editor.setCalendarPopup(True)
        editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss")

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        clear_button = button_box.addButton("Очистить", QDialogButtonBox.ActionRole)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(editor)
        layout.addWidget(button_box)

        clear_requested = False

        def clear_value() -> None:
            nonlocal clear_requested
            clear_requested = True
            dialog.accept()

        clear_button.clicked.connect(clear_value)

        if dialog.exec_() != QDialog.Accepted:
            return

        value = ""
        if not clear_requested:
            value = editor.dateTime().toPyDateTime().isoformat(timespec="seconds")
        self._store_metadata_text_value(metadata_key, value, field)

    def _handle_metadata_notes_changed(self) -> None:
        if self._updating_metadata_fields:
            return

        record = self._selected_project_image()
        if record is None:
            return

        self._normalize_record_metadata(record)
        notes = self.metadata_notes.toPlainText()
        if str(record.metadata.get("notes", "")) == notes:
            return

        record.set_metadata_value("notes", notes)
        self._schedule_project_save()

    def _handle_file_name_edit_finished(self) -> None:
        if self._updating_metadata_fields:
            return

        record = self._selected_project_image()
        if record is None:
            return

        new_name = self.property_file_name.text().strip()
        if new_name == record.display_name:
            return

        if self._rename_project_image(record, new_name):
            self._update_current_project_list_item(record)
            self._update_project_properties()
            self._update_window_title()
            self._save_project_silently(show_error=True)
            return

        with QSignalBlocker(self.property_file_name):
            self.property_file_name.setText(record.display_name)

    def _rename_file_to_image_id(self) -> None:
        record = self._selected_project_image()
        if record is None:
            return

        image_id = self.image_id.text().strip()
        if not self._store_record_id_value(image_id, self.image_id):
            return

        image_id = record.id.strip()
        if not image_id:
            self._show_warning(
                "Некорректный ID", "ID изображения не должен быть пустым."
            )
            return
        if "/" in image_id or "\\" in image_id:
            self._show_warning(
                "Некорректный ID",
                "ID изображения не должен содержать символы пути.",
            )
            return

        old_suffix = Path(record.relative_path).suffix
        target_name = image_id
        if old_suffix and not target_name.lower().endswith(old_suffix.lower()):
            target_name = f"{target_name}{old_suffix}"

        if target_name == record.display_name:
            self.statusBar().showMessage("Файл уже назван как ID.")
            return

        if self._rename_project_image(record, target_name):
            self._update_current_project_list_item(record)
            self._update_project_properties()
            self._update_window_title()
            self._save_project_silently(show_error=True)

    def _rename_project_image(self, record: ProjectImageRecord, new_name: str) -> bool:
        if not new_name:
            self._show_warning("Некорректное имя", "Имя файла не должно быть пустым.")
            return False

        new_file_name = Path(new_name).name
        if not new_file_name:
            self._show_warning("Некорректное имя", "Имя файла не должно быть пустым.")
            return False

        old_relative_path = Path(record.relative_path)
        old_suffix = old_relative_path.suffix
        candidate = Path(new_file_name)
        if not candidate.suffix:
            candidate = candidate.with_suffix(old_suffix)

        if candidate.suffix.lower() not in SUPPORTED_RASTER_SUFFIXES:
            self._show_warning(
                "Некорректное расширение",
                "Имя файла должно иметь поддерживаемое расширение изображения.",
            )
            return False

        image_path = self._project_image_path(record)
        if self.project_path is None:
            record.display_name = candidate.name
            return True

        if not image_path.exists():
            record.display_name = candidate.name
            self._show_warning(
                "Файл отсутствует",
                "Файл изображения не найден на диске. Изменено только отображаемое имя.",
            )
            self._schedule_project_save()
            return True

        target_path = image_path.with_name(candidate.name)
        if target_path.resolve() != image_path.resolve() and target_path.exists():
            self._show_warning(
                "Имя занято",
                "Файл с таким именем уже существует в папке проекта.",
            )
            return False

        try:
            if target_path.resolve() != image_path.resolve():
                image_path.rename(target_path)
        except OSError as exc:
            self._show_error(
                "Ошибка переименования", f"Не удалось переименовать файл: {exc}"
            )
            return False

        record.relative_path = portable_path_reference(
            target_path, self.project_path.parent
        )
        record.display_name = candidate.name
        if record.annotation is not None:
            record.annotation.image_path = record.relative_path
        self._schedule_project_save()
        self.statusBar().showMessage(f"Файл переименован: {record.display_name}")
        return True

    def _current_contour_stats(
        self,
        record: ProjectImageRecord,
    ) -> (
        tuple[
            tuple[int, int, int],
            tuple[int, int, int],
            tuple[int, int, int],
            tuple[int, int, int],
            tuple[int, int, int],
            int,
        ]
        | None
    ):
        if record.id != self._current_project_image_id:
            return None
        result = self._ensure_current_contour_analysis(record)
        if result is None or result.analysis is None:
            return None
        stats = result.analysis.stats
        return (
            stats.mean_rgb,
            stats.mean_lab,
            stats.mean_hsv,
            stats.mean_yuv,
            stats.mean_lms,
            stats.pixel_count,
        )

    def _set_average_color_swatch(self, rgb: tuple[int, int, int] | None) -> None:
        self._current_average_color_rgb = rgb
        if rgb is None:
            self.average_color_swatch.setStyleSheet(
                "QFrame#averageColorSwatch { border: 1px solid #95a3b8; border-radius: 4px; background: transparent; }"
            )
            self.average_color_swatch.setCursor(Qt.ArrowCursor)
            self.average_color_swatch.setToolTip("Средний цвет ещё не рассчитан.")
            return
        red, green, blue = rgb
        self.average_color_swatch.setStyleSheet(
            "QFrame#averageColorSwatch { "
            f"border: 1px solid #95a3b8; border-radius: 4px; background: rgb({red}, {green}, {blue}); "
            "}"
        )
        self.average_color_swatch.setCursor(Qt.PointingHandCursor)
        self.average_color_swatch.setToolTip("Сохранить средний цвет как PNG.")

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "О программе",
            (
                "MagicBorder\n\n"
                "Локальное приложение для выделения и редактирования контура листа растения.\n\n"
                "Разработчик: Дьяченко Р.А.\n\n"
                "Подсказки:\n"
                "- перетаскивайте узлы мышью;\n"
                "- двойной щелчок рядом с сегментом добавляет узел;\n"
                "- правая кнопка мыши по узлу или клавиша Delete удаляет узел;\n"
                "- Ctrl + колесо мыши изменяет масштаб."
            ),
        )

    def _build_annotation(self, annotation_path: Path) -> Annotation:
        image_path = self.canvas.current_image_path()
        image_size = self.canvas.image_size()
        if image_path is None or image_size is None:
            raise ValueError("Нет изображения для создания аннотации.")

        width, height = image_size
        return Annotation(
            image_path=portable_path_reference(image_path, annotation_path.parent),
            image_width=width,
            image_height=height,
            points=self.canvas.contour_points(),
            line_color=self.canvas.contour_line_color(),
            closed=True,
        )

    def _prepare_image_for_annotation(
        self, annotation: Annotation, annotation_path: Path
    ) -> bool:
        current_image_size = self.canvas.image_size()
        if current_image_size == (annotation.image_width, annotation.image_height):
            return True

        self._show_warning(
            "Аннотация не подходит",
            "Аннотацию можно загрузить только для выбранного изображения проекта с тем же размером.",
        )
        return False

    def _update_window_title(self) -> None:
        if self.project_document is not None:
            record = self._current_project_image()
            if record is not None:
                self.setWindowTitle(
                    f"{APP_TITLE} - {self.project_document.name} - {record.display_name}"
                )
                return
            self.setWindowTitle(f"{APP_TITLE} - {self.project_document.name}")
            return

        self.setWindowTitle(APP_TITLE)

    def _default_annotation_file_name(self) -> str:
        image_path = self.canvas.current_image_path()
        if image_path is None:
            return "annotation.json"
        return str(image_path.with_suffix(".json"))

    def _default_histogram_file_name(self, color_space: str) -> str:
        fallback_name = f"{color_space}_histogram.png"
        image_path = self.canvas.current_image_path()
        if image_path is None:
            return fallback_name
        return str(
            image_path.with_name(f"{image_path.stem}_{color_space}_histogram.png")
        )

    def closeEvent(self, event) -> None:
        self._store_current_angle_measurements()
        self._store_current_segment_measurements()
        self._save_current_project_annotation()
        if self._save_project_silently(show_error=True):
            self._shutdown_background_tasks()
            event.accept()
            return

        answer = QMessageBox.question(
            self,
            "Закрыть без сохранения?",
            "Проект не удалось сохранить. Закрыть приложение без сохранения проекта?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._shutdown_background_tasks()
            event.accept()
        else:
            event.ignore()

    def _shutdown_background_tasks(self) -> None:
        self._cancel_background_image_tasks()
        # Даём потокам закончить текущий шаг, чтобы они не отправляли сигналы
        # в уже разрушенное окно. Ожидание ограничено по времени.
        pools = {
            id(pool): pool
            for pool in (
                self._contour_analysis_thread_pool,
                self._contour_detection_thread_pool,
                self._project_color_stats_thread_pool,
                self._flatten_background_thread_pool,
                self._image_prepare_thread_pool,
            )
            if isinstance(pool, QThreadPool)
        }
        for pool in pools.values():
            pool.waitForDone(BACKGROUND_TASK_SHUTDOWN_TIMEOUT_MS)

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)


def _ensure_json_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    if path.suffix:
        return path.with_suffix(".json")
    return path.with_suffix(".json")


def _ensure_xlsx_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".xlsx":
        return path
    if path.suffix:
        return path.with_suffix(".xlsx")
    return path.with_suffix(".xlsx")


def _ensure_png_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".png":
        return path
    if path.suffix:
        return path.with_suffix(".png")
    return path.with_suffix(".png")


def _rgb_text(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"RGB({red}, {green}, {blue})"


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"#{red:02X}{green:02X}{blue:02X}"


def _calibration_points_from_signal(raw_points: object) -> tuple[Point, Point] | None:
    if not isinstance(raw_points, (list, tuple)) or len(raw_points) != 2:
        return None
    if not all(isinstance(point, Point) for point in raw_points):
        return None
    return raw_points[0], raw_points[1]


def _calibration_scale_text(calibration: ImageCalibration | None) -> str:
    if calibration is None:
        return "-"

    return (
        f"{_compact_float(calibration.pixel_length(), 2)} px = "
        f"{_compact_float(calibration.length_mm, 4)} мм; "
        f"{_compact_float(calibration.pixels_per_mm(), 2)} px/мм; "
        f"{_compact_float(calibration.mm_per_pixel(), 4)} мм/px"
    )


def _calibration_length_text(length_mm: float) -> str:
    return f"{_compact_float(length_mm, 4)} мм"


def _angle_by_id(
    record: ProjectImageRecord, angle_id: str
) -> ProjectAngleMeasurement | None:
    for angle in record.measurements.angles:
        if angle.id == angle_id:
            return angle
    return None


def _segment_by_id(
    record: ProjectImageRecord,
    segment_id: str,
) -> ProjectSegmentMeasurement | None:
    for segment in record.measurements.segments:
        if segment.id == segment_id:
            return segment
    return None


def _assessment_from_combo(field: QComboBox) -> ProjectMeasurementAssessment | None:
    code = normalize_oiv_code(field.currentData())
    if not code:
        return None
    return ProjectMeasurementAssessment(system=OIV_SYSTEM_NAME, code=code)


def _measurement_oiv_code(assessment: ProjectMeasurementAssessment | None) -> str:
    if assessment is None:
        return ""
    if assessment.system.strip().upper() != OIV_SYSTEM_NAME:
        return ""
    return normalize_oiv_code(assessment.code)


def _measurement_assessments_equal(
    left: ProjectMeasurementAssessment | None,
    right: ProjectMeasurementAssessment | None,
) -> bool:
    return _measurement_assessment_key(left) == _measurement_assessment_key(right)


def _measurement_assessment_key(
    assessment: ProjectMeasurementAssessment | None,
) -> tuple[str, str]:
    if assessment is None:
        return "", ""
    system = assessment.system.strip().upper()
    code = (
        normalize_oiv_code(assessment.code)
        if system == OIV_SYSTEM_NAME
        else assessment.code.strip()
    )
    return system, code


def _angle_canvas_records(
    record: ProjectImageRecord,
) -> list[tuple[str, Point, Point, Point, str, str, str]]:
    return [
        (
            angle.id,
            angle.first,
            angle.vertex,
            angle.second,
            angle.name,
            angle.line_color,
            angle.label_color,
        )
        for angle in record.measurements.angles
    ]


def _segment_canvas_records(
    record: ProjectImageRecord,
) -> list[tuple[str, Point, Point, str, str, str, str, str]]:
    return [
        (
            segment.id,
            segment.start,
            segment.end,
            segment.start_label,
            segment.end_label,
            segment.name,
            segment.line_color,
            segment.label_color,
        )
        for segment in record.measurements.segments
    ]


def _angle_measurements_equal(
    left: list[ProjectAngleMeasurement],
    right: list[ProjectAngleMeasurement],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        left_angle.id == right_angle.id
        and left_angle.first == right_angle.first
        and left_angle.vertex == right_angle.vertex
        and left_angle.second == right_angle.second
        and left_angle.name == right_angle.name
        and left_angle.line_color == right_angle.line_color
        and left_angle.label_color == right_angle.label_color
        and left_angle.note == right_angle.note
        and _measurement_assessments_equal(
            left_angle.assessment, right_angle.assessment
        )
        for left_angle, right_angle in zip(left, right, strict=True)
    )


def _segment_measurements_equal(
    left: list[ProjectSegmentMeasurement],
    right: list[ProjectSegmentMeasurement],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        left_segment.id == right_segment.id
        and left_segment.start == right_segment.start
        and left_segment.end == right_segment.end
        and left_segment.name == right_segment.name
        and left_segment.line_color == right_segment.line_color
        and left_segment.label_color == right_segment.label_color
        and left_segment.start_label == right_segment.start_label
        and left_segment.end_label == right_segment.end_label
        and left_segment.note == right_segment.note
        and _measurement_assessments_equal(
            left_segment.assessment, right_segment.assessment
        )
        for left_segment, right_segment in zip(left, right, strict=True)
    )


def _angle_display_name(angle: ProjectAngleMeasurement, angle_index: int) -> str:
    return angle.name or f"Угол {angle_index}"


def _segment_display_name(
    segment: ProjectSegmentMeasurement, segment_index: int
) -> str:
    return segment.name or f"Отрезок {segment_index}"


def _oiv_trait_option_text(trait: OivTrait) -> str:
    return f"{trait.code} - {trait.short_name}"


def _oiv_result_text(result: OivClassificationResult | None) -> str:
    if result is None:
        return "-"
    if not result.ok:
        return result.message or "-"

    value_text = _oiv_value_text(result.value, result.unit)
    score_text = str(result.score) if result.score is not None else "-"
    label_text = result.label or "-"
    if value_text:
        return f"{value_text}; оценка {score_text} - {label_text}"
    return f"оценка {score_text} - {label_text}"


def _oiv_value_text(value: float | None, unit: str) -> str:
    if value is None:
        return ""
    if unit == "deg":
        rounded = round(value, 1)
        if math.isclose(rounded, round(rounded), abs_tol=1e-9):
            return f"{int(round(rounded))}°"
        return f"{rounded:.1f}°"
    if unit == "mm":
        return f"{_compact_float(value, 2)} мм"
    unit_suffix = str(unit or "").strip()
    value_text = _compact_float(value, 2)
    return f"{value_text} {unit_suffix}".strip()


def _angle_measurement_value_text(angle: ProjectAngleMeasurement) -> str:
    value = _angle_degrees(angle.first, angle.vertex, angle.second)
    if value is None:
        return "-"
    rounded = round(value, 1)
    if math.isclose(rounded, round(rounded), abs_tol=1e-9):
        return f"{int(round(rounded))}°"
    return f"{rounded:.1f}°"


def _angle_degrees(first: Point, vertex: Point, second: Point) -> float | None:
    first_x = first.x - vertex.x
    first_y = first.y - vertex.y
    second_x = second.x - vertex.x
    second_y = second.y - vertex.y
    first_length = math.hypot(first_x, first_y)
    second_length = math.hypot(second_x, second_y)
    if first_length <= 1e-6 or second_length <= 1e-6:
        return None
    cosine = (first_x * second_x + first_y * second_y) / (first_length * second_length)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _point_text(point: Point) -> str:
    return f"x={_compact_float(point.x, 2)}, y={_compact_float(point.y, 2)}"


def _segment_measurement_length_text(
    segment: ProjectSegmentMeasurement,
    calibration: ImageCalibration | None,
) -> str:
    pixel_length = math.hypot(
        segment.end.x - segment.start.x, segment.end.y - segment.start.y
    )
    pixel_text = f"{_compact_float(pixel_length, 1)} px"
    if calibration is None:
        return pixel_text
    return f"{pixel_text} / {_compact_float(pixel_length * calibration.mm_per_pixel(), 2)} мм"


def _parse_calibration_length_mm(text: str) -> float | None:
    normalized = text.strip().lower().replace(",", ".")
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


def _contour_area_mm2_text(
    contour_pixel_count: int,
    calibration: ImageCalibration | None,
    calibration_error: str | None = None,
) -> str:
    if contour_pixel_count <= 0:
        return "-"
    if calibration_error:
        return "ошибка калибровки"
    if calibration is None:
        return "калибровка не произведена"

    area_mm2 = (
        contour_pixel_count * calibration.mm_per_pixel() * calibration.mm_per_pixel()
    )
    if area_mm2 >= 100:
        decimals = 1
    elif area_mm2 >= 1:
        decimals = 2
    else:
        decimals = 4
    return f"{_compact_float(area_mm2, decimals)} мм²"


def _compact_float(value: float, decimals: int) -> str:
    text = f"{float(value):.{decimals}f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _annotation_rgb_pixels(rgb_array: np.ndarray, annotation: Annotation) -> np.ndarray:
    return contour_rgb_pixels_from_points(rgb_array, annotation.points)


def _mean_lab_values(rgb_pixels: np.ndarray) -> tuple[int, int, int]:
    lab_values = _lab_values_from_rgb_pixels(rgb_pixels)
    mean_values = np.rint(lab_values.mean(axis=0)).astype(int)
    return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])


def _mean_hsv_values(rgb_pixels: np.ndarray) -> tuple[int, int, int]:
    hsv_values = _hsv_values_from_rgb_pixels(rgb_pixels)
    mean_values = np.rint(hsv_values.mean(axis=0)).astype(int)
    return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])


def _mean_yuv_values(rgb_pixels: np.ndarray) -> tuple[int, int, int]:
    yuv_values = _yuv_values_from_rgb_pixels(rgb_pixels)
    mean_values = np.rint(yuv_values.mean(axis=0)).astype(int)
    return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])


def _mean_lms_values(rgb_pixels: np.ndarray) -> tuple[int, int, int]:
    lms_values = rgb_to_lms(rgb_pixels)
    return _mean_lms_values_from_total(
        lms_values.sum(axis=0),
        lms_values.max(axis=0),
        int(lms_values.shape[0]),
    )


def _lab_values_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    return lab_values_from_rgb_pixels(rgb_pixels)


def _hsv_values_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    return hsv_values_from_rgb_pixels(rgb_pixels)


def _yuv_values_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    return yuv_values_from_rgb_pixels(rgb_pixels)


def _mean_lms_values_from_total(
    lms_total: np.ndarray,
    lms_max: np.ndarray,
    pixel_count: int,
) -> tuple[int, int, int]:
    return mean_lms_values_from_total(lms_total, lms_max, pixel_count)


def _circle_contour_points(width: int, height: int, node_count: int) -> list[Point]:
    bounded_count = max(3, min(128, int(node_count)))
    image_width = max(1.0, float(width))
    image_height = max(1.0, float(height))
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    radius = max(0.5, 0.35 * min(image_width, image_height))
    radius = min(radius, max(0.5, min(center_x, center_y) - 0.5))

    points: list[Point] = []
    for index in range(bounded_count):
        angle = -math.pi / 2.0 + 2.0 * math.pi * index / bounded_count
        x = center_x + math.cos(angle) * radius
        y = center_y + math.sin(angle) * radius
        x = min(max(x, 0.0), image_width)
        y = min(max(y, 0.0), image_height)
        points.append(Point(x, y))
    return points


def _current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _metadata_validation_error(metadata_key: str, value: str) -> str:
    if not value:
        return ""

    if metadata_key in ("added_at", "captured_at"):
        if (
            QDateTime.fromString(value, Qt.ISODate).isValid()
            or _datetime_from_text(value) is not None
        ):
            return ""
        return "Дата и время должны быть указаны в формате ISO 8601 или ГГГГ-ММ-ДД ЧЧ:ММ:СС."

    numeric_ranges = {
        "humidity": (0.0, 100.0, "Влажность должна быть числом в диапазоне 0..100."),
        "wind_speed": (0.0, None, "Скорость ветра не должна быть отрицательной."),
        "latitude": (-90.0, 90.0, "Широта должна быть числом в диапазоне -90..90."),
        "longitude": (
            -180.0,
            180.0,
            "Долгота должна быть числом в диапазоне -180..180.",
        ),
    }
    if metadata_key not in numeric_ranges:
        return ""

    minimum, maximum, message = numeric_ranges[metadata_key]
    try:
        number = float(value.replace(",", "."))
    except ValueError:
        return message

    if number < minimum:
        return message
    if maximum is not None and number > maximum:
        return message
    return ""


def _qdatetime_from_text(value: str) -> QDateTime:
    text = value.strip()
    if text:
        parsed = QDateTime.fromString(text, Qt.ISODate)
        if parsed.isValid():
            return parsed
        with_timezone = _datetime_from_text(text)
        if with_timezone is not None:
            return QDateTime(with_timezone)
    return QDateTime.currentDateTime()


def _datetime_from_text(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    for format_string in ("%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
        try:
            return datetime.strptime(text, format_string)
        except ValueError:
            continue
    return None


def _safe_project_name(raw_name: str) -> str:
    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_", " ") else "_"
        for char in raw_name.strip()
    )
    return "_".join(safe_name.split())


def _project_name_without_json_suffix(raw_name: str) -> str:
    name = str(raw_name or "").strip()
    if name.lower().endswith(".json"):
        name = name[:-5].strip()
    return name


def _project_name_from_path(project_path: Path | None, fallback: str = "") -> str:
    if project_path is not None and project_path.stem:
        return project_path.stem
    return _project_name_without_json_suffix(fallback)


def _project_name_from_field_text(raw_name: str) -> str | None:
    name = _project_name_without_json_suffix(raw_name)
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        return None
    safe_name = _safe_project_name(name)
    if not safe_name or safe_name in {".", ".."}:
        return None
    return safe_name


def _project_relative_path_key(value: str) -> str:
    return str(value or "").replace("\\", "/").strip("/")


def _downscaled_count_suffix(downscaled_count: int) -> str:
    return f" (уменьшено: {downscaled_count})" if downscaled_count else ""


def _unique_destination_path(
    directory: Path, file_name: str, *, reserved: set[Path] | None = None
) -> Path:
    source_name = Path(file_name).name
    stem = Path(source_name).stem or "image"
    suffix = Path(source_name).suffix
    candidate = directory / source_name
    index = 1
    while candidate.exists() or (reserved is not None and candidate in reserved):
        candidate = directory / f"{stem}_{index}{suffix}"
        index += 1
    if reserved is not None:
        reserved.add(candidate)
    return candidate
