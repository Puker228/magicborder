from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

PROJECT_FORMAT_VERSION = 2
CONTOUR_LINE_COLOR = "#0b84c6"
ANGLE_LINE_COLOR = "#22c55e"
ANGLE_LABEL_COLOR = "#166534"
SEGMENT_LINE_COLOR = "#f97316"
SEGMENT_LABEL_COLOR = "#9a3412"
_HEX_RGB_COLOR_PATTERN = re.compile(r"#[0-9a-fA-F]{6}")


def normalize_line_color(value: Any, default: str) -> str:
    text = str(value or "").strip()
    if _HEX_RGB_COLOR_PATTERN.fullmatch(text):
        return text.lower()
    fallback = str(default or "").strip()
    if _HEX_RGB_COLOR_PATTERN.fullmatch(fallback):
        return fallback.lower()
    return "#000000"


def _require_number(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Поле '{field_name}' должно быть числом.") from exc


def _require_int(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Поле '{field_name}' должно быть целым числом.") from exc
    if number <= 0:
        raise ValueError(f"Поле '{field_name}' должно быть положительным.")
    return number


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(float(self.x), 3), "y": round(float(self.y), 3)}

    @classmethod
    def from_dict(cls, data: Any) -> Point:
        if not isinstance(data, dict):
            raise ValueError(
                "Каждая точка контура должна быть объектом с полями 'x' и 'y'."
            )
        return cls(
            x=_require_number(data.get("x"), "x"),
            y=_require_number(data.get("y"), "y"),
        )


@dataclass(slots=True)
class ImageCalibration:
    start: Point
    end: Point
    length_mm: float

    def __post_init__(self) -> None:
        if not isinstance(self.start, Point):
            self.start = Point.from_dict(self.start)
        if not isinstance(self.end, Point):
            self.end = Point.from_dict(self.end)

        self.length_mm = _require_number(self.length_mm, "length_mm")
        if self.length_mm <= 0:
            raise ValueError("Длина калибровочного отрезка должна быть положительной.")
        if self.pixel_length() <= 1e-6:
            raise ValueError("Калибровочный отрезок должен иметь ненулевую длину.")

    def pixel_length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)

    def pixels_per_mm(self) -> float:
        return self.pixel_length() / self.length_mm

    def mm_per_pixel(self) -> float:
        return self.length_mm / self.pixel_length()

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "length_mm": float(self.length_mm),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ImageCalibration:
        if not isinstance(data, dict):
            raise ValueError("Калибровка должна быть объектом.")
        return cls(
            start=Point.from_dict(data.get("start")),
            end=Point.from_dict(data.get("end")),
            length_mm=_require_number(data.get("length_mm"), "length_mm"),
        )


def _new_angle_measurement_id() -> str:
    return f"angle-{uuid4()}"


def _new_segment_measurement_id() -> str:
    return f"segment-{uuid4()}"


@dataclass(slots=True)
class ProjectMeasurementAssessment:
    system: str = "OIV"
    code: str = ""

    def __post_init__(self) -> None:
        self.system = str(self.system or "OIV").strip() or "OIV"
        self.code = str(self.code or "").strip()

    def to_dict(self) -> dict[str, str]:
        return {
            "system": self.system,
            "code": self.code,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProjectMeasurementAssessment | None:
        if not isinstance(data, dict):
            return None
        code = str(data.get("code") or "").strip()
        if not code:
            return None
        return cls(
            system=str(data.get("system") or "OIV"),
            code=code,
        )


@dataclass(slots=True)
class ProjectAngleMeasurement:
    id: str
    first: Point
    vertex: Point
    second: Point
    name: str = ""
    note: str = ""
    assessment: ProjectMeasurementAssessment | None = None
    line_color: str = ANGLE_LINE_COLOR
    label_color: str = ANGLE_LABEL_COLOR

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip() or _new_angle_measurement_id()
        if not isinstance(self.first, Point):
            self.first = Point.from_dict(self.first)
        if not isinstance(self.vertex, Point):
            self.vertex = Point.from_dict(self.vertex)
        if not isinstance(self.second, Point):
            self.second = Point.from_dict(self.second)
        self.name = str(self.name or "").strip()
        self.line_color = normalize_line_color(self.line_color, ANGLE_LINE_COLOR)
        self.label_color = normalize_line_color(self.label_color, ANGLE_LABEL_COLOR)
        self.note = str(self.note or "")
        if not isinstance(self.assessment, ProjectMeasurementAssessment):
            self.assessment = ProjectMeasurementAssessment.from_dict(self.assessment)

    def to_dict(self) -> dict[str, Any]:
        self.line_color = normalize_line_color(self.line_color, ANGLE_LINE_COLOR)
        self.label_color = normalize_line_color(self.label_color, ANGLE_LABEL_COLOR)
        payload: dict[str, Any] = {"id": self.id}
        if self.name:
            payload["name"] = self.name
        payload.update(
            {
                "first": self.first.to_dict(),
                "vertex": self.vertex.to_dict(),
                "second": self.second.to_dict(),
                "line_color": self.line_color,
                "label_color": self.label_color,
                "note": self.note,
            }
        )
        if self.assessment is not None:
            payload["assessment"] = self.assessment.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> ProjectAngleMeasurement:
        if not isinstance(data, dict):
            raise ValueError("Измерение угла должно быть объектом.")
        return cls(
            id=str(data.get("id", "")),
            first=Point.from_dict(data.get("first")),
            vertex=Point.from_dict(data.get("vertex")),
            second=Point.from_dict(data.get("second")),
            name=str(data.get("name") or ""),
            line_color=normalize_line_color(data.get("line_color"), ANGLE_LINE_COLOR),
            label_color=normalize_line_color(
                data.get("label_color"), ANGLE_LABEL_COLOR
            ),
            note=str(data.get("note", "")),
            assessment=ProjectMeasurementAssessment.from_dict(data.get("assessment")),
        )


@dataclass(slots=True)
class ProjectSegmentMeasurement:
    id: str
    start: Point
    end: Point
    name: str = ""
    start_label: str = ""
    end_label: str = ""
    note: str = ""
    assessment: ProjectMeasurementAssessment | None = None
    line_color: str = SEGMENT_LINE_COLOR
    label_color: str = SEGMENT_LABEL_COLOR

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip() or _new_segment_measurement_id()
        if not isinstance(self.start, Point):
            self.start = Point.from_dict(self.start)
        if not isinstance(self.end, Point):
            self.end = Point.from_dict(self.end)
        self.name = str(self.name or "").strip()
        self.line_color = normalize_line_color(self.line_color, SEGMENT_LINE_COLOR)
        self.label_color = normalize_line_color(self.label_color, SEGMENT_LABEL_COLOR)
        self.start_label = str(self.start_label or "").strip()
        self.end_label = str(self.end_label or "").strip()
        self.note = str(self.note or "")
        if not isinstance(self.assessment, ProjectMeasurementAssessment):
            self.assessment = ProjectMeasurementAssessment.from_dict(self.assessment)

    def to_dict(self) -> dict[str, Any]:
        self.line_color = normalize_line_color(self.line_color, SEGMENT_LINE_COLOR)
        self.label_color = normalize_line_color(self.label_color, SEGMENT_LABEL_COLOR)
        payload: dict[str, Any] = {"id": self.id}
        if self.name:
            payload["name"] = self.name
        payload.update(
            {
                "start": self.start.to_dict(),
                "end": self.end.to_dict(),
                "line_color": self.line_color,
                "label_color": self.label_color,
                "start_label": self.start_label,
                "end_label": self.end_label,
                "note": self.note,
            }
        )
        if self.assessment is not None:
            payload["assessment"] = self.assessment.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> ProjectSegmentMeasurement:
        if not isinstance(data, dict):
            raise ValueError("Измерение отрезка должно быть объектом.")
        return cls(
            id=str(data.get("id", "")),
            start=Point.from_dict(data.get("start")),
            end=Point.from_dict(data.get("end")),
            name=str(data.get("name") or ""),
            line_color=normalize_line_color(data.get("line_color"), SEGMENT_LINE_COLOR),
            label_color=normalize_line_color(
                data.get("label_color"), SEGMENT_LABEL_COLOR
            ),
            start_label=str(data.get("start_label") or ""),
            end_label=str(data.get("end_label") or ""),
            note=str(data.get("note", "")),
            assessment=ProjectMeasurementAssessment.from_dict(data.get("assessment")),
        )


@dataclass(slots=True)
class ProjectImageMeasurements:
    angles: list[ProjectAngleMeasurement] = field(default_factory=list)
    segments: list[ProjectSegmentMeasurement] = field(default_factory=list)
    extra_groups: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra_groups)
        payload["angles"] = [angle.to_dict() for angle in self.angles]
        payload["segments"] = [segment.to_dict() for segment in self.segments]
        return payload

    def has_data(self) -> bool:
        return bool(self.angles or self.segments or self.extra_groups)

    @classmethod
    def from_dict(cls, data: Any) -> ProjectImageMeasurements:
        if not isinstance(data, dict):
            data = {}

        raw_angles = data.get("angles", [])
        angles: list[ProjectAngleMeasurement] = []
        if isinstance(raw_angles, list):
            for item in raw_angles:
                try:
                    angles.append(ProjectAngleMeasurement.from_dict(item))
                except ValueError:
                    continue

        raw_segments = data.get("segments", [])
        segments: list[ProjectSegmentMeasurement] = []
        if isinstance(raw_segments, list):
            for item in raw_segments:
                try:
                    segment = ProjectSegmentMeasurement.from_dict(item)
                except ValueError:
                    continue
                if (
                    math.hypot(
                        segment.end.x - segment.start.x, segment.end.y - segment.start.y
                    )
                    <= 1e-6
                ):
                    continue
                segments.append(segment)

        return cls(
            angles=angles,
            segments=segments,
            extra_groups={
                key: value
                for key, value in data.items()
                if key not in {"angles", "segments"}
            },
        )


@dataclass(slots=True)
class Annotation:
    image_path: str
    image_width: int
    image_height: int
    points: list[Point]
    closed: bool = True
    version: int = 1
    line_color: str = CONTOUR_LINE_COLOR

    def __post_init__(self) -> None:
        self.image_width = _require_int(self.image_width, "image_width")
        self.image_height = _require_int(self.image_height, "image_height")
        self.image_path = str(self.image_path or "")
        self.line_color = normalize_line_color(self.line_color, CONTOUR_LINE_COLOR)
        self.closed = bool(self.closed)
        self.version = max(1, int(self.version))
        if len(self.points) < 3:
            raise ValueError("Контур должен содержать минимум 3 точки.")

    def to_dict(self) -> dict[str, Any]:
        self.line_color = normalize_line_color(self.line_color, CONTOUR_LINE_COLOR)
        return {
            "version": self.version,
            "image_path": self.image_path,
            "image_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "closed": self.closed,
            "points": [point.to_dict() for point in self.points],
            "line_color": self.line_color,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Annotation:
        if not isinstance(data, dict):
            raise ValueError("JSON-аннотация должна содержать объект верхнего уровня.")

        image_size = data.get("image_size", {})
        if not isinstance(image_size, dict):
            image_size = {}

        image_width = image_size.get("width", data.get("image_width"))
        image_height = image_size.get("height", data.get("image_height"))
        raw_points = data.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("Поле 'points' должно содержать список точек.")

        return cls(
            image_path=str(data.get("image_path", "")),
            image_width=_require_int(image_width, "image_width"),
            image_height=_require_int(image_height, "image_height"),
            line_color=normalize_line_color(data.get("line_color"), CONTOUR_LINE_COLOR),
            closed=bool(data.get("closed", True)),
            version=int(data.get("version", 1)),
            points=[Point.from_dict(item) for item in raw_points],
        )


def _optional_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _normalize_project_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip("/")


PROJECT_IMAGE_FILE_METADATA_KEYS = ("added_at", "captured_at")
PROJECT_IMAGE_LOCATION_KEYS = (
    "illumination",
    "humidity",
    "wind_speed",
    "wind_direction",
    "latitude",
    "longitude",
)
PROJECT_IMAGE_DETAILS_KEYS = ("diagnosis", "notes")

PROJECT_IMAGE_METADATA_DEFAULTS: dict[str, str] = {
    "added_at": "",
    "captured_at": "",
    "illumination": "",
    "humidity": "",
    "wind_speed": "",
    "wind_direction": "",
    "latitude": "",
    "longitude": "",
    "diagnosis": "Не указано",
    "notes": "",
}


@dataclass(slots=True)
class ProjectInfo:
    general_info: str = ""

    def __post_init__(self) -> None:
        self.general_info = str(self.general_info or "")

    def to_dict(self) -> dict[str, str]:
        return {"general_info": self.general_info}

    @classmethod
    def from_dict(cls, data: Any) -> ProjectInfo:
        if not isinstance(data, dict):
            data = {}
        return cls(general_info=str(data.get("general_info", "")))


def default_project_image_metadata(
    *,
    added_at: str = "",
    captured_at: str = "",
) -> dict[str, str]:
    metadata = dict(PROJECT_IMAGE_METADATA_DEFAULTS)
    metadata["added_at"] = str(added_at or "")
    metadata["captured_at"] = str(captured_at or "")
    return metadata


@dataclass(slots=True)
class ProjectImageFileInfo:
    id: str
    path: str
    display_name: str
    image_width: int | None = None
    image_height: int | None = None
    added_at: str = ""
    captured_at: str = ""
    crop_reviewed: bool = False

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.path = _normalize_project_path(self.path)
        self.display_name = str(self.display_name or self.path.rsplit("/", 1)[-1])
        self.image_width = _optional_positive_int(self.image_width)
        self.image_height = _optional_positive_int(self.image_height)
        self.added_at = str(self.added_at or "")
        self.captured_at = str(self.captured_at or "")
        self.crop_reviewed = self.crop_reviewed is True

        if not self.id:
            raise ValueError("Запись изображения должна содержать id.")
        if not self.path:
            raise ValueError("Запись изображения должна содержать относительный путь.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "display_name": self.display_name,
            "image_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "added_at": self.added_at,
            "captured_at": self.captured_at,
            "crop_reviewed": self.crop_reviewed,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProjectImageFileInfo:
        if not isinstance(data, dict):
            raise ValueError("Группа 'file' в записи изображения должна быть объектом.")

        image_size = data.get("image_size", {})
        if not isinstance(image_size, dict):
            image_size = {}

        return cls(
            id=str(data.get("id", "")),
            path=str(data.get("path", "")),
            display_name=str(data.get("display_name", "")),
            image_width=_optional_positive_int(image_size.get("width")),
            image_height=_optional_positive_int(image_size.get("height")),
            added_at=str(data.get("added_at", "")),
            captured_at=str(data.get("captured_at", "")),
            crop_reviewed=data.get("crop_reviewed") is True,
        )


@dataclass(slots=True)
class ProjectImageContourInfo:
    annotation: Annotation | None = None
    annotation_error: str | None = None
    raw_annotation: Any | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"annotation": None}
        if self.annotation is not None:
            payload["annotation"] = self.annotation.to_dict()
        elif self.raw_annotation is not None:
            payload["annotation"] = self.raw_annotation
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> ProjectImageContourInfo:
        if not isinstance(data, dict):
            raise ValueError(
                "Группа 'contour' в записи изображения должна быть объектом."
            )

        annotation_payload = data.get("annotation")
        annotation = None
        annotation_error = None
        raw_annotation = None
        if annotation_payload is not None:
            raw_annotation = annotation_payload
            try:
                annotation = Annotation.from_dict(annotation_payload)
                raw_annotation = None
            except ValueError as exc:
                annotation_error = str(exc)

        return cls(
            annotation=annotation,
            annotation_error=annotation_error,
            raw_annotation=raw_annotation,
        )


@dataclass(slots=True)
class ProjectImageLocationInfo:
    illumination: str = ""
    humidity: str = ""
    wind_speed: str = ""
    wind_direction: str = ""
    latitude: str = ""
    longitude: str = ""

    def __post_init__(self) -> None:
        for key in PROJECT_IMAGE_LOCATION_KEYS:
            setattr(self, key, str(getattr(self, key) or ""))

    def to_dict(self) -> dict[str, str]:
        return {key: str(getattr(self, key)) for key in PROJECT_IMAGE_LOCATION_KEYS}

    @classmethod
    def from_dict(cls, data: Any) -> ProjectImageLocationInfo:
        if not isinstance(data, dict):
            data = {}
        return cls(
            **{key: str(data.get(key, "")) for key in PROJECT_IMAGE_LOCATION_KEYS}
        )


@dataclass(slots=True)
class ProjectImageDetailsInfo:
    diagnosis: str = "Не указано"
    notes: str = ""

    def __post_init__(self) -> None:
        self.diagnosis = "Не указано" if self.diagnosis is None else str(self.diagnosis)
        self.notes = str(self.notes or "")

    def to_dict(self) -> dict[str, str]:
        return {
            "diagnosis": self.diagnosis,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProjectImageDetailsInfo:
        if not isinstance(data, dict):
            data = {}
        return cls(
            diagnosis=str(data.get("diagnosis", "Не указано")),
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True, init=False)
class ProjectImageRecord:
    file: ProjectImageFileInfo
    contour: ProjectImageContourInfo
    location: ProjectImageLocationInfo
    details: ProjectImageDetailsInfo
    calibration: ImageCalibration | None
    calibration_error: str | None
    raw_calibration: Any | None
    measurements: ProjectImageMeasurements
    extra_groups: dict[str, Any]

    def __init__(
        self,
        *,
        id: str | None = None,
        relative_path: str | None = None,
        display_name: str | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        annotation: Annotation | None = None,
        annotation_error: str | None = None,
        metadata: dict[str, Any] | None = None,
        raw_annotation: Any | None = None,
        file: ProjectImageFileInfo | None = None,
        contour: ProjectImageContourInfo | None = None,
        location: ProjectImageLocationInfo | None = None,
        details: ProjectImageDetailsInfo | None = None,
        calibration: ImageCalibration | None = None,
        calibration_error: str | None = None,
        raw_calibration: Any | None = None,
        measurements: ProjectImageMeasurements | None = None,
        extra_groups: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata if isinstance(metadata, dict) else {}
        self.file = file or ProjectImageFileInfo(
            id=str(id or ""),
            path=str(relative_path or ""),
            display_name=str(display_name or ""),
            image_width=image_width,
            image_height=image_height,
            added_at=str(metadata.get("added_at", "")),
            captured_at=str(metadata.get("captured_at", "")),
        )
        self.contour = contour or ProjectImageContourInfo(
            annotation=annotation,
            annotation_error=annotation_error,
            raw_annotation=raw_annotation,
        )
        self.location = location or ProjectImageLocationInfo(
            **{key: str(metadata.get(key, "")) for key in PROJECT_IMAGE_LOCATION_KEYS}
        )
        self.details = details or ProjectImageDetailsInfo(
            diagnosis=str(metadata.get("diagnosis", "Не указано")),
            notes=str(metadata.get("notes", "")),
        )
        self.calibration = calibration
        self.calibration_error = calibration_error
        self.raw_calibration = raw_calibration
        self.measurements = measurements or ProjectImageMeasurements()
        self.extra_groups = dict(extra_groups or {})
        extra_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in PROJECT_IMAGE_METADATA_DEFAULTS
        }
        if extra_metadata:
            self.extra_groups["metadata_extra"] = extra_metadata

    @property
    def id(self) -> str:
        return self.file.id

    @id.setter
    def id(self, value: str) -> None:
        self.file.id = str(value or "").strip()

    @property
    def relative_path(self) -> str:
        return self.file.path

    @relative_path.setter
    def relative_path(self, value: str) -> None:
        self.file.path = _normalize_project_path(value)

    @property
    def display_name(self) -> str:
        return self.file.display_name

    @display_name.setter
    def display_name(self, value: str) -> None:
        self.file.display_name = str(value or "")

    @property
    def image_width(self) -> int | None:
        return self.file.image_width

    @image_width.setter
    def image_width(self, value: Any) -> None:
        self.file.image_width = _optional_positive_int(value)

    @property
    def image_height(self) -> int | None:
        return self.file.image_height

    @image_height.setter
    def image_height(self, value: Any) -> None:
        self.file.image_height = _optional_positive_int(value)

    @property
    def crop_reviewed(self) -> bool:
        return self.file.crop_reviewed

    @crop_reviewed.setter
    def crop_reviewed(self, value: bool) -> None:
        self.file.crop_reviewed = value is True

    @property
    def annotation(self) -> Annotation | None:
        return self.contour.annotation

    @annotation.setter
    def annotation(self, value: Annotation | None) -> None:
        self.contour.annotation = value

    @property
    def annotation_error(self) -> str | None:
        return self.contour.annotation_error

    @annotation_error.setter
    def annotation_error(self, value: str | None) -> None:
        self.contour.annotation_error = value

    @property
    def raw_annotation(self) -> Any | None:
        return self.contour.raw_annotation

    @raw_annotation.setter
    def raw_annotation(self, value: Any | None) -> None:
        self.contour.raw_annotation = value

    @property
    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = default_project_image_metadata(
            added_at=self.file.added_at,
            captured_at=self.file.captured_at,
        )
        for key in PROJECT_IMAGE_LOCATION_KEYS:
            metadata[key] = getattr(self.location, key)
        for key in PROJECT_IMAGE_DETAILS_KEYS:
            metadata[key] = getattr(self.details, key)
        extra_metadata = self.extra_groups.get("metadata_extra", {})
        if isinstance(extra_metadata, dict):
            metadata.update(extra_metadata)
        return metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        if not isinstance(value, dict):
            value = {}
        for key in PROJECT_IMAGE_METADATA_DEFAULTS:
            self.set_metadata_value(
                key, value.get(key, PROJECT_IMAGE_METADATA_DEFAULTS[key])
            )
        extra_metadata = {
            key: field_value
            for key, field_value in value.items()
            if key not in PROJECT_IMAGE_METADATA_DEFAULTS
        }
        if extra_metadata:
            self.extra_groups["metadata_extra"] = extra_metadata
        else:
            self.extra_groups.pop("metadata_extra", None)

    def set_metadata_value(self, key: str, value: Any) -> None:
        text = str(value or "")
        if key in PROJECT_IMAGE_FILE_METADATA_KEYS:
            setattr(self.file, key, text)
            return
        if key in PROJECT_IMAGE_LOCATION_KEYS:
            setattr(self.location, key, text)
            return
        if key == "diagnosis":
            self.details.diagnosis = text
            return
        if key == "notes":
            self.details.notes = text
            return
        extra_metadata = self.extra_groups.setdefault("metadata_extra", {})
        if not isinstance(extra_metadata, dict):
            extra_metadata = {}
            self.extra_groups["metadata_extra"] = extra_metadata
        extra_metadata[key] = value

    def has_annotation(self) -> bool:
        return self.annotation is not None

    def point_count(self) -> int:
        if self.annotation is None:
            return 0
        return len(self.annotation.points)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file": self.file.to_dict(),
            "contour": self.contour.to_dict(),
            "location": self.location.to_dict(),
            "details": self.details.to_dict(),
        }
        if self.calibration is not None:
            payload["calibration"] = self.calibration.to_dict()
        elif self.raw_calibration is not None:
            payload["calibration"] = self.raw_calibration
        if self.measurements.has_data():
            payload["measurements"] = self.measurements.to_dict()
        payload.update(self.extra_groups)
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> ProjectImageRecord:
        if not isinstance(data, dict):
            raise ValueError("Запись изображения в проекте должна быть объектом.")

        calibration_payload = data.get("calibration")
        calibration = None
        calibration_error = None
        raw_calibration = None
        if calibration_payload is not None:
            raw_calibration = calibration_payload
            try:
                calibration = ImageCalibration.from_dict(calibration_payload)
                raw_calibration = None
            except ValueError as exc:
                calibration_error = str(exc)

        measurements = ProjectImageMeasurements.from_dict(data.get("measurements", {}))

        known_groups = {
            "file",
            "contour",
            "location",
            "details",
            "calibration",
            "measurements",
        }
        return cls(
            file=ProjectImageFileInfo.from_dict(data.get("file")),
            contour=ProjectImageContourInfo.from_dict(data.get("contour", {})),
            location=ProjectImageLocationInfo.from_dict(data.get("location", {})),
            details=ProjectImageDetailsInfo.from_dict(data.get("details", {})),
            calibration=calibration,
            calibration_error=calibration_error,
            raw_calibration=raw_calibration,
            measurements=measurements,
            extra_groups={
                key: value for key, value in data.items() if key not in known_groups
            },
        )


@dataclass(slots=True)
class ProjectDocument:
    name: str
    images: list[ProjectImageRecord]
    version: int = PROJECT_FORMAT_VERSION
    images_dir: str = "images"
    project_info: ProjectInfo = field(default_factory=ProjectInfo)

    def __post_init__(self) -> None:
        self.name = str(self.name or "project").strip() or "project"
        self.version = _optional_positive_int(self.version) or PROJECT_FORMAT_VERSION
        self.images_dir = (
            _normalize_project_path(self.images_dir or "images") or "images"
        )
        if not isinstance(self.project_info, ProjectInfo):
            self.project_info = ProjectInfo.from_dict(self.project_info)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "images_dir": self.images_dir,
            "project_info": self.project_info.to_dict(),
            "images": [record.to_dict() for record in self.images],
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProjectDocument:
        if not isinstance(data, dict):
            raise ValueError("JSON проекта должен содержать объект верхнего уровня.")

        raw_images = data.get("images", [])
        if not isinstance(raw_images, list):
            raise ValueError("Поле 'images' должно содержать список изображений.")

        images: list[ProjectImageRecord] = []
        for item in raw_images:
            try:
                images.append(ProjectImageRecord.from_dict(item))
            except ValueError:
                continue

        return cls(
            name=str(data.get("name") or data.get("project_name") or "project"),
            version=_optional_positive_int(data.get("version"))
            or PROJECT_FORMAT_VERSION,
            images_dir=str(data.get("images_dir") or "images"),
            project_info=ProjectInfo.from_dict(data.get("project_info", {})),
            images=images,
        )
