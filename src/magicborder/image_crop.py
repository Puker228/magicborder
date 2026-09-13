from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .image_downscale import (
    RESIZE_REDUCING_GAP,
    DownscalePreset,
    _checked_raster_path,
    _resizable_image,
    _save_atomically,
    fit_size,
)
from .models import Point, ProjectImageRecord

MIN_CROP_SIDE = 8
_DEGENERATE_LENGTH = 1e-6


@dataclass(frozen=True, slots=True)
class CropBox:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def is_full(self, image_size: tuple[int, int]) -> bool:
        return (self.left, self.top, self.width, self.height) == (0, 0, *image_size)


def full_crop_box(image_size: tuple[int, int]) -> CropBox:
    width, height = image_size
    return CropBox(0, 0, width, height)


def normalize_crop_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    image_size: tuple[int, int],
) -> CropBox:
    """Приводит произвольный прямоугольник к целым пикселям внутри изображения."""
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Некорректный размер изображения.")
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    x0, x1 = _clamped_span(left, right, image_width)
    y0, y1 = _clamped_span(top, bottom, image_height)
    return CropBox(x0, y0, x1 - x0, y1 - y0)


def crop_output_size(box: CropBox, preset: DownscalePreset | None) -> tuple[int, int]:
    if preset is None:
        return box.size
    return fit_size(box.size, preset)


def is_noop_crop(
    box: CropBox, preset: DownscalePreset | None, image_size: tuple[int, int]
) -> bool:
    return box.is_full(image_size) and crop_output_size(box, preset) == image_size


def crop_image_file(
    source: str | Path,
    destination: str | Path,
    box: CropBox,
    preset: DownscalePreset | None = None,
) -> tuple[int, int]:
    """Сохраняет в destination обрезанную (и при необходимости уменьшенную) копию.

    Запись атомарная, поэтому source и destination могут совпадать.
    """
    source_path = _checked_raster_path(source)
    try:
        with Image.open(source_path) as image:
            if box.right > image.width or box.bottom > image.height:
                raise ValueError("Область обрезки выходит за границы изображения.")
            image_info = dict(image.info)
            working = _resizable_image(image).crop(
                (box.left, box.top, box.right, box.bottom)
            )
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Не удалось распознать файл как растровое изображение."
        ) from exc

    target_size = crop_output_size(box, preset)
    if working.size != target_size:
        working = working.resize(
            target_size,
            Image.Resampling.LANCZOS,
            reducing_gap=RESIZE_REDUCING_GAP,
        )
    _save_atomically(working, Path(destination), image_info)
    return working.size


def crop_point(point: Point, box: CropBox, output_size: tuple[int, int]) -> Point:
    output_width, output_height = output_size
    scale_x = output_width / box.width
    scale_y = output_height / box.height
    x = (float(point.x) - box.left) * scale_x
    y = (float(point.y) - box.top) * scale_y
    return Point(
        min(max(x, 0.0), float(output_width)),
        min(max(y, 0.0), float(output_height)),
    )


def apply_crop_to_record(
    record: ProjectImageRecord, box: CropBox, output_size: tuple[int, int]
) -> None:
    """Переносит контур, калибровку и измерения записи в координаты обрезанного кадра.

    Точки за пределами области прижимаются к краю; ставшие вырожденными
    элементы удаляются.
    """
    output_width, output_height = output_size

    def move(point: Point) -> Point:
        return crop_point(point, box, output_size)

    annotation = record.annotation
    if annotation is not None:
        points = [move(point) for point in annotation.points]
        if _polygon_area(points) <= _DEGENERATE_LENGTH:
            record.annotation = None
        else:
            record.annotation = replace(
                annotation,
                image_width=output_width,
                image_height=output_height,
                points=points,
            )

    calibration = record.calibration
    if calibration is not None:
        start, end = move(calibration.start), move(calibration.end)
        if _distance(start, end) <= _DEGENERATE_LENGTH:
            record.calibration = None
        else:
            record.calibration = replace(calibration, start=start, end=end)

    measurements = record.measurements
    angles = []
    for angle in measurements.angles:
        first, vertex, second = (
            move(angle.first),
            move(angle.vertex),
            move(angle.second),
        )
        if (
            _distance(first, vertex) > _DEGENERATE_LENGTH
            and _distance(second, vertex) > _DEGENERATE_LENGTH
        ):
            angles.append(replace(angle, first=first, vertex=vertex, second=second))
    measurements.angles = angles

    segments = []
    for segment in measurements.segments:
        start, end = move(segment.start), move(segment.end)
        if _distance(start, end) > _DEGENERATE_LENGTH:
            segments.append(replace(segment, start=start, end=end))
    measurements.segments = segments

    record.image_width = output_width
    record.image_height = output_height


def _clamped_span(start: float, end: float, limit: int) -> tuple[int, int]:
    low = min(max(round(start), 0), limit)
    high = min(max(round(end), 0), limit)
    min_side = min(MIN_CROP_SIDE, limit)
    if high - low < min_side:
        high = min(low + min_side, limit)
        low = high - min_side
    return low, high


def _distance(first: Point, second: Point) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)


def _polygon_area(points: list[Point]) -> float:
    doubled = sum(
        current.x * following.y - following.x * current.y
        for current, following in zip(points, points[1:] + points[:1], strict=True)
    )
    return abs(doubled) / 2.0
