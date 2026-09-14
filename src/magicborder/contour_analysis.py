from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from .histograms import (
    HistogramPlotData,
    channel_counts,
    hsv_histogram_from_counts,
    lab_histogram_from_counts,
    lms_histogram_from_values,
    rgb_histogram_from_counts,
    rgb_to_lms,
    yuv_histogram_from_counts,
)
from .models import Point

ContourSignature = tuple[tuple[float, float], ...]

_BIN_VALUES = np.arange(256, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ContourColorStats:
    mean_rgb: tuple[int, int, int]
    mean_lab: tuple[int, int, int]
    mean_hsv: tuple[int, int, int]
    mean_yuv: tuple[int, int, int]
    mean_lms: tuple[int, int, int]
    pixel_count: int


@dataclass(frozen=True, slots=True)
class ContourHistogramData:
    rgb: HistogramPlotData | None
    lab: HistogramPlotData | None
    hsv: HistogramPlotData | None
    yuv: HistogramPlotData | None
    lms: HistogramPlotData | None


@dataclass(frozen=True, slots=True)
class ContourAnalysis:
    stats: ContourColorStats
    histograms: ContourHistogramData


@dataclass(frozen=True, slots=True)
class ContourColorSums:
    """Суммы значений пикселей контура в каждом цветовом пространстве.

    Хранятся суммы, а не средние: суммы разных изображений складываются,
    и среднее по проекту получается взвешенным по числу пикселей. Объект
    маленький (несколько векторов из 3 чисел), поэтому его удобно кэшировать
    по файлу вместо того, чтобы заново декодировать изображение.
    """

    rgb: np.ndarray
    lab: np.ndarray
    hsv: np.ndarray
    yuv: np.ndarray
    lms: np.ndarray
    lms_max: np.ndarray
    pixel_count: int


@dataclass(frozen=True, slots=True)
class _ConvertedPixels:
    """Пиксели контура, переведённые в каждое пространство ровно один раз.

    Раньше Lab/HSV/YUV/LMS считались дважды: для статистики и для гистограмм.
    """

    rgb_counts: np.ndarray
    lab_counts: np.ndarray
    hsv_counts: np.ndarray
    yuv_counts: np.ndarray
    lms: np.ndarray
    pixel_count: int

    @classmethod
    def from_rgb_pixels(cls, rgb_pixels: np.ndarray) -> _ConvertedPixels:
        # Столбец N×1×3 — формат, который cvtColor принимает без копирования.
        column = rgb_pixels.reshape((-1, 1, 3))
        return cls(
            rgb_counts=channel_counts(rgb_pixels),
            lab_counts=_cv_channel_counts(column, cv2.COLOR_RGB2LAB),
            hsv_counts=_cv_channel_counts(column, cv2.COLOR_RGB2HSV),
            yuv_counts=_cv_channel_counts(column, cv2.COLOR_RGB2YUV),
            lms=rgb_to_lms(rgb_pixels),
            pixel_count=int(rgb_pixels.shape[0]),
        )

    def sums(self) -> ContourColorSums:
        # Суммы 8-битных каналов получаются из счётчиков точно (в целых),
        # без float32-накопления по миллионам пикселей.
        rgb_sum = self.rgb_counts @ _BIN_VALUES
        lab_sum = self.lab_counts @ _BIN_VALUES
        hsv_sum = self.hsv_counts @ _BIN_VALUES
        yuv_sum = self.yuv_counts @ _BIN_VALUES
        count = float(self.pixel_count)
        return ContourColorSums(
            rgb=rgb_sum,
            # Та же шкала, что в lab_values_from_rgb_pixels: L 0..100, a/b со сдвигом -128.
            lab=np.array(
                [
                    lab_sum[0] * (100.0 / 255.0),
                    lab_sum[1] - 128.0 * count,
                    lab_sum[2] - 128.0 * count,
                ]
            ),
            # H у OpenCV 0..179, в градусах это ×2.
            hsv=np.array([hsv_sum[0] * 2.0, hsv_sum[1], hsv_sum[2]]),
            yuv=yuv_sum,
            lms=self.lms.sum(axis=0, dtype=np.float64),
            lms_max=self.lms.max(axis=0).astype(np.float64),
            pixel_count=self.pixel_count,
        )

    def histograms(self) -> ContourHistogramData:
        return ContourHistogramData(
            rgb=rgb_histogram_from_counts(self.rgb_counts, self.pixel_count),
            lab=lab_histogram_from_counts(self.lab_counts, self.pixel_count),
            hsv=hsv_histogram_from_counts(self.hsv_counts, self.pixel_count),
            yuv=yuv_histogram_from_counts(self.yuv_counts, self.pixel_count),
            lms=lms_histogram_from_values(self.lms),
        )


def contour_signature(points: list[Point]) -> ContourSignature:
    return tuple(
        (round(float(point.x), 3), round(float(point.y), 3)) for point in points
    )


def contour_rgb_pixels_from_points(
    rgb_array: np.ndarray, points: list[Point]
) -> np.ndarray:
    if len(points) < 3:
        return np.empty((0, 3), dtype=np.uint8)

    mask = contour_mask_from_points(rgb_array.shape[:2], points)
    pixels = rgb_array[mask > 0]
    return np.ascontiguousarray(pixels.reshape((-1, 3)))


def contour_mask_from_points(shape: tuple[int, ...], points: list[Point]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(points) < 3:
        return mask
    polygon = np.array(
        [[int(round(point.x)), int(round(point.y))] for point in points],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def flatten_background_outside_contour(
    rgb_array: np.ndarray, points: list[Point]
) -> np.ndarray:
    """Возвращает копию кадра с белым фоном за пределами контура.

    Одна копия массива. Раньше в GUI-потоке их было три, и ещё одна появлялась
    при повторном чтении сохранённого файла.
    """
    if len(points) < 3:
        raise ValueError("Сначала постройте или загрузите контур.")
    result = np.array(rgb_array, dtype=np.uint8, copy=True)
    mask = contour_mask_from_points(result.shape, points)
    result[mask == 0] = 255
    return result


def build_contour_analysis(
    rgb_array: np.ndarray, points: list[Point]
) -> ContourAnalysis | None:
    pixels = contour_rgb_pixels_from_points(rgb_array, points)
    if pixels.size == 0:
        return None

    converted = _ConvertedPixels.from_rgb_pixels(pixels)
    return ContourAnalysis(
        stats=color_stats_from_sums(converted.sums()),
        histograms=converted.histograms(),
    )


def contour_color_sums(
    rgb_array: np.ndarray, points: list[Point]
) -> ContourColorSums | None:
    """Суммы для статистики проекта: без построения гистограмм."""
    pixels = contour_rgb_pixels_from_points(rgb_array, points)
    if pixels.size == 0:
        return None
    return _ConvertedPixels.from_rgb_pixels(pixels).sums()


def combine_color_sums(parts: Iterable[ContourColorSums]) -> ContourColorSums | None:
    combined: ContourColorSums | None = None
    for part in parts:
        if part.pixel_count <= 0:
            continue
        if combined is None:
            combined = part
            continue
        combined = ContourColorSums(
            rgb=combined.rgb + part.rgb,
            lab=combined.lab + part.lab,
            hsv=combined.hsv + part.hsv,
            yuv=combined.yuv + part.yuv,
            lms=combined.lms + part.lms,
            lms_max=np.maximum(combined.lms_max, part.lms_max),
            pixel_count=combined.pixel_count + part.pixel_count,
        )
    return combined


def color_stats_from_sums(sums: ContourColorSums) -> ContourColorStats:
    count = max(1, sums.pixel_count)
    return ContourColorStats(
        mean_rgb=_rounded_mean(sums.rgb, count),
        mean_lab=_rounded_mean(sums.lab, count),
        mean_hsv=_rounded_mean(sums.hsv, count),
        mean_yuv=_rounded_mean(sums.yuv, count),
        mean_lms=_mean_lms_values_from_total(sums.lms, sums.lms_max, count),
        pixel_count=sums.pixel_count,
    )


def build_contour_color_stats(rgb_pixels: np.ndarray) -> ContourColorStats:
    return color_stats_from_sums(_ConvertedPixels.from_rgb_pixels(rgb_pixels).sums())


def lab_values_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    lab_pixels = cv2.cvtColor(
        rgb_pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2LAB
    ).reshape((-1, 3))
    lab_pixels = lab_pixels.astype(np.float32)
    return np.column_stack(
        (
            lab_pixels[:, 0] * (100.0 / 255.0),
            lab_pixels[:, 1] - 128.0,
            lab_pixels[:, 2] - 128.0,
        )
    )


def hsv_values_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    hsv_pixels = cv2.cvtColor(
        rgb_pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2HSV
    ).reshape((-1, 3))
    hsv_values = hsv_pixels.astype(np.float32)
    hsv_values[:, 0] *= 2.0
    return hsv_values


def yuv_values_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    yuv_pixels = cv2.cvtColor(
        rgb_pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2YUV
    ).reshape((-1, 3))
    return yuv_pixels.astype(np.float32)


def mean_lms_values_from_total(
    lms_total: np.ndarray,
    lms_max: np.ndarray,
    pixel_count: int,
) -> tuple[int, int, int]:
    return _mean_lms_values_from_total(lms_total, lms_max, pixel_count)


def _cv_channel_counts(column: np.ndarray, conversion: int) -> np.ndarray:
    converted = cv2.cvtColor(column, conversion).reshape((-1, 3))
    return channel_counts(converted)


def _rounded_mean(total: np.ndarray, pixel_count: int) -> tuple[int, int, int]:
    mean_values = np.rint(total / pixel_count).astype(int)
    return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])


def _mean_lms_values_from_total(
    lms_total: np.ndarray,
    lms_max: np.ndarray,
    pixel_count: int,
) -> tuple[int, int, int]:
    max_values = np.maximum(lms_max, 1e-9)
    normalized_mean = (lms_total / max(1, pixel_count)) / max_values * 255.0
    mean_values = np.rint(normalized_mean).astype(int)
    mean_values = np.clip(mean_values, 0, 255)
    return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])
