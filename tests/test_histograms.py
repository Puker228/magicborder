from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt5.QtGui import QColor

from magicborder import histograms
from magicborder.histograms import (
    HistogramCanvas,
    HistogramPanel,
    HistogramPlotData,
    HistogramSeries,
    _ensure_png_suffix,
    _short_count,
    build_hsv_histogram,
    build_lab_histogram,
    build_lms_histogram,
    build_rgb_histogram,
    build_yuv_histogram,
    normalize_lms_for_display,
    rgb_to_lms,
)

BUILDERS = (
    build_rgb_histogram,
    build_lab_histogram,
    build_hsv_histogram,
    build_yuv_histogram,
    build_lms_histogram,
)


def _pixels(*colors: tuple[int, int, int]) -> np.ndarray:
    return np.array(colors, dtype=np.uint8)


def _plot_data(*, series_count: int = 3, max_value: int = 10) -> HistogramPlotData:
    palette = ("#ff5b63", "#35c46f", "#4aa3ff")
    series = []
    for index in range(series_count):
        values = np.zeros(256, dtype=np.int64)
        values[index * 10] = max_value
        series.append(
            HistogramSeries(
                name=f"Канал {index}",
                values=values,
                color=QColor(palette[index % len(palette)]),
            )
        )
    return HistogramPlotData(
        series=tuple(series),
        x_ticks=((0, "0"), (128, "128"), (255, "255")),
        x_label="Тестовая шкала",
        sample_count=1234,
    )


class TestEmptyInput:
    @pytest.mark.parametrize("builder", BUILDERS)
    def test_empty_pixels_return_none(self, builder) -> None:
        assert builder(np.empty((0, 3), dtype=np.uint8)) is None


class TestBuildRgbHistogram:
    def test_series_shape_and_names(self) -> None:
        data = build_rgb_histogram(_pixels((10, 20, 30), (10, 20, 30), (255, 0, 0)))

        assert data is not None
        assert [item.name for item in data.series] == ["R", "G", "B"]
        assert all(len(item.values) == 256 for item in data.series)
        assert data.sample_count == 3

    def test_counts_match_channel_values(self) -> None:
        data = build_rgb_histogram(_pixels((10, 20, 30), (10, 20, 30)))

        assert data is not None
        red, green, blue = data.series
        assert red.values[10] == 2
        assert green.values[20] == 2
        assert blue.values[30] == 2


class TestBuildLabHistogram:
    def test_lightness_is_scaled_to_0_100(self) -> None:
        data = build_lab_histogram(_pixels((255, 255, 255), (255, 255, 255)))

        assert data is not None
        lightness = data.series[0]
        assert lightness.name == "L"
        # L = 255 в OpenCV -> 100.0 -> последний бин диапазона (0, 100).
        assert lightness.values[-1] == 2
        assert lightness.values[:-1].sum() == 0

    def test_a_and_b_are_shifted_by_128(self) -> None:
        data = build_lab_histogram(_pixels((128, 128, 128), (128, 128, 128)))

        assert data is not None
        _, a_series, b_series = data.series
        assert (a_series.name, b_series.name) == ("a", "b")
        # Нейтральный серый: a = b = 128 -> 0.0 -> бин 128 диапазона (-128, 128).
        assert a_series.values[128] == 2
        assert b_series.values[128] == 2

    def test_declared_scale_is_mentioned_in_label(self) -> None:
        data = build_lab_histogram(_pixels((10, 20, 30)))

        assert data is not None
        assert data.x_label == "Шкала: L 0..100; a,b -128..127"
        assert data.x_ticks == ((0, "0"), (128, "128"), (255, "255"))


class TestBuildHsvHistogram:
    @pytest.mark.parametrize(
        ("color", "expected_bin"),
        [
            ((255, 0, 0), 0),  # hue 0
            ((0, 255, 0), 85),  # hue 60 -> round(60 * 255 / 179)
            ((0, 0, 255), 171),  # hue 120 -> round(120 * 255 / 179)
        ],
    )
    def test_hue_is_rescaled_from_179_to_255(
        self,
        color: tuple[int, int, int],
        expected_bin: int,
    ) -> None:
        data = build_hsv_histogram(_pixels(color))

        assert data is not None
        assert data.series[0].name == "H"
        assert data.series[0].values[expected_bin] == 1

    def test_rescaled_hue_stays_within_0_255(self) -> None:
        # Полная развёртка оттенков через HSV -> RGB, чтобы поймать верхнюю границу.
        import cv2

        hsv = np.stack(
            [
                np.arange(180, dtype=np.uint8),
                np.full(180, 255, np.uint8),
                np.full(180, 255, np.uint8),
            ],
            axis=1,
        ).reshape((-1, 1, 3))
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape((-1, 3))

        data = build_hsv_histogram(rgb)

        assert data is not None
        assert len(data.series[0].values) == 256
        assert int(data.series[0].values.sum()) == 180


class TestBuildYuvHistogram:
    def test_series_names_and_ticks(self) -> None:
        data = build_yuv_histogram(_pixels((10, 20, 30)))

        assert data is not None
        assert [item.name for item in data.series] == ["Y", "U", "V"]
        assert data.x_ticks == (
            (0, "0"),
            (64, "64"),
            (128, "128"),
            (192, "192"),
            (255, "255"),
        )
        assert data.x_label == "Y яркость; U,V цветность, нейтраль около 128"


class TestRgbToLms:
    def test_black_pixel_is_zero(self) -> None:
        assert np.allclose(rgb_to_lms(_pixels((0, 0, 0))), 0.0)

    def test_white_pixel_matches_reference_values(self) -> None:
        lms = rgb_to_lms(_pixels((255, 255, 255)))

        assert lms[0] == pytest.approx([0.941428, 1.040317, 1.089532], rel=1e-4)

    def test_both_srgb_linearisation_branches(self) -> None:
        # 10/255 = 0.0392 попадает в линейную ветку, 128/255 = 0.502 — в степенную.
        pixels = _pixels((10, 10, 10), (128, 128, 128))
        rgb = pixels.astype(np.float32) / 255.0
        expected_linear = np.where(
            rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
        )

        assert expected_linear[0, 0] == pytest.approx(10 / 255 / 12.92, rel=1e-5)
        assert expected_linear[1, 0] == pytest.approx(
            ((128 / 255 + 0.055) / 1.055) ** 2.4, rel=1e-5
        )

        lms = rgb_to_lms(pixels)
        assert lms[1, 0] > lms[0, 0]

    def test_output_is_clipped_to_non_negative(self) -> None:
        rng = np.random.default_rng(17)
        pixels = rng.integers(0, 256, size=(512, 3), dtype=np.uint8)

        lms = rgb_to_lms(pixels)

        assert (lms >= 0.0).all()


class TestNormalizeLmsForDisplay:
    def test_zero_channel_maximum_is_guarded(self) -> None:
        normalized = normalize_lms_for_display(np.zeros((4, 3), dtype=np.float32))

        assert normalized.dtype == np.uint8
        assert not normalized.any()

    def test_values_are_clipped_and_cast_to_uint8(self) -> None:
        lms = np.array([[0.0, 0.5, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        normalized = normalize_lms_for_display(lms)

        assert normalized.dtype == np.uint8
        assert normalized.max() == 255
        assert normalized.min() == 0
        assert normalized[0, 1] == 128

    def test_lms_histogram_uses_normalised_channels(self) -> None:
        data = build_lms_histogram(_pixels((255, 255, 255), (0, 0, 0)))

        assert data is not None
        assert [item.name for item in data.series] == ["L", "M", "S"]
        for item in data.series:
            assert item.values[255] == 1
            assert item.values[0] == 1


def _reference_histogram_values(name: str, pixels: np.ndarray) -> list[np.ndarray]:
    """Прежняя попиксельная реализация гистограмм — эталон для оптимизированной."""
    column = pixels.reshape((-1, 1, 3))
    if name == "rgb":
        return [np.bincount(pixels[:, c], minlength=256)[:256] for c in range(3)]
    if name == "lab":
        lab = cv2.cvtColor(column, cv2.COLOR_RGB2LAB).reshape((-1, 3))
        lab = lab.astype(np.float32)
        return [
            np.histogram(lab[:, 0] * (100.0 / 255.0), bins=256, range=(0.0, 100.0))[0],
            np.histogram(lab[:, 1] - 128.0, bins=256, range=(-128.0, 128.0))[0],
            np.histogram(lab[:, 2] - 128.0, bins=256, range=(-128.0, 128.0))[0],
        ]
    if name == "hsv":
        hsv = cv2.cvtColor(column, cv2.COLOR_RGB2HSV).reshape((-1, 3))
        hue = np.rint(hsv[:, 0].astype(np.float32) * (255.0 / 179.0))
        hue = np.clip(hue, 0, 255).astype(np.uint8)
        return [
            np.bincount(hue, minlength=256)[:256],
            np.bincount(hsv[:, 1], minlength=256)[:256],
            np.bincount(hsv[:, 2], minlength=256)[:256],
        ]
    yuv = cv2.cvtColor(column, cv2.COLOR_RGB2YUV).reshape((-1, 3))
    return [np.bincount(yuv[:, c], minlength=256)[:256] for c in range(3)]


def _every_rgb_color() -> np.ndarray:
    channels = np.meshgrid(*(np.arange(256, dtype=np.uint8),) * 3, indexing="ij")
    return np.stack(channels, axis=-1).reshape((-1, 3))


class TestOptimizedHistogramsMatchReference:
    @pytest.mark.parametrize("name", ["rgb", "lab", "hsv", "yuv"])
    def test_counts_match_per_pixel_reference_for_every_color(self, name: str) -> None:
        pixels = _every_rgb_color()
        builder = getattr(histograms, f"build_{name}_histogram")

        data = builder(pixels)

        assert data is not None
        for series, expected in zip(
            data.series, _reference_histogram_values(name, pixels), strict=True
        ):
            assert np.array_equal(series.values, expected)

    def test_lut_lms_matches_float_formula_for_every_color(self) -> None:
        pixels = _every_rgb_color()

        # float-вход идёт по прежнему пути с pow, uint8 — через таблицу.
        assert np.array_equal(rgb_to_lms(pixels), rgb_to_lms(pixels.astype(np.float32)))


class TestShortCount:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (999, "999"),
            (1_000, "1.0K"),
            (1_500, "1.5K"),
            (9_999, "10.0K"),
            (10_000, "10K"),
            (15_000, "15K"),
            (999_999, "1000K"),
            (1_000_000, "1.0M"),
            (2_300_000, "2.3M"),
        ],
    )
    def test_thresholds(self, value: int, expected: str) -> None:
        assert _short_count(value) == expected


class TestEnsurePngSuffix:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("histogram", "histogram.png"),
            ("histogram.jpg", "histogram.png"),
            ("histogram.png", "histogram.png"),
            ("histogram.PNG", "histogram.PNG"),
        ],
    )
    def test_suffix_normalisation(self, source: str, expected: str) -> None:
        assert _ensure_png_suffix(Path(source)).name == expected


class TestHistogramCanvas:
    def test_plot_data_lifecycle(self, qapp) -> None:
        canvas = HistogramCanvas()

        assert canvas.has_plot_data() is False

        canvas.set_plot_data(_plot_data())
        assert canvas.has_plot_data() is True

        canvas.clear_plot_data("Нет данных для гистограммы.")
        assert canvas.has_plot_data() is False
        assert canvas._empty_message == "Нет данных для гистограммы."

    def test_save_png_writes_file(self, qapp, tmp_path: Path) -> None:
        canvas = HistogramCanvas()
        canvas.resize(320, 200)
        canvas.set_plot_data(_plot_data())
        output_path = tmp_path / "histogram.png"

        canvas.save_png(output_path)

        assert output_path.is_file()
        assert output_path.stat().st_size > 0

    def test_save_png_failure_raises_value_error(self, qapp, tmp_path: Path) -> None:
        canvas = HistogramCanvas()
        canvas.resize(120, 90)

        with pytest.raises(ValueError, match="Не удалось сохранить гистограмму"):
            canvas.save_png(tmp_path / "нет" / "такой" / "папки.png")

    def test_paint_event_renders_empty_state(self, qapp) -> None:
        canvas = HistogramCanvas()
        canvas.resize(240, 120)

        pixmap = canvas.grab()

        assert not pixmap.isNull()

    @pytest.mark.parametrize(
        ("width", "height", "expected_margins"),
        [
            (240, 120, (36, 18, 10, 34)),
            (400, 300, (48, 24, 14, 46)),
            (400, 120, (36, 18, 10, 34)),
        ],
    )
    def test_plot_margins_branches(
        self,
        qapp,
        width: int,
        height: int,
        expected_margins: tuple[int, int, int, int],
    ) -> None:
        canvas = HistogramCanvas()
        canvas.resize(width, height)

        assert canvas._plot_margins() == expected_margins
        assert not canvas.grab().isNull()

    def test_paint_event_survives_all_zero_series(self, qapp) -> None:
        canvas = HistogramCanvas()
        canvas.resize(400, 300)
        zero_series = HistogramSeries(
            name="Z",
            values=np.zeros(256, dtype=np.int64),
            color=QColor("#ff0000"),
        )
        canvas.set_plot_data(
            HistogramPlotData(
                series=(zero_series,),
                x_ticks=((0, "0"),),
                x_label="Пусто",
                sample_count=0,
            )
        )

        image = canvas.grab().toImage()

        assert image.width() == 400
        # Кривая не рисуется: в области графика нет пикселей цвета серии.
        assert not any(
            image.pixelColor(x, y).name() == "#ff0000"
            for x in range(60, 380, 7)
            for y in range(40, 250, 7)
        )

    def test_legend_is_truncated_on_narrow_widget(self, qapp) -> None:
        data = _plot_data(series_count=3)

        def legend_colors(width: int) -> set[str]:
            canvas = HistogramCanvas()
            canvas.resize(width, 220)
            canvas.set_plot_data(data)
            image = canvas.grab().toImage()
            return {
                image.pixelColor(x, y).name()
                for x in range(width)
                for y in range(0, 18)
            }

        wide_colors = legend_colors(600)
        narrow_colors = legend_colors(220)

        assert {"#ff5b63", "#35c46f", "#4aa3ff"} <= wide_colors
        assert "#4aa3ff" not in narrow_colors


class TestHistogramPanel:
    def test_save_button_follows_histogram_state(self, qapp) -> None:
        panel = HistogramPanel("RGB", "rgb.png")

        assert panel._save_button.isEnabled() is False

        panel.set_histogram(_plot_data())
        assert panel._save_button.isEnabled() is True

        panel.clear_histogram("Нет контура.")
        assert panel._save_button.isEnabled() is False

    def test_default_file_name_uses_static_value(self, qapp) -> None:
        panel = HistogramPanel("RGB", "rgb.png")

        assert panel.default_file_name() == "rgb.png"

    def test_default_file_name_uses_provider(self, qapp) -> None:
        panel = HistogramPanel(
            "RGB",
            "rgb.png",
            default_file_name_provider=lambda: "leaf_rgb_histogram.png",
        )

        assert panel.default_file_name() == "leaf_rgb_histogram.png"

    def test_save_is_skipped_without_plot_data(
        self,
        qapp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        panel = HistogramPanel("RGB", "rgb.png")
        calls: list[str] = []
        monkeypatch.setattr(
            histograms.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: calls.append("dialog") or ("", ""),
        )

        panel._save_histogram()

        assert calls == []

    def test_cancelled_dialog_writes_nothing(
        self,
        qapp,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        panel = HistogramPanel("RGB", str(tmp_path / "rgb.png"))
        panel.set_histogram(_plot_data())
        monkeypatch.setattr(
            histograms.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: ("", ""),
        )

        panel._save_histogram()

        assert list(tmp_path.iterdir()) == []

    def test_accepted_dialog_writes_png(
        self,
        qapp,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        panel = HistogramPanel("RGB", "rgb.png")
        panel.canvas.resize(320, 200)
        panel.set_histogram(_plot_data())
        target = tmp_path / "saved"
        monkeypatch.setattr(
            histograms.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(target), ""),
        )

        panel._save_histogram()

        assert (tmp_path / "saved.png").is_file()

    def test_save_error_shows_critical_message(
        self,
        qapp,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        panel = HistogramPanel("RGB", "rgb.png")
        panel.set_histogram(_plot_data())
        shown: list[tuple[str, str]] = []

        def fail_save(_path: Path) -> None:
            raise ValueError("Не удалось сохранить гистограмму в PNG.")

        monkeypatch.setattr(panel.canvas, "save_png", fail_save)
        monkeypatch.setattr(
            histograms.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(tmp_path / "rgb.png"), ""),
        )
        monkeypatch.setattr(
            histograms.QMessageBox,
            "critical",
            staticmethod(
                lambda _parent, title, message: shown.append((title, message))
            ),
        )

        panel._save_histogram()

        assert shown == [
            ("Ошибка сохранения", "Не удалось сохранить гистограмму в PNG.")
        ]
