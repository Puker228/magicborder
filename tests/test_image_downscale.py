from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from magicborder.image_downscale import (
    DOWNSCALE_PRESETS,
    DownscalePreset,
    downscale_image_file,
    fit_size,
    is_large_image,
    read_image_size,
    verify_raster_image,
)
from magicborder.io_utils import read_image_captured_at

FULL_HD, HD = DOWNSCALE_PRESETS


class TestSizeRules:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            ((2559, 1919), False),
            ((1920, 1080), False),
            ((2560, 1000), True),
            ((1000, 2560), True),
            ((2000, 1920), True),
            ((3840, 2160), True),
        ],
    )
    def test_is_large_image(self, size: tuple[int, int], expected: bool) -> None:
        assert is_large_image(size) is expected

    @pytest.mark.parametrize(
        ("size", "preset", "expected"),
        [
            ((6000, 4000), FULL_HD, (1620, 1080)),
            ((4000, 6000), FULL_HD, (1080, 1620)),
            ((5000, 2000), FULL_HD, (1920, 768)),
            ((3840, 2160), FULL_HD, (1920, 1080)),
            ((3840, 2160), HD, (1280, 720)),
            ((2560, 1920), FULL_HD, (1440, 1080)),
            ((800, 600), FULL_HD, (800, 600)),
            ((10000, 10), HD, (1280, 1)),
        ],
    )
    def test_fit_size_keeps_aspect_ratio_without_upscaling(
        self,
        size: tuple[int, int],
        preset: DownscalePreset,
        expected: tuple[int, int],
    ) -> None:
        assert fit_size(size, preset) == expected

    def test_fit_size_rejects_empty_size(self) -> None:
        with pytest.raises(ValueError):
            fit_size((0, 10), FULL_HD)


class TestReadAndVerify:
    def test_read_image_size_reads_header(self, tmp_path: Path) -> None:
        path = tmp_path / "leaf.png"
        Image.new("RGB", (30, 20)).save(path)

        assert read_image_size(path) == (30, 20)
        assert verify_raster_image(path) == (30, 20)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_image_size(tmp_path / "нет.png")

    def test_unsupported_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "leaf.gif"
        path.write_bytes(b"GIF89a")

        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            read_image_size(path)

    def test_broken_file(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")

        with pytest.raises(ValueError, match="Не удалось распознать"):
            read_image_size(path)
        with pytest.raises(ValueError, match="Не удалось распознать"):
            verify_raster_image(path)

    def test_verify_detects_truncated_data(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.png"
        Image.effect_noise((64, 64), 50).convert("RGB").save(path)
        path.write_bytes(path.read_bytes()[:200])

        assert read_image_size(path) == (64, 64)
        with pytest.raises(ValueError, match="Не удалось открыть изображение"):
            verify_raster_image(path)


class TestDownscaleImageFile:
    def test_png_is_resized_with_same_colors(self, tmp_path: Path) -> None:
        source = tmp_path / "big.png"
        Image.new("RGB", (3000, 2000), (120, 80, 40)).save(source)
        destination = tmp_path / "out" / "big.png"
        destination.parent.mkdir()

        assert downscale_image_file(source, destination, FULL_HD) == (1620, 1080)

        with Image.open(destination) as image:
            assert image.size == (1620, 1080)
            assert image.mode == "RGB"
            assert image.getpixel((810, 540)) == (120, 80, 40)
        with Image.open(source) as image:
            assert image.size == (3000, 2000)

    def test_jpeg_keeps_exif_and_quality(self, tmp_path: Path) -> None:
        source = tmp_path / "photo.jpg"
        exif = Image.Exif()
        exif[36867] = "2021:01:02 03:04:05"
        Image.new("RGB", (4000, 3000), (10, 200, 30)).save(
            source, exif=exif, quality=95
        )
        destination = tmp_path / "small.jpg"

        assert downscale_image_file(source, destination, HD) == (960, 720)

        assert read_image_captured_at(destination) == "2021-01-02T03:04:05"
        with Image.open(destination) as image:
            red, green, blue = image.getpixel((480, 360))
        assert abs(red - 10) <= 3 and abs(green - 200) <= 3 and abs(blue - 30) <= 3

    def test_rgba_png_keeps_alpha(self, tmp_path: Path) -> None:
        source = tmp_path / "alpha.png"
        Image.new("RGBA", (2600, 2600), (1, 2, 3, 128)).save(source)

        downscale_image_file(source, source, HD)

        with Image.open(source) as image:
            assert image.mode == "RGBA"
            assert image.size == (720, 720)
            pixel = image.getpixel((10, 10))
        assert all(
            abs(actual - expected) <= 1
            for actual, expected in zip(pixel, (1, 2, 3, 128), strict=True)
        )

    def test_palette_image_is_converted_for_quality_resize(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "palette.png"
        Image.new("RGB", (2600, 1000), (200, 10, 10)).convert("P").save(source)

        assert downscale_image_file(source, source, HD) == (1280, 492)

        with Image.open(source) as image:
            assert image.mode == "RGB"

    def test_overwrite_in_place_leaves_no_temp_files(self, tmp_path: Path) -> None:
        source = tmp_path / "leaf.tif"
        Image.new("L", (2600, 2000), 77).save(source)

        downscale_image_file(source, source, FULL_HD)

        assert [path.name for path in tmp_path.iterdir()] == ["leaf.tif"]
        with Image.open(source) as image:
            assert image.size == (1404, 1080)
            assert image.getpixel((0, 0)) == 77

    def test_failed_save_removes_temp_file(self, tmp_path: Path) -> None:
        source = tmp_path / "leaf.png"
        Image.new("RGB", (2600, 2000)).save(source)
        destination = tmp_path / "missing-dir" / "leaf.png"

        with pytest.raises(OSError):
            downscale_image_file(source, destination, FULL_HD)

        assert [path.name for path in tmp_path.iterdir()] == ["leaf.png"]

    def test_broken_source(self, tmp_path: Path) -> None:
        source = tmp_path / "broken.jpg"
        source.write_bytes(b"not an image")

        with pytest.raises(ValueError, match="Не удалось распознать"):
            downscale_image_file(source, tmp_path / "out.jpg", FULL_HD)
