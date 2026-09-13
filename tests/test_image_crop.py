from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from magicborder.image_crop import (
    MIN_CROP_SIDE,
    CropBox,
    apply_crop_to_record,
    crop_image_file,
    crop_output_size,
    crop_point,
    full_crop_box,
    is_noop_crop,
    normalize_crop_box,
)
from magicborder.image_downscale import DOWNSCALE_PRESETS
from magicborder.models import (
    Annotation,
    ImageCalibration,
    Point,
    ProjectAngleMeasurement,
    ProjectImageMeasurements,
    ProjectImageRecord,
    ProjectSegmentMeasurement,
)

PRESET_800 = DOWNSCALE_PRESETS[2]


class TestNormalizeCropBox:
    def test_rounds_and_orders_coordinates(self) -> None:
        assert normalize_crop_box(30.6, 20.2, 10.4, 5.5, (100, 80)) == CropBox(
            10, 6, 21, 14
        )

    def test_clamps_to_image_bounds(self) -> None:
        assert normalize_crop_box(-15, -3, 140, 95, (100, 80)) == full_crop_box(
            (100, 80)
        )

    def test_enforces_minimum_side_inside_image(self) -> None:
        box = normalize_crop_box(98, 79, 99, 80, (100, 80))

        assert box.size == (MIN_CROP_SIDE, MIN_CROP_SIDE)
        assert box.right == 100
        assert box.bottom == 80

    def test_tiny_image_uses_its_own_size_as_minimum(self) -> None:
        assert normalize_crop_box(1, 1, 1, 1, (4, 3)).size == (4, 3)

    def test_rejects_empty_image(self) -> None:
        with pytest.raises(ValueError):
            normalize_crop_box(0, 0, 1, 1, (0, 10))


class TestOutputSize:
    def test_without_preset_keeps_box_size(self) -> None:
        assert crop_output_size(CropBox(0, 0, 3000, 1000), None) == (3000, 1000)

    def test_preset_fits_without_upscaling(self) -> None:
        assert crop_output_size(CropBox(0, 0, 3200, 2400), PRESET_800) == (800, 600)
        assert crop_output_size(CropBox(0, 0, 400, 300), PRESET_800) == (400, 300)

    def test_noop_crop_detection(self) -> None:
        size = (400, 300)
        assert is_noop_crop(full_crop_box(size), None, size)
        assert is_noop_crop(full_crop_box(size), PRESET_800, size)
        assert not is_noop_crop(CropBox(1, 0, 399, 300), None, size)
        assert not is_noop_crop(full_crop_box((3200, 2400)), PRESET_800, (3200, 2400))


class TestCropImageFile:
    def test_crops_rgb_png_in_place(self, tmp_path: Path) -> None:
        path = tmp_path / "leaf.png"
        image = Image.new("RGB", (40, 30), (0, 0, 255))
        image.paste((255, 0, 0), (10, 5, 30, 25))
        image.save(path)

        size = crop_image_file(path, path, CropBox(10, 5, 20, 20))

        assert size == (20, 20)
        with Image.open(path) as cropped:
            assert cropped.size == (20, 20)
            assert cropped.getpixel((0, 0)) == (255, 0, 0)
            assert cropped.getpixel((19, 19)) == (255, 0, 0)
        assert list(tmp_path.glob(".*magicborder-*")) == []

    def test_keeps_alpha_channel(self, tmp_path: Path) -> None:
        path = tmp_path / "leaf.png"
        Image.new("RGBA", (40, 30), (10, 20, 30, 40)).save(path)

        crop_image_file(path, path, CropBox(0, 0, 16, 16))

        with Image.open(path) as cropped:
            assert cropped.mode == "RGBA"
            assert cropped.getpixel((3, 3)) == (10, 20, 30, 40)

    def test_resizes_to_preset_and_keeps_exif(self, tmp_path: Path) -> None:
        source = tmp_path / "source.jpg"
        destination = tmp_path / "result.jpg"
        exif = Image.Exif()
        exif[0x0110] = "Test camera"
        Image.new("RGB", (3600, 2600), (90, 160, 60)).save(source, exif=exif)

        size = crop_image_file(
            source, destination, CropBox(100, 100, 3200, 2400), PRESET_800
        )

        assert size == (800, 600)
        with Image.open(destination) as result:
            assert result.size == (800, 600)
            assert result.getexif().get(0x0110) == "Test camera"
        with Image.open(source) as original:
            assert original.size == (3600, 2600)

    def test_rejects_box_outside_image(self, tmp_path: Path) -> None:
        path = tmp_path / "leaf.png"
        Image.new("RGB", (40, 30)).save(path)

        with pytest.raises(ValueError):
            crop_image_file(path, path, CropBox(30, 0, 20, 10))

        with Image.open(path) as unchanged:
            assert unchanged.size == (40, 30)


class TestGeometry:
    def test_crop_point_shifts_scales_and_clamps(self) -> None:
        box = CropBox(100, 50, 200, 100)

        assert crop_point(Point(150, 75), box, (200, 100)) == Point(50, 25)
        assert crop_point(Point(150, 75), box, (100, 50)) == Point(25, 12.5)
        assert crop_point(Point(0, 500), box, (200, 100)) == Point(0, 100)

    def test_apply_crop_to_record_moves_and_drops_degenerate_items(self) -> None:
        record = ProjectImageRecord(
            id="image-1",
            relative_path="images/leaf.png",
            image_width=400,
            image_height=300,
            annotation=Annotation(
                image_path="images/leaf.png",
                image_width=400,
                image_height=300,
                points=[Point(110, 60), Point(190, 60), Point(190, 140)],
                line_color="#123456",
            ),
            calibration=ImageCalibration(Point(120, 80), Point(160, 80), 10.0),
            measurements=ProjectImageMeasurements(
                angles=[
                    ProjectAngleMeasurement(
                        "kept", Point(110, 70), Point(150, 100), Point(190, 70)
                    ),
                    ProjectAngleMeasurement(
                        "outside", Point(0, 0), Point(10, 10), Point(20, 0)
                    ),
                ],
                segments=[
                    ProjectSegmentMeasurement(
                        "kept", Point(120, 90), Point(180, 90), name="length"
                    ),
                    ProjectSegmentMeasurement("outside", Point(0, 290), Point(50, 295)),
                ],
            ),
        )

        apply_crop_to_record(record, CropBox(100, 50, 100, 100), (50, 50))

        assert (record.image_width, record.image_height) == (50, 50)
        assert record.annotation is not None
        assert (record.annotation.image_width, record.annotation.image_height) == (
            50,
            50,
        )
        assert record.annotation.points == [Point(5, 5), Point(45, 5), Point(45, 45)]
        assert record.annotation.line_color == "#123456"
        assert record.calibration is not None
        assert record.calibration.start == Point(10, 15)
        assert record.calibration.end == Point(30, 15)
        assert record.calibration.length_mm == 10.0
        assert [angle.id for angle in record.measurements.angles] == ["kept"]
        assert record.measurements.angles[0].vertex == Point(25, 25)
        assert [segment.id for segment in record.measurements.segments] == ["kept"]
        assert record.measurements.segments[0].name == "length"

    def test_contour_outside_crop_is_removed(self) -> None:
        record = ProjectImageRecord(
            id="image-1",
            relative_path="images/leaf.png",
            annotation=Annotation(
                image_path="images/leaf.png",
                image_width=100,
                image_height=100,
                points=[Point(1, 1), Point(5, 1), Point(5, 5)],
            ),
            calibration=ImageCalibration(Point(1, 1), Point(4, 1), 1.0),
        )

        apply_crop_to_record(record, CropBox(50, 50, 50, 50), (50, 50))

        assert record.annotation is None
        assert record.calibration is None
