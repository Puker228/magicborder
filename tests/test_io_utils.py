from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from PIL import Image

from magicborder import io_utils
from magicborder.io_utils import (
    SUPPORTED_RASTER_SUFFIXES,
    _clean_xml_text,
    _normalize_xlsx_color,
    _parse_exif_datetime,
    _xlsx_column_name,
    image_open_filter,
    load_annotation,
    load_project,
    load_raster_image,
    load_rgb_array,
    loaded_image_from_rgb_array,
    read_image_captured_at,
    save_annotation,
    save_project,
    save_rgb_array_atomically,
    write_xlsx_table,
)
from magicborder.models import Annotation, Point, ProjectDocument, ProjectImageRecord

SHEET_NAMESPACE = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _sheet_xml(path: Path) -> str:
    with zipfile.ZipFile(path) as workbook:
        return workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")


def _sheet_rows(path: Path) -> list[list[str]]:
    root = ElementTree.fromstring(_sheet_xml(path))
    rows: list[list[str]] = []
    for row in root.findall("s:sheetData/s:row", SHEET_NAMESPACE):
        cells: list[str] = []
        for cell in row.findall("s:c", SHEET_NAMESPACE):
            text_node = cell.find("s:is/s:t", SHEET_NAMESPACE)
            cells.append("" if text_node is None else (text_node.text or ""))
        rows.append(cells)
    return rows


def _cell_styles(path: Path) -> dict[str, str]:
    root = ElementTree.fromstring(_sheet_xml(path))
    return {
        cell.attrib["r"]: cell.attrib.get("s", "0")
        for cell in root.findall("s:sheetData/s:row/s:c", SHEET_NAMESPACE)
    }


def _annotation() -> Annotation:
    return Annotation(
        image_path="изображения/лист.png",
        image_width=40,
        image_height=30,
        points=[Point(1, 1), Point(20, 1), Point(20, 20)],
    )


class TestLoadRasterImage:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Файл не найден"):
            load_raster_image(tmp_path / "нет.png")

    @pytest.mark.parametrize("suffix", [".gif", ".webp"])
    def test_unsupported_suffix_is_rejected(self, tmp_path: Path, suffix: str) -> None:
        path = tmp_path / f"image{suffix}"
        path.write_bytes(b"whatever")

        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            load_raster_image(path)

    def test_corrupted_file_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.png"
        path.write_bytes(b"not really a png")

        with pytest.raises(
            ValueError, match="не удалось распознать|Не удалось распознать"
        ):
            load_raster_image(path)

    def test_os_error_is_wrapped_into_value_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "image.png"
        Image.new("RGB", (4, 4)).save(path)

        def raise_os_error(*_args, **_kwargs):
            raise OSError("диск недоступен")

        monkeypatch.setattr(io_utils.Image, "open", raise_os_error)

        with pytest.raises(ValueError, match="диск недоступен"):
            load_raster_image(path)

    @pytest.mark.parametrize("suffix", list(SUPPORTED_RASTER_SUFFIXES))
    def test_every_supported_suffix_loads(
        self,
        qapp,
        tmp_path: Path,
        suffix: str,
    ) -> None:
        path = tmp_path / f"image{suffix}"
        Image.new("RGB", (12, 8), (10, 20, 30)).save(path)

        loaded = load_raster_image(path)

        assert (loaded.width, loaded.height) == (12, 8)
        assert loaded.rgb_array.shape == (8, 12, 3)
        assert tuple(loaded.rgb_array[0, 0]) == (10, 20, 30)

    @pytest.mark.parametrize("mode", ["P", "L", "RGBA"])
    def test_non_rgb_modes_are_converted(self, qapp, tmp_path: Path, mode: str) -> None:
        path = tmp_path / "image.png"
        Image.new("RGB", (9, 7), (200, 100, 50)).convert(mode).save(path)

        loaded = load_raster_image(path)

        assert loaded.rgb_array.dtype == np.uint8
        assert loaded.rgb_array.shape == (7, 9, 3)
        assert (loaded.width, loaded.height) == (9, 7)


class TestLightweightImageLoading:
    def test_loaded_rgb_array_is_read_only(self, qapp, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        Image.new("RGB", (6, 4), (10, 20, 30)).save(path)

        loaded = load_raster_image(path)

        # Массив можно без копии отдавать в рабочие потоки: изменить его нельзя.
        assert loaded.rgb_array.flags.writeable is False
        with pytest.raises(ValueError):
            loaded.rgb_array[0, 0] = (1, 2, 3)

    def test_opaque_image_pixmap_matches_file_pixels(
        self, qapp, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(5)
        rgb = rng.integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
        path = tmp_path / "noise.png"
        Image.fromarray(rgb).save(path)

        loaded = load_raster_image(path)

        image = loaded.pixmap.toImage()
        for x, y in ((0, 0), (12, 8), (6, 4)):
            color = image.pixelColor(x, y)
            assert (color.red(), color.green(), color.blue()) == tuple(rgb[y, x])

    def test_transparent_png_keeps_alpha_on_screen(self, qapp, tmp_path: Path) -> None:
        path = tmp_path / "alpha.png"
        Image.new("RGBA", (4, 4), (200, 100, 50, 0)).save(path)

        loaded = load_raster_image(path)

        assert loaded.qimage.pixelColor(1, 1).alpha() == 0
        assert tuple(loaded.rgb_array[1, 1]) == (200, 100, 50)

    @pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
    def test_load_rgb_array_matches_pil_conversion(
        self, tmp_path: Path, mode: str
    ) -> None:
        path = tmp_path / "image.png"
        Image.new("RGB", (5, 3), (200, 100, 50)).convert(mode).save(path)

        rgb_array = load_rgb_array(path)

        with Image.open(path) as image:
            expected = np.asarray(image.convert("RGB"))
        assert rgb_array.dtype == np.uint8
        assert np.array_equal(rgb_array, expected)

    def test_load_rgb_array_validates_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_rgb_array(tmp_path / "нет.png")
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not a png")
        with pytest.raises(ValueError, match="Не удалось распознать"):
            load_rgb_array(broken)

    @pytest.mark.parametrize("suffix", [".png", ".tif", ".bmp"])
    def test_atomic_save_is_lossless_and_leaves_no_temp_file(
        self, tmp_path: Path, suffix: str
    ) -> None:
        rng = np.random.default_rng(9)
        rgb = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
        path = tmp_path / f"image{suffix}"
        Image.new("RGB", (16, 12)).save(path)

        save_rgb_array_atomically(rgb, path)

        with Image.open(path) as image:
            assert np.array_equal(np.asarray(image.convert("RGB")), rgb)
        assert [item.name for item in tmp_path.iterdir()] == [path.name]

    def test_atomic_save_failure_keeps_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "image.png"
        Image.new("RGB", (4, 4), (1, 2, 3)).save(path)

        def failing_save(*_args, **_kwargs):
            raise OSError("диск переполнен")

        monkeypatch.setattr(io_utils.Image.Image, "save", failing_save)

        with pytest.raises(OSError, match="диск переполнен"):
            save_rgb_array_atomically(np.zeros((4, 4, 3), dtype=np.uint8), path)
        monkeypatch.undo()
        with Image.open(path) as image:
            assert image.getpixel((0, 0)) == (1, 2, 3)
        assert [item.name for item in tmp_path.iterdir()] == ["image.png"]


class TestImageOpenFilter:
    def test_filter_lists_every_supported_suffix(self) -> None:
        filter_text = image_open_filter()

        for suffix in SUPPORTED_RASTER_SUFFIXES:
            assert f"*{suffix}" in filter_text
        assert "All files (*)" in filter_text


class TestLoadedImageFromRgbArray:
    def test_non_uint8_dtype_is_converted(self, qapp) -> None:
        rgb_array = np.full((4, 5, 3), 128.0, dtype=np.float64)

        loaded = loaded_image_from_rgb_array(Path("a.png"), rgb_array)

        assert loaded.rgb_array.dtype == np.uint8
        assert (loaded.rgb_array == 128).all()

    def test_non_contiguous_array_is_accepted(self, qapp) -> None:
        source = np.arange(4 * 5 * 6, dtype=np.uint8).reshape((4, 5, 6))
        non_contiguous = source[:, :, ::2]
        assert not non_contiguous.flags["C_CONTIGUOUS"]

        loaded = loaded_image_from_rgb_array(Path("a.png"), non_contiguous)

        assert loaded.rgb_array.shape == (4, 5, 3)
        assert np.array_equal(loaded.rgb_array, non_contiguous)

    @pytest.mark.parametrize(
        "shape",
        [(4, 5), (4, 5, 4)],
    )
    def test_wrong_shape_is_rejected(self, qapp, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="три канала|тремя каналами"):
            loaded_image_from_rgb_array(Path("a.png"), np.zeros(shape, dtype=np.uint8))

    def test_result_is_detached_copy(self, qapp) -> None:
        source = np.zeros((4, 5, 3), dtype=np.uint8)

        loaded = loaded_image_from_rgb_array(Path("a.png"), source)
        source[0, 0] = (255, 255, 255)

        assert tuple(loaded.rgb_array[0, 0]) == (0, 0, 0)


class TestProjectPersistence:
    def test_save_project_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        project_path = tmp_path / "project.json"

        save_project(project_path, ProjectDocument(name="p", images=[]))

        assert project_path.exists()
        assert list(tmp_path.glob(".*.tmp")) == []

    def test_save_project_creates_parent_directories(self, tmp_path: Path) -> None:
        project_path = tmp_path / "deep" / "nested" / "project.json"

        save_project(project_path, ProjectDocument(name="p", images=[]))

        assert project_path.is_file()

    def test_save_project_overwrites_existing_file(self, tmp_path: Path) -> None:
        project_path = tmp_path / "project.json"
        save_project(project_path, ProjectDocument(name="first", images=[]))

        save_project(
            project_path,
            ProjectDocument(
                name="second",
                images=[ProjectImageRecord(id="a", relative_path="images/a.png")],
            ),
        )

        document = load_project(project_path)
        assert document.name == "second"
        assert [record.id for record in document.images] == ["a"]

    def test_missing_project_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Файл проекта не найден"):
            load_project(tmp_path / "нет.json")

    def test_invalid_project_json_raises_value_error(self, tmp_path: Path) -> None:
        project_path = tmp_path / "project.json"
        project_path.write_text("{ broken", encoding="utf-8")

        with pytest.raises(ValueError, match="Некорректный JSON проекта"):
            load_project(project_path)


class TestAnnotationPersistence:
    def test_cyrillic_is_written_without_escapes(self, tmp_path: Path) -> None:
        annotation_path = tmp_path / "annotation.json"

        save_annotation(annotation_path, _annotation())

        raw_text = annotation_path.read_text(encoding="utf-8")
        assert "изображения/лист.png" in raw_text
        assert "\\u" not in raw_text

    def test_round_trip(self, tmp_path: Path) -> None:
        annotation_path = tmp_path / "annotation.json"
        save_annotation(annotation_path, _annotation())

        loaded = load_annotation(annotation_path)

        assert loaded.image_path == "изображения/лист.png"
        assert len(loaded.points) == 3

    def test_missing_annotation_file_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Файл аннотации не найден"):
            load_annotation(tmp_path / "нет.json")

    def test_invalid_annotation_json_raises_value_error(self, tmp_path: Path) -> None:
        annotation_path = tmp_path / "annotation.json"
        annotation_path.write_text("[oops", encoding="utf-8")

        with pytest.raises(ValueError, match="Некорректный JSON"):
            load_annotation(annotation_path)


class TestReadImageCapturedAt:
    def _jpeg_with_exif(self, path: Path, tags: dict[int, str]) -> None:
        exif = Image.Exif()
        for tag, value in tags.items():
            exif[tag] = value
        Image.new("RGB", (8, 8), (1, 2, 3)).save(path, exif=exif)

    def test_image_without_exif_returns_empty_string(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.png"
        Image.new("RGB", (8, 8)).save(path)

        assert read_image_captured_at(path) == ""

    def test_non_image_file_returns_empty_string(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("просто текст", encoding="utf-8")

        assert read_image_captured_at(path) == ""

    def test_missing_file_returns_empty_string(self, tmp_path: Path) -> None:
        assert read_image_captured_at(tmp_path / "нет.jpg") == ""

    def test_date_time_original_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.jpg"
        self._jpeg_with_exif(
            path,
            {
                36867: "2021:01:02 03:04:05",  # DateTimeOriginal
                36868: "2022:02:03 04:05:06",  # DateTimeDigitized
                306: "2023:03:04 05:06:07",  # DateTime
            },
        )

        assert read_image_captured_at(path) == "2021-01-02T03:04:05"

    def test_date_time_digitized_is_second_choice(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.jpg"
        self._jpeg_with_exif(
            path,
            {
                36868: "2022:02:03 04:05:06",
                306: "2023:03:04 05:06:07",
            },
        )

        assert read_image_captured_at(path) == "2022-02-03T04:05:06"

    def test_date_time_is_last_choice(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.jpg"
        self._jpeg_with_exif(path, {306: "2023:03:04 05:06:07"})

        assert read_image_captured_at(path) == "2023-03-04T05:06:07"


class TestParseExifDatetime:
    def test_colon_format(self) -> None:
        assert _parse_exif_datetime("2024:05:01 10:20:30") == "2024-05-01T10:20:30"

    def test_dash_format(self) -> None:
        assert _parse_exif_datetime("2024-05-01 10:20:30") == "2024-05-01T10:20:30"

    def test_unrecognized_text_is_returned_as_is(self) -> None:
        assert _parse_exif_datetime("вчера вечером") == "вчера вечером"

    @pytest.mark.parametrize("value", [None, "", "   ", 0])
    def test_blank_values_return_empty_string(self, value: object) -> None:
        assert _parse_exif_datetime(value) == ""


class TestXlsxColumnName:
    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (1, "A"),
            (26, "Z"),
            (27, "AA"),
            (52, "AZ"),
            (53, "BA"),
            (702, "ZZ"),
            (703, "AAA"),
        ],
    )
    def test_column_names(self, index: int, expected: str) -> None:
        assert _xlsx_column_name(index) == expected


class TestNormalizeXlsxColor:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("#a1b2c3", "FFA1B2C3"),
            ("a1b2c3", "FFA1B2C3"),
            ("ffa1b2c3", "FFA1B2C3"),
            ("#FFA1B2C3", "FFA1B2C3"),
        ],
    )
    def test_accepted_values(self, value: str, expected: str) -> None:
        assert _normalize_xlsx_color(value) == expected

    @pytest.mark.parametrize("value", ["#abc", "#gggggg", "zzzzzzzz", "", "#12345"])
    def test_rejected_values(self, value: str) -> None:
        assert _normalize_xlsx_color(value) is None


class TestCleanXmlText:
    @pytest.mark.parametrize("char", ["\x00", "\x0b", "\x1f"])
    def test_forbidden_control_characters_are_stripped(self, char: str) -> None:
        assert _clean_xml_text(f"a{char}b") == "ab"

    def test_allowed_whitespace_survives(self) -> None:
        assert _clean_xml_text("a\tb\nc\rd") == "a\tb\nc\rd"

    def test_astral_plane_character_survives(self) -> None:
        assert _clean_xml_text("лист 🍃") == "лист 🍃"


class TestWriteXlsxTable:
    def test_empty_rows_produce_header_only_workbook(self, tmp_path: Path) -> None:
        output_path = tmp_path / "table.xlsx"

        write_xlsx_table(output_path, ["Имя", "Значение"], [], sheet_name="Лист")

        assert zipfile.is_zipfile(output_path)
        assert _sheet_rows(output_path) == [["Имя", "Значение"]]

    def test_empty_fieldnames(self, tmp_path: Path) -> None:
        output_path = tmp_path / "table.xlsx"

        write_xlsx_table(output_path, [], [{"a": "b"}], sheet_name="Лист")

        assert _sheet_rows(output_path) == [[], []]

    def test_more_than_26_columns_use_two_letter_names(self, tmp_path: Path) -> None:
        output_path = tmp_path / "wide.xlsx"
        fieldnames = [f"col{index}" for index in range(53)]

        write_xlsx_table(output_path, fieldnames, [], sheet_name="Wide")

        sheet_xml = _sheet_xml(output_path)
        assert 'r="AA1"' in sheet_xml
        assert 'r="AZ1"' in sheet_xml
        assert 'r="BA1"' in sheet_xml
        assert '<dimension ref="A1:BA1"/>' in sheet_xml

    def test_bold_row_with_fill_uses_bold_fill_style(self, tmp_path: Path) -> None:
        output_path = tmp_path / "styled.xlsx"

        write_xlsx_table(
            output_path,
            ["a", "b"],
            [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}],
            sheet_name="Стили",
            bold_rows={0},
            cell_fills={(0, "a"): "#ff0000", (1, "b"): "#00ff00"},
        )

        styles = _cell_styles(output_path)
        # 2 цвета заливки: обычные стили 2 и 3, жирные с заливкой — 4 и 5.
        assert styles["A2"] == "4"  # жирная строка + заливка
        assert styles["B2"] == "1"  # жирная строка без заливки
        assert styles["B3"] == "3"  # заливка без жирного
        assert styles["A3"] == "0"

    def test_invalid_fill_colors_are_dropped(self, tmp_path: Path) -> None:
        output_path = tmp_path / "bad-fill.xlsx"

        write_xlsx_table(
            output_path,
            ["a"],
            [{"a": "1"}],
            sheet_name="Стили",
            cell_fills={(0, "a"): "#abc"},
        )

        assert _cell_styles(output_path)["A2"] == "0"

    def test_missing_key_becomes_empty_cell(self, tmp_path: Path) -> None:
        output_path = tmp_path / "sparse.xlsx"

        write_xlsx_table(output_path, ["a", "b"], [{"a": "1"}], sheet_name="Лист")

        assert _sheet_rows(output_path)[1] == ["1", ""]

    def test_special_characters_are_escaped(self, tmp_path: Path) -> None:
        output_path = tmp_path / "escaped.xlsx"

        write_xlsx_table(
            output_path,
            ["a"],
            [{"a": "R&D < 5"}],
            sheet_name="Лист",
        )

        assert "&amp;" in _sheet_xml(output_path)
        assert "&lt;" in _sheet_xml(output_path)
        assert _sheet_rows(output_path)[1] == ["R&D < 5"]

    def test_control_characters_are_removed_from_cells(self, tmp_path: Path) -> None:
        output_path = tmp_path / "control.xlsx"

        write_xlsx_table(output_path, ["a"], [{"a": "x\x00y\x0bz"}], sheet_name="Лист")

        assert _sheet_rows(output_path)[1] == ["xyz"]

    def test_sheet_name_is_quoted_in_workbook_xml(self, tmp_path: Path) -> None:
        output_path = tmp_path / "quoted.xlsx"

        write_xlsx_table(output_path, ["a"], [], sheet_name='R&D "лист"')

        with zipfile.ZipFile(output_path) as workbook:
            workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")

        assert "&amp;" in workbook_xml
        # quoteattr берёт одинарные кавычки, чтобы не экранировать двойные внутри значения.
        assert "'R&amp;D \"лист\"'" in workbook_xml
        root = ElementTree.fromstring(workbook_xml)
        sheet = root.find("s:sheets/s:sheet", SHEET_NAMESPACE)
        assert sheet is not None
        assert sheet.attrib["name"] == 'R&D "лист"'


def test_project_round_trip_keeps_json_readable(tmp_path: Path) -> None:
    project_path = tmp_path / "проект.json"
    save_project(project_path, ProjectDocument(name="проект", images=[]))

    payload = json.loads(project_path.read_text(encoding="utf-8"))

    assert payload["name"] == "проект"


def test_exif_without_date_tags_returns_empty_string(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[271] = "MagicBorder"  # Make, но ни одного тега с датой
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path, exif=exif)

    assert read_image_captured_at(path) == ""
