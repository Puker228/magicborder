from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# saxutils.escape/quoteattr only escape output, they do not parse XML.
from xml.sax.saxutils import escape, quoteattr  # nosec B406

import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError
from PyQt5.QtGui import QImage, QPixmap

from .models import Annotation, ProjectDocument

SUPPORTED_RASTER_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
PNG_FAST_COMPRESS_LEVEL = 1
JPEG_SAVE_QUALITY = 95


@dataclass(slots=True)
class LoadedImage:
    path: Path
    qimage: QImage
    pixmap: QPixmap
    rgb_array: np.ndarray
    width: int
    height: int


def image_open_filter() -> str:
    patterns = " ".join(f"*{suffix}" for suffix in SUPPORTED_RASTER_SUFFIXES)
    return f"Raster images ({patterns});;All files (*)"


def load_raster_image(path: str | Path) -> LoadedImage:
    image_path = _checked_raster_path(path)
    try:
        with Image.open(image_path) as pil_image:
            # Альфа нужна только для отображения прозрачных PNG/TIFF. У обычных
            # фотографий её нет, и промежуточный RGBA-кадр (+33% памяти и ещё
            # одна конверсия) не создаётся.
            rgba_image = pil_image.convert("RGBA") if _has_alpha(pil_image) else None
            rgb_image = pil_image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Не удалось распознать файл как растровое изображение."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Не удалось открыть изображение: {exc}") from exc

    # np.asarray забирает буфер PIL без второй копии; массив свежий, и больше
    # на него никто не ссылается, поэтому защитная копия не нужна.
    rgb_array = np.asarray(rgb_image, dtype=np.uint8)
    return loaded_image_from_rgb_array(
        image_path, rgb_array, rgba_image=rgba_image, copy=False
    )


def load_rgb_array(path: str | Path) -> np.ndarray:
    """Декодирует только RGB-массив, без QImage/QPixmap.

    QPixmap можно создавать только в GUI-потоке, а для статистики он не нужен,
    поэтому эту функцию безопасно вызывать из рабочих потоков.
    """
    image_path = _checked_raster_path(path)
    try:
        with Image.open(image_path) as pil_image:
            rgb_image = (
                pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")
            )
            rgb_image.load()
            return np.asarray(rgb_image, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Не удалось распознать файл как растровое изображение."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Не удалось открыть изображение: {exc}") from exc


def save_rgb_array_atomically(rgb_array: np.ndarray, image_path: str | Path) -> None:
    """Сохраняет RGB-массив через временный файл, чтобы не испортить исходник при сбое."""
    target_path = Path(image_path)
    temp_path = target_path.with_name(
        f".{target_path.stem}.magicborder-{uuid4().hex}{target_path.suffix}"
    )
    save_kwargs: dict[str, object] = {}
    suffix = target_path.suffix.lower()
    if suffix == ".png":
        # Сжатие PNG без потерь на любом уровне; уровень 1 кодирует кадр
        # 1920×1080 в разы быстрее уровня 6 по умолчанию, файл чуть больше.
        save_kwargs["compress_level"] = PNG_FAST_COMPRESS_LEVEL
    elif suffix in (".jpg", ".jpeg"):
        save_kwargs["quality"] = JPEG_SAVE_QUALITY
    try:
        Image.fromarray(np.asarray(rgb_array, dtype=np.uint8), mode="RGB").save(
            temp_path, **save_kwargs
        )
        temp_path.replace(target_path)
    except (OSError, ValueError):
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def save_annotation(path: str | Path, annotation: Annotation) -> None:
    annotation_path = Path(path)
    payload = json.dumps(annotation.to_dict(), ensure_ascii=False, indent=2)
    annotation_path.write_text(payload, encoding="utf-8")


def load_annotation(path: str | Path) -> Annotation:
    annotation_path = Path(path)
    try:
        raw_text = annotation_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Файл аннотации не найден: {annotation_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON: {exc}") from exc
    return Annotation.from_dict(payload)


def save_project(path: str | Path, project: ProjectDocument) -> None:
    project_path = Path(path)
    project_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(project.to_dict(), ensure_ascii=False, indent=2)
    temp_path = project_path.with_name(f".{project_path.name}.tmp")
    temp_path.write_text(payload, encoding="utf-8")
    os.replace(temp_path, project_path)


def load_project(path: str | Path) -> ProjectDocument:
    project_path = Path(path)
    try:
        raw_text = project_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Файл проекта не найден: {project_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON проекта: {exc}") from exc
    return ProjectDocument.from_dict(payload)


def write_xlsx_table(
    output_path: str | Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    *,
    sheet_name: str,
    bold_rows: set[int] | None = None,
    cell_fills: dict[tuple[int, str], str] | None = None,
) -> None:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    normalized_cell_fills = _normalize_xlsx_cell_fills(cell_fills or {})
    fill_colors = list(dict.fromkeys(normalized_cell_fills.values()))
    archive_parts = {
        "[Content_Types].xml": _xlsx_content_types_xml(),
        "_rels/.rels": _xlsx_root_relationships_xml(),
        "docProps/app.xml": _xlsx_app_properties_xml(sheet_name),
        "docProps/core.xml": _xlsx_core_properties_xml(created_at),
        "xl/workbook.xml": _xlsx_workbook_xml(sheet_name),
        "xl/_rels/workbook.xml.rels": _xlsx_workbook_relationships_xml(),
        "xl/styles.xml": _xlsx_styles_xml(fill_colors),
        "xl/worksheets/sheet1.xml": _xlsx_sheet_xml(
            fieldnames,
            rows,
            bold_rows=bold_rows or set(),
            cell_fills=normalized_cell_fills,
            fill_colors=fill_colors,
        ),
    }
    with zipfile.ZipFile(
        Path(output_path), "w", compression=zipfile.ZIP_DEFLATED
    ) as workbook:
        for name, content in archive_parts.items():
            workbook.writestr(name, content)


def read_image_captured_at(path: str | Path) -> str:
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
    except (OSError, UnidentifiedImageError):
        return ""
    return captured_at_from_exif(exif)


def captured_at_from_exif(exif: Image.Exif | None) -> str:
    if not exif:
        return ""

    tag_by_name = {name: tag for tag, name in ExifTags.TAGS.items()}
    for tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        raw_value = exif.get(tag_by_name.get(tag_name))
        parsed_value = _parse_exif_datetime(raw_value)
        if parsed_value:
            return parsed_value
    return ""


def loaded_image_from_rgb_array(
    path: str | Path,
    rgb_array: np.ndarray,
    *,
    rgba_image: Image.Image | None = None,
    copy: bool = True,
) -> LoadedImage:
    image_path = Path(path).resolve()
    normalized_rgb = _normalize_rgb_array(rgb_array, copy=copy)

    if rgba_image is not None:
        qimage = _pil_rgba_to_qimage(rgba_image)
    else:
        qimage = _rgb_array_to_qimage(normalized_rgb)
    pixmap = QPixmap.fromImage(qimage)

    return LoadedImage(
        path=image_path,
        qimage=qimage,
        pixmap=pixmap,
        rgb_array=normalized_rgb,
        width=qimage.width(),
        height=qimage.height(),
    )


def _rgb_array_to_qimage(rgb_array: np.ndarray) -> QImage:
    height, width = rgb_array.shape[:2]
    # QImage поверх буфера numpy (без tobytes и без RGBA-кадра). copy() делает
    # QImage владельцем данных, иначе он ссылался бы на память массива.
    qimage = QImage(rgb_array.data, width, height, 3 * width, QImage.Format_RGB888)
    return qimage.copy()


def _pil_rgba_to_qimage(image: Image.Image) -> QImage:
    data = image.tobytes("raw", "RGBA")
    qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
    return qimage.copy()


def _xlsx_content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""


def _xlsx_root_relationships_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def _xlsx_app_properties_xml(sheet_name: str) -> str:
    title = escape(_clean_xml_text(sheet_name))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
    xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>MagicBorder</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>1</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr">
      <vt:lpstr>{title}</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>"""


def _xlsx_core_properties_xml(created_at: str) -> str:
    created = escape(_clean_xml_text(created_at))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:dcterms="http://purl.org/dc/terms/"
    xmlns:dcmitype="http://purl.org/dc/dcmitype/"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>MagicBorder</dc:creator>
  <cp:lastModifiedBy>MagicBorder</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>"""


def _xlsx_workbook_xml(sheet_name: str) -> str:
    quoted_sheet_name = quoteattr(_clean_xml_text(sheet_name))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name={quoted_sheet_name} sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""


def _xlsx_workbook_relationships_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _xlsx_styles_xml(fill_colors: list[str] | None = None) -> str:
    fill_colors = fill_colors or []
    color_fills_xml = "\n".join(
        (
            '    <fill><patternFill patternType="solid">'
            f'<fgColor rgb="{color}"/><bgColor indexed="64"/>'
            "</patternFill></fill>"
        )
        for color in fill_colors
    )
    fills_count = 2 + len(fill_colors)
    fill_xfs_xml = "\n".join(
        f'    <xf numFmtId="0" fontId="0" fillId="{2 + index}" borderId="0" xfId="0" applyFill="1"/>'
        for index, _color in enumerate(fill_colors)
    )
    bold_fill_xfs_xml = "\n".join(
        (
            f'    <xf numFmtId="0" fontId="1" fillId="{2 + index}" borderId="0" '
            'xfId="0" applyFont="1" applyFill="1"/>'
        )
        for index, _color in enumerate(fill_colors)
    )
    cell_xfs_count = 2 + (2 * len(fill_colors))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="{fills_count}">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
{color_fills_xml}
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="{cell_xfs_count}">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
{fill_xfs_xml}
{bold_fill_xfs_xml}
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="0"/>
  <tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>
</styleSheet>"""


def _xlsx_sheet_xml(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    *,
    bold_rows: set[int],
    cell_fills: dict[tuple[int, str], str],
    fill_colors: list[str],
) -> str:
    fill_style_ids = {color: 2 + index for index, color in enumerate(fill_colors)}
    bold_fill_style_ids = {
        color: 2 + len(fill_colors) + index for index, color in enumerate(fill_colors)
    }
    table_rows: list[tuple[list[str], list[int]]] = [
        (fieldnames, [0 for _field in fieldnames])
    ]
    for row_index, row in enumerate(rows):
        is_bold = row_index in bold_rows
        values = [row.get(field, "") for field in fieldnames]
        style_ids: list[int] = []
        for field in fieldnames:
            fill_color = cell_fills.get((row_index, field))
            if fill_color:
                style_ids.append(
                    bold_fill_style_ids[fill_color]
                    if is_bold
                    else fill_style_ids[fill_color]
                )
            else:
                style_ids.append(1 if is_bold else 0)
        table_rows.append((values, style_ids))
    rows_xml = "\n".join(
        _xlsx_row_xml(row_values, row_index, style_ids=style_ids)
        for row_index, (row_values, style_ids) in enumerate(table_rows, start=1)
    )
    last_column = _xlsx_column_name(len(fieldnames))
    last_row = max(1, len(table_rows))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:{last_column}{last_row}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
      <selection pane="bottomLeft" activeCell="A2" sqref="A2"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>
    <col min="1" max="1" width="38" customWidth="1"/>
    <col min="2" max="3" width="28" customWidth="1"/>
    <col min="4" max="10" width="18" customWidth="1"/>
  </cols>
  <sheetData>
{rows_xml}
  </sheetData>
  <autoFilter ref="A1:{last_column}{last_row}"/>
  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>
</worksheet>"""


def _xlsx_row_xml(values: list[str], row_index: int, *, style_ids: list[int]) -> str:
    cells_xml = "".join(
        _xlsx_cell_xml(
            value,
            _xlsx_column_name(column_index),
            row_index,
            style_id=style_ids[column_index - 1]
            if column_index - 1 < len(style_ids)
            else 0,
        )
        for column_index, value in enumerate(values, start=1)
    )
    return f'    <row r="{row_index}">{cells_xml}</row>'


def _xlsx_cell_xml(
    value: str, column_name: str, row_index: int, *, style_id: int = 0
) -> str:
    cell_ref = f"{column_name}{row_index}"
    text = escape(_clean_xml_text(str(value)))
    style_attribute = f' s="{style_id}"' if style_id else ""
    return f'<c r="{cell_ref}"{style_attribute} t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def _xlsx_column_name(column_index: int) -> str:
    name = ""
    while column_index:
        column_index, remainder = divmod(column_index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _normalize_xlsx_cell_fills(
    cell_fills: dict[tuple[int, str], str],
) -> dict[tuple[int, str], str]:
    normalized: dict[tuple[int, str], str] = {}
    for cell_key, color in cell_fills.items():
        normalized_color = _normalize_xlsx_color(color)
        if normalized_color:
            normalized[cell_key] = normalized_color
    return normalized


def _normalize_xlsx_color(color: str) -> str | None:
    value = str(color).strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 6:
        value = f"FF{value}"
    if len(value) != 8:
        return None
    if any(char not in "0123456789abcdefABCDEF" for char in value):
        return None
    return value.upper()


def _clean_xml_text(value: str) -> str:
    return "".join(char for char in value if _is_valid_xml_character(ord(char)))


def _is_valid_xml_character(codepoint: int) -> bool:
    return (
        codepoint in (0x09, 0x0A, 0x0D)
        or 0x20 <= codepoint <= 0xD7FF
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )


def _parse_exif_datetime(value: object) -> str:
    if not value:
        return ""
    raw_text = str(value).strip()
    if not raw_text:
        return ""
    for format_string in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw_text, format_string).isoformat(
                timespec="seconds"
            )
        except ValueError:
            continue
    return raw_text


def _normalize_rgb_array(rgb_array: np.ndarray, *, copy: bool = True) -> np.ndarray:
    source = np.asarray(rgb_array)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Ожидается RGB-массив изображения с тремя каналами.")
    # ascontiguousarray копирует только при другом dtype или разрывной памяти.
    array = np.ascontiguousarray(source, dtype=np.uint8)
    # Защитная копия нужна, только если буфер общий с массивом вызывающего кода.
    if copy and np.shares_memory(array, source):
        array = array.copy()
    # Массив только для чтения: его можно отдавать в рабочие потоки без копий,
    # не опасаясь, что кто-то изменит пиксели во время расчёта.
    array.flags.writeable = False
    return array


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        "transparency" in image.info
    )


def _checked_raster_path(path: str | Path) -> Path:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_RASTER_SUFFIXES:
        raise ValueError("Неподдерживаемый формат изображения.")
    return image_path
