from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from .io_utils import SUPPORTED_RASTER_SUFFIXES, captured_at_from_exif

LARGE_IMAGE_THRESHOLD = (2560, 1920)
JPEG_SUFFIXES = (".jpg", ".jpeg")
JPEG_QUALITY = 95
RESIZE_REDUCING_GAP = 3.0
_KEEP_MODES = ("RGB", "RGBA", "L", "LA")


@dataclass(frozen=True, slots=True)
class DownscalePreset:
    label: str
    long_side: int
    short_side: int


DOWNSCALE_PRESETS: tuple[DownscalePreset, ...] = (
    DownscalePreset("1920×1080 (Full HD)", 1920, 1080),
    DownscalePreset("1280×720 (HD)", 1280, 720),
    DownscalePreset("800×600 (SVGA)", 800, 600),
    DownscalePreset("600×450", 600, 450),
)
RECOMMENDED_PRESET_INDEX = 2


def read_image_size(path: str | Path) -> tuple[int, int]:
    """Возвращает размер изображения, читая только заголовок файла."""
    image_path = _checked_raster_path(path)
    try:
        with Image.open(image_path) as image:
            return image.size
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Не удалось распознать файл как растровое изображение."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Не удалось открыть изображение: {exc}") from exc


def read_image_header(path: str | Path) -> tuple[tuple[int, int], str]:
    """Размер и дата съёмки за одно открытие файла (раньше файл открывался дважды)."""
    image_path = _checked_raster_path(path)
    try:
        with Image.open(image_path) as image:
            return image.size, captured_at_from_exif(image.getexif())
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Не удалось распознать файл как растровое изображение."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Не удалось открыть изображение: {exc}") from exc


def verify_raster_image(path: str | Path) -> tuple[int, int]:
    """Полностью декодирует изображение без Qt-объектов и возвращает его размер."""
    image_path = _checked_raster_path(path)
    try:
        with Image.open(image_path) as image:
            image.load()
            return image.size
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Не удалось распознать файл как растровое изображение."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Не удалось открыть изображение: {exc}") from exc


def is_large_image(size: tuple[int, int]) -> bool:
    long_side, short_side = max(size), min(size)
    threshold_long, threshold_short = LARGE_IMAGE_THRESHOLD
    return long_side >= threshold_long or short_side >= threshold_short


def fit_size(size: tuple[int, int], preset: DownscalePreset) -> tuple[int, int]:
    """Вписывает размер в рамку пресета с учётом ориентации, без увеличения."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Некорректный размер изображения.")
    scale = min(
        preset.long_side / max(width, height),
        preset.short_side / min(width, height),
        1.0,
    )
    return max(1, round(width * scale)), max(1, round(height * scale))


def downscale_image_file(
    source: str | Path,
    destination: str | Path,
    preset: DownscalePreset,
) -> tuple[int, int]:
    """Сохраняет в destination уменьшенную копию source и возвращает её размер.

    Запись атомарная, поэтому source и destination могут совпадать.
    """
    source_path = _checked_raster_path(source)
    destination_path = Path(destination)
    try:
        with Image.open(source_path) as image:
            target_size = fit_size(image.size, preset)
            if image.format == "JPEG":
                image.draft(image.mode, target_size)
            image_info = dict(image.info)
            working = _resizable_image(image)
            if working.size != target_size:
                working = working.resize(
                    target_size,
                    Image.Resampling.LANCZOS,
                    reducing_gap=RESIZE_REDUCING_GAP,
                )
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Не удалось распознать файл как растровое изображение."
        ) from exc

    _save_atomically(working, destination_path, image_info)
    return working.size


def _checked_raster_path(path: str | Path) -> Path:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_RASTER_SUFFIXES:
        raise ValueError("Неподдерживаемый формат изображения.")
    return image_path


def _resizable_image(image: Image.Image) -> Image.Image:
    if image.mode in _KEEP_MODES:
        image.load()
        return image
    has_alpha = image.mode in ("PA", "RGBa", "La") or "transparency" in image.info
    return image.convert("RGBA" if has_alpha else "RGB")


def _save_atomically(
    image: Image.Image, destination: Path, image_info: dict[str, object]
) -> None:
    suffix = destination.suffix
    save_kwargs: dict[str, object] = {}
    icc_profile = image_info.get("icc_profile")
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    if suffix.lower() in JPEG_SUFFIXES:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        save_kwargs.update(quality=JPEG_QUALITY, subsampling=0)
        exif = image_info.get("exif")
        if exif:
            save_kwargs["exif"] = exif

    temp_path = destination.with_name(
        f".{destination.stem}.magicborder-{uuid4().hex}{suffix}"
    )
    try:
        image.save(temp_path, **save_kwargs)
        temp_path.replace(destination)
    except (OSError, ValueError):
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
