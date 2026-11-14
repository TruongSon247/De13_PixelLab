import base64
import struct
from pathlib import Path

import cv2
import numpy as np

ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageInputError(ValueError):
    pass


def detect_image_format(raw: bytes):
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if raw.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if raw.startswith(b"BM"):
        return "BMP"
    if raw.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "TIFF"
    return None


def _header_dimensions(raw: bytes):
    """Read dimensions from common image headers before full decode where possible."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        return struct.unpack(">II", raw[16:24])
    if raw.startswith(b"BM") and len(raw) >= 26:
        width, height = struct.unpack("<ii", raw[18:26])
        return abs(width), abs(height)
    if raw.startswith(b"\xff\xd8"):
        i = 2
        frame_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while i + 4 <= len(raw):
            while i < len(raw) and raw[i] != 0xFF:
                i += 1
            while i < len(raw) and raw[i] == 0xFF:
                i += 1
            if i >= len(raw):
                break
            marker = raw[i]; i += 1
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
                continue
            if i + 2 > len(raw):
                break
            length = struct.unpack(">H", raw[i:i + 2])[0]
            if length < 2 or i + length > len(raw):
                break
            if marker in frame_markers and length >= 7:
                height, width = struct.unpack(">HH", raw[i + 3:i + 7])
                return width, height
            i += length
    if len(raw) >= 8 and raw[:4] in (b"II*\x00", b"MM\x00*"):
        endian = "<" if raw[:2] == b"II" else ">"
        offset = struct.unpack(endian + "I", raw[4:8])[0]
        if offset + 2 <= len(raw):
            count = struct.unpack(endian + "H", raw[offset:offset + 2])[0]
            if count <= 4096 and offset + 2 + count * 12 <= len(raw):
                dimensions = {}
                for pos in range(offset + 2, offset + 2 + count * 12, 12):
                    tag, kind, n = struct.unpack(endian + "HHI", raw[pos:pos + 8])
                    if tag not in (256, 257) or n != 1:
                        continue
                    field = raw[pos + 8:pos + 12]
                    if kind == 3:
                        dimensions[tag] = struct.unpack(endian + "H", field[:2])[0]
                    elif kind == 4:
                        dimensions[tag] = struct.unpack(endian + "I", field)[0]
                if 256 in dimensions and 257 in dimensions:
                    return dimensions[256], dimensions[257]
    if len(raw) >= 16 and raw[:4] in (b"II+\x00", b"MM\x00+"):
        endian = "<" if raw[:2] == b"II" else ">"
        offset = struct.unpack(endian + "Q", raw[8:16])[0]
        if offset + 8 <= len(raw):
            count = struct.unpack(endian + "Q", raw[offset:offset + 8])[0]
            if count <= 4096 and offset + 8 + count * 20 <= len(raw):
                dimensions = {}
                for pos in range(offset + 8, offset + 8 + count * 20, 20):
                    tag, kind = struct.unpack(endian + "HH", raw[pos:pos + 4])
                    n = struct.unpack(endian + "Q", raw[pos + 4:pos + 12])[0]
                    if tag not in (256, 257) or n != 1:
                        continue
                    field = raw[pos + 12:pos + 20]
                    if kind == 3:
                        dimensions[tag] = struct.unpack(endian + "H", field[:2])[0]
                    elif kind == 4:
                        dimensions[tag] = struct.unpack(endian + "I", field[:4])[0]
                    elif kind == 16:
                        dimensions[tag] = struct.unpack(endian + "Q", field)[0]
                if 256 in dimensions and 257 in dimensions:
                    return dimensions[256], dimensions[257]
    return None


def decode_image(raw: bytes, filename: str, max_pixels: int = 16_000_000,
                 max_dimension: int = 10_000) -> np.ndarray:
    """Decode actual image bytes and return uint8 gray, BGR, or BGRA pixels.

    16-bit images are linearly mapped from their observed min/max to uint8;
    constant images retain their value (high byte). Input bytes are never mutated.
    """
    if Path(filename).suffix.lower() not in ALLOWED:
        raise ImageInputError("Định dạng hỗ trợ: PNG, JPG, JPEG, BMP, TIFF.")
    if not raw:
        raise ImageInputError("Tệp ảnh đang trống.")
    actual_format = detect_image_format(raw)
    expected = {".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF", ".tiff": "TIFF"}
    if actual_format is None:
        raise ImageInputError("Không nhận diện được nội dung ảnh PNG, JPEG, BMP hoặc TIFF.")
    suffix = Path(filename).suffix.lower()
    expected_format = expected.get(suffix, suffix[1:].upper())
    if actual_format != expected_format:
        raise ImageInputError(f"Phần mở rộng tệp không khớp định dạng ảnh thực tế ({actual_format}).")
    dimensions = _header_dimensions(raw)
    if dimensions:
        width, height = dimensions
        if width <= 0 or height <= 0 or width > max_dimension or height > max_dimension or width * height > max_pixels:
            raise ImageInputError(f"Ảnh vượt giới hạn {max_dimension} px mỗi chiều hoặc {max_pixels:,} pixel.")
    try:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ImageInputError("Không giải mã được tệp ảnh.") from exc
    if image is None or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ImageInputError("Tệp không chứa ảnh hợp lệ hoặc có số kênh không hỗ trợ.")
    height, width = image.shape[:2]
    if width > max_dimension or height > max_dimension or width * height > max_pixels:
        raise ImageInputError(f"Ảnh vượt giới hạn {max_dimension} px mỗi chiều hoặc {max_pixels:,} pixel.")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.dtype == np.uint16:
        low, high = int(image.min()), int(image.max())
        if low == high:
            image = (image >> 8).astype(np.uint8)
        else:
            image = np.rint((image.astype(np.float32) - low) * (255.0 / (high - low))).clip(0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ImageInputError("Ảnh có kiểu dữ liệu hoặc giá trị pixel không hợp lệ.")
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(image)


def image_info(image, filename, size, actual_format=None):
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {"name": Path(filename).name, "width": int(image.shape[1]), "height": int(image.shape[0]),
            "channels": channels, "format": actual_format or Path(filename).suffix[1:].upper(), "size": int(size),
            "pixels": int(image.shape[0] * image.shape[1]), "alpha": channels == 4}


def analysis_image(image):
    """Use a consistent white-background composite for all analysis of alpha images."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.rint(bgr * alpha + 255.0 * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return image


def intensity_gray(image):
    visible = analysis_image(image)
    if visible.ndim == 2:
        return visible
    return cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY)


def image_data_url(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Image encoding failed")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def thumbnail(image, max_size=800):
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale == 1.0:
        return image
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
