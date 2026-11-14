import cv2
import numpy as np

from services.image_service import analysis_image

METHODS = {"stretch", "equalize", "clahe", "gamma"}


def _stretch(channel):
    low, high = int(channel.min()), int(channel.max())
    if low == high:
        return channel.copy()
    return np.rint((channel.astype(np.float32) - low) * (255.0 / (high - low))).clip(0, 255).astype(np.uint8)


def _equalize(channel, return_details=False):
    hist = np.bincount(channel.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = int(cdf[cdf > 0][0])
    if cdf_min == channel.size:
        lut = np.arange(256, dtype=np.uint8)
        result = channel.copy()
    else:
        lut = np.rint((cdf - cdf_min) * 255.0 / (channel.size - cdf_min)).clip(0, 255).astype(np.uint8)
        result = lut[channel]
    if not return_details:
        return result
    details = {
        "histogram": hist.tolist(),
        "pdf": (hist.astype(np.float64) / channel.size).tolist(),
        "cdf": (cdf.astype(np.float64) / channel.size).tolist(),
        "lut": lut.tolist(),
        "cdf_min": cdf_min / channel.size,
        "total_pixels": int(channel.size),
        "min": int(channel.min()),
        "max": int(channel.max()),
        "constant": cdf_min == channel.size,
    }
    return result, details


def _odd_int(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"{name} phải là số nguyên.")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} phải nằm trong khoảng {low}–{high}.")
    return value


def enhance(image, method="stretch", params=None, include_details=False):
    if method not in METHODS:
        raise ValueError("Phương pháp nâng ảnh không hợp lệ.")
    params = params or {}
    source = analysis_image(image)
    if source.ndim == 2:
        light = source
        lab = None
    else:
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        light = lab[:, :, 0]

    details = None
    if method == "stretch":
        changed = _stretch(light)
        used = {"method": method, "input_min": int(light.min()), "input_max": int(light.max())}
    elif method == "equalize":
        if include_details:
            changed, details = _equalize(light, return_details=True)
        else:
            changed = _equalize(light)
        used = {"method": method}
    elif method == "clahe":
        clip_limit = float(params.get("clip_limit", 2.0))
        if not np.isfinite(clip_limit) or not 0.1 <= clip_limit <= 10:
            raise ValueError("Giới hạn clip của CLAHE phải nằm trong khoảng 0,1–10.")
        tile = _odd_int(params.get("tile_size", 8), "Kích thước ô", 2, 32)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        changed = clahe.apply(light)
        used = {"method": method, "clip_limit": clip_limit, "tile_size": tile}
    else:
        gamma = float(params.get("gamma", 1.0))
        if not np.isfinite(gamma) or not 0.1 <= gamma <= 5:
            raise ValueError("Gamma phải nằm trong khoảng 0.1–5.")
        lut = np.rint(((np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma)) * 255.0).clip(0, 255).astype(np.uint8)
        changed = lut[light]
        used = {"method": method, "gamma": gamma}

    if lab is not None and np.array_equal(changed, light):
        result = source.copy()
        return (result, used, details) if include_details else (result, used)
    if lab is None:
        result = changed
    else:
        lab[:, :, 0] = changed
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return (result, used, details) if include_details else (result, used)
