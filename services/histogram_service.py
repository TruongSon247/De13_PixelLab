import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

from services.image_service import analysis_image, intensity_gray

COLORSPACES = {"gray", "rgb", "hsv", "lab", "ycrcb"}


def _planes(image, space):
    image = analysis_image(image)
    if space == "gray":
        return {"Intensity": intensity_gray(image)}
    if image.ndim == 2:
        raise ValueError("Không gian màu chỉ áp dụng cho ảnh màu.")
    if space == "rgb":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {name: rgb[:, :, i] for i, name in enumerate(("R", "G", "B"))}
    if space == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return {name: converted[:, :, i] for i, name in enumerate(("H", "S", "V"))}
    if space == "lab":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return {name: converted[:, :, i] for i, name in enumerate(("L", "a", "b"))}
    if space == "ycrcb":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        return {name: converted[:, :, i] for i, name in enumerate(("Y", "Cr", "Cb"))}
    raise ValueError("Không gian màu không hợp lệ.")


def _stats(channel, value_max=255):
    flat = channel.ravel()
    hist = np.bincount(flat, minlength=256).astype(np.int64)
    probs = hist[hist > 0].astype(np.float64) / flat.size
    mean = float(flat.mean())
    std = float(flat.std())
    return {"min": int(flat.min()), "max": int(flat.max()), "mean": mean,
            "median": float(np.median(flat)), "std": std,
            "dynamic_range": int(flat.max() - flat.min()), "entropy": float(-(probs * np.log2(probs)).sum()),
            "contrast": std, "pixels": int(flat.size), "value_max": int(value_max), "histogram": hist.tolist()}


def histogram_data(image, space="gray"):
    planes = _planes(image, space)
    channels = {name: _stats(plane, 179 if space == "hsv" and name == "H" else 255) for name, plane in planes.items()}
    return {"space": space, "channels": channels, "pixels": int(next(iter(planes.values())).size)}


def all_histograms(image):
    result = {"gray": histogram_data(image, "gray")}
    if image.ndim == 3:
        for space in ("rgb", "hsv", "lab", "ycrcb"):
            result[space] = histogram_data(image, space)
    return result


def histogram_plot(histogram, space="gray", threshold=None, before=None, selected_channels=None, x_range=None):
    available = list(histogram["channels"])
    names = available if selected_channels is None else list(selected_channels)
    if not names or any(name not in available for name in names):
        raise ValueError("Danh sách kênh histogram không hợp lệ.")
    fig, ax = plt.subplots(figsize=(9, 3.3), dpi=120)
    colors = {"R": "#df5861", "G": "#2b9a70", "B": "#517bd8", "Intensity": "#6658d5",
              "H": "#d85db5", "S": "#e69f32", "V": "#517bd8", "L": "#6658d5",
              "a": "#da625c", "b": "#cf9d28", "Y": "#6658d5", "Cr": "#df5861", "Cb": "#517bd8"}
    for name in names:
        row = histogram["channels"][name]
        values = np.asarray(row["histogram"])
        ax.plot(values, color=colors.get(name, "#6658d5"), lw=1.15, label=name)
        if before is not None:
            ax.plot(np.asarray(before["channels"][name]["histogram"]), color=colors.get(name, "#6658d5"),
                    lw=1, alpha=.55, linestyle="--", label=f"{name} · ảnh gốc")
    if threshold is not None and space == "gray":
        ax.axvline(threshold, color="#222333", linestyle="--", lw=1, label=f"Ngưỡng Otsu {threshold:.0f}")
    ax.set_xlim(*(x_range if x_range is not None else (0, 255))); ax.set_xlabel("Mức cường độ/kênh (Hue HSV của OpenCV: 0–179)"); ax.set_ylabel("Số Pixel")
    ax.legend(frameon=False, fontsize=8, ncol=min(5, len(names)))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.14); fig.tight_layout()
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        plt.close(fig)
