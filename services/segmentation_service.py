import cv2
import numpy as np

from services.image_service import intensity_gray

MORPHOLOGY = {"none", "erode", "dilate", "open", "close"}
SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}


def otsu(image, mode="binary"):
    if mode not in {"binary", "binary_inv"}:
        raise ValueError("Chế độ Otsu không hợp lệ.")
    gray = intensity_gray(image)
    flag = cv2.THRESH_BINARY if mode == "binary" else cv2.THRESH_BINARY_INV
    threshold, mask = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    prob = hist / total
    levels = np.arange(256, dtype=np.float64)
    global_mean = float(np.dot(prob, levels))
    cumulative = np.cumsum(prob)
    means = np.cumsum(prob * levels)
    w0 = cumulative[int(threshold)]
    w1 = 1.0 - w0
    between = 0.0 if w0 <= 0 or w1 <= 0 else float(w0 * w1 * ((means[int(threshold)] / w0) - ((global_mean - means[int(threshold)]) / w1)) ** 2)
    foreground = float(np.count_nonzero(mask == 255) / total)
    degenerate = int(gray.min()) == int(gray.max()) or foreground in (0.0, 1.0)
    return mask, {"threshold": float(threshold), "between_class_variance": between,
                  "foreground_ratio": foreground, "background_ratio": 1.0 - foreground,
                  "degenerate": degenerate}


def morphology(mask, operation="none", size=3, shape="ellipse", iterations=1):
    if operation not in MORPHOLOGY:
        raise ValueError("Phép hình thái học không hợp lệ.")
    if shape not in SHAPES:
        raise ValueError("Hình dạng Kernel không hợp lệ.")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or size > 31 or size % 2 == 0:
        raise ValueError("Kích thước Kernel phải là số lẻ từ 1 đến 31.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1 or iterations > 10:
        raise ValueError("Số lần lặp phải nằm trong khoảng 1–10.")
    if operation == "none":
        return mask.copy()
    kernel = cv2.getStructuringElement(SHAPES[shape], (size, size))
    if operation == "erode":
        return cv2.erode(mask, kernel, iterations=iterations)
    if operation == "dilate":
        return cv2.dilate(mask, kernel, iterations=iterations)
    morph_op = cv2.MORPH_OPEN if operation == "open" else cv2.MORPH_CLOSE
    return cv2.morphologyEx(mask, morph_op, kernel, iterations=iterations)
