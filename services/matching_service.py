import cv2
import numpy as np

from services.image_service import analysis_image


def _luminance(image, keep_lab=False):
    visible = analysis_image(image)
    if visible.ndim == 2:
        return visible, None
    lab = cv2.cvtColor(visible, cv2.COLOR_BGR2LAB)
    if keep_lab:
        return lab[:, :, 0], lab
    return lab[:, :, 0].copy(), None


def _hist_cdf(channel):
    hist = np.bincount(channel.ravel(), minlength=256).astype(np.int64)
    cdf = np.cumsum(hist, dtype=np.float64) / channel.size
    return hist, cdf


def histogram_match(source, reference):
    """Match gray levels, or LAB L for color, using a NumPy CDF lookup table."""
    source_l, source_lab = _luminance(source, keep_lab=True)
    ref_l, _ = _luminance(reference)
    source_hist, source_cdf = _hist_cdf(source_l)
    ref_hist, ref_cdf = _hist_cdf(ref_l)

    # For each possible source intensity, select the reference intensity with
    # the nearest cumulative probability. searchsorted keeps this O(256 log 256).
    right = np.searchsorted(ref_cdf, source_cdf, side="left").clip(0, 255)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(source_cdf - ref_cdf[left]) <= np.abs(ref_cdf[right] - source_cdf)
    lut = np.where(use_left, left, right).astype(np.uint8)
    matched_l = lut[source_l]

    visible_source = analysis_image(source)
    if source_lab is None:
        matched = matched_l
    elif np.array_equal(matched_l, source_l):
        matched = visible_source.copy()
    else:
        source_lab[:, :, 0] = matched_l
        matched = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

    matched_hist, matched_cdf = _hist_cdf(matched_l)
    return matched, {
        "lut": lut.tolist(),
        "source_histogram": source_hist.tolist(),
        "reference_histogram": ref_hist.tolist(),
        "matched_histogram": matched_hist.tolist(),
        "source_cdf": source_cdf.tolist(),
        "reference_cdf": ref_cdf.tolist(),
        "matched_cdf": matched_cdf.tolist(),
        "source_pixels": int(source_l.size),
        "reference_pixels": int(ref_l.size),
        "source_min": int(source_l.min()),
        "source_max": int(source_l.max()),
        "reference_min": int(ref_l.min()),
        "reference_max": int(ref_l.max()),
        "mean_cdf_distance_before": float(np.mean(np.abs(source_cdf - ref_cdf))),
        "mean_cdf_distance_after": float(np.mean(np.abs(matched_cdf - ref_cdf))),
        "channel": "L (LAB)" if source_lab is not None else "Intensity",
        "metric_note": "Trung bình độ lệch tuyệt đối giữa hai CDF trên 256 mức; chỉ mô tả độ gần phân bố, không đánh giá chất lượng cảm nhận.",
    }
