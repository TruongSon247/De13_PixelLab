def compare_metrics(before, after, threshold_before=None, threshold_after=None):
    return {"mean_delta": after["mean"] - before["mean"],
            "std_delta": after["std"] - before["std"],
            "threshold_delta": None if threshold_before is None or threshold_after is None else threshold_after - threshold_before,
            "contrast_ratio": (after["std"] / before["std"]) if before["std"] else None,
            "interpretation": "Mô tả phân bố pixel; không tự kết luận chất lượng cảm nhận được cải thiện."}


def mask_metrics(mask):
    total = int(mask.size)
    foreground = int((mask == 255).sum())
    ratio = foreground / total if total else 0.0
    return {"foreground_pixels": foreground, "background_pixels": total - foreground,
            "foreground_ratio": ratio, "background_ratio": 1.0 - ratio}
