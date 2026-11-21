import logging
import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request, send_file
import io

from services.enhancement_service import METHODS, enhance
from services.evaluation_service import compare_metrics, mask_metrics
from services.histogram_service import COLORSPACES, all_histograms, histogram_data, histogram_plot
from services.image_service import ImageInputError, decode_image, detect_image_format, image_data_url, image_info, intensity_gray, thumbnail
from services.matching_service import histogram_match
from services.segmentation_service import morphology, otsu

images = Blueprint("images", __name__)
logger = logging.getLogger(__name__)


@images.get("/")
def index():
    return render_template("dashboard.html")


@images.post("/api/analyze")
def analyze():
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return jsonify(ok=False, error={"code": "missing_image", "message": "Chọn một tệp ảnh trước khi phân tích."}), 400
    raw = upload.read()
    try:
        original = decode_image(raw, upload.filename, current_app.config["MAX_IMAGE_PIXELS"], current_app.config["MAX_IMAGE_DIMENSION"])
        method = request.form.get("method", "stretch")
        if method not in METHODS:
            raise ValueError("Phương pháp nâng ảnh không hợp lệ.")
        params = {}
        for key in ("gamma", "clip_limit", "tile_size"):
            if key in request.form:
                value = request.form[key]
                try:
                    params[key] = float(value) if key != "tile_size" else int(value)
                except ValueError as exc:
                    raise ValueError("Tham số số học không hợp lệ.") from exc
        mode = request.form.get("mode", "binary")
        if method == "equalize":
            enhanced, used_params, equalization = enhance(original, method, params, include_details=True)
        else:
            enhanced, used_params = enhance(original, method, params)
            equalization = None
        source_mask, source_otsu = otsu(original, mode)
        result_mask, result_otsu = otsu(enhanced, mode)
        source_inverse = cv2.bitwise_not(source_mask)
        result_inverse = cv2.bitwise_not(result_mask)
        morph_args = {"operation": request.form.get("morphology", "none"),
                      "size": int(request.form.get("kernel_size", 3)),
                      "shape": request.form.get("kernel_shape", "ellipse"),
                      "iterations": int(request.form.get("iterations", 1))}
        source_post = morphology(source_mask, **morph_args)
        result_post = morphology(result_mask, **morph_args)
        source_hist = all_histograms(original)
        result_hist = all_histograms(enhanced)
        variants = {}
        variant_images = {}
        for variant in sorted(METHODS):
            if variant == method:
                variant_image, variant_params = enhanced, used_params
            else:
                try:
                    variant_image, variant_params = enhance(original, variant, params)
                except ValueError:
                    variant_image, variant_params = enhance(original, variant)
            variant_hist = histogram_data(variant_image, "gray")
            variant_images[variant] = variant_image
            variants[variant] = {"preview": image_data_url(thumbnail(variant_image)),
                                 "params": variant_params, "stats": variant_hist["channels"]["Intensity"]}
        experiment = {}
        for key, variant_image in (("original", original), ("stretch", variant_images["stretch"]),
                                   ("equalize", variant_images["equalize"])):
            variant_mask, variant_otsu = otsu(variant_image, mode)
            variant_stats = histogram_data(variant_image, "gray")["channels"]["Intensity"]
            experiment[key] = {
                "image": image_data_url(thumbnail(variant_image)),
                "histogram": variant_stats["histogram"],
                "stats": {name: variant_stats[name] for name in
                          ("min", "max", "mean", "median", "std", "dynamic_range", "entropy", "contrast")},
                "mask": image_data_url(thumbnail(variant_mask)),
                "otsu": {name: variant_otsu[name] for name in
                         ("threshold", "between_class_variance", "foreground_ratio", "background_ratio", "degenerate")},
            }
        data = {
            "info": image_info(original, upload.filename, len(raw), detect_image_format(raw)), "method": method, "params": used_params,
            "mode": mode, "morphology": morph_args,
            "original": image_data_url(thumbnail(original)), "enhanced": image_data_url(thumbnail(enhanced)),
            "mask_before": image_data_url(thumbnail(source_mask)), "mask_after": image_data_url(thumbnail(result_mask)),
            "mask_before_morph": image_data_url(thumbnail(source_post)), "mask_after_morph": image_data_url(thumbnail(result_post)),
            "mask_before_inverse": image_data_url(thumbnail(source_inverse)), "mask_after_inverse": image_data_url(thumbnail(result_inverse)),
            "grayscale_before": image_data_url(thumbnail(intensity_gray(original))),
            "otsu_before": source_otsu, "otsu_after": result_otsu,
            "mask_metrics_before": mask_metrics(source_mask), "mask_metrics_after": mask_metrics(result_mask),
            "stats_before": source_hist["gray"]["channels"]["Intensity"],
            "stats_after": result_hist["gray"]["channels"]["Intensity"],
            "histograms_before": source_hist, "histograms_after": result_hist,
            "equalization": equalization,
            "variants": variants,
            "experimental_comparison": experiment,
            "comparison": compare_metrics(source_hist["gray"]["channels"]["Intensity"], result_hist["gray"]["channels"]["Intensity"], source_otsu["threshold"], result_otsu["threshold"]),
            "warnings": (["Ảnh đồng nhất hoặc mask chỉ có tiền cảnh/hậu cảnh; kết quả Otsu có thể không hữu ích cho phân đoạn."] if source_otsu["degenerate"] or result_otsu["degenerate"] else []) + (["Kênh alpha được ghép trên nền trắng khi phân tích; ảnh tăng cường xuất ra không có độ trong suốt."] if original.ndim == 3 and original.shape[2] == 4 else []),
        }
        return jsonify(ok=True, data=data)
    except (ImageInputError, ValueError, cv2.error) as exc:
        return jsonify(ok=False, error={"code": "invalid_input", "message": str(exc)}), 400
    except Exception:
        logger.exception("Image analysis failed for upload %r", upload.filename)
        return jsonify(ok=False, error={"code": "processing_failed", "message": "Không thể xử lý ảnh này. Hãy kiểm tra ảnh và thử lại."}), 422


@images.post("/api/export/histogram")
def export_histogram():
    body = request.get_json(silent=True) or {}
    try:
        if not isinstance(body, dict):
            raise ValueError("Yêu cầu phải có định dạng JSON hợp lệ.")
        histogram = body.get("histogram") or body.get("after")
        before = body.get("before")
        space = body.get("space", "gray")
        threshold = body.get("threshold")
        selected_channels = body.get("channels")
        x_range = body.get("range")
        if space not in COLORSPACES:
            raise ValueError("Không gian màu không hợp lệ.")
        if selected_channels is not None and (not isinstance(selected_channels, list) or not selected_channels or any(not isinstance(name, str) for name in selected_channels)):
            raise ValueError("Danh sách kênh phải có ít nhất một tên hợp lệ.")
        valid_names = {"gray": {"Intensity"}, "rgb": {"R", "G", "B"}, "hsv": {"H", "S", "V"},
                       "lab": {"L", "a", "b"}, "ycrcb": {"Y", "Cr", "Cb"}}[space]

        def validate_histogram(value, field):
            if not isinstance(value, dict) or set(value) - {"space", "channels", "pixels"} or "channels" not in value:
                raise ValueError(f"{field} phải chứa dữ liệu Histogram hợp lệ với trường channels.")
            channels = value["channels"]
            if not isinstance(channels, dict) or not channels or len(channels) > 3:
                raise ValueError(f"{field}.channels phải là dữ liệu hợp lệ gồm 1–3 kênh.")
            if any(not isinstance(name, str) or name not in valid_names for name in channels):
                raise ValueError(f"{field}.channels chứa tên kênh không hợp lệ cho {space}.")
            for name, channel in channels.items():
                if not isinstance(channel, dict) or set(channel) - {"histogram", "min", "max", "mean", "median", "std", "dynamic_range", "entropy", "contrast", "pixels", "value_max"} or "histogram" not in channel:
                    raise ValueError(f"{field}.channels.{name} phải chứa trường histogram hợp lệ.")
                bins = channel["histogram"]
                if not isinstance(bins, list) or len(bins) != 256 or any(isinstance(n, bool) or not isinstance(n, (int, float)) for n in bins):
                    raise ValueError(f"{field}.channels.{name}.histogram phải có đúng 256 số.")
                values = np.asarray(bins, dtype=np.float64)
                if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0 or values.sum() > current_app.config["MAX_IMAGE_PIXELS"]:
                    raise ValueError("Số pixel histogram phải hữu hạn, dương và nằm trong giới hạn ảnh.")
            if "pixels" in value and (isinstance(value["pixels"], bool) or not isinstance(value["pixels"], int) or not 1 <= value["pixels"] <= current_app.config["MAX_IMAGE_PIXELS"]):
                raise ValueError(f"{field}.pixels không hợp lệ.")

        validate_histogram(histogram, "histogram")
        if before is not None:
            validate_histogram(before, "before")
            if set(before["channels"]) != set(histogram["channels"]):
                raise ValueError("Histogram trước và sau phải có cùng các kênh.")
        if selected_channels is not None and any(name not in histogram["channels"] for name in selected_channels):
            raise ValueError("Danh sách kênh không khớp với dữ liệu Histogram.")
        if x_range is not None:
            if not isinstance(x_range, list) or len(x_range) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in x_range) or not 0 <= x_range[0] < x_range[1] <= 255:
                raise ValueError("Khoảng giá trị phải hợp lệ trong giới hạn 0–255.")
        if threshold is not None and (not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 255):
            raise ValueError("Ngưỡng histogram phải nằm trong khoảng 0–255.")
        plot = histogram_plot(histogram, space, threshold, before, selected_channels, x_range)
        return jsonify(ok=True, data={"image": plot})
    except (TypeError, ValueError) as exc:
        return jsonify(ok=False, error={"code": "invalid_histogram", "message": str(exc)}), 400


@images.post("/api/match")
def match_histogram():
    source_upload = request.files.get("image")
    reference_upload = request.files.get("reference")
    if not source_upload or not source_upload.filename or not reference_upload or not reference_upload.filename:
        return jsonify(ok=False, error={"code": "missing_image", "message": "Chọn cả ảnh nguồn và ảnh tham chiếu."}), 400
    try:
        max_pixels = current_app.config["MAX_IMAGE_PIXELS"]
        max_dimension = current_app.config["MAX_IMAGE_DIMENSION"]
        source = decode_image(source_upload.read(), source_upload.filename, max_pixels, max_dimension)
        reference = decode_image(reference_upload.read(), reference_upload.filename, max_pixels, max_dimension)
        matched, details = histogram_match(source, reference)
        data = {
            "source": image_data_url(thumbnail(source)),
            "reference": image_data_url(thumbnail(reference)),
            "matched": image_data_url(thumbnail(matched)),
            "details": details,
        }
        return jsonify(ok=True, data=data)
    except (ImageInputError, ValueError, cv2.error) as exc:
        return jsonify(ok=False, error={"code": "invalid_input", "message": str(exc)}), 400
    except Exception:
        logger.exception("Histogram matching failed")
        return jsonify(ok=False, error={"code": "matching_failed", "message": "Không thể khớp Histogram cho các ảnh đã chọn."}), 422


@images.post("/api/export/match")
def export_histogram_match():
    source_upload = request.files.get("image")
    reference_upload = request.files.get("reference")
    if not source_upload or not source_upload.filename or not reference_upload or not reference_upload.filename:
        return jsonify(ok=False, error={"code": "missing_image", "message": "Chọn cả ảnh nguồn và ảnh tham chiếu."}), 400
    try:
        max_pixels = current_app.config["MAX_IMAGE_PIXELS"]
        max_dimension = current_app.config["MAX_IMAGE_DIMENSION"]
        source = decode_image(source_upload.read(), source_upload.filename, max_pixels, max_dimension)
        reference = decode_image(reference_upload.read(), reference_upload.filename, max_pixels, max_dimension)
        matched, _details = histogram_match(source, reference)
        ok, encoded = cv2.imencode(".png", matched)
        if not ok:
            raise ValueError("Không thể mã hóa ảnh Histogram Matching.")
        return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/png", as_attachment=True,
                         download_name="histogram-matched.png", max_age=0)
    except (ImageInputError, ValueError, cv2.error) as exc:
        return jsonify(ok=False, error={"code": "invalid_input", "message": str(exc)}), 400
    except Exception:
        logger.exception("Matched image export failed")
        return jsonify(ok=False, error={"code": "matching_export_failed", "message": "Không thể xuất ảnh đã khớp Histogram."}), 422


@images.post("/api/export/image")
def export_image():
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return jsonify(ok=False, error={"code": "missing_image", "message": "Chọn ảnh trước khi xuất."}), 400
    try:
        original = decode_image(upload.read(), upload.filename, current_app.config["MAX_IMAGE_PIXELS"], current_app.config["MAX_IMAGE_DIMENSION"])
        asset = request.form.get("asset", "enhanced")
        params = {}
        for key in ("gamma", "clip_limit", "tile_size"):
            if key in request.form:
                try:
                    params[key] = int(request.form[key]) if key == "tile_size" else float(request.form[key])
                except ValueError as exc:
                    raise ValueError("Tham số số học không hợp lệ.") from exc
        if asset == "enhanced":
            image, _ = enhance(original, request.form.get("method", "stretch"), params)
        elif asset in {"mask_before", "mask_after", "mask_before_morph", "mask_after_morph"}:
            is_after = asset in {"mask_after", "mask_after_morph"}
            base = original
            if is_after:
                base, _ = enhance(original, request.form.get("method", "stretch"), params)
            image, _ = otsu(base, request.form.get("mode", "binary"))
            if asset.endswith("_morph"):
                image = morphology(image, request.form.get("morphology", "none"),
                    int(request.form.get("kernel_size", 3)), request.form.get("kernel_shape", "ellipse"),
                    int(request.form.get("iterations", 1)))
        else:
            raise ValueError("Loại ảnh xuất không hợp lệ.")
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Không mã hóa được ảnh đầu ra.")
        return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/png", as_attachment=True,
                         download_name=f"{asset}.png", max_age=0)
    except (ImageInputError, ValueError, cv2.error) as exc:
        return jsonify(ok=False, error={"code": "invalid_export", "message": str(exc)}), 400
    except Exception:
        logger.exception("Image export failed for upload %r", upload.filename)
        return jsonify(ok=False, error={"code": "export_failed", "message": "Không thể xuất ảnh."}), 422
