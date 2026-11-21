import base64
import io
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import app
from services.enhancement_service import enhance
from services.histogram_service import histogram_data
from services.segmentation_service import otsu


def png_bytes(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode test fixture")
    return encoded.tobytes()


class ApiTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def post_image(self, image, **fields):
        return self.client.post("/api/analyze", data={
            "image": (io.BytesIO(png_bytes(image)), "sample.png"), **fields
        }, content_type="multipart/form-data")

    def test_grayscale_and_color_analysis(self):
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        response = self.post_image(gray, method="equalize", mode="binary_inv")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])
        self.assertEqual(sum(response.json["data"]["histograms_before"]["gray"]["channels"]["Intensity"]["histogram"]), 256)
        equalization = response.json["data"]["equalization"]
        self.assertEqual(len(equalization["lut"]), 256)
        self.assertEqual(len(response.json["data"]["otsu_after"]["variance_curve"]), 256)
        self.assertTrue(response.json["data"]["mask_after_inverse"].startswith("data:image/png;base64,"))
        color = np.zeros((16, 20, 3), np.uint8); color[:, :10] = (20, 80, 220)
        response = self.post_image(color, method="clahe", clip_limit="2.5", tile_size="8")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json["data"]["variants"]), {"stretch", "equalize", "clahe", "gamma"})
        self.assertIsNone(response.json["data"]["equalization"])

    def test_gamma_variant_uses_submitted_parameters(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        response = self.post_image(image, method="equalize", gamma="2.0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["data"]["variants"]["gamma"]["params"]["gamma"], 2.0)

    def test_experimental_comparison_has_independent_three_way_results(self):
        image = np.array([[10, 20, 30, 40], [80, 100, 180, 240]], dtype=np.uint8)
        original_copy = image.copy()
        response = self.post_image(image, method="gamma", mode="binary_inv")
        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        experiment = data["experimental_comparison"]
        self.assertEqual(set(experiment), {"original", "stretch", "equalize"})
        expected = {
            "original": image,
            "stretch": enhance(image, "stretch")[0],
            "equalize": enhance(image, "equalize")[0],
        }
        for name, variant in expected.items():
            item = experiment[name]
            self.assertEqual(variant.shape, image.shape)
            self.assertEqual(variant.dtype, np.uint8)
            self.assertGreaterEqual(int(variant.min()), 0)
            self.assertLessEqual(int(variant.max()), 255)
            gray = variant if variant.ndim == 2 else cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY)
            pixels = gray.astype(np.float64).ravel()
            counts = np.bincount(pixels.astype(np.uint8), minlength=256)
            probabilities = counts[counts > 0] / pixels.size
            reference_stats = {
                "min": int(pixels.min()), "max": int(pixels.max()), "mean": float(pixels.mean()),
                "median": float(np.median(pixels)), "std": float(pixels.std()),
                "dynamic_range": int(pixels.max() - pixels.min()),
                "entropy": float(-(probabilities * np.log2(probabilities)).sum()),
                "contrast": float(pixels.std()),
            }
            self.assertEqual(item["histogram"], counts.tolist())
            self.assertEqual(len(item["histogram"]), 256)
            self.assertEqual(sum(item["histogram"]), pixels.size)
            self.assertEqual(set(item["stats"]), set(reference_stats))
            for key, expected_value in reference_stats.items():
                actual = item["stats"][key]
                self.assertIsInstance(actual, (int, float))
                self.assertFalse(isinstance(actual, bool))
                self.assertTrue(np.isfinite(actual))
                self.assertAlmostEqual(actual, expected_value, places=10)
            self.assertEqual(item["stats"]["dynamic_range"], item["stats"]["max"] - item["stats"]["min"])

            mask, otsu_stats = otsu(variant, "binary_inv")
            mask_url = item["mask"]
            self.assertTrue(mask_url.startswith("data:image/png;base64,"))
            decoded_mask = cv2.imdecode(np.frombuffer(base64.b64decode(mask_url.split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(np.array_equal(decoded_mask, mask))
            self.assertEqual(set(np.unique(mask)).issubset({0, 255}), True)
            self.assertEqual(item["otsu"]["threshold"], otsu_stats["threshold"])
            threshold_candidates = []
            for threshold in range(256):
                lower, upper = pixels[pixels <= threshold], pixels[pixels > threshold]
                if not lower.size or not upper.size:
                    threshold_candidates.append(0.0)
                else:
                    w0, w1 = lower.size / pixels.size, upper.size / pixels.size
                    threshold_candidates.append(w0 * w1 * (lower.mean() - upper.mean()) ** 2)
            best_threshold = int(np.argmax(threshold_candidates))
            self.assertEqual(item["otsu"]["threshold"], best_threshold)
            self.assertAlmostEqual(item["otsu"]["between_class_variance"], threshold_candidates[best_threshold], places=10)
            foreground = float(np.count_nonzero(mask == 255) / mask.size)
            background = float(np.count_nonzero(mask == 0) / mask.size)
            self.assertAlmostEqual(item["otsu"]["foreground_ratio"], foreground)
            self.assertAlmostEqual(item["otsu"]["background_ratio"], background)
            self.assertAlmostEqual(item["otsu"]["foreground_ratio"] + item["otsu"]["background_ratio"], 1.0)
            for key in ("threshold", "between_class_variance", "foreground_ratio", "background_ratio"):
                value = item["otsu"][key]
                self.assertIsInstance(value, (int, float))
                self.assertFalse(isinstance(value, bool))
                self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(item["otsu"]["foreground_ratio"], 0.0)
            self.assertLessEqual(item["otsu"]["foreground_ratio"], 1.0)
            self.assertTrue(item["image"].startswith("data:image/png;base64,"))
        self.assertTrue(np.array_equal(image, original_copy))

    def test_experimental_comparison_otsu_mode_is_synchronized(self):
        image = np.array([[0, 0, 20, 30], [40, 90, 160, 255]], dtype=np.uint8)
        binary = self.post_image(image, mode="binary").json["data"]["experimental_comparison"]
        inverse = self.post_image(image, mode="binary_inv").json["data"]["experimental_comparison"]
        for branch in ("original", "stretch", "equalize"):
            first = cv2.imdecode(np.frombuffer(base64.b64decode(binary[branch]["mask"].split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
            second = cv2.imdecode(np.frombuffer(base64.b64decode(inverse[branch]["mask"].split(",", 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(np.array_equal(first.astype(np.uint16) + second.astype(np.uint16), np.full(first.shape, 255, np.uint16)))
            self.assertAlmostEqual(binary[branch]["otsu"]["foreground_ratio"], inverse[branch]["otsu"]["background_ratio"])

    def test_analyze_rejects_missing_empty_and_invalid_image_safely(self):
        missing = self.client.post("/api/analyze", data={}, content_type="multipart/form-data")
        self.assertEqual(missing.status_code, 400)
        self.assertFalse(missing.json["ok"])
        empty = self.client.post("/api/analyze", data={"image": (io.BytesIO(b""), "empty.png")}, content_type="multipart/form-data")
        self.assertEqual(empty.status_code, 400)
        self.assertFalse(empty.json["ok"])
        invalid = self.client.post("/api/analyze", data={"image": (io.BytesIO(b"not an image"), "invalid.png")}, content_type="multipart/form-data")
        self.assertEqual(invalid.status_code, 400)
        self.assertFalse(invalid.json["ok"])

    def test_invalid_selected_numeric_parameter_has_localized_error(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        response = self.post_image(image, method="gamma", gamma="abc")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "invalid_input")
        self.assertEqual(response.json["error"]["message"], "Tham số số học không hợp lệ.")

    def test_histogram_matching_api_returns_comparison_data(self):
        source = np.arange(100, dtype=np.uint8).reshape(10, 10)
        reference = np.full((7, 11), 210, np.uint8)
        response = self.client.post("/api/match", data={
            "image": (io.BytesIO(png_bytes(source)), "source.png"),
            "reference": (io.BytesIO(png_bytes(reference)), "reference.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        details = response.json["data"]["details"]
        self.assertEqual(len(details["lut"]), 256)
        self.assertEqual(details["reference_pixels"], reference.size)
        self.assertTrue(response.json["data"]["matched"].startswith("data:image/png;base64,"))
        exported = self.client.post("/api/export/match", data={
            "image": (io.BytesIO(png_bytes(source)), "source.png"),
            "reference": (io.BytesIO(png_bytes(reference)), "reference.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(exported.status_code, 200)
        full_output = cv2.imdecode(np.frombuffer(exported.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(full_output.shape, source.shape)

    def test_histogram_matching_api_validates_both_inputs(self):
        response = self.client.post("/api/match", data={"image": (io.BytesIO(b"bad"), "bad.png")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "missing_image")
        response = self.client.post("/api/match", data={
            "image": (io.BytesIO(b"bad"), "source.png"),
            "reference": (io.BytesIO(png_bytes(np.zeros((2, 2), np.uint8))), "reference.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "invalid_input")

    def test_workspace_page_serves_controls(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"id=\"histogram\"", response.data)
        self.assertIn(b"id=\"undo\"", response.data)
        self.assertIn(b"id=\"equalization-explorer\"", response.data)
        self.assertIn(b"id=\"otsu-explorer\"", response.data)
        self.assertIn(b"id=\"matching-explorer\"", response.data)
        for label in ("Ảnh gốc", "Kéo giãn tương phản", "Cân bằng Histogram", "Thống kê ảnh", "Thông số phân đoạn Otsu"):
            self.assertIn(label.encode("utf-8"), response.data)
        frontend = Path(app.root_path, "static", "js", "app.js").read_text(encoding="utf-8")
        for label in ("Ảnh gốc", "Kéo giãn tương phản", "Cân bằng Histogram", "Nhỏ nhất", "Lớn nhất", "Trung bình", "Trung vị", "Độ lệch chuẩn", "Dải động", "Độ tương phản", "Biểu đồ Histogram", "Mặt nạ phân đoạn Otsu", "Nhị phân đảo"):
            self.assertIn(label, frontend)

    def test_invalid_upload_and_pixel_limit_return_json(self):
        bad = self.client.post("/api/analyze", data={"image": (io.BytesIO(b"bad"), "bad.png")})
        self.assertEqual(bad.status_code, 400)
        self.assertFalse(bad.json["ok"])
        old = app.config["MAX_IMAGE_PIXELS"]
        try:
            app.config["MAX_IMAGE_PIXELS"] = 10
            response = self.post_image(np.zeros((4, 4), np.uint8))
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["error"]["code"], "invalid_input")
        finally:
            app.config["MAX_IMAGE_PIXELS"] = old

    def test_upload_size_limit_returns_json_413(self):
        old = app.config["MAX_CONTENT_LENGTH"]
        try:
            app.config["MAX_CONTENT_LENGTH"] = 128
            response = self.client.post("/api/analyze", data={"image": (io.BytesIO(b"x" * 200), "large.png")})
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json["error"]["code"], "upload_too_large")
        finally:
            app.config["MAX_CONTENT_LENGTH"] = old

    def test_exports(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        response = self.post_image(image)
        hist = response.json["data"]["histograms_after"]["gray"]
        chart = self.client.post("/api/export/histogram", json={"histogram": hist, "space": "gray", "threshold": 120})
        self.assertEqual(chart.status_code, 200)
        exported = self.client.post("/api/export/image", data={"image": (io.BytesIO(png_bytes(image)), "sample.png"), "asset": "mask_after"})
        self.assertEqual(exported.status_code, 200)
        decoded = cv2.imdecode(np.frombuffer(exported.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(decoded.shape, image.shape)

    def test_histogram_export_rejects_malformed_channel_structures(self):
        malformed = [
            {"channels": "RGB"},
            {"channels": ["R", "G", "B"]},
            {"channels": {"R": "bad"}},
            {"channels": {"R": 4}},
            {"channels": {"R": None}},
            {"channels": {"R": {"histogram": "not bins"}}},
            {"channels": {"R": {"histogram": [1] * 255}}},
            {},
        ]
        for value in malformed:
            with self.subTest(value=repr(value)[:60]):
                response = self.client.post("/api/export/histogram", json={"histogram": value, "space": "rgb"})
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.json["ok"])
                self.assertEqual(response.json["error"]["code"], "invalid_histogram")
        response = self.client.post("/api/export/histogram", json={"histogram": {"channels": {}}, "space": "rgb", "channels": "R"})
        self.assertEqual(response.status_code, 400)

    def test_histogram_export_matches_selected_channel_and_zoom(self):
        image = np.zeros((20, 30, 3), np.uint8)
        image[:, :15] = (20, 80, 220)
        analysis = self.post_image(image).json["data"]
        response = self.client.post("/api/export/histogram", json={
            "after": analysis["histograms_after"]["rgb"],
            "before": analysis["histograms_before"]["rgb"],
            "space": "rgb", "channels": ["R"], "range": [20, 80]
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["data"]["image"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
