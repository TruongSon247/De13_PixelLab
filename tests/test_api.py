import io
import unittest

import cv2
import numpy as np

from app import app


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
