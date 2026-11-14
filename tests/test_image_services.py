import io
import struct
import unittest

import cv2
import numpy as np

from services.enhancement_service import _equalize, enhance
from services.histogram_service import all_histograms, histogram_data
from services.image_service import ImageInputError, analysis_image, decode_image, intensity_gray
from services.matching_service import histogram_match
from services.segmentation_service import morphology, otsu


def png_bytes(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode test fixture")
    return encoded.tobytes()


class ImageServiceTests(unittest.TestCase):
    def test_gray_histogram_has_256_bins_and_pixel_total(self):
        image = np.array([[0, 1, 1], [127, 255, 255]], dtype=np.uint8)
        row = histogram_data(image)["channels"]["Intensity"]
        self.assertEqual(len(row["histogram"]), 256)
        self.assertEqual(sum(row["histogram"]), image.size)
        self.assertEqual((row["min"], row["max"], row["median"]), (0, 255, 64.0))

    def test_rgb_order_and_all_spaces(self):
        bgr = np.array([[[10, 40, 230], [10, 40, 230]]], dtype=np.uint8)
        spaces = all_histograms(bgr)
        self.assertEqual(set(spaces), {"gray", "rgb", "hsv", "lab", "ycrcb"})
        channels = spaces["rgb"]["channels"]
        self.assertEqual(channels["R"]["histogram"][230], 2)
        self.assertEqual(channels["G"]["histogram"][40], 2)
        self.assertEqual(channels["B"]["histogram"][10], 2)
        self.assertEqual(spaces["hsv"]["channels"]["H"]["value_max"], 179)

    def test_alpha_is_composited_on_white_for_analysis(self):
        bgra = np.array([[[0, 255, 0, 0], [0, 0, 255, 255]]], dtype=np.uint8)
        visible = analysis_image(bgra)
        self.assertEqual(visible.shape, (1, 2, 3))
        self.assertTrue(np.array_equal(visible[0, 0], [255, 255, 255]))
        self.assertLess(intensity_gray(bgra)[0, 1], 100)

    def test_16bit_uses_observed_range_and_constant_keeps_high_byte(self):
        spread = np.array([[1000, 3000, 5000]], dtype=np.uint16)
        converted = decode_image(png_bytes(spread), "spread.png")
        self.assertEqual(converted.tolist(), [[0, 128, 255]])
        constant = np.full((2, 2), 0xABCD, dtype=np.uint16)
        self.assertTrue(np.all(decode_image(png_bytes(constant), "flat.png") == 0xAB))

    def test_decode_rejects_invalid_and_oversized_images(self):
        with self.assertRaises(ImageInputError):
            decode_image(b"not an image", "bad.png")
        with self.assertRaises(ImageInputError):
            decode_image(png_bytes(np.zeros((4, 5), np.uint8)), "large.png", max_pixels=19)

    def test_large_png_dimensions_rejected_before_decode(self):
        oversized_header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 20_000, 20) + b"\x08\x02\x00\x00\x00"
        with self.assertRaises(ImageInputError):
            decode_image(oversized_header, "large.png", max_pixels=16_000_000)

    def test_real_format_must_match_filename_extension(self):
        with self.assertRaises(ImageInputError):
            decode_image(png_bytes(np.zeros((2, 2), np.uint8)), "renamed.jpg")

    def test_stretch_and_equalize_leave_constant_image_unchanged(self):
        image = np.full((8, 9), 83, dtype=np.uint8)
        self.assertTrue(np.array_equal(enhance(image, "stretch")[0], image))
        self.assertTrue(np.array_equal(enhance(image, "equalize")[0], image))

    def test_equalization_details_match_the_applied_lut(self):
        image = np.array([[0, 0, 7, 7, 200, 255]], dtype=np.uint8)
        output, details = _equalize(image, return_details=True)
        self.assertEqual(sum(details["histogram"]), image.size)
        self.assertAlmostEqual(sum(details["pdf"]), 1.0)
        self.assertTrue(np.all(np.diff(details["cdf"]) >= 0))
        self.assertTrue(np.array_equal(output, np.asarray(details["lut"], np.uint8)[image]))
        enhanced, _params, api_details = enhance(image, "equalize", include_details=True)
        self.assertTrue(np.array_equal(output, enhanced))
        self.assertEqual(api_details["lut"], details["lut"])
        constant = np.full((3, 4), 73, np.uint8)
        result, info = _equalize(constant, return_details=True)
        self.assertTrue(np.array_equal(result, constant))
        self.assertTrue(info["constant"])

    def test_color_equalization_details_use_lab_luminance(self):
        source = np.zeros((4, 6, 3), np.uint8)
        source[:, :2] = (10, 40, 210)
        source[:, 2:4] = (90, 130, 30)
        source[:, 4:] = (190, 80, 120)
        light = cv2.cvtColor(analysis_image(source), cv2.COLOR_BGR2LAB)[:, :, 0]
        _result, _params, details = enhance(source, "equalize", include_details=True)
        self.assertEqual(details["histogram"], np.bincount(light.ravel(), minlength=256).tolist())
        self.assertEqual(details["total_pixels"], source.shape[0] * source.shape[1])

    def test_histogram_matching_gray_sparse_constant_and_different_shapes(self):
        source = np.array([[10, 10, 40], [80, 80, 200]], np.uint8)
        reference = np.array([[100, 120], [220, 240], [250, 250]], np.uint8)
        result, details = histogram_match(source, reference)
        self.assertEqual(result.shape, source.shape)
        self.assertTrue(np.all(np.diff(details["lut"]) >= 0))
        self.assertEqual(len(details["source_histogram"]), 256)
        self.assertEqual(details["reference_pixels"], reference.size)
        self.assertLessEqual(details["mean_cdf_distance_after"], details["mean_cdf_distance_before"])
        flat_result, flat_details = histogram_match(np.full((2, 3), 40, np.uint8), reference)
        self.assertTrue(np.all(flat_result == reference.max()))
        self.assertEqual(flat_details["source_min"], flat_details["source_max"])

    def test_histogram_matching_color_uses_lab_l_channel(self):
        source = np.zeros((8, 9, 3), np.uint8)
        source[:, :4] = (30, 80, 180)
        source[:, 4:] = (90, 150, 240)
        reference = np.full((5, 6, 3), (30, 180, 70), np.uint8)
        result, details = histogram_match(source, reference)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(details["channel"], "L (LAB)")

    def test_low_contrast_dark_bright_noisy_and_large_images(self):
        low = np.array([[100, 102, 105, 110]], np.uint8)
        stretched, _ = enhance(low, "stretch")
        self.assertEqual((int(stretched.min()), int(stretched.max())), (0, 255))
        dark = np.full((32, 32), 18, np.uint8); dark[::3, ::3] = 25
        bright = np.full((32, 32), 238, np.uint8); bright[::3, ::3] = 250
        noisy = np.random.default_rng(4).integers(80, 176, (64, 64), dtype=np.uint8)
        for sample in (dark, bright, noisy, np.zeros((128, 128), np.uint8)):
            output, _ = enhance(sample, "equalize")
            self.assertEqual(output.shape, sample.shape)
            self.assertEqual(output.dtype, np.uint8)

    def test_color_enhancement_keeps_bgr_convention_and_alpha_is_flattened(self):
        red_bgr = np.full((3, 3, 3), (0, 0, 255), np.uint8)
        unchanged, _ = enhance(red_bgr, "equalize")
        self.assertTrue(np.array_equal(unchanged, red_bgr))
        rgba = np.full((3, 3, 4), (0, 0, 255, 128), np.uint8)
        result, _ = enhance(rgba, "gamma", {"gamma": 1.4})
        self.assertEqual(result.shape, (3, 3, 3))

    def test_enhancement_parameter_validation(self):
        with self.assertRaises(ValueError):
            enhance(np.arange(16, dtype=np.uint8).reshape(4, 4), "gamma", {"gamma": 0})
        with self.assertRaises(ValueError):
            enhance(np.arange(16, dtype=np.uint8).reshape(4, 4), "clahe", {"tile_size": 100})

    def test_otsu_binary_inverse_and_degenerate_warning(self):
        image = np.array([[0, 0, 255, 255]], dtype=np.uint8)
        binary, stats = otsu(image)
        inverse, _ = otsu(image, "binary_inv")
        self.assertEqual(stats["threshold"], 0)
        self.assertTrue(np.array_equal(binary.astype(int) + inverse.astype(int), np.full(image.shape, 255)))
        _, flat_stats = otsu(np.full((5, 5), 10, np.uint8))
        self.assertTrue(flat_stats["degenerate"])

    def test_morphology_is_independent_and_validated(self):
        mask = np.zeros((25, 25), np.uint8)
        mask[7:18, 7:18] = 255
        mask[10, 10] = 0
        mask[2, 2] = 255
        shape_flags = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}
        for shape, flag in shape_flags.items():
            kernel = cv2.getStructuringElement(flag, (3, 3))
            self.assertTrue(np.array_equal(morphology(mask, "erode", 3, shape, 1), cv2.erode(mask, kernel, iterations=1)))
            self.assertTrue(np.array_equal(morphology(mask, "dilate", 3, shape, 2), cv2.dilate(mask, kernel, iterations=2)))
            self.assertTrue(np.array_equal(morphology(mask, "open", 3, shape, 1), cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)))
            self.assertTrue(np.array_equal(morphology(mask, "close", 3, shape, 1), cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)))
        self.assertTrue(np.array_equal(morphology(mask, "none"), mask))
        self.assertEqual(morphology(mask, "dilate", size=1, iterations=1).shape, mask.shape)
        for kwargs in ({"operation": "unknown"}, {"shape": "triangle"}, {"size": 0}, {"size": 4}, {"size": 33}, {"iterations": 0}, {"iterations": 11}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                morphology(mask, **kwargs)

    def test_otsu_between_class_variance_against_independent_reference(self):
        image = np.array([[10, 10, 20, 20, 200, 200, 240, 240]], dtype=np.uint8)
        _mask, stats = otsu(image)
        pixels = image.astype(np.float64).ravel()
        candidates = []
        for threshold in range(256):
            lower, upper = pixels[pixels <= threshold], pixels[pixels > threshold]
            if not lower.size or not upper.size:
                variance = 0.0
            else:
                w0, w1 = lower.size / pixels.size, upper.size / pixels.size
                variance = w0 * w1 * (lower.mean() - upper.mean()) ** 2
            candidates.append(variance)
        reference_threshold = int(np.argmax(candidates))
        self.assertEqual(stats["threshold"], reference_threshold)
        self.assertAlmostEqual(stats["between_class_variance"], candidates[reference_threshold], places=10)
        self.assertEqual(len(stats["variance_curve"]), 256)
        self.assertAlmostEqual(max(stats["variance_curve"]), candidates[reference_threshold], places=10)
        self.assertAlmostEqual(stats["foreground_ratio"] + stats["background_ratio"], 1.0)

    def test_otsu_uniform_and_single_foreground_level_are_degenerate(self):
        mask, stats = otsu(np.full((3, 4), 77, np.uint8))
        self.assertTrue(stats["degenerate"])
        self.assertEqual(stats["between_class_variance"], 0.0)
        self.assertEqual(np.unique(mask).size, 1)
        self.assertIn(stats["foreground_ratio"], (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
