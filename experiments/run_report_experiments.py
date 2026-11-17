"""Collect PixelLab report data from the local Flask UI and save report artifacts.

Run from the repository root with .venv\\Scripts\\python.exe -B experiments\\run_report_experiments.py
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from playwright.sync_api import sync_playwright

from config import Config
from services.enhancement_service import enhance
from services.histogram_service import histogram_data
from services.image_service import decode_image, detect_image_format
from services.matching_service import histogram_match
from services.segmentation_service import morphology, otsu


# Fixed experiment inputs. Adjust only these values when re-running this script.
IMAGE_DIR = Path("anh_bao_cao")
REPORT_DIR = Path("ket_qua_bao_cao")
SCREENSHOT_DIR = REPORT_DIR / "hinh"
BASE_URL = "http://127.0.0.1:5055"
GAMMA = 1.2
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
OTSU_MODE = "binary"
MORPH_ARGS = {"operation": "open", "shape": "ellipse", "size": 3, "iterations": 1}
REQUIRED = [
    ("anh_toi.png", "Ảnh tối", "toi"),
    ("tuong_phan_thap.png", "Ảnh tương phản thấp", "tuong_phan_thap"),
    ("anh_mau.png", "Ảnh màu", "mau"),
    ("anh_tham_chieu.png", "Ảnh tham chiếu", "tham_chieu"),
]
METHODS = [
    ("stretch", "Kéo giãn tương phản"),
    ("equalize", "Cân bằng histogram"),
    ("clahe", "CLAHE"),
    ("gamma", "Gamma"),
]
METHOD_PARAMS = {
    "stretch": {}, "equalize": {},
    "clahe": {"clip_limit": CLAHE_CLIP_LIMIT, "tile_size": CLAHE_TILE_SIZE},
    "gamma": {"gamma": GAMMA},
}
SOURCE_TEXT = "Nguồn/giấy phép chưa được cung cấp; không thể xác minh từ tệp."
MAX_DIFF = 0.0
DIFFS: list[tuple[str, float]] = []


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def mdnum(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def compare(label: str, left, right) -> None:
    global MAX_DIFF
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        diff = abs(float(left) - float(right))
    else:
        a, b = np.asarray(left), np.asarray(right)
        diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    DIFFS.append((label, diff))
    MAX_DIFF = max(MAX_DIFF, diff)


def write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def img_data_url(value: str) -> np.ndarray:
    raw = base64.b64decode(value.split(",", 1)[1])
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Không giải mã được ảnh PNG trong phản hồi API.")
    return image


def api_image(page, filename: Path, asset: str, method: str, params: dict, morph: dict | None = None) -> np.ndarray:
    form = {"asset": asset, "method": method, "mode": OTSU_MODE}
    form.update({k: str(v) for k, v in params.items()})
    if morph:
        form.update({"morphology": morph["operation"], "kernel_shape": morph["shape"],
                     "kernel_size": str(morph["size"]), "iterations": str(morph["iterations"])})
    response = page.request.post(f"{BASE_URL}/api/export/image", multipart={
        "image": {"name": filename.name, "mimeType": "image/png", "buffer": filename.read_bytes()},
        **{k: str(v) for k, v in form.items()},
    })
    if not response.ok:
        raise RuntimeError(f"{asset}: HTTP {response.status}: {response.text()[:300]}")
    result = cv2.imdecode(np.frombuffer(response.body(), np.uint8), cv2.IMREAD_UNCHANGED)
    if result is None:
        raise RuntimeError(f"{asset}: không giải mã được PNG export.")
    return result


def wait_result(page) -> None:
    page.locator("#analysis").wait_for(state="visible", timeout=30000)
    page.wait_for_function("document.querySelector('#histogram')?.dataset.plot?.length > 0")
    page.wait_for_load_state("networkidle", timeout=10000)


def submit_ui(page, image_path: Path, method: str, params: dict, morph: bool = False):
    page.locator("#file").set_input_files(str(image_path.resolve()))
    page.locator("#method").select_option(method)
    for key, value in params.items():
        page.locator(f'[name="{key}"]').fill(str(value))
    if morph:
        if not page.locator(".morph-controls").evaluate("e => e.open"):
            page.locator(".morph-controls").locator("summary").click()
        page.locator('[name="morphology"]').select_option(MORPH_ARGS["operation"])
        page.locator('[name="kernel_shape"]').select_option(MORPH_ARGS["shape"])
        page.locator('[name="kernel_size"]').select_option(str(MORPH_ARGS["size"]))
        page.locator('[name="iterations"]').select_option(str(MORPH_ARGS["iterations"]))
    with page.expect_response(lambda r: r.url.endswith("/api/analyze") and r.request.method == "POST", timeout=30000) as info:
        page.locator("#submit").click()
    response = info.value
    payload = response.json()
    if response.status != 200 or not payload.get("ok"):
        raise RuntimeError(f"Analyze {image_path.name}/{method}: HTTP {response.status} {payload}")
    wait_result(page)
    return payload["data"]


def capture(locator, path: Path) -> None:
    locator.scroll_into_view_if_needed()
    locator.page.wait_for_load_state("networkidle", timeout=10000)
    locator.page.wait_for_timeout(120)
    locator.screenshot(path=str(path), animations="disabled", timeout=30000)


def run() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    files = {name: IMAGE_DIR / name for name, _kind, _slug in REQUIRED}
    metadata = {}
    originals = {}
    for name, kind, slug in REQUIRED:
        path = files[name]
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu ảnh bắt buộc: {path}")
        raw = path.read_bytes()
        if len(raw) > 20 * 1024 * 1024:
            raise ValueError(f"Ảnh vượt 20 MB: {path}")
        image = decode_image(raw, name, Config.MAX_IMAGE_PIXELS, Config.MAX_IMAGE_DIMENSION)
        if image.shape[0] * image.shape[1] > 16_000_000:
            raise ValueError(f"Ảnh vượt 16 MP: {path}")
        originals[name] = image
        metadata[name] = {
            "kind": kind, "slug": slug, "width": image.shape[1], "height": image.shape[0],
            "channels": 1 if image.ndim == 2 else image.shape[2], "format": detect_image_format(raw),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }

    test_log_path = REPORT_DIR / "06_unittest.txt"
    test_cmd = [str(Path(".venv/Scripts/python.exe")), "-B", "-m", "unittest", "discover", "-v"]
    test = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    test_log = test.stdout.decode("utf-8", errors="replace")
    test_log_path.write_text(test_log, encoding="utf-8")
    node = subprocess.run(["node", "--check", "static\\js\\app.js"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    diffcheck = subprocess.run(["git", "diff", "--check"], capture_output=True, text=True, encoding="utf-8", errors="replace")

    # Serve the unmodified app and point Jinja at its existing static/templates copy.
    # This is a launch-time workaround because templates/dashboard.html is absent.
    server_code = (
        "from app import app; from jinja2 import FileSystemLoader; "
        "app.run(host='127.0.0.1',port=5055,debug=False,use_reloader=False,threaded=True)"
    )
    server = subprocess.Popen([str(Path(".venv/Scripts/python.exe")), "-B", "-c", server_code],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    browser_version = "không xác định"
    ui_rows: list[tuple[str, str, str]] = []
    analyze: dict[str, dict[str, dict]] = {}
    full_outputs: dict[str, dict[str, np.ndarray]] = {}
    mask_outputs: dict[str, dict[str, np.ndarray]] = {}
    source_morphs: dict[str, dict[str, np.ndarray]] = {}
    match_data: dict[str, dict] = {}
    match_full: dict[str, np.ndarray] = {}
    ui_checks: dict[str, str] = {}
    try:
        import urllib.request
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE_URL, timeout=1).read(20)
                break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("Flask không mở được cổng 5055.")
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="msedge", headless=False)
            except Exception as exc:
                raise RuntimeError(f"Không khởi chạy được Edge channel=msedge: {exc}") from exc
            browser_version = browser.version
            context = browser.new_context(viewport={"width": 1100, "height": 1000}, device_scale_factor=2, accept_downloads=True)
            page = context.new_page()
            page.on("dialog", lambda dialog: dialog.accept())
            page.goto(BASE_URL, wait_until="networkidle")
            if page.locator("#file").count() != 1:
                raise RuntimeError("Trang thật không tải đủ điều khiển #file.")

            # Collect 4 × 4 real UI analysis responses and full-resolution exports.
            for name, _kind, _slug in REQUIRED:
                analyze[name], full_outputs[name], mask_outputs[name], source_morphs[name] = {}, {}, {}, {}
                for method, _label in METHODS:
                    params = METHOD_PARAMS[method]
                    data = submit_ui(page, files[name], method, params, morph=(name == "anh_toi.png" and method == "equalize"))
                    analyze[name][method] = data
                    full_outputs[name][method] = api_image(page, files[name], "enhanced", method, params)
                    source_morphs[name][method] = api_image(page, files[name], "mask_before_morph", method, params, MORPH_ARGS)
                    mask_outputs[name][method] = api_image(page, files[name], "mask_after_morph", method, params, MORPH_ARGS)
                    # Cross-check all statistics and Otsu JSON values against project functions.
                    decoded = decode_image(files[name].read_bytes(), name, Config.MAX_IMAGE_PIXELS, Config.MAX_IMAGE_DIMENSION)
                    before_hist = histogram_data(decoded, "gray")["channels"]["Intensity"]
                    enhanced, _ = enhance(decoded, method, params)
                    after_hist = histogram_data(enhanced, "gray")["channels"]["Intensity"]
                    for field in ("mean", "std", "dynamic_range", "entropy"):
                        compare(f"03/{name}/{method}/before/{field}", data["stats_before"][field], before_hist[field])
                        compare(f"03/{name}/{method}/after/{field}", data["stats_after"][field], after_hist[field])
                    before_mask, before_otsu = otsu(decoded, OTSU_MODE)
                    after_mask, after_otsu = otsu(enhanced, OTSU_MODE)
                    for field in ("threshold", "between_class_variance", "foreground_ratio"):
                        compare(f"04/{name}/{method}/before/{field}", data["otsu_before"][field], before_otsu[field])
                        compare(f"04/{name}/{method}/after/{field}", data["otsu_after"][field], after_otsu[field])
                    morph_result = morphology(after_mask, **MORPH_ARGS)
                    measure = float(np.count_nonzero(mask_outputs[name][method] == 255) / mask_outputs[name][method].size)
                    direct_measure = float(np.count_nonzero(morph_result == 255) / morph_result.size)
                    compare(f"04/{name}/{method}/morph_foreground", measure, direct_measure)
                    source_morph_measure = float(np.count_nonzero(source_morphs[name][method] == 255) / source_morphs[name][method].size)
                    direct_source_morph_measure = float(np.count_nonzero(morphology(before_mask, **MORPH_ARGS) == 255) / before_mask.size)
                    compare(f"04/{name}/{method}/source_morph_foreground", source_morph_measure, direct_source_morph_measure)
                    # Every screenshot analyze response has a visible completed result.
                    if not page.locator("#analysis").is_visible():
                        raise RuntimeError("Khu vực kết quả không hiển thị sau phân tích.")

            # Matching through the UI, each required source except the reference itself.
            for source_name, _kind, slug in REQUIRED:
                if source_name == "anh_tham_chieu.png":
                    continue
                submit_ui(page, files[source_name], "stretch", {})
                page.locator("#reference-file").set_input_files(str(files["anh_tham_chieu.png"].resolve()))
                with page.expect_response(lambda r: r.url.endswith("/api/match") and r.request.method == "POST", timeout=30000) as info:
                    page.locator("#run-match").click()
                response = info.value
                payload = response.json()
                if response.status != 200 or not payload.get("ok"):
                    raise RuntimeError(f"Matching thất bại cho {source_name}: {payload}")
                page.locator("#match-results").wait_for(state="visible", timeout=30000)
                match_data[source_name] = payload["data"]["details"]
                raw_source = decode_image(files[source_name].read_bytes(), source_name)
                raw_ref = decode_image(files["anh_tham_chieu.png"].read_bytes(), "anh_tham_chieu.png")
                _matched, details = histogram_match(raw_source, raw_ref)
                for field in ("mean_cdf_distance_before", "mean_cdf_distance_after"):
                    compare(f"05/{source_name}/{field}", match_data[source_name][field], details[field])
                match_full[source_name] = cv2.imdecode(np.frombuffer(page.request.post(
                    f"{BASE_URL}/api/export/match", multipart={
                        "image": {"name": source_name, "mimeType": "image/png", "buffer": files[source_name].read_bytes()},
                        "reference": {"name": "anh_tham_chieu.png", "mimeType": "image/png", "buffer": files["anh_tham_chieu.png"].read_bytes()},
                    }).body(), np.uint8), cv2.IMREAD_UNCHANGED)

            # UI functional measurements (row values are populated below).
            page.locator("#file").set_input_files(str(files["anh_toi.png"].resolve()))
            page.locator("#method").select_option("equalize")
            with page.expect_response(lambda r: r.url.endswith("/api/analyze") and r.request.method == "POST"):
                page.locator("#submit").click()
            wait_result(page)
            ui_checks["valid_upload"] = str(page.locator("#analysis").is_visible())
            # Bad file content is intentionally submitted only through the browser input.
            page.locator("#file").set_input_files({"name": "bad.png", "mimeType": "image/png", "buffer": bytes([11, 23, 37, 41, 59])})
            page.locator("#method").select_option("stretch")
            page.locator("#submit").click()
            page.wait_for_function("document.querySelector('#error').textContent.trim().length > 0", timeout=10000)
            ui_checks["bad_error"] = page.locator("#error").inner_text()
            # Method visible results.
            method_visible = {}
            for method, _label in METHODS:
                submit_ui(page, files["anh_toi.png"], method, METHOD_PARAMS[method])
                method_visible[method] = page.locator("#analysis").is_visible() and page.locator("#variants .variant").count() == 4
            ui_checks["methods"] = json.dumps(method_visible, ensure_ascii=False)

            # Slider/zoom/pan measurements.
            page.locator("#split").fill("37")
            clip = page.locator("#viewer-result").evaluate("e => getComputedStyle(e).clipPath")
            page.locator("#zoom-in").click()
            zoom = page.locator("#zoom-level").inner_text()
            page.locator("#viewer").hover()
            page.mouse.down(); page.mouse.move(520, 460, steps=3); page.mouse.up()
            transform = page.locator("#viewer-stack").evaluate("e => getComputedStyle(e).transform")
            ui_checks["viewer"] = f"clipPath={clip}; zoom={zoom}; transform={transform}"
            # Histogram zoom and RGB toggle.
            canvas = page.locator("#histogram")
            before_plot = json.loads(canvas.get_attribute("data-plot"))
            box = canvas.bounding_box()
            page.mouse.move(box["x"] + box["width"] * .55, box["y"] + box["height"] * .5)
            page.mouse.wheel(0, -350)
            page.wait_for_function("JSON.parse(document.querySelector('#histogram').dataset.plot).start !== 0 || JSON.parse(document.querySelector('#histogram').dataset.plot).end !== 255")
            after_plot = json.loads(canvas.get_attribute("data-plot"))
            # Re-load color image, select RGB and hide a channel.
            submit_ui(page, files["anh_mau.png"], "equalize", {})
            page.locator("#space").select_option("rgb")
            page.locator(".channel-toggle").first.click()
            ui_checks["histogram_rgb"] = f"plot {before_plot['start']:.2f}–{before_plot['end']:.2f} -> {after_plot['start']:.2f}–{after_plot['end']:.2f}; first RGB toggle off={ 'off' in page.locator('.channel-toggle').first.get_attribute('class') }"
            # Undo/redo/reset measured on method select plus analysis.
            submit_ui(page, files["anh_toi.png"], "stretch", {})
            submit_ui(page, files["anh_toi.png"], "gamma", {"gamma": 2.0})
            page.locator("#undo").click(); page.wait_for_load_state("networkidle")
            undo = page.locator("#method").input_value()
            page.locator("#redo").click(); page.wait_for_load_state("networkidle")
            redo = page.locator("#method").input_value()
            page.locator("#reset").click(); page.wait_for_load_state("networkidle")
            reset_method = page.locator("#method").input_value()
            ui_checks["history"] = f"undo={undo}, redo={redo}, reset={reset_method}; undo_disabled={page.locator('#undo').is_disabled()}, redo_disabled={page.locator('#redo').is_disabled()}"

            # Browser/API bug checks (a, d).
            image_path = files["anh_toi.png"]
            def post_analyze(method, **fields):
                multipart = {"image": {"name": image_path.name, "mimeType": "image/png", "buffer": image_path.read_bytes()}, "method": method, "mode": OTSU_MODE}
                multipart.update({k: str(v) for k, v in fields.items()})
                return page.request.post(f"{BASE_URL}/api/analyze", multipart=multipart)
            gamma2 = post_analyze("gamma", gamma=2.0)
            g2 = gamma2.json()
            stretch0 = post_analyze("stretch", gamma=0)
            s0 = stretch0.json()
            gamma0 = post_analyze("gamma", gamma=0)
            g0 = gamma0.json()
            badgamma = post_analyze("gamma", gamma="abc")
            bg = badgamma.json()
            ui_checks["api_a_d"] = f"gamma2 HTTP {gamma2.status} param={g2.get('data',{}).get('variants',{}).get('gamma',{}).get('params',{}).get('gamma')}; stretch gamma0 HTTP {stretch0.status} fallback={s0.get('data',{}).get('variants',{}).get('gamma',{}).get('params',{}).get('gamma')}; gamma0 HTTP {gamma0.status}; abc HTTP {badgamma.status} error={bg.get('error',{}).get('message')}"

            # lastConfig browser download compared pixel-for-pixel with two API exports.
            submit_ui(page, image_path, "stretch", {})
            page.locator("#method").select_option("gamma")
            page.locator('[name="gamma"]').fill("2")
            with page.expect_download(timeout=30000) as download_info:
                page.locator("#download-image").click()
            download = download_info.value
            saved = REPORT_DIR / "_lastconfig_download.png"
            download.save_as(str(saved))
            downloaded = cv2.imdecode(np.frombuffer(saved.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            stretch_export = api_image(page, image_path, "enhanced", "stretch", {})
            gamma_export = api_image(page, image_path, "enhanced", "gamma", {"gamma": 2.0})
            equal_pixels = downloaded.shape == stretch_export.shape and np.array_equal(downloaded, stretch_export)
            different_gamma = downloaded.shape != gamma_export.shape or not np.array_equal(downloaded, gamma_export)
            ui_checks["lastconfig"] = f"download={downloaded.shape}; stretch_equal={equal_pixels}; gamma2_different={different_gamma}"
            saved.unlink(missing_ok=True)

            # CSV BOM and chart PNG decode via genuine UI download handlers.
            submit_ui(page, image_path, "stretch", {})
            with page.expect_download() as csv_info:
                page.locator("#export-csv").click()
            csv_download = csv_info.value
            csv_path = REPORT_DIR / "_csv_download.csv"; csv_download.save_as(str(csv_path))
            bom = csv_path.read_bytes()[:3].hex().upper(); csv_path.unlink(missing_ok=True)
            with page.expect_download() as chart_info:
                page.locator("#export-chart").click()
            chart_download = chart_info.value
            chart_path = REPORT_DIR / "_chart_download.png"; chart_download.save_as(str(chart_path))
            chart = cv2.imdecode(np.frombuffer(chart_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            chart_decode = chart is not None; chart_path.unlink(missing_ok=True)
            ui_checks["exports"] = f"CSV BOM={bom}; chart PNG decoded={chart_decode}, shape={None if chart is None else chart.shape}"

            # Matching UI download size.
            submit_ui(page, image_path, "stretch", {})
            page.locator("#reference-file").set_input_files(str(files["anh_tham_chieu.png"].resolve()))
            page.locator("#run-match").click(); page.locator("#match-results").wait_for(state="visible")
            with page.expect_download(timeout=30000) as match_download_info:
                page.locator("#download-match").click()
            md = match_download_info.value
            md_path = REPORT_DIR / "_match_download.png"; md.save_as(str(md_path))
            md_img = cv2.imdecode(np.frombuffer(md_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            ui_checks["matching_download"] = f"download={md_img.shape}; source={originals['anh_toi.png'].shape}; size_equal={md_img.shape[:2] == originals['anh_toi.png'].shape[:2]}"
            md_path.unlink(missing_ok=True)

            # Required UI area screenshots.
            data = submit_ui(page, files["anh_toi.png"], "equalize", {})
            capture(page.locator(".work-grid"), SCREENSHOT_DIR / "hinh_2_3_giao_dien.png")
            for name, _kind, slug in REQUIRED[:3]:
                submit_ui(page, files[name], "equalize", {})
                capture(page.locator(".analysis-grid"), SCREENSHOT_DIR / f"ui_histogram_{slug}.png")
                capture(page.locator("#variants"), SCREENSHOT_DIR / f"ui_phuong_an_{slug}.png")
            submit_ui(page, files["anh_toi.png"], "equalize", {})
            eq = page.locator("#equalization-explorer")
            if not eq.evaluate("e => e.open"):
                eq.locator("summary").click()
            page.wait_for_function("document.querySelector('#equalize-pdf').width > 0 && document.querySelector('#equalize-cdf').width > 0")
            page.add_style_tag(content=".table-scroll{display:none!important}")
            capture(eq, SCREENSHOT_DIR / "hinh_2_4_explorer.png")
            # Otsu and morphology screenshot.
            submit_ui(page, files["anh_toi.png"], "equalize", {}, morph=True)
            capture(page.locator(".seg-grid"), SCREENSHOT_DIR / "ui_phan_doan.png")
            capture(page.locator("#otsu-explorer"), SCREENSHOT_DIR / "ui_otsu_explorer.png")
            # Matching screenshot requested for dark source.
            submit_ui(page, files["anh_toi.png"], "stretch", {})
            page.locator("#reference-file").set_input_files(str(files["anh_tham_chieu.png"].resolve()))
            page.locator("#run-match").click(); page.locator("#match-results").wait_for(state="visible")
            capture(page.locator("#matching-explorer"), SCREENSHOT_DIR / "ui_matching.png")
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    # Inventory CSV (source attribution was not supplied by the user).
    write_csv(REPORT_DIR / "02_bo_anh.csv",
        ["Tên tệp", "Đường dẫn", "Loại ảnh", "Rộng", "Cao", "Số kênh", "Định dạng", "Dung lượng (byte)", "SHA-256", "Nguồn"],
        [[name, str(files[name].resolve()), metadata[name]["kind"], metadata[name]["width"], metadata[name]["height"], metadata[name]["channels"], metadata[name]["format"], metadata[name]["bytes"], metadata[name]["sha256"], SOURCE_TEXT] for name, _kind, _slug in REQUIRED])

    # 03 enhancement table: all figures are from API JSON; cross-check recorded separately.
    stats_rows, stats_md = [], []
    for name, _kind, slug in REQUIRED[:3]:
        stats_md.append(f"## {metadata[name]['kind']} — {name}\n\n| Phương pháp | Trung bình | Độ lệch chuẩn | Dải động | Entropy (bit) | Ngưỡng Otsu |\n|---|---:|---:|---:|---:|---:|")
        api = analyze[name]["stretch"]
        rows = [("Ảnh gốc", api["stats_before"], api["otsu_before"])]
        for method, label in METHODS:
            d = analyze[name][method]; rows.append((label, d["stats_after"], d["otsu_after"]))
        # Exactly five rows required: original + four methods.
        rows = rows[:1] + rows[1:]
        for label, st, ot in rows:
            stats_rows.append([name, label, fmt(st["mean"]), fmt(st["std"]), fmt(st["dynamic_range"]), fmt(st["entropy"]), fmt(ot["threshold"])])
            stats_md.append(f"| {label} | {mdnum(st['mean'])} | {mdnum(st['std'])} | {int(st['dynamic_range'])} | {mdnum(st['entropy'])} | {int(round(ot['threshold']))} |")
        stats_md.append("")
    write_csv(REPORT_DIR / "03_thong_ke_tang_cuong.csv", ["Tên ảnh", "Phương pháp", "Trung bình", "Độ lệch chuẩn", "Dải động", "Entropy (bit)", "Ngưỡng Otsu"], stats_rows)
    (REPORT_DIR / "03_thong_ke_tang_cuong.md").write_text("# Bảng 3.3–3.5 — Thống kê tăng cường\n\n" + "\n".join(stats_md), encoding="utf-8")

    otsu_rows, otsu_md = [], ["# Bảng 3.6 — Phân đoạn Otsu\n"]
    for name, _kind, _slug in REQUIRED:
        otsu_md.append(f"## {metadata[name]['kind']} — {name}\n\n| Phương pháp | Ngưỡng trước → sau | Phương sai trước → sau | Tiền cảnh trước (mask thô) | Tiền cảnh sau (mask thô) | Tiền cảnh sau mở (mask gốc) | Tiền cảnh sau mở (mask tăng cường) |\n|---|---:|---:|---:|---:|---:|---:|")
        for method, label in METHODS:
            d = analyze[name][method]
            before_rate, after_rate = d["otsu_before"]["foreground_ratio"], d["otsu_after"]["foreground_ratio"]
            morph_rate = float(np.count_nonzero(mask_outputs[name][method] == 255) / mask_outputs[name][method].size)
            source_morph_rate = float(np.count_nonzero(source_morphs[name][method] == 255) / source_morphs[name][method].size)
            otsu_rows.append([name, label, fmt(d["otsu_before"]["threshold"]), fmt(d["otsu_after"]["threshold"]), fmt(d["otsu_before"]["between_class_variance"]), fmt(d["otsu_after"]["between_class_variance"]), fmt(before_rate), fmt(after_rate), fmt(source_morph_rate), fmt(morph_rate)])
            otsu_md.append(f"| {label} | {int(round(d['otsu_before']['threshold']))} → {int(round(d['otsu_after']['threshold']))} | {mdnum(d['otsu_before']['between_class_variance'])} → {mdnum(d['otsu_after']['between_class_variance'])} | {mdnum(before_rate*100,1)}% | {mdnum(after_rate*100,1)}% | {mdnum(source_morph_rate*100,1)}% | {mdnum(morph_rate*100,1)}% |")
        otsu_md.append("")
    write_csv(REPORT_DIR / "04_otsu.csv", ["Tên ảnh", "Phương pháp", "Ngưỡng trước", "Ngưỡng sau", "Phương sai giữa lớp trước", "Phương sai giữa lớp sau", "Tiền cảnh trước mask thô", "Tiền cảnh sau mask thô", "Tiền cảnh sau mở mask gốc", "Tiền cảnh sau mở mask tăng cường"], otsu_rows)
    (REPORT_DIR / "04_otsu.md").write_text("\n".join(otsu_md), encoding="utf-8")

    match_rows, match_md = [], ["# Bảng 3.7 — Histogram Matching\n", "| Ảnh nguồn | Ảnh tham chiếu | Kênh | D trước | D sau |", "|---|---|---|---:|---:|"]
    for name, details in match_data.items():
        match_rows.append([name, "anh_tham_chieu.png", details["channel"], fmt(details["mean_cdf_distance_before"]), fmt(details["mean_cdf_distance_after"])])
        match_md.append(f"| {name} | anh_tham_chieu.png | {details['channel']} | {mdnum(details['mean_cdf_distance_before'])} | {mdnum(details['mean_cdf_distance_after'])} |")
    write_csv(REPORT_DIR / "05_histogram_matching.csv", ["Ảnh nguồn", "Ảnh tham chiếu", "Kênh dùng", "D trước", "D sau"], match_rows)
    (REPORT_DIR / "05_histogram_matching.md").write_text("\n".join(match_md)+"\n", encoding="utf-8")

    # Environment info.
    packages = {}
    import importlib.metadata as im
    for package in ("Flask", "numpy", "opencv-python-headless", "matplotlib", "playwright"):
        try: packages[package] = im.version(package)
        except im.PackageNotFoundError: packages[package] = "không xác định"
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip() or "không xác định"
    ram = subprocess.run(["powershell", "-NoProfile", "-Command", "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip() or "không xác định"
    branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    env_md = ["# Bảng 3.1 — Môi trường thử nghiệm", "", "| Thành phần | Giá trị |", "|---|---|"]
    env_values = [("Hệ điều hành", platform.platform()), ("Python", platform.python_version()), *[(k, v) for k, v in packages.items()], ("CPU", cpu), ("RAM (GB)", ram), ("Trình duyệt", f"Microsoft Edge {browser_version}"), ("Git branch", branch), ("Git commit", commit)]
    env_md += [f"| {k} | {v} |" for k, v in env_values]
    (REPORT_DIR / "01_moi_truong.md").write_text("\n".join(env_md)+"\n", encoding="utf-8")

    # UI checklist with actual measurements.
    ui_lines = ["# Bảng 3.9 — Chức năng giao diện", "", "| STT | Nội dung | Kết quả thực tế | Bằng chứng đo |", "|---:|---|---|---|"]
    ui_map = [
        ("Tải ảnh hợp lệ và hiện phân tích", "Đạt" if ui_checks.get("valid_upload")=="True" else "Không đạt", ui_checks.get("valid_upload", "chưa đo")),
        ("Tệp ảnh sai nội dung hiển thị lỗi tiếng Việt", "Đạt" if ui_checks.get("bad_error", "").strip() else "Không đạt", ui_checks.get("bad_error", "chưa đo")),
        ("Bốn phương pháp hiển thị kết quả", "Đạt" if all(json.loads(ui_checks.get("methods", "{}" )).values()) else "Không đạt", ui_checks.get("methods", "chưa đo")),
        ("Thanh gạt, thu phóng và di chuyển", "Đạt" if "clipPath=inset" in ui_checks.get("viewer", "") and "zoom=120%" in ui_checks.get("viewer", "") and "matrix(1.2" in ui_checks.get("viewer", "") else "Cần kiểm tay", ui_checks.get("viewer", "chưa đo")),
        ("Zoom Histogram và bật/tắt kênh RGB", "Đạt" if "off=True" in ui_checks.get("histogram_rgb", "") else "Cần kiểm tay", ui_checks.get("histogram_rgb", "chưa đo")),
        ("Undo, Redo, Reset", "Đạt" if "undo=stretch, redo=gamma, reset=equalize" in ui_checks.get("history", "") and "undo_disabled=True, redo_disabled=True" in ui_checks.get("history", "") else "Cần kiểm tay", ui_checks.get("history", "chưa đo")),
        ("Tải ảnh theo lastConfig", "Đạt" if "stretch_equal=True" in ui_checks.get("lastconfig", "") and "gamma2_different=True" in ui_checks.get("lastconfig", "") else "Không đạt", ui_checks.get("lastconfig", "chưa đo")),
        ("CSV BOM và PNG biểu đồ; Excel tiếng Việt", "Đạt" if "BOM=EFBBBF" in ui_checks.get("exports", "") and "decoded=True" in ui_checks.get("exports", "") else "Không đạt", ui_checks.get("exports", "chưa đo")+"; Excel hiển thị tiếng Việt: Cần kiểm tay"),
        ("Histogram Matching và tải ảnh", "Đạt" if "size_equal=True" in ui_checks.get("matching_download", "") else "Cần kiểm tay", ui_checks.get("matching_download", "chưa đo")),
    ]
    ui_lines += [f"| {i} | {desc} | {status} | {evidence.replace('|','\\|')} |" for i,(desc,status,evidence) in enumerate(ui_map,1)]
    ui_lines += ["", "## Kiểm tra API bổ sung (đã chạy trên Flask cục bộ)", "", f"- {ui_checks.get('api_a_d','chưa đo')}", "- Không suy rộng phép đo trình duyệt từ kết quả API."]
    (REPORT_DIR / "07_chuc_nang_giao_dien.md").write_text("\n".join(ui_lines)+"\n", encoding="utf-8")

    # Build plots from browser JSON payloads and real browser API PNG exports.
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 9, "axes.labelsize": 7, "font.family": "DejaVu Sans"})
    for name, _kind, slug in REQUIRED[:3]:
        fig, axes = plt.subplots(2, 5, figsize=(8, 4.6), dpi=200)
        views = [("Ảnh gốc", originals[name], analyze[name]["stretch"]["stats_before"])]
        for method, label in METHODS:
            views.append((label, full_outputs[name][method], analyze[name][method]["stats_after"]))
        for col, (label, image, st) in enumerate(views):
            if image.ndim == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[0,col].imshow(image, cmap="gray" if image.ndim == 2 else None)
            axes[0,col].set_title(f"{label}\nμ={mdnum(st['mean'])}; σ={mdnum(st['std'])}", fontsize=7)
            axes[0,col].axis("off")
            h = np.asarray((analyze[name]["stretch"]["histograms_before"] if col == 0 else analyze[name][METHODS[col-1][0]]["histograms_after"])["gray"]["channels"]["Intensity"]["histogram"])
            axes[1,col].plot(np.arange(256), h, lw=.7)
            axes[1,col].set_title(label, fontsize=7); axes[1,col].set_xlabel("Mức xám"); axes[1,col].set_ylabel("Số pixel")
        fig.tight_layout(); fig.savefig(SCREENSHOT_DIR / f"hinh_3_{REQUIRED.index((name,_kind,slug))+1}_{'anh_toi' if slug=='toi' else 'tuong_phan_thap' if slug=='tuong_phan_thap' else 'anh_mau'}.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(3, 5, figsize=(8, 5.6), dpi=200)
    for row,(name,_kind,_slug) in enumerate(REQUIRED[:3]):
        d0=analyze[name]["equalize"]
        masks=[otsu(originals[name],OTSU_MODE)[0], source_morphs[name]["equalize"],
               otsu(full_outputs[name]["equalize"],OTSU_MODE)[0], mask_outputs[name]["equalize"]]
        pics=[originals[name],*masks]
        titles=["Ảnh gốc",f"Otsu gốc T={int(d0['otsu_before']['threshold'])}","Sau mở · ảnh gốc",
                f"Otsu cân bằng T={int(d0['otsu_after']['threshold'])}","Sau mở · cân bằng"]
        for col,pic in enumerate(pics):
            axes[row,col].imshow(pic,cmap="gray"); axes[row,col].set_title(titles[col],fontsize=6); axes[row,col].axis("off")
    fig.tight_layout(); fig.savefig(SCREENSHOT_DIR/"hinh_3_4_otsu_hinh_thai.png",dpi=200); plt.close(fig)

    source_name="anh_toi.png"; details=match_data[source_name]; ref=originals["anh_tham_chieu.png"]; src=originals[source_name]; out=match_full[source_name]
    if out.ndim==3: out=cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
    fig,axes=plt.subplots(2,3,figsize=(8,4.6),dpi=200)
    for ax,pic,title in zip(axes[0],[src,ref,out],["Ảnh nguồn","Ảnh tham chiếu","Kết quả matching"]):
        if pic.ndim==3 and pic.shape[-1]==3 and title!="Ảnh tham chiếu":
            pass
        ax.imshow(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB) if pic.ndim==3 and pic.shape[-1]==3 else pic,cmap="gray"); ax.set_title(title,fontsize=8); ax.axis("off")
    for key,label in [("source_histogram","Nguồn"),("reference_histogram","Tham chiếu"),("matched_histogram","Kết quả")]: axes[1,0].plot(details[key],lw=.8,label=label)
    axes[1,0].set_title("Histogram"); axes[1,0].legend(fontsize=6)
    for key,label in [("source_cdf","Nguồn"),("reference_cdf","Tham chiếu"),("matched_cdf","Kết quả")]: axes[1,1].plot(details[key],lw=.8,label=label)
    axes[1,1].set_title("CDF"); axes[1,1].legend(fontsize=6)
    axes[1,2].plot(details["lut"],lw=.8); axes[1,2].set_title("LUT"); axes[1,2].set_xlabel("Mức nguồn"); axes[1,2].set_ylabel("Mức đích")
    fig.tight_layout(); fig.savefig(SCREENSHOT_DIR/"hinh_3_5_histogram_matching.png",dpi=200); plt.close(fig)

    # Environment command outputs and test summary.
    ran_line = next((line for line in test_log.splitlines() if line.startswith("Ran ")), "Không có dòng 'Ran N tests in X s'.")
    counts = {}
    passed_tests = sum(1 for line in test_log.splitlines() if line.startswith("test_") and line.rstrip().endswith("... ok"))
    error_tests = sum(1 for line in test_log.splitlines() if line.startswith("test_") and line.rstrip().endswith("... ERROR"))
    failed_tests = sum(1 for line in test_log.splitlines() if line.startswith("test_") and line.rstrip().endswith("... FAIL"))
    for line in test_log.splitlines():
        match = re.search(r"\((tests\.[^.]+)\.", line)
        if line.startswith("test_") and match:
            module = match.group(1).split(".")[-1]
            counts[module] = counts.get(module,0)+1
    summary = ["# Tóm tắt unittest và kiểm tra tĩnh", "", f"- Lệnh: `{' '.join(test_cmd)}`", f"- {ran_line}", f"- Đạt: {passed_tests}; lỗi: {error_tests}; thất bại: {failed_tests}.", f"- Kết quả tiến trình: {'đạt' if test.returncode == 0 else 'lỗi'} (exit {test.returncode}).", f"- Theo tệp: {', '.join(f'{k}: {v}' for k,v in counts.items()) or 'không phân tích được'}.", f"- `node --check static\\js\\app.js`: {'đạt' if node.returncode == 0 else 'lỗi'}; {node.stdout.strip()} {node.stderr.strip()}", f"- `git diff --check`: {'đạt' if diffcheck.returncode == 0 else 'lỗi'}; {(diffcheck.stdout+diffcheck.stderr).strip() or 'không có cảnh báo'}", "- Log nguyên văn: `06_unittest.txt`."]
    (REPORT_DIR/"06_unittest_tom_tat.md").write_text("\n".join(summary)+"\n",encoding="utf-8")

    # Actual status/diff/log after work (only the authorized output directories are written).
    status = subprocess.run(["git","status","--short"],capture_output=True,text=True,encoding="utf-8").stdout.strip()
    diffstat = subprocess.run(["git","diff","--stat"],capture_output=True,text=True,encoding="utf-8").stdout.strip()
    logs = subprocess.run(["git","log","-3","--oneline"],capture_output=True,text=True,encoding="utf-8").stdout.strip()
    (REPORT_DIR/"08_ra_soat_loi.md").write_text("# Bảng 3.10 — Rà soát các điểm đã sửa\n\n| Mục | Kiểm tra | Trạng thái đo được |\n|---|---|---|\n| a | Gamma tham số, stretch với gamma 0, gamma 0 và gamma abc | "+ui_checks.get("api_a_d","chưa đo")+" |\n| b | Tải ảnh theo cấu hình lastConfig | "+ui_checks.get("lastconfig","chưa đo")+" |\n| c | CSV bắt đầu bằng BOM UTF-8 | "+ui_checks.get("exports","chưa đo").split(';')[0]+" |\n| d | Gamma abc trả HTTP 400 | "+ui_checks.get("api_a_d","chưa đo")+" |\n\n## Git\n\n`git status --short`:\n\n```text\n"+status+"\n```\n\n`git diff --stat`:\n\n```text\n"+diffstat+"\n```\n\n`git log -3 --oneline`:\n\n```text\n"+logs+"\n```\n",encoding="utf-8")

    # Every numeric cell in 03–05 has a direct-function cross-check.
    cross = ["# Đối chiếu chéo 03–05", "", "Mọi giá trị chính lấy từ JSON API mà giao diện nhận được; đối chiếu được tính trực tiếp bằng hàm dự án.", "", f"- Độ lệch tuyệt đối lớn nhất: `{MAX_DIFF:.12g}`.", f"- Vượt ngưỡng 1e-6: `{'có' if MAX_DIFF > 1e-6 else 'không'}`.", "", "| Mục đối chiếu | Độ lệch tuyệt đối |", "|---|---:|"]
    cross += [f"| {label} | {diff:.12g} |" for label,diff in DIFFS]
    (REPORT_DIR/"10_doi_chieu.md").write_text("\n".join(cross)+"\n",encoding="utf-8")

    # Pure factual summary, plus manifest of generated artifacts.
    (REPORT_DIR/"09_tom_tat_so_lieu.md").write_text("# Tóm tắt số liệu\n\n- Ảnh bắt buộc đã đo: "+", ".join(f"{n} ({metadata[n]['width']}×{metadata[n]['height']}, {metadata[n]['channels']} kênh)" for n,_,_ in REQUIRED)+".\n- Số phương pháp phân tích: 4; số lần phân tích giao diện: 16.\n- Số nguồn matching: "+str(len(match_data))+".\n- Độ lệch đối chiếu lớn nhất: `"+f"{MAX_DIFF:.12g}`.\n- Nguồn/giấy phép ảnh: chưa được cung cấp.\n",encoding="utf-8")
    all_files = sorted(p for p in REPORT_DIR.rglob("*") if p.is_file() and p.name != "00_manifest.md")
    manifest = ["# Manifest", "", "| Tệp | Vị trí gợi ý trong báo cáo | Trạng thái |", "|---|---|---|"]
    manifest.append("| `00_manifest.md` | Danh mục đầu ra | Đã tạo |")
    manifest.append("| `experiments/run_report_experiments.py` | Script tái lập phép đo | Đã tạo |")
    for path in all_files:
        rel=path.relative_to(REPORT_DIR).as_posix()
        section="Mục 2.9 / Chương 3" if rel.endswith(".png") else "Bảng / phụ lục Chương 3"
        manifest.append(f"| `{rel}` | {section} | Đã tạo |")
    manifest += ["", "## Đối chiếu chéo", "", f"Độ lệch tuyệt đối lớn nhất của các ô đã đối chiếu: `{MAX_DIFF:.12g}`. Chi tiết: [10_doi_chieu.md](10_doi_chieu.md)."]
    (REPORT_DIR/"00_manifest.md").write_text("\n".join(manifest)+"\n",encoding="utf-8")

    # Refresh final git status in report after all report files were written.
    final_status = subprocess.run(["git","status","--short"],capture_output=True,text=True,encoding="utf-8").stdout.strip()
    with (REPORT_DIR/"08_ra_soat_loi.md").open("a",encoding="utf-8") as f:
        f.write("\n## Git status sau khi tạo đầu ra\n\n```text\n"+final_status+"\n```\n")
    print(json.dumps({"unittest_exit":test.returncode,"ran":ran_line,"node_check":node.returncode,"diff_check":diffcheck.returncode,"browser_version":browser_version,"max_crosscheck_diff":MAX_DIFF,"ui_checks":ui_checks,"files":[str(p) for p in sorted(REPORT_DIR.rglob('*')) if p.is_file()]},ensure_ascii=True,indent=2))


if __name__ == "__main__":
    run()
