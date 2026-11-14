# PixelLab — Phân tích và xử lý ảnh

PixelLab là ứng dụng Flask để phân tích Histogram, tăng cường ảnh và khám phá các thuật toán xử lý ảnh. Giao diện dùng HTML, CSS và JavaScript thuần; NumPy/OpenCV xử lý dữ liệu ảnh, Matplotlib xuất biểu đồ.

## Cài đặt và chạy (Windows PowerShell)

Yêu cầu Python 3.13.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
~~~

Mở http://127.0.0.1:5000. Giới hạn upload là 20 MB; ảnh giải mã tối đa 16 megapixel và 10.000 pixel mỗi chiều. Ảnh nguồn và ảnh tham chiếu của Histogram Matching cùng chịu giới hạn request 20 MB.

## Chức năng

### Analysis Dashboard

- Histogram ảnh xám và các không gian RGB, HSV, LAB, YCrCb; chọn kênh, xem trước/sau trên cùng biểu đồ, hover số pixel và zoom.
- So sánh Min, Max, Mean, Median, Standard Deviation, Dynamic Range, Entropy và Contrast giữa ảnh gốc/sau xử lý. Bảng có chênh lệch tuyệt đối và phần trăm khi giá trị gốc khác 0.
- Contrast Stretching, Histogram Equalization, CLAHE và Gamma Correction. Ảnh màu được xử lý trên kênh sáng LAB L.
- Otsu Binary/Binary Inverse với threshold, between-class variance, tỷ lệ foreground/background và ảnh mask.
- Erosion, Dilation, Opening, Closing; cấu hình Kernel Shape/Size và số lần lặp.
- So sánh ảnh trước/sau, zoom/pan, thanh gạt, toàn màn hình, các phương án enhancement, lịch sử Undo/Redo giới hạn 14 bước và Reset.
- Xuất ảnh PNG, Histogram PNG và thống kê CSV. Tiêu đề cột CSV là tiếng Việt; các giá trị mã hóa không gian màu vẫn giữ nguyên.

Các thống kê mô tả phân bố pixel; chúng không tự chứng minh chất lượng cảm nhận được cải thiện. Tỷ lệ foreground/background không phải độ chính xác phân đoạn nếu không có Ground Truth.

### Histogram Equalization Explorer

Khi phương pháp đang chọn là Histogram Equalization, phần giải thích hiển thị Histogram đầu vào, PDF, CDF, LUT 256 mức và tra cứu một mức xám. Dữ liệu trung gian được trả về từ đúng lần tính Histogram/CDF/LUT tạo ra ảnh kết quả; ảnh hằng được giữ nguyên. Với ảnh màu, thuật toán áp dụng lên kênh L của LAB.

### Otsu Analysis Explorer

Hiển thị đường cong phương sai giữa hai lớp theo 256 ngưỡng, ngưỡng trước/sau, tỷ lệ mask, trọng số và trung bình mỗi lớp, ảnh xám, Binary và Binary Inverse. Ngưỡng tối ưu có thể không duy nhất; ảnh suy biến được cảnh báo.

### Histogram Matching

Chọn ảnh tham chiếu riêng, rồi chạy matching trên ảnh nguồn đang tải trong workspace. Thuật toán tự cài bằng NumPy: tính CDF hai ảnh, tạo LUT bằng mức tham chiếu có CDF gần nhất và áp dụng LUT. Ảnh màu dùng kênh L của LAB; ảnh nguồn và ảnh tham chiếu có thể khác kích thước.

Kết quả gồm ảnh xem trước, Histogram/CDF của nguồn, tham chiếu và ảnh khớp, cùng metric mean_cdf_distance — trung bình độ lệch tuyệt đối giữa hai CDF trên 256 mức. Metric chỉ mô tả độ gần phân bố, không đánh giá ảnh có đẹp hơn hay không. Tải kết quả PNG xuất ảnh ở kích thước nguồn. Matching không thay đổi lịch sử Undo/Redo và ảnh không được ghi lên ổ đĩa.

## API

- GET / — giao diện PixelLab.
- POST /api/analyze — multipart form có trường image; trả ảnh xem trước, Histogram/thống kê, Otsu, variants và kết quả xử lý. Trường equalization chứa Histogram/PDF/CDF/LUT khi method là equalize; nếu không, trường này là null. Mỗi kết quả Otsu có thêm variance_curve.
- POST /api/export/histogram — JSON Histogram và tùy chọn không gian/kênh/khoảng xem; trả biểu đồ PNG dạng data URL.
- POST /api/export/image — ảnh PNG enhanced hoặc mask morphology.
- POST /api/match — multipart form image và reference; trả ảnh xem trước cùng Histogram, CDF và LUT matching.
- POST /api/export/match — cùng hai trường ảnh; trả ảnh PNG matching đầy đủ kích thước.

API giữ nguyên các endpoint, khóa JSON và giá trị enum hiện có; dữ liệu Explorer được bổ sung vào phản hồi phân tích.

## Quy ước ảnh

- Định dạng nhận diện từ nội dung ảnh: PNG, JPG/JPEG, BMP và TIFF.
- Ảnh 16-bit được ánh xạ tuyến tính theo min/max quan sát được; ảnh hằng giữ byte cao. Tệp tải lên không bị sửa.
- RGBA/BGRA được ghép trên nền trắng khi phân tích. Ảnh enhanced đầu ra không có alpha; ảnh gốc trong trình duyệt vẫn nguyên trạng.
- Hue HSV của OpenCV dùng miền 0–179; các kênh uint8 còn lại dùng 0–255.
- Contrast được báo cáo bằng độ lệch chuẩn cường độ. IoU/Dice không được báo cáo khi không có Ground Truth.

## Kiểm thử

~~~powershell
.\.venv\Scripts\python.exe -B -m unittest discover -v
node --check static\js\app.js
git diff --check
~~~

Bộ unittest bao phủ upload/API, Histogram, equalization LUT, Otsu/variance, morphology, matching grayscale/màu/ảnh hằng/kích thước khác nhau và export.

## Cấu trúc

- app.py, config.py: ứng dụng Flask, giới hạn request và xử lý lỗi JSON.
- routes/image_routes.py: phân tích, matching và export API.
- services/image_service.py: giải mã ảnh, giới hạn, alpha và mã hóa preview.
- services/histogram_service.py: Histogram nhiều không gian màu, thống kê và biểu đồ Matplotlib.
- services/enhancement_service.py: Stretch, equalization tự cài, CLAHE và Gamma.
- services/segmentation_service.py: Otsu, đường cong phương sai và morphology tách biệt.
- services/matching_service.py: Histogram Matching NumPy trên grayscale hoặc LAB L.
- services/evaluation_service.py: thống kê mô tả và chỉ số mask.
- templates/ và static/: giao diện responsive và tương tác.
- tests/: kiểm thử thuật toán và Flask API.