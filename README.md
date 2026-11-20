# Hệ thống đếm linh kiện bằng AI

Hướng dẫn cài đặt, khởi động và sử dụng hệ thống nhận diện/đếm **đai ốc (nut), ốc vít (screw) và long đền (washer)** bằng YOLO26n kết hợp camera IP Hikvision.

---

## 1. Giới thiệu

Hệ thống sử dụng camera Hikvision truyền hình ảnh qua RTSP, mô hình YOLO26n nhận diện linh kiện và giao diện web Flask để người vận hành thực hiện thao tác đếm.

Các chức năng chính:

- Xem hình ảnh trực tiếp từ camera IP.
- Hiển thị vùng nhận diện (ROI) trên khung hình.
- Đếm ba nhóm linh kiện: `nut`, `screw`, `washer`.
- Mỗi lần đếm có thời gian chờ 3 giây, sau đó lấy 5 khung hình để tổng hợp kết quả.
- Lưu ảnh kết quả và báo cáo JSON của phiên đếm.
- Đánh dấu những vật thể cần người dùng kiểm tra lại.

> **Lưu ý:** Đây là hệ thống hỗ trợ kiểm đếm bằng thị giác máy tính. Người vận hành cần kiểm tra các vật thể được đánh dấu “Chờ xác minh” và đối chiếu kết quả trong những trường hợp linh kiện chồng lấn, che khuất hoặc nằm ngoài vùng camera quan sát.

## 2. Yêu cầu trước khi sử dụng

### Phần cứng

- Máy tính chạy Windows hoặc Linux.
- Camera IP Hikvision có bật dịch vụ RTSP.
- Máy tính và camera kết nối cùng mạng LAN (hoặc có định tuyến mạng phù hợp).
- Nguồn sáng ổn định, hạn chế phản xạ/chói trên bề mặt kim loại.
- Giá đỡ camera chắc chắn, không thay đổi góc nhìn trong quá trình vận hành.

### Phần mềm

- Python tương thích với phiên bản Ultralytics/PyTorch đang sử dụng.
- Các thư viện Python:
  - `ultralytics`
  - `opencv-python`
  - `flask`
  - `numpy`
  - `Pillow`
  - `scipy` (khuyến nghị; dùng cho ghép đối tượng bằng thuật toán Hungarian)

Có thể cài các thư viện cơ bản bằng lệnh:

```bash
pip install -r requirements.txts
```

**PyTorch:** Cài phiên bản phù hợp với máy tính và GPU (nếu có) theo hướng dẫn cài đặt chính thức của PyTorch. Nếu không có GPU NVIDIA tương thích, có thể chạy bằng CPU nhưng thời gian xử lý sẽ lâu hơn.

## 3. Cấu trúc thư mục

Đặt các tệp theo cấu trúc tương tự dưới đây:

```text
project/
├── app.py                 # Mã nguồn Flask và xử lý nhận diện
├── best.pt                 # Trọng số mô hình YOLO đã huấn luyện
├── templates/
│   └── index.html          # Giao diện web
└── static/
    └── saved_images/       # Ảnh kết quả được hệ thống tự tạo
        ├── review/         # Các phiên cần kiểm tra
        └── debug/          # Dữ liệu debug (nếu bật)
```

Tên tệp Python có thể khác `app.py`; khi đó hãy dùng đúng tên tệp trong lệnh chạy.

**Quan trọng:** Tệp `best.pt` phải nằm cùng thư mục với tệp Python đang chạy. Nếu thiếu mô hình, chương trình sẽ dừng với lỗi `FileNotFoundError`. Thư mục `templates/index.html` phải tồn tại để Flask hiển thị giao diện.

## 4. Cấu hình camera

Mặc định mã nguồn sử dụng:

- Camera IP: `192.168.1.233`
- Tài khoản: `admin`
- Cổng RTSP: `554`
- Luồng hình ảnh: Hikvision **Channel 101** (Main Stream)
- Đường dẫn RTSP: `/Streaming/Channels/101`

Nếu thông số camera của bạn khác, cần cấu hình lại trước khi chạy.

### Cấu hình bằng biến môi trường (Windows PowerShell)

Ví dụ:

```powershell
$env:CAMERA_IP="192.168.1.233"
$env:CAMERA_USERNAME="admin"
$env:CAMERA_PASSWORD="MAT_KHAU_CAMERA"
$env:RTSP_PORT="554"
python app.py
```

Thay `MAT_KHAU_CAMERA` bằng mật khẩu camera thực tế. Không chia sẻ mật khẩu camera, không đưa mật khẩu thật lên GitHub hoặc tài liệu công khai.

> **Khuyến nghị bảo mật trước khi triển khai:** Trong mã nguồn được bàn giao, hãy bỏ mật khẩu mặc định viết trực tiếp trong `app.py` và chỉ đọc mật khẩu từ biến môi trường `CAMERA_PASSWORD`. Nếu biến này chưa được thiết lập, chương trình nên báo lỗi thay vì dùng mật khẩu mặc định.

### Kiểm tra kết nối

1. Kiểm tra máy tính có truy cập được địa chỉ IP camera hay không.
2. Kiểm tra camera đã bật RTSP và tài khoản có quyền xem luồng hình ảnh.
3. Đảm bảo đúng IP, tên đăng nhập, mật khẩu và cổng RTSP.
4. Nếu không kết nối được, kiểm tra firewall, VLAN và cấu hình mạng camera.

## 5. Khởi động hệ thống

Mở Terminal/PowerShell tại thư mục dự án.

Nếu sử dụng môi trường ảo:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install ultralytics opencv-python flask numpy pillow scipy
```

Thiết lập thông tin camera như mục 4, sau đó chạy:

```powershell
python app.py
```

Khi khởi động thành công, Flask chạy tại:

```text
http://127.0.0.1:5000/pick
```

Mở địa chỉ trên bằng trình duyệt Chrome hoặc Edge trên chính máy tính đang chạy chương trình.

> **Truy cập từ máy khác:** Mã nguồn hiện tại khởi chạy Flask với `host="127.0.0.1"`, nên chỉ truy cập được từ máy đang chạy server. Muốn truy cập bằng điện thoại hoặc máy tính khác trong mạng LAN, cần được quản trị viên thay host thành `0.0.0.0`, cấu hình firewall và kiểm soát truy cập phù hợp. Khi đó dùng `http://IP_MAY_CHAY_SERVER:5000/pick`. Không mở cổng trực tiếp ra Internet nếu chưa có biện pháp bảo mật.

## 6. Hướng dẫn sử dụng giao diện

### Bước 1 – Mở trang hệ thống

Truy cập trang `/pick`. Kiểm tra giao diện đã tải đầy đủ và không có thông báo lỗi.

### Bước 2 – Bật camera

1. Nhấn nút **Bật camera** trên giao diện.
2. Chờ hệ thống kết nối đến camera Hikvision.
3. Khi trạng thái báo camera đã sẵn sàng và hình ảnh trực tiếp xuất hiện, bạn có thể bắt đầu đếm.
4. Quan sát khung hình và vùng ROI màu xanh lá.

Nếu chưa có hình ảnh hoặc camera chưa sẵn sàng, không nhấn bắt đầu đếm.

### Bước 3 – Chuẩn bị linh kiện

Trước khi đếm:

- Đặt linh kiện vào khu vực nằm trong khung ROI màu xanh.
- Cố gắng trải linh kiện để các vật thể không che khuất nhau.
- Tránh để linh kiện nằm ngoài vùng quan sát.
- Không di chuyển khay hoặc camera trong lúc hệ thống đang lấy ảnh.
- Hạn chế ánh sáng phản chiếu mạnh làm mất đường biên của linh kiện.

### Bước 4 – Bắt đầu đếm

1. Nhấn nút **Bắt đầu đếm**.
2. Hệ thống đếm ngược 3 giây để người vận hành ổn định vị trí linh kiện.
3. Sau đó hệ thống tự động thu thập 5 khung hình, mỗi khung cách nhau khoảng 0,15 giây.
4. Mô hình nhận diện trên các khung hình và tổng hợp kết quả.
5. Chờ đến khi trạng thái chuyển sang hoàn tất.

Trong lúc đếm hoặc đang xử lý, không bắt đầu một phiên đếm mới.

### Bước 5 – Đọc kết quả

Kết quả được phân thành:

- **Đai ốc (nut):** Số lượng đai ốc được phát hiện.
- **Ốc vít (screw):** Số lượng ốc vít được phát hiện.
- **Long đền (washer):** Số lượng long đền được phát hiện.
- **Tổng phát hiện:** Tổng số vật thể, bao gồm cả vật thể đang chờ xác minh.
- **Đã xác nhận:** Những vật thể được hệ thống xác nhận tự động.
- **Chờ xác minh:** Những vật thể cần người vận hành kiểm tra lại.

Trên ảnh kết quả:

- **Khung màu xanh lá:** Vật thể đã được xác nhận.
- **Khung màu cam có nhãn `?1`, `?2`, ...:** Vật thể đang chờ xác minh.

Khi có vật thể màu cam, hãy phóng to ảnh kết quả và kiểm tra xem đó có phải linh kiện thật hay không. Không nên mặc định cộng tất cả vật thể chờ xác minh vào số lượng cuối cùng nếu chưa kiểm tra.

### Bước 6 – Lưu và kết thúc

- Ảnh kết quả của mỗi phiên được tự động lưu.
- Báo cáo JSON tương ứng được lưu cùng tên với ảnh kết quả.
- Nếu phiên có cảnh báo cần kiểm tra, hệ thống lưu thêm ảnh vào thư mục `review`.
- Khi hoàn tất công việc, nhấn **Tắt camera** để dừng luồng camera.

## 7. Nơi lưu kết quả

Dữ liệu được lưu trong:

```text
static/saved_images/
```

| Loại dữ liệu | Vị trí | Mô tả |
|---|---|---|
| Ảnh kết quả | `static/saved_images/result_*.jpg` | Ảnh có khung nhận diện và bảng tổng kết |
| Báo cáo phiên đếm | `static/saved_images/result_*.json` | Thời gian, số lượng, trạng thái, thông tin nhận diện và cảnh báo |
| Ảnh cần review | `static/saved_images/review/` | Ảnh gốc và ảnh kết quả của phiên cần kiểm tra |
| Dữ liệu debug | `static/saved_images/debug/` | 5 frame RAW và dữ liệu chi tiết khi bật chế độ debug |

Các tệp `result_N.jpg` và `result_N.json` được ghép cặp theo cùng số thứ tự. Nên sao lưu thư mục `saved_images` định kỳ nếu cần lưu lịch sử kiểm đếm.

## 8. Các trạng thái thường gặp

| Trạng thái | Ý nghĩa | Cách xử lý |
|---|---|---|
| Đã dừng | Camera chưa được bật | Nhấn Bật camera |
| Đang kết nối camera IP | Hệ thống đang mở luồng RTSP | Chờ kết nối |
| Camera IP đang bật | Camera đã kết nối | Có thể bắt đầu đếm |
| Chuẩn bị đếm ngược 3 giây | Đang chờ trước khi lấy ảnh | Giữ nguyên khay linh kiện |
| Đang lấy 5 frame | Đang thu thập ảnh | Không di chuyển vật thể |
| Đang chạy YOLO | Đang nhận diện và tổng hợp | Chờ xử lý xong |
| Hoàn tất | Đã có kết quả | Kiểm tra số lượng và ảnh |
| Lỗi nhận diện | Có lỗi trong quá trình xử lý | Xem cửa sổ Terminal để biết chi tiết |

## 9. Xử lý sự cố

### Không mở được trang web

- Kiểm tra cửa sổ Terminal có đang chạy `python app.py` hay không.
- Kiểm tra cổng 5000 có bị chương trình khác sử dụng không.
- Mở đúng địa chỉ `http://127.0.0.1:5000/pick` trên máy chạy server.

### Camera không kết nối

- Kiểm tra nguồn điện và dây mạng camera.
- Kiểm tra IP camera, tài khoản, mật khẩu và cổng RTSP.
- Kiểm tra RTSP đã được bật trong cấu hình camera.
- Kiểm tra máy tính và camera có cùng mạng hoặc có thể định tuyến đến nhau.
- Kiểm tra firewall và quyền truy cập luồng Main Stream.

### Không thấy hình ảnh camera

- Chờ hệ thống kết nối lại.
- Kiểm tra trạng thái camera trong giao diện.
- Nếu vẫn không có hình, kiểm tra log RTSP trong Terminal và thử lại thông tin kết nối.

### Không tìm thấy mô hình

Nếu xuất hiện lỗi tương tự:

```text
FileNotFoundError: Khong tim thay model: .../best.pt
```

Hãy đặt đúng tệp `best.pt` vào cùng thư mục với tệp Python chạy ứng dụng.

### Không tải được giao diện

Kiểm tra tệp:

```text
templates/index.html
```

Tên thư mục và tên tệp phải chính xác.

### Đếm thiếu hoặc đếm dư

- Kiểm tra linh kiện có bị chồng lấn hoặc che khuất không.
- Kiểm tra toàn bộ linh kiện có nằm trong ROI không.
- Kiểm tra ảnh có bị mờ, rung, lóa sáng hoặc thiếu sáng không.
- Cố định lại camera và khay, điều chỉnh ánh sáng tán xạ.
- Kiểm tra các khung màu cam và ảnh trong thư mục `review`.
- Nếu lỗi lặp lại với một loại linh kiện, lưu ảnh thực tế để đánh giá và cải thiện dữ liệu/mô hình.

### Camera bị mất kết nối trong lúc đếm

Phiên đếm đang thực hiện có thể bị hủy. Chờ camera kết nối lại, xác nhận trạng thái sẵn sàng rồi thực hiện một phiên đếm mới.

## 10. Khuyến nghị vận hành để đạt độ chính xác tốt

1. **Cố định camera:** Không thay đổi độ cao, góc nghiêng, tiêu cự hoặc zoom sau khi đã căn chỉnh.
2. **Cố định vùng đặt linh kiện:** Dùng khay hoặc mặt bàn có ranh giới rõ ràng, đặt đúng vị trí ROI.
3. **Ổn định ánh sáng:** Ưu tiên ánh sáng tán xạ, tránh phản chiếu trực tiếp lên bề mặt kim loại.
4. **Giữ ảnh sắc nét:** Vệ sinh ống kính, kiểm tra lấy nét và hạn chế rung.
5. **Giảm chồng lấn:** Nếu quy trình cho phép, trải linh kiện để giảm che khuất. Với các vật thể giao nhau, luôn kiểm tra kết quả.
6. **Kiểm tra vật thể màu cam:** Đây là các trường hợp hệ thống yêu cầu xác minh, không nên bỏ qua.
7. **Đối chiếu định kỳ:** So sánh kết quả với số lượng thực tế, ghi nhận ảnh đếm sai để cải thiện mô hình.

## 11. Thông tin kỹ thuật dành cho quản trị viên

Các cấu hình dưới đây được đọc từ biến môi trường. Người vận hành thông thường không cần thay đổi.

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `CAMERA_IP` | `192.168.1.233` | Địa chỉ IP camera |
| `CAMERA_USERNAME` | `admin` | Tài khoản RTSP |
| `CAMERA_PASSWORD` | Không nên đặt mặc định trong mã nguồn | Mật khẩu camera |
| `RTSP_PORT` | `554` | Cổng RTSP |
| `COUNT_PROFILE` | `v3` | Profile xử lý (`baseline`, `v2`, `v3`) |
| `COUNT_DEBUG` | `0` | Bật/tắt lưu dữ liệu debug |
| `CONF_LOW` | `0.30` | Ngưỡng suy luận thấp cho profile v2/v3 |
| `SHAPE_CORRECT` | `0` | Cho phép OpenCV điều chỉnh class nut/washer |
| `IMAGE_SIZE` | `1600` | Kích thước suy luận dùng cho profile không đa scale |
| `SCALES` | `1280,1600,1920,1600,1280` | Kích thước suy luận tương ứng 5 frame của v3 |
| `REFINE` | `1` | Bật/tắt suy luận lại vùng nghi ngờ trong v3 |

Ví dụ bật debug trên PowerShell:

```powershell
$env:COUNT_DEBUG="1"
python app.py
```

Để trở về chế độ thông thường:

```powershell
$env:COUNT_DEBUG="0"
```

Chỉ người có chuyên môn nên thay đổi các tham số xử lý. Sau mỗi lần thay đổi, cần kiểm tra lại bằng tập ảnh có số lượng thực tế đã biết.

---

## 12. Giới hạn sử dụng

- Kết quả phụ thuộc vào chất lượng ảnh, góc camera, ánh sáng, kích thước linh kiện và dữ liệu huấn luyện.
- Vật thể bị che khuất hoàn toàn hoặc chồng lên nhau phức tạp có thể bị nhận diện thiếu.
- Vật thể ngoài ROI không được đưa vào vùng suy luận.
- Số lượng “Chờ xác minh” cần người vận hành kiểm tra.
- Kết quả AI không thay thế hoàn toàn việc kiểm tra thực tế trong các công đoạn yêu cầu độ chính xác tuyệt đối.

**Khuyến nghị:** Trước khi đưa vào dây chuyền chính thức, hãy chạy thử với nhiều lô linh kiện và nhiều tình huống bố trí khác nhau; so sánh số đếm với số lượng chuẩn để đánh giá độ chính xác thực tế.
