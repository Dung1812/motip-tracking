# ĐẶC TẢ KỸ THUẬT DỮ LIỆU TỔNG QUÁT (GENERALIZED MOT DATASET SPECIFICATION)
**Đề tài:** Nghiên cứu theo dõi đa đối tượng (Multiple Object Tracking - MOT) dựa trên mô hình Deep Learning  
**Phạm vi áp dụng:** Chuẩn hóa cho mọi bộ dữ liệu MOT dạng video (DanceTrack, SportsMOT, BFT, MOT17/20 và dữ liệu tùy biến)  
**Tài liệu tham chiếu:** `dataset_spec.md` 

---

## 1. Nguyên lý tổng quát & Phạm vi tài liệu
Tài liệu này chuẩn hóa giao thức dữ liệu đầu vào cho toàn bộ hệ thống MOT đa kịch bản (người đi bộ, thể thao, đàn chim/động vật, phương tiện giao thông). Mọi tập dữ liệu từ các nguồn khác nhau khi đưa vào hệ thống đều phải được chuyển đổi và đóng gói về **cấu trúc chuẩn MOTChallenge mở rộng** để tái sử dụng toàn bộ pipeline nạp dữ liệu (`Dataset/DataLoader`), kiểm tra chất lượng nhãn (`Sanity Check`), và công cụ chấm điểm (`TrackEval`).

---

## 2. Cấu trúc cây thư mục tổng quát (Directory Hierarchy)
Mọi bộ dữ liệu MOT được tổ chức phân cấp theo cây thư mục độc lập với tên miền dữ liệu:

```text
{DATASET_ROOT}/
├── train/
│   ├── <seq_name_01>/
│   │   ├── img1/
│   │   │   ├── 00000001.jpg
│   │   │   ├── 00000002.jpg
│   │   │   └── ...
│   │   ├── gt/
│   │   │   └── gt.txt
│   │   └── seqinfo.ini
│   ├── <seq_name_02>/
│   └── ...
├── val/
│   ├── <seq_name_k>/
│   │   ├── img1/
│   │   │   └── ...
│   │   ├── gt/
│   │   │   └── gt.txt
│   │   └── seqinfo.ini
│   └── ...
├── test/
│   ├── <seq_name_m>/
│   │   ├── img1/
│   │   │   └── ...
│   │   └── seqinfo.ini
│   └── ...
├── train_seqmap.txt
├── val_seqmap.txt
├── test_seqmap.txt
└── TrackEval/
```

### Nguyên tắc định danh & quy chuẩn:
* **Tên thư mục chuỗi (`<seq_name>`):** Chuỗi ký tự duy nhất đại diện cho video (ví dụ: `dancetrack0001`, `v_sports_01`, `bft_flock_03`).
* **Định dạng khung hình (Frames):** Tệp ảnh định dạng JPG hoặc PNG, tên tệp được đệm số 0 (Zero-padding) tối thiểu 6 đến 8 chữ số (`%06d.jpg` hoặc `%08d.jpg`), đánh số tăng dần liên tục bắt đầu từ 1.
* **Tệp danh mục (`*_seqmap.txt`):** File văn bản thuần liệt kê danh sách tên các thư mục chuỗi video cần nạp trong phân vùng (mỗi dòng một tên video, dòng đầu tiên là tiêu đề `name`).

---

## 3. Đặc tả siêu dữ liệu chuỗi video (`seqinfo.ini`)
Tệp `seqinfo.ini` đặt trong thư mục gốc của từng chuỗi video, cung cấp siêu dữ liệu cần thiết cho việc tiền xử lý và suy luận:

```ini
[Sequence]
name=<seq_name>
imDir=img1
frameRate=20
seqLength=1200
imWidth=1920
imHeight=1080
imExt=.jpg
```

| Tham số | Kiểu dữ liệu | Phạm vi giá trị | Mô tả & Ý nghĩa trong hệ thống |
| :--- | :--- | :--- | :--- |
| `name` | String | Ký tự chữ & số | Định danh duy nhất của chuỗi video trong toàn bộ benchmark |
| `imDir` | String | Tên thư mục | Thư mục chứa frame ảnh (thường mặc định là `img1`) |
| `frameRate` | Integer/Float | $10 \le \text{fps} \le 120$ | Tốc độ khung hình thực tế; dùng cho mô hình chuyển động (Kalman/Velocity) |
| `seqLength` | Integer | $\ge 1$ | Tổng số lượng ảnh khung hình; kiểm soát vòng lặp đọc dữ liệu |
| `imWidth` | Integer | $\ge 100$ | Chiều rộng ảnh gốc (pixel) phục vụ chuẩn hóa tọa độ |
| `imHeight` | Integer | $\ge 100$ | Chiều cao ảnh gốc (pixel) phục vụ chuẩn hóa tọa độ |
| `imExt` | String | `.jpg` / `.png` | Phần mở rộng của tệp ảnh để bộ nạp quét đúng mẫu |

---

## 4. Đặc tả nhãn Ground Truth chuẩn hóa (`gt/gt.txt`)
Mỗi dòng trong `gt.txt` biểu diễn trạng thái của **một đối tượng tại một thời điểm**, phân tách bằng dấu phẩy theo chuẩn quốc tế 9 cột:

```text
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
```
### Chi tiết các trường dữ liệu:
1. **`frame` (int):** Chỉ số khung hình (1-indexed, $1 \le \text{frame} \le \text{seqLength}$).
2. **`id` (int):** Mã định danh duy nhất của cá thể đối tượng trong chuỗi đó (1-indexed). Duy trì cố định xuyên suốt vòng đời của cá thể trong cùng một video.
3. **`bb_left` ($x_{\min}$):** Tọa độ hoành độ góc trên-trái của hộp giới hạn (pixel gốc, hệ tọa độ ảnh với gốc $(0,0)$ ở góc trên cùng bên trái).
4. **`bb_top` ($y_{\min}$):** Tọa độ tung độ góc trên-trái của hộp giới hạn (pixel gốc).
5. **`bb_width` ($w$):** Chiều rộng của hộp giới hạn (pixel).
6. **`bb_height` ($h$):** Chiều cao của hộp giới hạn (pixel).
7. **`conf` (int/float):** Cờ đánh giá hoặc độ tin cậy của nhãn:
   * `1`: Nhãn hợp lệ (được đưa vào tính hàm mất mát và chấm điểm metric).
   * `0`: Nhãn bị bỏ qua (Ignore region/Vùng nhiễu/Vật thể ngoài phạm vi quan tâm), không phạt Loss và không tính vào FP/FN khi đánh giá.
8. **`class` (int):** Mã lớp đối tượng theo bảng ánh xạ phân loại:
   * `1`: Lớp đối tượng quan tâm chính (Person đối với DanceTrack/SportsMOT; Bird đối với BFT; Pedestrian đối với MOT17).
   * `2, 3, ...`: Các lớp phụ (phương tiện, đồ vật) hoặc các trạng thái không đánh giá.
9. **`visibility` (float):** Tỷ lệ phần trăm quan sát được của đối tượng, nhận giá trị trong đoạn $[0.0, 1.0]$:
   * `1.0`: Rõ nét hoàn toàn, không bị vật cản che khuất.
   * `0.0`: Bị che khuất hoàn toàn (Full occlusion).

---

## 5. Quy chuẩn toán học chuyển đổi tọa độ & Tính toán Loss
Các kiến trúc học sâu End-to-End (Transformer/DETR-based) yêu cầu biểu diễn hình học độc lập với độ phân giải tuyệt đối của ảnh.

### 5.1. Công thức chuẩn hóa không gian (Normalization)
Gọi kích thước ảnh gốc từ `seqinfo.ini` là $(W = \text{imWidth}, H = \text{imHeight})$. Hộp giới hạn gốc $(x_{\min}, y_{\min}, w, h)$ được chuẩn hóa thành vector hộp 4 chiều $\mathbf{b} \in [0, 1]^4$:

$$
\hat{cx} = \frac{x_{\min} + \frac{w}{2}}{W}, \quad \hat{cy} = \frac{y_{\min} + \frac{h}{2}}{H}
$$

$$
\hat{w} = \frac{w}{W}, \quad \hat{h} = \frac{h}{H}
$$

$$
\mathbf{b}_{gt} = [\hat{cx}, \hat{cy}, \hat{w}, \hat{h}]
$$

### 5.2. Chuyển đổi ngược phục vụ suy luận (De-normalization)
Khi mô hình suy luận đưa ra dự đoán $\mathbf{b}_{pred} = [\hat{cx}_p, \hat{cy}_p, \hat{w}_p, \hat{h}_p]$, tọa độ xuất file kết quả được khôi phục về pixel tuyệt đối:

$$
x_{\min} = \left(\hat{cx}_p - \frac{\hat{w}_p}{2}\right) \times W, \quad y_{\min} = \left(\hat{cy}_p - \frac{\hat{h}_p}{2}\right) \times H
$$

$$
w = \hat{w}_p \times W, \quad h = \hat{h}_p \times H
$$

### 5.3. Liên kết hàm mất mát trong huấn luyện
* **Hồi quy hộp ($L_{\text{L1}}$, $L_{\text{GIoU}}$):** Sử dụng trực tiếp vector đã chuẩn hóa $\mathbf{b}_{\text{gt}}$ để tính khoảng cách và diện tích bao lồi.
* **Phân loại đối tượng ($L_{\text{Focal}}$):** Dựa vào trường `class` và cờ `conf = 1` để định danh nhãn Positive trong bài toán cân bằng mẫu âm/dương.
* **Dự đoán định danh ($L_{\text{id}}$ hay $L_{\text{ID-CE}}$):** Mã `id` được ánh xạ cục bộ thành phân phối nhãn danh tính trong tập cửa sổ thời gian trượt (Temporal Window Clip).

---

## 6. Bộ tiêu chí kiểm tra tính toàn vẹn dữ liệu (Universal Sanity Checks)
Trước khi đưa dữ liệu vào hệ thống, module tiền xử lý dữ liệu (`prepare_dataset.py`) bắt buộc phải kiểm duyệt các điều kiện tiên quyết sau:

1. **Tính suy biến hình học (Non-degeneracy):**
   $$w > 0 \quad \text{và} \quad h > 0$$
   Loại bỏ mọi dòng nhãn có diện tích bằng $0$ hoặc giá trị tọa độ âm bất thường.
2. **Cắt biên an toàn (Bounding Box Clipping):**
   Nếu hộp nhãn vượt quá khung hình do góc quay camera, tự động áp dụng hàm kẹp biên:
   $$x_{\min} = \max(0, x_{\min}), \quad y_{\min} = \max(0, y_{\min})$$
   $$w = \min(W - x_{\min}, w), \quad h = \min(H - y_{\min}, h)$$
3. **Tính đơn nhất của định danh (Frame-wise Unique ID):**
   Trong cùng một chỉ số `frame`, mỗi giá trị `id` chỉ được xuất hiện duy nhất 1 lần. Nếu trùng lặp ID trong cùng frame, cảnh báo lỗi xung đột nhãn.
4. **Kiểm tra tính liên tục của khung hình (Sequence Completeness):**
   Đếm thực tế số lượng tệp ảnh trong `img1/` và đối soát:
   $$\text{count}(\text{files in } img1/) \equiv \text{seqLength}$$
   Mọi frame ảnh từ $1$ đến $\text{seqLength}$ phải tồn tại và đọc được mà không bị lỗi tệp rỗng (corrupted file).

---

## 7. Bảng tham chiếu các biến thể Benchmark phổ biến

| Tiêu chí | DanceTrack | SportsMOT | BFT (Bird Flock Tracking) |
| :--- | :--- | :--- | :--- |
| **Đối tượng mục tiêu** | Người nhảy (Dancers) | Vận động viên (Players) | Đàn chim bay (Birds) |
| **Mã lớp (`class`)** | `1` (Person) | `1` (Player/Referee) | `1` (Bird) |
| **Độ phân giải phổ biến** | $1920 \times 1080$ | $1920 \times 1080$ | Đa dạng ($1280 \times 720 \dots 4\text{K}$) |
| **Tốc độ khung hình (FPS)** | 20 FPS | 25 – 30 FPS | 30 – 60 FPS |
| **Đặc trưng chuyển động** | Phi tuyến tính, uốn lượn | Tốc độ cao, đổi hướng đột ngột | Chuyển động 3D, dày đặc, phân tán |
| **Đặc trưng ngoại hình** | Đồng phục tương đồng | Áo thi đấu giống nhau | Cá thể gần như giống nhau 100% |
| **Thách thức cốt lõi** | Che khuất chéo liên tục | Camera lia góc cực nhanh | Tỷ lệ che khuất dày đặc, mật độ lớn |

---

## 8. Hướng dẫn tích hợp dữ liệu mới (Custom Dataset Adaptation)
Để tích hợp một bộ dữ liệu bất kỳ chưa có chuẩn MOT vào hệ thống:
1. Đặt video hoặc tách video thành thư mục chuỗi ảnh `img1/` đánh số liên tục từ `00000001.jpg`.
2. Tạo file `seqinfo.ini` ghi đúng thông số kích thước ảnh, độ dài frame và FPS.
3. Chuyển đổi file nhãn gốc (từ COCO JSON, Pascal VOC XML, hay YOLO TXT) sang file `gt.txt` tuân thủ đúng thứ tự 9 cột ở Mục 4.
4. Ghi tên thư mục chuỗi video vào tệp `train_seqmap.txt` hoặc `val_seqmap.txt` tương ứng.
