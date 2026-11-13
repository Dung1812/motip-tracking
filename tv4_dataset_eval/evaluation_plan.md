# KẾ HOẠCH & QUY TRÌNH ĐÁNH GIÁ (EVALUATION PLAN)
**Đề tài:** Nghiên cứu Multiple Object Tracking dựa trên Deep Learning ID Prediction  
**Tập trung đo lường:** HOTA, AssA, DetA, IDF1, MOTA, IDSW  
**Công cụ chuẩn hóa:** TrackEval (Official Evaluation Protocol)  
**Tài liệu tham chiếu:** `evaluation_plan.md`

---

## 1. Mục tiêu 
Xây dựng giao thức kiểm thử thống nhất, công bằng và có thể tái lập cho toàn bộ các thí nghiệm của nhóm. Đảm bảo mọi kết quả thực nghiệm từ baseline đến mô hình cải tiến đều được đo lường tự động trên cùng một tập dữ liệu, cùng một hệ thống metric chuẩn quốc tế và triệt tiêu sai số chủ quan.

---

## 2. Bản chất toán học & Phương pháp tính toán Metric

Khác với các bài toán thị giác thông thường, theo dõi đa đối tượng (MOT) đòi hỏi đồng thời cả hai yếu tố: **Bắt đúng vật thể (Detection)** và **Gán đúng danh tính theo thời gian (Association)**.

### 2.1. Thước đo HOTA (Higher Order Tracking Accuracy)
HOTA là tiêu chuẩn vàng đánh giá MOT hiện nay, được thiết kế để cân bằng độc lập giữa chất lượng Detection và chất lượng Association mà không bị thiên lệch.

1. **Ngưỡng tương đồng không gian ($\alpha$):**
   * Đánh giá được tính trung bình trên 19 ngưỡng IoU khác nhau: $\alpha \in \{0.05, 0.10, \dots, 0.90, 0.95\}$.

2. **Khớp nối tại từng frame:**
   * Tại mỗi ngưỡng $\alpha$, thuật toán so khớp giữa các hộp dự đoán ($Pr$) và nhãn thật ($Gt$) để chia thành các tập: $TP_\alpha$ (True Positives), $FP_\alpha$ (False Positives), và $FN_\alpha$ (False Negatives).

3. **Đo lường độ chính xác phát hiện (Detection Accuracy - DetA):**

$$
\text{DetA}_\alpha = \frac{|TP_\alpha|}{|TP_\alpha| + |FN_\alpha| + |FP_\alpha|}
$$

4. **Đo lường độ chính xác liên kết quỹ đạo (Association Accuracy - AssA):**
   * Với mỗi điểm khớp $c \in TP_\alpha$, xét toàn bộ quỹ đạo của đối tượng đó xuyên suốt video:
     * $TPA(c)$: Các frame mà cả nhãn thật và dự đoán đều khớp đúng cùng một ID đó.
     * $FPA(c)$: Các frame mà mô hình gán sai ID đó cho đối tượng khác.
     * $FNA(c)$: Các frame mà ID đó bị bỏ sót hoặc nhảy sang ID khác.

$$
\mathcal{A}_\alpha(c) = \frac{|TPA(c)|}{|TPA(c)| + |FPA(c)| + |FNA(c)|}
$$

$$
\text{AssA}_\alpha = \frac{1}{|TP_\alpha|} \sum_{c \in TP_\alpha} \mathcal{A}_\alpha(c)
$$

5. **Công thức HOTA tổng hợp:**

$$
\text{HOTA}_\alpha = \sqrt{\frac{\sum_{c \in TP_\alpha} \mathcal{A}_\alpha(c)}{|TP_\alpha| + |FN_\alpha| + |FP_\alpha|}} = \sqrt{\text{DetA}_\alpha \times \text{AssA}_\alpha}
$$

$$
\text{HOTA} = \frac{1}{19} \sum_{\alpha} \text{HOTA}_\alpha
$$

---

### 2.2. Thước đo IDF1 (Identification F1-Score)
IDF1 đo lường khả năng duy trì danh tính dài hạn thông qua thuật toán Hungarian toàn cục giữa quỹ đạo dự đoán và quỹ đạo thật:
$$\text{IDF1} = \frac{2 \cdot IDTP}{2 \cdot IDTP + IDFP + IDFN}$$
* $IDTP$: Tổng số khung hình mà đối tượng được gán đúng ID chủ đạo khớp tốt nhất.
* $IDFP / IDFN$: Số khung hình bị gán nhầm sang ID khác hoặc bị bỏ sót.
* **Ý nghĩa:** IDF1 đặc biệt nhạy cảm với khả năng ghi nhớ quỹ đạo sau khi đối tượng bị che khuất thời gian dài.

---

### 2.3. Các thước đo bổ trợ truyền thống (CLEAR MOT)
* **IDSW (ID Switches - Số lần tráo ID):** Đếm số lần một quỹ đạo đối tượng bị gián đoạn và đổi sang mã ID mới. Mục tiêu tối ưu là kéo IDSW về càng gần 0 càng tốt.
* **MOTA (Multiple Object Tracking Accuracy):**
  $$\text{MOTA} = 1 - \frac{\sum (FN + FP + IDSW)}{\sum GT}$$
  * Nhược điểm của MOTA: Quá phụ thuộc vào chất lượng phát hiện (Detection), bỏ qua gần như toàn bộ độ mượt và tính nhất quán dài hạn của quỹ đạo. Giữ MOTA để đối chiếu lịch sử.

---

## 3. Quy chuẩn định dạng đầu ra của Mô hình (Model Output Protocol)
Khi chạy suy luận trên tập dữ liệu kiểm định (`val`), mô hình bắt buộc phải xuất ra các file văn bản có tên trùng với tên video: `<sequence_name>.txt` (ví dụ: `dancetrack0004.txt`).

Cấu trúc mỗi dòng trong file kết quả (chuẩn MOTChallenge 10 cột):
```text
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
