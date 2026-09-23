# KIỂM CHỨNG SỐ LIỆU DẪN LẠI TỪ TÀI LIỆU

Tệp này giữ các đoạn trích nguyên văn từ bản toàn văn của những công trình mà
đồ án dẫn số liệu. Mục đích là để mọi con số dẫn lại đều truy được về đúng câu,
đúng bảng trong bài gốc, thay vì chép lại từ tóm tắt hay từ một bài khác.

Quy tắc: số đo của đồ án lấy từ `docs/KET_QUA_DO_LUONG.md`, số của công trình đã
công bố thì chép từ bảng trong bài gốc kèm số hiệu bảng và điều kiện đo, và hai
loại không bao giờ trộn vào cùng một cột.

Danh mục tài liệu tham khảo đang dùng nằm ở
`reports/tai_lieu_tham_khao/TAI_LIEU_THAM_KHAO.md`, gồm 23 mục tách tiếng
Việt và tiếng Anh, xếp alphabet theo Phụ lục 2.

---

Phần này trích nguyên văn từ bản toàn văn của ba công trình. Mỗi mục dưới đây
dùng làm căn cứ cho một cấu hình thí nghiệm cụ thể trong `notebooks/`.

## 1. Anomal-E — hướng phát triển do chính tác giả nêu

Nguyên văn câu cuối phần kết luận:

> "We also plan on investigating the effects of **temporal-based training** to
> further the exploration of Anomal-E's performance outside synthetic network
> conditions."

Đây chính là hướng mà đồ án đang theo đuổi. Điểm này dùng được cho phần mở đầu
và phần tổng quan của báo cáo: hướng tiếp cận đồ thị thời gian không phải do đồ
án tự nghĩ ra mà là bước tiếp theo được nhóm tác giả Anomal-E nêu ra.

Bộ thí nghiệm hiện hành **không có bản tái lập Anomal-E**. Sáu cấu hình đã
chạy đều là kiến trúc hai giai đoạn của đồ án, chạy trên hai giao thức chia tập
`graphids` và `chronological_80_10_10`. Hàm `split_anomal_e` vẫn còn trong
`src/tgn_nids/data/protocols.py` như một giao thức đăng ký được, nhưng không
cấu hình nào gọi tới nó.

Hệ quả cho báo cáo: con số của Anomal-E chỉ được trích từ bài gốc kèm số hiệu
bảng và điều kiện đo, tuyệt đối không được trình bày như một phép đo của đồ án.
Giai đoạn một của đồ án cũng học tự giám sát trên riêng lưu lượng lành tính,
nhưng đó là encoder của đồ án chứ không phải Anomal-E, nên đặt hai con số
cạnh nhau phải nói rõ chúng đến từ hai mô hình khác nhau.

**Chi tiết bắt buộc phải nêu kèm khi trích số 92,35% Macro F1:** đó là kết quả
của biến thể Anomal-E kết hợp Isolation Forest ở **mức nhiễm bẩn 4%** (Bảng 8
của bài gốc). Cùng biến thể ở mức nhiễm bẩn 0% (Bảng 7) chỉ đạt **85,62%**.
Trích con số mà bỏ tham số nhiễm bẩn là trình bày thiếu điều kiện.

## 2. GraphIDS — dải siêu tham số công bố và hạn chế tác giả tự nêu

Dải siêu tham số ghi trong bài:

| Tham số | Giá trị công bố |
|---|---|
| Số chiều nhúng GNN | 64 đến 96 |
| Số hop | 1 |
| Fan-out | 32.768 |
| Tỷ lệ bỏ học GNN | 0,5 đến 0,75 |
| Số lớp Transformer | 1 |
| Số attention head | 4 |
| Số chiều nhúng Transformer | 32 đến 48 |
| Kích thước cửa sổ | 512 |
| Tỷ lệ che | 0,15 |
| Learning rate | 7,4 × 10⁻⁵ đến 1 × 10⁻⁴ |
| Bộ tối ưu | AdamW |
| Số epoch tối đa | 100 |

Cấu hình hiện hành dùng learning rate `1 × 10⁻⁴`, nằm đúng ở giới hạn trên của
dải công bố, và tỷ lệ che 0,15 trùng khớp. Ngân sách vòng lặp là 150 cho giai
đoạn một và 100 cho giai đoạn hai; vòng tốt nhất của cả sáu cấu hình đều nằm
trong ngân sách, cao nhất là 58 trên 100, nên không lần nào bị cắt khi còn đang
tiến bộ.

Hạn chế do chính tác giả nêu:

> "Like other anomaly detection models, GraphIDS assumes relatively stable
> network behavior, and its performance may **degrade under abrupt distribution
> shifts**, potentially leading to increased false positives or missed
> detections."

Ba hướng phát triển tác giả đề xuất: học trực tuyến để thích nghi liên tục mà
không cần huấn luyện lại toàn bộ; tích hợp dữ liệu đa phương thức; đo độ trễ
trong điều kiện triển khai thật.

Bộ thí nghiệm hiện hành không có cặp đối chứng nào giữa cấu hình của đồ án và
cấu hình tác giả công bố. Hạn chế về distribution shift mà tác giả tự nêu thì có
kiểm: phần đánh giá out-of-distribution ở mục 4 của `notebooks/twoDTS/
NB3_phan_tich.ipynb` đưa ba mô hình sang NF-ToN-IoT-v3 và đo mức suy giảm.

## 3. TE-G-SAGE — cấu hình thật và luận điểm trung tâm

Cấu hình tốt nhất sau tìm kiếm lưới của nhóm tác giả:

| Tham số | Giá trị |
|---|---|
| Độ sâu | **2** (cố định từ bước tinh chỉnh sơ bộ) |
| Fan-out | (25, 15) |
| Hidden size | 128 |
| Tỷ lệ bỏ học | 0,4 |
| Kích thước batch | 512 |
| Bộ tối ưu | Adam |
| Nhúng đỉnh | hằng số học được, **không giữ danh tính máy chủ** |

Luận điểm trung tâm ở phần kết luận:

> "TE-G-SAGE showed that **multi-hop**, edge-aware modeling **improves recall**
> under evolving traffic."

Lý do tác giả cố định độ sâu bằng 2: độ sâu lớn hơn làm tăng số đỉnh và cạnh
phải lấy mẫu mỗi mini-batch và tăng nguy cơ làm trơn quá mức trên đồ thị lưu
lượng có nhãn pha trộn.

Lý do dùng nhúng đỉnh hằng số: "enables the model to treat nodes as anonymous
routers of flow context and **avoids leaking host labels into the encoder**".
Điểm này biện minh trực tiếp cho thí nghiệm phân rã thành phần "bỏ định danh
đỉnh" của đồ án — đó không phải một phép thử tuỳ hứng mà là tái hiện một quyết
định thiết kế có chủ đích của TE-G-SAGE.

Thí nghiệm tương ứng: `tegsage_binary_v3` và `tegsage_multiclass_v3` đã sửa
sang độ sâu 2 cho đúng bài gốc; `proposed_multiclass_depth2_v3` kiểm luận điểm
multi-hop trên kiến trúc của đồ án.

## 4. Xác nhận độc lập cho trích dẫn TCG-IDS

Mục [20] trong danh mục tài liệu tham khảo của TE-G-SAGE ghi:

> "Wu, C.; Sun, J.; Chen, J.; Alazab, M.; Liu, Y.; Xiang, Y. TCG-IDS: Robust
> Network Intrusion Detection via Temporal Contrastive Graph Learning. IEEE
> Trans. Inf. Forensics Secur. 2025, 20, 1475–1486."

Đây là xác nhận từ một nguồn thứ ba độc lập cho thông tin thư mục của TCG-IDS.
Đồ án không dẫn công trình này, lý do loại trừ ghi ở mục 2.6 của báo cáo.
