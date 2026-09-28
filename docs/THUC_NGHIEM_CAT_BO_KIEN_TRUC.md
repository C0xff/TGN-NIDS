# THỰC NGHIỆM CẮT BỎ KIẾN TRÚC: CẤU HÌNH ĐÃ CHẠY VÀ GIỚI HẠN DIỄN GIẢI

Bốn cấu hình đã chạy trên NF-UNSW-NB15-v3. Kết quả nằm ở
`models/saved/twoDTS_ablation/`; bảng đầy đủ kèm điều kiện đo ở
`docs/SO_LIEU_DAU_RA_CHI_TIET.md`. Tệp này ghi **cách làm** và **những ràng buộc
đã áp dụng khi thực hiện, không lặp lại con số và không phải hướng dẫn chạy lại.

Thí nghiệm này **có trong báo cáo**, với đúng bốn cấu hình đã chạy. Phần mở rộng
sang nhiều hạt giống thì không: nó thuộc Hướng phát triển, ghi ở mục 6. Vì vậy
mọi con số trích ra đây phải đi kèm giới hạn phát hiện ở mục 5.

---

## 1. Hai loại ablation, không được lẫn

**Ablation thuộc tính**, tiếng Anh là Feature Ablation: lần lượt đặt từng cột đặc
trưng về giá trị không rồi đo độ lệch của xác suất đầu ra, để xếp hạng mức đóng
góp của từng đặc trưng. Đây là **công cụ giải thích**, cài trong
`src/tgn_nids/explainers/`, kết quả ở
`models/saved/twoDTS_phan_tich/tables/04_giai_thich/`.

**Ablation kiến trúc**, thứ tệp này nói tới: vô hiệu hoá một thành phần của kiến
trúc rồi huấn luyện lại từ đầu và đo lại toàn bộ chỉ số.

Lẫn hai thứ này từng làm một bảng trong báo cáo dẫn sai nguồn sang ô giải thích
của notebook phân tích.

---

## 2. Bốn cấu hình và cách vô hiệu hoá từng thành phần

| Cấu hình | Thành phần bị vô hiệu hoá | Cách làm |
|---|---|---|
| `baseline` | không | mốc đối chiếu trong cùng phiên thực thi |
| `random_nodes` | danh tính đỉnh | cờ `ablate_node_identity` của `runner.py`, gán lại ngẫu nhiên đỉnh nguồn và đích theo hạt giống |
| `no_iat` | tám cột đặc trưng thời gian IAT | cờ `ablate_features` của `runner.py`, loại tám cột khỏi bảng dữ liệu |
| `no_propagation` | phép tổng hợp từ đỉnh lân cận | đặt `num_layers = 0` cho `TemporalEmbeddingModule`, giữ nguyên vector nhớ |

Điểm đáng chú ý: `no_propagation` **không cần sửa `runner.py`**. `num_layers` là
tham số sẵn có; đặt bằng không thì không còn tầng chú ý nào và vector nhớ đi
thẳng ra ngoài. Notebook có ô tiền kiểm dựng một `TemporalEmbeddingModule` thử,
khẳng định `len(gat_layers) == 0` và đầu ra vẫn đúng hình dạng, trước khi chạy
thật.

Mốc đối chiếu được chạy trong cùng phiên với ba cấu hình còn lại, nên chênh lệch
trong bảng không ghép từ các phiên có môi trường khác nhau.

---

## 3. Bảo toàn kết quả đã chốt

`runner.py` nằm trong phạm vi khoá và đã sinh ra toàn bộ kết quả hiện hành. Hai
cờ `ablate_features`, `ablate_node_identity` cùng tham số `num_layers` là các
thiết lập đã dùng cho bốn cấu hình trong báo cáo.

Không chạy lại thí nghiệm đã công bố và không ghi đè kết quả đã lưu. Nếu sau này
có một nghiên cứu mở rộng, nó cần dùng một quy trình độc lập và một cây kết quả
mới; số đo mới không được trộn với bốn cấu hình của báo cáo.

Kết quả của phép cắt bỏ được giữ riêng tại `models/saved/twoDTS_ablation/`, tách
khỏi `twoDTS_train_v2`, `twoDTS_train_v3` và `twoDTS_phan_tich`.

---

## 4. Ba điểm đã kiểm khi trích số

**Trường `ablation` phải khớp cấu hình đã đặt.** Đọc lại trường đó trong từng tệp
kết quả; cấu hình ablation nào mà trường vẫn rỗng thì lần chạy đó đã không bật
chế độ ablation và kết quả vô nghĩa.

**Vòng tốt nhất không được nằm sát cuối ngân sách.** Nằm sát cuối nghĩa là mô
hình vẫn đang tiến bộ khi bị cắt, nên con số của nó không đọc được như con số của
một mô hình đã hội tụ. Notebook tự tính cột `near_epoch_limit` và in cảnh báo.

**Tên cột phải là tên đại lượng thật.** Xuất tên cột từ chính khoá của tệp kết
quả, đừng gõ tay.

---

## 5. Chỉ số phải ghi kèm

Macro F1 phải luôn đứng cạnh **recall của lớp tấn công**: hai tiêu chí chỉ tách
nhau khi mô hình thất bại, và một bảng chỉ có Macro F1 che mất đúng cái thất bại
cần nêu.

Mọi F1 nhị phân phải kèm **mốc phân loại tầm thường** `2r/(1+r)` với `r` là tỷ lệ
tấn công của đúng tập đang đo.

---

## 6. Thiết kế bắt cặp nhiều hạt giống, thuộc Hướng phát triển

Đây là cách tách hiệu ứng thật khỏi dao động giữa các lần chạy. Chưa chạy, ghi
lại ở đây để phần Hướng phát triển của báo cáo có nội dung cụ thể thay vì một câu
hứa chung.

**Thiết kế.** Chọn năm hạt giống, ví dụ 42 tới 46. Với mỗi hạt giống, chạy đủ hai
cấu hình `baseline` và `no_propagation` trong cùng điều kiện, cùng phép chia tập,
cùng ngân sách epoch. Mỗi hạt giống vì vậy cho một **cặp** giá trị Macro F1.

**Vì sao bắt cặp.** So hai giá trị trung bình của hai nhóm độc lập thì dao động
giữa các lần chạy vào thẳng phép so. Bắt cặp theo hạt giống thì mỗi cặp chịu đúng
một nguồn ngẫu nhiên, và phép so chỉ xét **dấu của hiệu trong từng cặp**.

**Phép kiểm.** Dùng phép kiểm dấu. Nếu cả năm cặp cùng dấu thì xác suất xảy ra do
tình cờ là `2 · 0,5^5`, xấp xỉ 6,3% cho kiểm hai phía và 3,1% cho kiểm một phía.
Năm cặp là số tối thiểu để phép kiểm còn nói được điều gì; bốn cặp cùng dấu chỉ
cho 12,5% một phía.

**Phải báo cáo kèm giới hạn phát hiện.** Năm cặp không phát hiện được hiệu ứng nhỏ
hơn biên độ dao động. Nên cùng với kết luận phải ghi biên độ dao động quan sát
được giữa các hạt giống, để người đọc biết phép đo này phát hiện được tới mức nào.

Một phép đo mở rộng cần lưu kết quả vào cây riêng và tách khỏi
`models/saved/twoDTS_ablation/`.

---

## 7. Nguồn thực nghiệm

Notebook đã thực thi kèm đầu ra từng ô là
`notebooks/executed/b-twodts-ablation.ipynb`, chạy trên Kaggle ở kernel
`julonao/b-twodts-ablation`.
