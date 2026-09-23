# THỰC NGHIỆM CẮT BỎ KIẾN TRÚC: CÁCH ĐÃ CHẠY VÀ RÀNG BUỘC KHI CHẠY LẠI

Bốn cấu hình đã chạy trên NF-UNSW-NB15-v3. Kết quả nằm ở
`models/saved/twoDTS_ablation/`; bảng đầy đủ kèm điều kiện đo ở
`docs/SO_LIEU_DAU_RA_CHI_TIET.md`. Tệp này ghi **cách làm** và **những ràng buộc
phải giữ** nếu chạy lại hoặc mở rộng, không lặp lại con số.

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
| `baseline` | không | mốc đối chiếu, chạy lại trong cùng phiên |
| `random_nodes` | danh tính đỉnh | cờ `ablate_node_identity` của `runner.py`, gán lại ngẫu nhiên đỉnh nguồn và đích theo hạt giống |
| `no_iat` | tám cột đặc trưng thời gian IAT | cờ `ablate_features` của `runner.py`, loại tám cột khỏi bảng dữ liệu |
| `no_propagation` | phép tổng hợp từ đỉnh lân cận | đặt `num_layers = 0` cho `TemporalEmbeddingModule`, giữ nguyên vector nhớ |

Điểm đáng chú ý: `no_propagation` **không cần sửa `runner.py`**. `num_layers` là
tham số sẵn có; đặt bằng không thì không còn tầng chú ý nào và vector nhớ đi
thẳng ra ngoài. Notebook có ô tiền kiểm dựng một `TemporalEmbeddingModule` thử,
khẳng định `len(gat_layers) == 0` và đầu ra vẫn đúng hình dạng, trước khi chạy
thật.

Mốc đối chiếu **phải chạy lại trong cùng phiên**, không lấy lại số của lần chạy
trước, nếu không thì chênh lệch đọc được lẫn cả sai khác môi trường.

---

## 3. Ràng buộc bắt buộc: không được làm hỏng kết quả đang có

**Thứ nhất, không sửa `runner.py` khi chưa hỏi.** Hai cờ `ablate_features` và
`ablate_node_identity` đã đủ cho hai cấu hình, và cấu hình thứ ba dùng tham số
sẵn có. Nếu một cấu hình mới cần thêm cờ, hoặc cần đổi hành vi của cờ cũ, thì
phải **dừng lại và hỏi trước**: `runner.py` nằm trong phạm vi khoá và là tệp đã
sinh ra toàn bộ kết quả hiện hành.

**Thứ hai, mọi sửa đổi trong `runner.py` phải giữ nguyên đường chạy khi không bật
ablation.** Sáu cấu hình huấn luyện gốc đều đi theo đường mặc định. Một thay đổi
làm đổi đường đó khiến chúng không còn dựng lại được, tức hỏng khả năng tái lập
của cả đồ án. Phép kiểm sau khi sửa: chạy lại một cấu hình cũ với cùng hạt giống
rồi so chỉ số với tệp kết quả đang lưu.

**Thứ ba, không ghi vào thư mục kết quả cũ.** Lần chạy ablation ghi vào cây riêng
`models/saved/twoDTS_ablation/`, không chạm `twoDTS_train_v2`, `twoDTS_train_v3`
hay `twoDTS_phan_tich`. Sau khi chạy, đối chiếu lại kết quả
phải báo các cấu hình cũ còn nguyên vẹn; tệp mới hiện ra dưới dạng `THÊM` và chỉ
chốt vào bản kê khi đã duyệt.

---

## 4. Ba phép tự kiểm phải chạy trước khi trích số

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

**Ràng buộc.** Lần chạy mới phải ghi vào một cây kết quả riêng, không ghi vào
`models/saved/twoDTS_ablation/`, theo đúng mục 3 tệp này.

---

## 7. Nguồn của notebook

Bản nguồn là `notebooks/twoDTS/NB4_ablation.ipynb`, bản đã thực thi kèm đầu ra
từng ô là `notebooks/executed/b-twodts-ablation.ipynb`, chạy trên Kaggle ở kernel
`julonao/b-twodts-ablation`.
