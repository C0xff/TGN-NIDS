# Quy tắc sinh tập dữ liệu demo cho dashboard

Tệp này quy định cách sinh các tệp trong `data/samples/`, giải thích **vì sao**
mỗi đặc điểm của chúng được chọn, và phân tích điều gì xảy ra khi một bộ dữ liệu
thu thập thật ngoài đời **không** có những đặc điểm đó.

Bộ sinh nằm ở `scripts/pipeline/build_demo_samples.py`. Mọi tham số nêu trong tệp này đều
đủ để dựng lại ba tệp demo từ dữ liệu thô mà không cần đọc mã.

---

## 1. Tập demo phải trả lời câu hỏi gì

Một tập demo không phải một mẩu dữ liệu bất kỳ cắt ra cho đẹp. Nó là công cụ để
người xem tự kiểm chứng ba điều, theo đúng thứ tự:

Thứ nhất, **hệ thống có chạy thật không**. Người xem cần thấy mô hình nhận vào
dữ liệu thô và cho ra quyết định, chứ không phải một màn hình dựng sẵn.

Thứ hai, **con số trên màn hình có khớp con số trong báo cáo không**. Nếu báo
cáo ghi độ chính xác 99,949% mà demo hiện một con số khác hẳn thì một trong hai
chỗ sai, và người chấm sẽ hỏi ngay.

Thứ ba, **hệ thống hỏng ở đâu**. Một demo chỉ cho thấy phần chạy tốt là một demo
không trung thực. Người chấm luôn hỏi giới hạn của hệ thống, nên tốt hơn là cho
thấy giới hạn đó một cách có kiểm soát.

Ba câu hỏi này quyết định vì sao có ba tệp demo chứ không phải một.

---

## 2. Bốn ràng buộc bắt buộc, và lý do của từng ràng buộc

### 2.1. Không được chứa dữ liệu đã dùng để huấn luyện

Ràng buộc này áp dụng cho **tệp demo chấm trên chính mô hình đã học bộ dữ liệu
đó**. Trong ba tệp hiện hành chỉ có `demo_1` thuộc trường hợp ấy, nên chỉ
`demo_1` rút từ một tập xác định, cụ thể là 10% cuối theo thời gian, tức đúng
tập kiểm thử. Bộ sinh còn kiểm thêm rằng mọi địa chỉ trong tệp đều có ô nhớ đã
học, và dừng với lỗi nếu không.

`demo_2` và `demo_3` rút đều trên toàn bộ bộ dữ liệu nguồn, **có chủ đích**.
Hai tệp này chấm bằng checkpoint `nf_unsw_nb15_v3`, vốn chưa từng huấn luyện
trên CSE-CIC-IDS2018 ở bất kỳ phiên bản nào. Vì vậy việc một dòng nằm ở phần
train hay phần test của CSE không làm chỉ số lạc quan lên, và ràng buộc cắt
tập không cần thiết ở đây.

### 2.2. Không được trùng với bộ dữ liệu dùng cho phần đánh giá out-of-distribution

Phần đánh giá khả năng khái quát dùng `NF-ToN-IoT-v3`. Nếu tệp demo cũng lấy từ
bộ đó thì phần trình diễn và phần đo lường không còn độc lập: người xem không
phân biệt được đâu là minh hoạ và đâu là bằng chứng, còn nhóm thực hiện thì mất
khả năng dùng bộ ấy làm phép thử độc lập về sau.

### 2.3. Mọi địa chỉ phải có ô nhớ riêng trong bảng memory đã học

Đây là ràng buộc riêng của kiến trúc TGN và là ràng buộc dễ bỏ sót nhất.

Mô hình cấp cho mỗi máy chủ một vector nhớ, cập nhật mỗi khi có luồng của máy đó
đi qua. Bảng nhớ có kích thước cố định từ lúc huấn luyện, ở đây là 44 ô. Một địa
chỉ chưa từng gặp không có ô riêng, nên phần suy luận buộc phải băm nó vào một ô
đã có chủ: `zlib.crc32(địa_chỉ) % 44`.

Hệ quả là **nhiều máy không liên quan gì nhau dùng chung một vector nhớ**. Mô
hình đọc lịch sử của một máy khác rồi ra quyết định cho máy đang xét. Đo trên ba
tệp demo hiện hành, số liệu từ
`models/saved/twoDTS_demo/tables/do_tap_demo.csv`:

| Tệp demo | Số cột | Số đỉnh | Có ô nhớ riêng |
|---|---:|---:|---:|
| `demo_1_cung_phan_phoi` | 55 | 40 | 40 trên 40 |
| `demo_2_bo_du_lieu_la` | 55 | 2.426 | 0 trên 2.426 |
| `demo_3_thieu_cot_dia_chi` | 43 | không có cột địa chỉ | — |

Bảng nhớ có 44 ô và cả 44 ô đều đã có chủ từ checkpoint. Trên `demo_2`, toàn bộ
2.426 địa chỉ đều là địa chỉ lạ nên bị băm vào 44 ô đó, tức bộ nhớ theo đỉnh mất
sạch tác dụng. Mà bộ nhớ theo đỉnh chính là điểm phân biệt TGN với một mạng đồ
thị tĩnh thông thường, nên trên tệp đó phần demo không minh hoạ được đúng thứ đồ
án muốn trình bày.

### 2.4. Phải giữ đủ các lớp tấn công và giữ nguyên tỷ lệ lớp

Rút mẫu đều trên toàn bộ dòng làm biến mất các lớp hiếm, mà lớp hiếm lại là chỗ
mô hình yếu nhất và là chỗ người xem muốn nhìn. Vì vậy phép rút mẫu phân tầng
theo cột `Attack`, giữ nguyên tỷ lệ giữa các lớp của khung nguồn.

---

## 3. Ba tệp demo và vai trò của từng tệp

Tham số chung: 10.000 luồng mỗi tệp, rút mẫu phân tầng theo cột `Attack`, hạt
giống ngẫu nhiên cố định 42.

### 3.1. `demo_1_cung_phan_phoi.csv`

Nguồn: `NF-UNSW-NB15-v3`, lấy từ **tập kiểm thử**, tức 10% cuối sau khi sắp toàn
bộ theo `FLOW_START_MILLISECONDS`.

| Đặc điểm | Giá trị |
|---|---|
| Số luồng | 10.000 |
| Số cột | 55 |
| Tỷ lệ tấn công | 8,500% |
| Số lớp | 10 |
| Đỉnh có ô nhớ riêng | 40 trên 40 |

Đây là tệp trả lời câu hỏi thứ hai ở mục 1.

**Giao thức đo, bắt buộc nêu kèm mọi con số dưới đây.** Gọi `reset_memory` rồi
nạp đúng một tệp. `reset_memory` khôi phục bảng nhớ theo đỉnh về đúng trạng thái
lưu trong checkpoint, nên kết quả không phụ thuộc thứ tự các tệp đã mở trước đó.
Đây đúng là thứ dashboard làm trước mỗi tệp mới.

Đã kiểm tính tất định bằng ba bối cảnh khác nhau: chạy đầu tiên, chạy sau một
tệp khác, và chạy hai lần liên tiếp cùng một tệp. Cả ba cho cùng một kết quả tới
từng chữ số.

Giao thức phải nêu vì mô hình mang bộ nhớ theo đỉnh: kết quả phụ thuộc trạng
thái bộ nhớ lúc bắt đầu. Mục 3.4 đo cụ thể mức phụ thuộc đó.

Đo bằng chính dashboard:

| Chỉ số | Trên tệp demo | Đã ghi trong kết quả huấn luyện |
|---|---:|---:|
| Độ chính xác nhị phân | 99,940% | 99,949% |
| Phát hiện tấn công | 100,000% | 99,955% |
| Tỷ lệ báo động giả | 0,066% | 0,052% |

Ba cặp số khớp nhau trong sai số lấy mẫu, nên người xem đối chiếu màn hình với
báo cáo là thấy trùng.

Lưu ý khi đọc: các con số trên là **nhị phân quy đổi**, tức chỉ phân biệt có tấn
công hay không. Chỉ số đa lớp của cùng mô hình thấp hơn nhiều, độ chính xác
97,136% và Macro F1 55,961%. Hai bộ số đo hai bài toán khác nhau, không được đặt
cạnh nhau như thể so sánh được.

### 3.2. `demo_2_bo_du_lieu_la.csv`

Nguồn: `NF-CSE-CIC-IDS2018-v3`. Bộ này không phải tập huấn luyện của mô hình đang
công bố, cũng không phải bộ dùng cho phần đánh giá out-of-distribution, nên dùng làm
demo không đụng vào phép đo nào của báo cáo.

| Đặc điểm | Giá trị |
|---|---|
| Phạm vi rút | toàn bộ bộ dữ liệu nguồn |
| Số luồng | 10.000 |
| Số cột | 55 |
| Tỷ lệ tấn công | 12,930% |
| Số lớp | 15 |
| Đỉnh có ô nhớ riêng | 0 trên 2.426 |

Kết quả đo theo đúng giao thức nêu ở mục 3.1, trạng thái bảng nhớ khôi phục từ
checkpoint: độ chính xác 74,560%, phát hiện tấn công 5,800%, báo động giả
15,229%. Nguồn là `models/saved/twoDTS_demo/metrics/do_tap_demo.json`, sinh bởi
`scripts/analysis/do_tap_demo.py` đọc chính tệp demo hiện hành.

Tệp này trả lời câu hỏi thứ ba. Nó cho thấy một điều quan trọng mà bảng chỉ số
không nói được: mô hình **không sụp thành báo bừa**, mà sụp theo hướng bỏ sót.
Nó vẫn giữ được tỷ lệ báo động giả ở mức hai chữ số, nhưng gần như không còn
nhận ra tấn công nào. Đó là hành vi cần giải thích trong báo cáo, và nó bắt
nguồn từ đúng hai nguyên nhân: phân bố đặc trưng khác, và toàn bộ 2.426 địa chỉ
đều không có ô nhớ riêng.

### 3.3. `demo_3_thieu_cot_dia_chi.csv`

Nguồn: `NF-CSE-CIC-IDS2018-v2`, bản đặc trưng thứ hai, chỉ có 43 cột và **không
có cột địa chỉ IP**.

| Đặc điểm | Giá trị |
|---|---|
| Phạm vi rút | toàn bộ bộ dữ liệu nguồn |
| Số luồng | 10.000 |
| Số cột | 43 |
| Tỷ lệ tấn công | 11,840% |
| Đỉnh | không có cột địa chỉ |

Kết quả đo, theo đúng giao thức nêu ở mục 3.1: độ chính xác 59,970%, phát hiện
tấn công 4,814%, báo động giả 32,623%.

Giữ tệp này **có chủ đích**. Thiếu cột địa chỉ là tình huống rất phổ biến với dữ
liệu thu thập thật, và cần cho người xem thấy hệ thống phản ứng ra sao. Dashboard
khi đó tổng hợp định danh đỉnh từ số cổng, đặt tiền tố `ANONYMOUS_` và ghi cờ
`endpoint_identity_source = "synthetic"` để mọi chỗ hiển thị đều biết danh tính
đỉnh không phải quan sát được.

---

### 3.4. Kết quả phụ thuộc trạng thái bộ nhớ, và mức phụ thuộc đo được

Mô hình mang một vector nhớ cho mỗi đỉnh và cập nhật vector đó theo từng luồng
đi qua. Vì vậy cùng một tệp, cùng một bộ trọng số, vẫn cho kết quả khác nhau
tuỳ trạng thái bộ nhớ lúc bắt đầu. Đây là tính chất của kiến trúc chứ không
phải sai sót, nhưng nó buộc mọi con số phải nêu kèm giao thức đo.

Ba trạng thái khởi đầu khả dĩ, đo trên `demo_1_cung_phan_phoi.csv`:

| Bảng nhớ lúc bắt đầu | Độ chính xác | Phát hiện | Báo động giả |
|---|---:|---:|---:|
| khôi phục từ checkpoint | 99,940% | 100,000% | 0,066% |
| xoá trắng về không | 96,870% | 100,000% | 3,421% |
| còn dư sau một tệp khác | tuỳ tệp trước, không tái lập |  |  |

Chênh lệch ở tỷ lệ báo động giả là **52 lần**. Con số của lần chạy đã ghi trong
`result_*.json` là 99,949% và 0,052%, tức nhánh khôi phục là nhánh khớp báo cáo.

**Vì sao chốt nhánh khôi phục.** Bảng nhớ 44 x 192 trong tệp checkpoint là một
phần của thứ đã học: nó ghi trạng thái của từng máy chủ mà mô hình đã quan sát.
Xoá trắng bảng đó khiến mô hình bắt đầu ở tình trạng không biết gì về máy nào,
tức bỏ đi một phần năng lực đã huấn luyện, và cho ra con số không khớp bất kỳ
phép đo nào của đồ án.

**Hạn chế phải nêu khi trình bày, không được giấu.** Trình tự trong `runner.py`
là: xoá bảng nhớ, chạy qua tập huấn luyện, chạy qua tập kiểm định, rồi chạy qua
tập kiểm thử; checkpoint được lưu **sau** bước cuối. Bảng nhớ trong tệp `.pt` vì
vậy đã mang thông tin của tập kiểm thử.

`demo_1` rút từ chính tập kiểm thử đó. Nên con số 99,940% trên màn hình là con
số của một mô hình **đã triển khai và đã quen các máy chủ trong mạng**, không
phải con số của một phép đánh giá độc lập. Khi trình bày dashboard trước hội
đồng, phải nói rõ điều này.

Phép đánh giá độc lập nằm ở `models/saved/`, do `runner.py` sinh ra theo trình
tự thời gian: mỗi luồng kiểm thử được chấm điểm bằng bộ nhớ chỉ chứa những gì
xảy ra **trước** nó. Đó mới là con số đưa vào bảng kết quả của báo cáo.

`demo_2` và `demo_3` không vướng hạn chế này vì chúng đến từ bộ dữ liệu khác;
không đỉnh nào của chúng có ô nhớ riêng trong bảng đã học.

---

## 4. Nếu dữ liệu thu thập thật không có những đặc điểm trên thì sao

Đây là phần quan trọng nhất của tệp này, vì ba tệp demo ở trên đều cắt ra từ các
bộ dữ liệu nghiên cứu đã được làm sạch. Dữ liệu thu từ một mạng thật hiếm khi
giống vậy. Bốn tình huống dưới đây xếp theo mức độ ảnh hưởng.

### 4.1. Thiếu cột địa chỉ máy

**Xảy ra khi nào.** Bộ thu thập chỉ xuất thống kê mức luồng; hoặc địa chỉ bị gỡ
vì lý do riêng tư trước khi dữ liệu rời khỏi tổ chức.

**Mô hình bị gì.** Không dựng được đồ thị máy chủ. Toàn bộ phần đóng góp của
kiến trúc đồ thị biến mất, mô hình tụt về mức của một bộ phân loại chỉ dùng đặc
trưng luồng. Đo được trên `demo_3`: báo động giả nhảy lên 32,623%, tức cứ ba
luồng lành tính thì gần một luồng bị báo nhầm.

**Làm gì.** Không thay bằng định danh tổng hợp rồi báo cáo chỉ số như thường.
Định danh đỉnh quyết định toàn bộ cấu trúc đồ thị nên nó tham gia vào mọi chỉ
số; một topology bịa ra sẽ cho những con số nói về một mạng không tồn tại. Hoặc
đổi sang mô hình không cần đồ thị, hoặc yêu cầu bổ sung trường địa chỉ ở khâu
thu thập.

### 4.2. Địa chỉ hoàn toàn mới so với lúc huấn luyện

**Xảy ra khi nào.** Triển khai sang một mạng khác; mạng dùng cấp phát địa chỉ
động nên địa chỉ đổi theo thời gian; hoặc chỉ đơn giản là đã lâu kể từ lần huấn
luyện gần nhất.

**Mô hình bị gì.** Mỗi địa chỉ mới bị băm vào một ô nhớ đã có chủ và thừa hưởng
lịch sử của một máy không liên quan. Với bảng 44 ô và vài nghìn địa chỉ mới, số
máy dùng chung một ô lên tới hàng chục, như đo được ở mục 2.3. Kết quả là điểm
bất thường mang nhiều thông tin của máy khác hơn là của máy đang xét.

**Làm gì.** Ba hướng, xếp theo chi phí tăng dần. Một, huấn luyện lại trên dữ
liệu của chính mạng sẽ triển khai. Hai, đặt bảng nhớ lớn hơn hẳn số máy dự kiến
ngay từ lúc huấn luyện, đổi lại tốn bộ nhớ và có nhiều ô không bao giờ được cập
nhật. Ba, thêm cơ chế cấp ô nhớ động cho đỉnh mới kèm chính sách thu hồi ô cũ,
đây là thay đổi kiến trúc chứ không phải chỉnh tham số.

Trong mọi trường hợp, giao diện phải hiện rõ tỷ lệ đỉnh có ô nhớ đúng, để người
vận hành biết kết quả đang đáng tin tới đâu.

### 4.3. Thiếu một phần các cột đặc trưng

**Xảy ra khi nào.** Bộ thu thập dùng phiên bản đặc trưng cũ. Bản đặc trưng thứ
hai của họ NetFlow này có 43 cột, bản thứ ba có 55; tám cột chênh lệch là các
đại lượng về khoảng cách thời gian giữa các gói.

**Mô hình bị gì.** Phần tiền xử lý điền giá trị thay thế cho cột vắng mặt, nên
mô hình vẫn chạy nhưng đọc một vector đặc trưng khác hẳn vector nó đã học. Mức
suy giảm phụ thuộc vào việc các cột thiếu có nằm trong nhóm đặc trưng quan trọng
hay không, và điều đó đo được bằng ô giải thích trong notebook phân tích.

**Làm gì.** Dashboard đã hiện số cột thiếu ngay trên bảng số liệu. Ngưỡng chấp
nhận nên đặt theo phần đóng góp tích luỹ của các cột thiếu, chứ không theo số
lượng cột: thiếu tám cột ít quan trọng khác hẳn thiếu hai cột nằm trong nhóm
đầu bảng.

### 4.4. Không có nhãn

**Xảy ra khi nào.** Đây là tình huống mặc định của dữ liệu thật. Không ai gán
nhãn tấn công cho lưu lượng đang chảy.

**Mô hình bị gì.** Bản thân việc phát hiện vẫn chạy, vì giai đoạn một là học tự
giám sát và chỉ cần lưu lượng lành tính để huấn luyện. Cái mất là **khả năng đo
chất lượng**: không có nhãn thì không tính được độ chính xác, phát hiện tấn
công, hay tỷ lệ báo động giả.

**Làm gì.** Không hiển thị các chỉ số đó dưới dạng số. Đây chính là lý do phần
suy luận trả về giá trị rỗng kèm lý do thay vì một con số mặc định trông như
phép đo. Muốn có chỉ số thì phải có một tập nhỏ được gán nhãn tay, và mọi con số
báo cáo phải ghi rõ nó đo trên tập nhỏ đó chứ không phải trên toàn bộ lưu lượng.

---

## 5. Kiểm tra trước khi công bố một tệp demo mới

Năm phép kiểm dưới đây chạy được bằng mã, nên nếu về sau có thêm tệp demo thì
làm đủ cả năm.

Một, không dòng nào của tệp demo trùng với tập huấn luyện. Đối chiếu theo toàn
bộ các cột đặc trưng, không chỉ vài cột khoá.

Hai, tỷ lệ tấn công của tệp demo khớp tỷ lệ của phần dữ liệu nguồn tương ứng,
sai lệch dưới một phần vạn.

Ba, mọi lớp tấn công có mặt trong nguồn thì cũng phải có mặt trong tệp demo.

Bốn, với tệp thuộc nhóm cùng phân phối, tỷ lệ đỉnh có ô nhớ đúng phải bằng 100%.
Với tệp thuộc nhóm minh hoạ giới hạn, tỷ lệ này phải được ghi rõ trên giao diện.

Năm, chạy thử qua dashboard và đối chiếu ít nhất ba chỉ số với kết quả đã lưu
của lần chạy huấn luyện tương ứng.
