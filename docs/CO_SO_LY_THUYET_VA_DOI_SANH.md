# Cơ sở lý thuyết và ma trận đối sánh

Đây là nguồn cho **Chương 2** của báo cáo: bối cảnh khoa học, ba công trình đối
sánh, công trình bị loại khỏi phạm vi, đặc tả NetFlow và hệ thống chỉ số. Chỉ
những số liệu đã kiểm tra trong bản toàn văn mới được đưa vào báo cáo.

Số liệu đo được của đồ án không nằm ở đây. Chúng nằm ở
`docs/SO_LIEU_DAU_RA_CHI_TIET.md`, sinh trực tiếp từ `models/saved`.

---

## 1. Bối cảnh khoa học

### 1.1. Vì sao chuyển từ bảng tĩnh sang đồ thị

Các thế hệ hệ thống phát hiện xâm nhập trước đây dựa trên nhận diện chữ ký,
hoặc trên các mô hình học máy phân tích lưu lượng NetFlow dưới dạng **bảng
tĩnh**. Cách này xem mỗi luồng là một thực thể độc lập, tách khỏi bối cảnh giao
tiếp của toàn hệ thống. Random Forest, SVM hay mạng nơ-ron nhiều tầng đều học
được đặc trưng thống kê của từng luồng, nhưng đánh mất hai tính chất bản thể của
không gian mạng: **liên kết cấu trúc** giữa các máy, và **sự tiến hoá hành vi
theo thời gian**.

Chuyển dữ liệu sang dạng đồ thị giải quyết vế thứ nhất. Địa chỉ IP thành đỉnh,
luồng giao tiếp thành cạnh, và mạng nơ-ron đồ thị trích được đặc trưng tô-pô cục
bộ qua cơ chế truyền tin giữa các đỉnh lân cận.

### 1.2. Vì sao đồ thị tĩnh chưa đủ

Các kiến trúc đồ thị thế hệ đầu giả định đồ thị là **tĩnh và đồng nhất**: dựng
một đồ thị duy nhất bao trùm toàn bộ dữ liệu lịch sử rồi huấn luyện trên đó.
Cách này phá vỡ tính nhân quả của dòng thời gian và dẫn tới **rò rỉ dữ liệu**:
các luồng của cùng một chiến dịch tấn công xuất hiện đan xen ở cả tập huấn luyện
lẫn tập kiểm thử, khiến mô hình ghi nhớ địa chỉ độc hại thay vì học mẫu hình
hành vi.

Đồ án này gặp đúng vấn đề đó ở dạng cực đoan nhất: trên NF-UNSW-NB15-v3, toàn bộ
tấn công chỉ xuất phát từ bốn địa chỉ nguồn, nên một luật chỉ xét địa chỉ nguồn
đã tách được gần trọn hai lớp. Đây không còn là nguy cơ lý thuyết mà là tính chất
đo được của bộ dữ liệu.

### 1.3. Mạng đồ thị thời gian

Hướng nghiên cứu hiện hành dịch sang **mạng nơ-ron đồ thị động theo thời gian**.
Khung TGN do Rossi và cộng sự đề xuất xử lý mạng như một chuỗi sự kiện mang dấu
thời gian, kèm một mô-đun bộ nhớ dựng trên đơn vị hồi tiếp có cổng. Mô-đun này
cập nhật trạng thái của từng đỉnh sau mỗi tương tác, nhờ đó hệ thống "nhớ" được
rằng một địa chỉ đã quét cổng vài giờ trước khi thực hiện tấn công từ chối dịch
vụ hiện tại.

Cách xử lý này phản ánh đúng môi trường của một trung tâm điều hành an ninh, nơi
nhật ký mạng đổ về liên tục dưới dạng chuỗi sự kiện tuần tự.

### 1.4. Vì sao phải giải thích được

Rào cản lớn nhất khi triển khai học sâu vào môi trường an toàn thông tin là bản
chất hộp đen. Đầu ra của một mạng đồ thị phức tạp thường chỉ là một nhãn hoặc một
điểm số bất thường, không kèm cơ sở diễn giải. Với chuyên gia phân tích, thiếu
minh bạch đồng nghĩa với không xác minh được cảnh báo, dẫn tới kéo dài thời gian
phản ứng. Vì vậy đồ án bắt buộc tích hợp kỹ thuật trí tuệ nhân tạo giải thích
được, cụ thể là trích xuất subgraph ego-network quanh một địa chỉ bị nghi ngờ.

---

## 2. Ba công trình đối sánh và một công trình bị loại

Đồ án đối sánh trực tiếp với Anomal-E, GraphIDS và TE-G-SAGE. TCG-IDS có trong
đề cương ban đầu nhưng bị loại khỏi báo cáo định lượng vì không có bản toàn văn
để kiểm chứng điều kiện đo.

Các kiến trúc đồ thị đời đầu học biểu diễn cho **đỉnh**, trong khi bản chất tấn
công mạng nằm ở **hành vi giao tiếp**, tức ở cạnh. Hướng học **hướng cạnh** ra
đời để khắc phục, và là nền của cả Anomal-E lẫn GraphIDS. Đồ án không cài đặt
hay đối sánh riêng với các mô hình học giám sát hướng cạnh, nên chúng chỉ được
nhắc tới ở đây như bối cảnh.

### 2.1. Anomal-E, tự giám sát trên đồ thị

Caville, Lo, Layeghy, Portmann. **Knowledge-Based Systems**, tập 258, bài 110030,
2022. arXiv:2207.06819.

Dùng hạt nhân E-GraphSAGE kết hợp cơ chế tối đa hoá thông tin tương hỗ Deep Graph
Infomax. Điểm sáng tạo nằm ở cách sinh đồ thị âm: thay vì xáo trộn đặc trưng
đỉnh như DGI nguyên bản, hệ thống xáo trộn **đặc trưng cạnh**, buộc mô hình phân
biệt cấu trúc thật với cấu trúc nhiễu.

Kết quả trên NF-UNSW-NB15-v2, lấy mẫu 10% rồi chia ngẫu nhiên 70/30:

| Biến thể | Bảng | Accuracy | Macro F1 | Detection Rate |
|---|---|---|---|---|
| Isolation Forest, nhiễm bẩn 4% | Bảng 4 | 98,66% | 92,35% | 98,77% |
| HBOS, nhiễm bẩn 4% | Bảng 8 | 98,18% | 88,45% | 80,36% |

Cả hai hàng đều đã đối chiếu bản gốc. Đồ án dùng hàng Isolation Forest. **Phải ghi rõ biến thể mỗi khi
trích**, vì bốn biến thể chênh nhau tới sáu điểm Macro F1.

### 2.2. GraphIDS, đỉnh cao hiện tại của hướng tự giám sát

**Guerra**, Chapuis, Duc, Mozharovskyi, Nguyen. NeurIPS 2025. arXiv:2509.16625.

> Tác giả đầu của GraphIDS là **Lorenzo Guerra**; Mozharovskyi là tác giả thứ tư.

Khác với các phương pháp tự giám sát dựa trên học tương phản, GraphIDS dùng một
khung hợp nhất đầu cuối: một mạng đồ thị nắm bối cảnh tô-pô cục bộ, cộng một bộ
mã hoá tự động che giấu nền Transformer học các mẫu đồng xuất hiện trên toàn cục.
Khi huấn luyện, mô hình phục hồi biểu diễn luồng bị che; khi suy luận, sai số tái
thiết cao bất thường bị coi là tấn công.

Trên NF-UNSW-NB15-v3: **PR-AUC 99,98%, Macro F1 99,61%**, vượt các đường cơ sở từ
5 tới 25 điểm phần trăm.

GraphIDS chia 80/10/10 phân tầng theo loại tấn công, tập huấn luyện chỉ chứa
Benign và chọn ngưỡng bằng cực đại Macro F1 trên tập kiểm định. Đồ án dựng lại
giao thức này trên UNSW-v2. Trên UNSW-v3, đồ án dùng lát cắt theo thời gian nên
hai kết quả không cùng giao thức. Hai phương pháp còn khác ở cách chuẩn hóa và
kiến trúc encoder.

### 2.3. TCG-IDS, học đối nghịch theo thời gian

Wu, Sun, Chen, Alazab, Liu, Xiang. IEEE Transactions on Information Forensics and
Security, tập 20, trang 1475-1486, 2025.

Tích hợp ba chiến lược học tương phản: theo thời gian để nắm chuỗi phụ thuộc hành
vi, bất đối xứng để phân tích giao tiếp không đồng nhất, và che giấu để nâng chất
lượng biểu diễn đỉnh.

> **Loại khỏi bảng đối sánh.** Bài nằm sau tường phí IEEE, không có bản tiền ấn
> mở, nhóm không có bản toàn văn. Không mở được bảng kết quả thì không xác nhận
> được điều kiện đo của từng con số. Đề cương đã nộp có nêu bài này, nên báo cáo
> **vẫn nhắc tới** ở dạng ghi chú về giới hạn phạm vi.

### 2.4. TE-G-SAGE, giải thích được và đánh giá theo thời gian

**Luša**, Pintar, Vranić. **Modelling**, MDPI, tập 6, số 4, bài 165, 2025.

Kết hợp kiến trúc học cạnh có yếu tố thời gian với thuật toán SHAP, hiệu chỉnh để
áp lên subgraph tính toán trong phạm vi hai hop. Nhờ đó xuất được giải
thích chỉ đích danh đặc trưng nào và kết nối nào dẫn tới cảnh báo.

Đóng góp phương pháp luận quan trọng hơn cả kết quả: nhóm tác giả chỉ ra rằng phần
lớn mô hình đồ thị hiện tại lạm dụng chia tách ngẫu nhiên, gây rò rỉ dữ liệu tương
lai và thổi phồng hiệu năng. Khi đánh giá lại theo trình tự thời gian nghiêm ngặt,
điểm số trở về mức thực: Accuracy 0,95586 nhưng Macro F1 chỉ **0,49057**.

Sự sụt giảm này phản ánh nhầm lẫn chéo giữa các lớp tấn công thiểu số, do cơ chế
truyền tin của mạng đồ thị khuếch đại các tín hiệu dùng chung.

Triết lý thiết kế của họ: trong an ninh mạng, thà chấp nhận một tỷ lệ cảnh báo giả
nhất định để đẩy độ thu hồi lên cao, còn hơn bỏ sót tấn công. Đồ án kế thừa định
hướng này và dùng phần giải thích làm bộ lọc trực quan giúp người trực rà soát
cảnh báo.

**Hai hàng của Bảng 7 mà báo cáo bắt buộc phải có:** XGBoost
đạt Macro F1 **0,56782** và GCN đạt **0,38781**, đo trên cùng bộ dữ liệu và cùng
lát cắt. XGBoost cao hơn mô hình đồ án.

---

## 3. Đặc tả trường dữ liệu NetFlow

**Nhóm tô-pô mạng.** `IPV4_SRC_ADDR` và `IPV4_DST_ADDR` là địa chỉ nguồn và đích,
trở thành đỉnh của đồ thị động. `L4_SRC_PORT` và `L4_DST_PORT` là cổng dịch vụ,
giúp nhận diện dịch vụ đứng sau: 80 cho HTTP, 443 cho HTTPS, 22 cho SSH, 53 cho
DNS. `PROTOCOL` là giao thức tầng mạng: 6 cho TCP, 17 cho UDP, 1 cho ICMP.

**Nhóm khối lượng.** `IN_BYTES` và `OUT_BYTES` là tổng byte theo hai chiều;
`IN_PKTS` và `OUT_PKTS` là tổng gói tin. Nhóm `NUM_PKTS_UP_TO_128_BYTES` tới
`NUM_PKTS_1024_TO_1514_BYTES` cho phân bố kích thước gói, giúp phân biệt lưu lượng
tải tệp lớn với luồng điều khiển nhỏ.

**Nhóm trạng thái kết nối.** `TCP_FLAGS` là tổ hợp cờ SYN, ACK, FIN, RST, PSH,
URG, then chốt để phát hiện SYN Flood hoặc quét cổng. `CLIENT_TCP_FLAGS` và
`SERVER_TCP_FLAGS` tách theo chiều. `RETRANSMITTED_IN_BYTES` và
`RETRANSMITTED_OUT_BYTES` đếm byte truyền lại.

**Nhóm thời gian, chỉ có ở bộ đặc trưng v3.** `FLOW_START_MILLISECONDS` và
`FLOW_END_MILLISECONDS` là dấu thời gian chính xác tới mili giây;
`FLOW_DURATION_MILLISECONDS` là thời gian tồn tại của phiên. Tám cột IAT
(`SRC_TO_DST_IAT_MIN/MAX/AVG/STDDEV` và bốn cột tương ứng chiều ngược lại) ghi
thống kê khoảng cách giữa các gói tin liên tiếp.

**Nhãn.** `Label` là nhãn nhị phân, 0 cho lành tính và 1 cho tấn công. `Attack`
là nhãn đa lớp.

> **Số cột, đo trên chính tệp dữ liệu.**
> NF-UNSW-NB15-v2 có **45 cột**, NF-UNSW-NB15-v3 có **55 cột**, và v3 hơn v2
> **12 cột**: tám cột IAT, hai cột mốc thời gian, hai cột địa
> chỉ. Riêng NF-CSE-CIC-IDS2018-v2 chỉ có 43 cột, vì bản UQ phát hành cho bộ đó
> **không kèm cột địa chỉ**.

---

## 4. Hệ thống chỉ số đánh giá

### 4.1. Tám chỉ số và lý do chọn từng chỉ số

Bộ NF-UNSW-NB15 mất cân bằng cực đoan: hơn 94% là lưu lượng lành tính. Chỉ dùng
Accuracy là nguy hiểm, vì một mô hình gán nhãn lành tính cho mọi luồng vẫn đạt
trên 94% mà hoàn toàn vô dụng. Vì vậy đồ án dùng tám chỉ số sau.

- **Accuracy** chỉ dùng làm thước đo nền.
- **Precision** cho biết trong số cảnh báo phát ra, bao nhiêu phần là thật. Cao
  thì giảm tải công việc vô ích cho người trực.
- **Recall** đo khả năng bắt được tấn công đang diễn ra. Bỏ lỡ một vụ xâm nhập
  nặng hơn nhiều so với một cảnh báo nhầm, nên hệ thống ưu tiên chỉ số này.
- **Macro F1** tính F1 độc lập cho từng lớp rồi lấy trung bình cộng, nên lớp
  hiếm như Worms có cùng trọng số với lớp đông. Đây là trục đối sánh chính.
- **FAR**, còn gọi FPR, đo xác suất hiểu nhầm lưu lượng hợp pháp thành tấn công.
  Phải giữ cực thấp để tránh hiện tượng người trực mệt mỏi vì cảnh báo.
- **PR-AUC** không phụ thuộc ngưỡng và không bị số lượng âm tính đúng chi phối
  như ROC-AUC, nên khách quan hơn trên dữ liệu mất cân bằng.
- **Độ trễ suy luận và thông lượng**, để mô tả chi phí của suy luận theo lô.
  Phép đo này không đủ để kết luận hệ thống vận hành thời gian thực.
- **Mốc của bộ phân loại tầm thường**, `2·rate/(1+rate)`. Không có mốc này thì
  một F1 cao vẫn có thể nằm dưới mức đoán bừa.

### 4.2. Ba trục đối sánh

**Trục một, sức bền dưới ràng buộc rò rỉ dữ liệu.** Đối chiếu trực tiếp với phát
hiện của TE-G-SAGE: Macro F1 của họ rơi xuống 49,057% khi buộc phải tôn trọng
trình tự thời gian. Mô hình của đồ án chịu cùng ràng buộc trên NF-UNSW-NB15-v3.

**Trục hai, tốc độ vận hành.** Đồ án báo độ trễ và thông lượng của chính mô hình
trên Tesla T4. Không đối sánh tốc độ với TCG-IDS vì không có toàn văn để xác
nhận phần cứng, cách đo và điều kiện chạy.

**Trục ba, chất lượng giải thích.** TE-G-SAGE dùng SHAP trên subgraph tính toán.
Đồ án kết hợp độ nhạy thuộc tính, GNNExplainer chạy ngoại tuyến và đồ thị lân
cận trên giao diện. Phần này trình bày cơ chế và ví dụ, chưa có chỉ số chung để
xếp hạng chất lượng giải thích giữa hai hệ thống.

---

## 5. Thiết kế phần đối sánh

**Hai bảng tách bạch, không gộp làm một.** Đã cài đặt trong ô đối sánh của cả
hai notebook huấn luyện.

Lý do không gộp: giai đoạn một giải bài toán hai lớp, ma trận nhầm lẫn 2x2; giai
đoạn hai giải bài toán nhiều lớp, ma trận 10x10 hoặc 15x15. Hai giá trị cùng
mang tên Macro F1 nhưng đo trên hai bài toán khác nhau, đặt chung một trục thì
cột nhị phân trông như mạnh gần gấp đôi cột đa lớp trong khi hai con số không so
được với nhau. Các công trình đối sánh cũng chia đúng hai nhóm như vậy: GraphIDS
và Anomal-E công bố số nhị phân, TE-G-SAGE công bố số đa lớp.

Bộ dữ liệu nào không có công trình nào công bố số cho một bài toán thì bỏ hẳn
biểu đồ của bài toán đó, vì một biểu đồ đối sánh chỉ có đúng một cột của chính
đồ án không đối sánh với thứ gì; giá trị vẫn được ghi xuống bảng CSV.

**Ràng buộc kèm theo:** mọi chỗ hiển thị Macro F1 phải hiển thị kèm **recall của
lớp tấn công**. Macro F1 cho lớp lành tính một phiếu ngang lớp tấn công, mà lớp
lành tính vừa đông vừa dễ, nên nó có thể cao trong khi hệ thống bỏ sót phần lớn
tấn công.

**Bảng một, nhị phân.** Đặt kết quả giai đoạn một cạnh GraphIDS và Anomal-E,
kèm đúng giao thức của từng hàng. Phần thảo luận bổ sung mốc tầm thường và các
baseline dạng bảng để tránh đọc F1 cao như bằng chứng riêng cho lợi thế của đồ
thị.

**Bảng hai, đa lớp.** Bảng dùng số của GCN, TE-G-SAGE và XGBoost từ Bảng 7 của
bài TE-G-SAGE, đặt cạnh kết quả giai đoạn hai của đồ án. Ba mô hình công bố dùng
lát cắt 60/30/10, còn kết quả chính của đồ án dùng 80/10/10. Báo cáo ghi rõ
khác biệt này thay vì xem các hàng là một phép xếp hạng tuyệt đối.

Phần chỉ số trích từ PDF gốc của ba công trình đối sánh nằm ở mục 2 của chính
tệp này, và đã đối chiếu với bản gốc của từng bài.

---

## 6. Điều kiện thực nghiệm của ba công trình, trích trực tiếp từ PDF

Phần này tồn tại để mọi phát biểu so sánh trong Chương 4 và Chương 5 đều dẫn
được về một dòng cụ thể trong bài gốc.

### 6.1. Quy mô dữ liệu, GraphIDS Bảng 1

| Bộ dữ liệu | Số luồng | Số máy | Tỷ lệ tấn công |
|---|---:|---:|---:|
| NF-UNSW-NB15-v2 | 2.390.275 | 44 | 3,98% |
| NF-UNSW-NB15-v3 | 2.365.424 | 44 | 5,40% |
| NF-CSE-CIC-IDS2018-v2 | 18.893.708 | 255.042 | 11,95% |
| NF-CSE-CIC-IDS2018-v3 | 20.115.529 | 205.801 | 12,93% |

**GraphIDS chạy toàn bộ CIC-IDS2018.** Đồ án cắt còn 11,76% vì trần 12 giờ của
Kaggle, còn 2.365.423 luồng và 76.060 máy. Hai con số CIC vì vậy **không so trực
tiếp được**, và bảng đối sánh CIC bắt buộc ghi kèm ba dòng điều kiện này. Nơi so
ngang hàng thật sự là NF-UNSW-NB15, nơi đồ án chạy trọn bộ đúng bằng họ.

### 6.2. Siêu tham số, GraphIDS Bảng 14

| Tham số | UNSW-v3 | CIC-2018-v3 | UNSW-v2 | CIC-2018-v2 |
|---|---:|---:|---:|---:|
| `edim_out` của GNN | 96 | 64 | 72 | 64 |
| `embed_dim` của Transformer | 48 | 32 | 48 | 32 |
| `num_heads` | 4 | 4 | 4 | 4 |
| `num_layers`, `nhops` | 1 | 1 | 1 | 1 |
| `mask_ratio` | 0,15 | 0,15 | 0,15 | 0,15 |
| `dropout` của GNN | 0,6 | 0,5 | 0,75 | 0,5 |
| `learning_rate` | 1e-4 | 1e-4 | 1,1e-5 | 7,4e-5 |
| `gnn_weight_decay` | 0,6 | 0,6 | 0,6 | 0,6 |
| `ae_weight_decay` | 0,04 | 0,04 | 0,046 | 0,011 |

Cách tìm: quét lưới thô rồi tối ưu Bayes, chọn theo PR-AUC trên tập kiểm định.
Tổng chi phí toàn bộ thí nghiệm của họ là **976 giờ GPU** trên A100.

Anomal-E Bảng 1 dùng một cấu hình duy nhất cho mọi bộ: 1 lớp, 256 đơn vị ẩn,
learning rate 1e-3, ReLU, **không dropout**. TE-G-SAGE quét lưới nhỏ rồi chốt
hidden 128, fan-out (25,15), dropout 0,4, batch 512, cũng một cấu hình cho mọi bộ.

**Hai trên ba công trình không tinh chỉnh riêng từng bộ dữ liệu.** Việc đồ án cố
định một bộ tham số vì vậy là lựa chọn phổ biến, không phải thiếu sót; và chính
nó là điều kiện để phát biểu "cùng cấu hình, ba bộ cho kết quả khác nhau".

### 6.3. Năm điểm giao thức đồ án đã trùng

Dùng được làm căn cứ khi bảo vệ, thay cho câu "nhóm thấy nên chọn vậy".

| Điểm | GraphIDS | Đồ án |
|---|---|---|
| `mask_ratio` | 0,15 cho cả bốn bộ | 0,15 |
| Số hop | `nhops = 1` | `num_layers = 1` |
| Cột thời gian trong đặc trưng | loại bỏ | loại bỏ, 7 cột |
| Đại lượng dừng sớm | PR-AUC, patience 20 | PR-AUC, patience 25 |
| Kích thước lô | 512 | 512, trùng cả TE-G-SAGE |

### 6.4. Hai điểm khác biệt phải nêu là hạn chế

**Một, đồ án không dò siêu tham số bằng thí nghiệm.** Giá trị 192 cho `memory_dim`
và `embedding_dim` đi từ 100 lên 128 rồi 192 qua các phiên làm việc, không lần nào
có đối chứng. Cả ba công trình đều dò, dù ở mức độ khác nhau.

**Hai, chính quy hoá của đồ án nhẹ hơn nhiều.** `weight_decay` 1e-5 so với 0,6 của
GraphIDS, `dropout` 0,1 so với 0,5 đến 0,75. Lý do họ nêu ở phụ lục C.3: *"the GNN
is prone to overfitting in smaller network environments"*, tức mạng ít máy thì khối
đồ thị dễ học thuộc, và NF-UNSW-NB15 chỉ có 44 máy.

Con số của họ **không bê thẳng sang được**: họ tách hai bộ tối ưu riêng cho GNN và
cho bộ tái thiết, còn đồ án dùng một; và kiến trúc của họ không có khối bộ nhớ TGN
nên chưa ai biết phạt mạnh khối đó cho ra kết quả gì. Chỉ hướng là mang sang được,
không phải giá trị.

---

## 7. Đối tượng nghiên cứu của đề cương, ánh xạ sang tên lớp của bộ dữ liệu

Đề cương nêu đối tượng nghiên cứu gồm `Các kỹ thuật Evasion Attack (Port Scan,
Lateral Movement)`. Ba bộ dữ liệu NetFlow-v3 không dùng chính hai cụm từ đó mà
đặt tên lớp theo cách riêng, nên phần này ghi rõ chỗ nối để người đọc báo cáo
đối chiếu được giữa mục tiêu và bảng kết quả.

### 7.1. Bảng ánh xạ

| Kỹ thuật nêu trong đề cương | Lớp tương ứng trong dữ liệu | Bộ dữ liệu |
|---|---|---|
| Port Scan, dò quét dịch vụ | Reconnaissance | NF-UNSW-NB15-v2 và v3 |
| Lateral Movement, di chuyển ngang | Infilteration | NF-CSE-CIC-IDS2018-v3 |

`Reconnaissance` trong họ UNSW-NB15 gộp các hành vi dò quét cổng và thu thập
thông tin dịch vụ trước khi khai thác. `Infilteration` trong CSE-CIC-IDS2018 mô
tả giai đoạn sau khi đã có chỗ đứng trong mạng, khi kẻ tấn công mở rộng phạm vi
sang các máy khác, tức đúng nội dung của di chuyển ngang.

### 7.2. Kết quả đo được trên hai lớp này

| Lớp | Bộ dữ liệu | Precision | Recall | F1 | Số mẫu |
|---|---|---:|---:|---:|---:|
| Reconnaissance | NF-UNSW-NB15-v2 | 68,154% | 78,873% | 73,123% | 1.278 |
| Reconnaissance | NF-UNSW-NB15-v3 | 74,864% | 59,761% | 66,465% | 2.761 |
| Infilteration | NF-CSE-CIC-IDS2018-v3 | 81,073% | 11,613% | 20,316% | 2.213 |

Nguồn: trường `metrics.per_class` trong ba tệp
`result_stage2_nf_unsw_nb15_v2.json`, `result_stage2_nf_unsw_nb15_v3.json` và
`result_stage2_nf_cse_cic_ids2018_v3.json`, đo trên tập kiểm thử.

### 7.3. Hạn chế phải nêu, không được để người phản biện phát hiện trước

Lớp `Infilteration` đạt recall 11,613%, tức mô hình bỏ sót gần chín phần mười số
luồng di chuyển ngang. Đây là chỉ số thấp nhất trong toàn bộ các lớp được báo
cáo, và nó rơi đúng vào một kỹ thuật mà đề cương nêu đích danh, nên báo cáo phải
trình bày con số này ở phần kết quả chứ không để nó nằm im trong bảng phụ lục.

Ba điểm giải thích, đều kiểm chứng được từ dữ liệu.

Thứ nhất, precision của lớp này là 81,073%, cao hơn hẳn recall. Mô hình không
đoán bừa: khi nó gọi một luồng là `Infilteration` thì phần lớn là đúng, nhưng nó
chỉ dám gọi tên trong một phần nhỏ số trường hợp. Đây là dạng thất bại thiên về
bỏ sót, khác hẳn dạng thất bại báo động giả tràn lan.

Thứ hai, di chuyển ngang về bản chất là lưu lượng nội bộ trông giống lưu lượng
quản trị hợp lệ. Đặc trưng NetFlow ở mức luồng không chứa nội dung gói tin, nên
ranh giới giữa một phiên quản trị từ xa hợp lệ và một phiên di chuyển ngang
thường không nằm trong tập đặc trưng mà mô hình quan sát được.

Thứ ba, đây là hạn chế của bài toán chứ không riêng của mô hình đề xuất. Báo cáo
nên nêu như một giới hạn của biểu diễn dữ liệu ở mức luồng, kèm hướng khắc phục
là bổ sung đặc trưng theo phiên hoặc theo chuỗi hành vi của cùng một cặp máy.

### 7.4. Các lớp khác cùng họ, để bảng kết quả đọc được theo ngữ cảnh

Bốn lớp dưới đây thuộc cùng chuỗi tấn công với hai lớp trên, ghi kèm để phần
phân tích có đủ ngữ cảnh, đo trên NF-UNSW-NB15-v3.

| Lớp | Precision | Recall | F1 | Số mẫu |
|---|---:|---:|---:|---:|
| Exploits | 65,714% | 68,104% | 66,888% | 6.515 |
| Shellcode | 60,248% | 72,388% | 65,763% | 268 |
| Analysis | 33,438% | 75,352% | 46,320% | 142 |
| Backdoor | 71,429% | 0,406% | 0,808% | 1.231 |

Lớp `Backdoor` có recall 0,406% trên v3 trong khi đạt 71,429% recall trên v2.
Chênh lệch này phải nêu kèm số mẫu: v3 có 1.231 mẫu còn v2 có 217, nên khác biệt
không đến từ việc thiếu dữ liệu mà đến từ phân bố đặc trưng của chính bộ v3.
