# TGN-NIDS

**Hệ thống phát hiện xâm nhập mạng dựa trên mạng đồ thị thời gian kết hợp mô
hình giải thích.**

Đây là đồ án tốt nghiệp ngành Trí tuệ nhân tạo tại Trường Đại học Công nghệ
Thông tin, ĐHQG-HCM. Hệ thống sử dụng Temporal Graph Network (TGN) để phân tích
dữ liệu NetFlow, phát hiện luồng bất thường, nhận diện loại tấn công và cung
cấp thông tin giải thích cho cảnh báo.

## Tổng quan

TGN-NIDS biểu diễn lưu lượng mạng dưới dạng đồ thị động liên tục:

- mỗi địa chỉ IP là một đỉnh;
- mỗi luồng NetFlow là một cạnh có hướng, kèm thời gian và thuộc tính kết nối;
- bộ nhớ của từng đỉnh được cập nhật sau mỗi sự kiện để giữ lại lịch sử liên
  lạc.

Hệ thống thực hiện hai giai đoạn học trên cùng một bộ mã hóa:

1. **Phát hiện bất thường tự giám sát:** học từ lưu lượng lành tính thông qua
   nhiệm vụ tái thiết đặc trưng bị che. Sai số tái thiết được dùng làm điểm bất
   thường.
2. **Phân loại tấn công có giám sát:** đóng băng bộ mã hóa đã học và huấn luyện
   đầu phân loại cho các lớp hành vi của từng bộ dữ liệu. UNSW-v3 và UNSW-v2
   có 10 lớp, còn bộ CSE có 15 lớp.

Khối giải thích hỗ trợ phân tích độ nhạy đặc trưng và trích xuất đồ thị lân cận
quanh endpoint cần điều tra. GNNExplainer được chạy ngoại tuyến cho một cảnh
báo đại diện, không nằm trong dashboard.

## Kết quả chính

| Nội dung đánh giá | Bộ dữ liệu | Kết quả |
| --- | --- | ---: |
| Phát hiện bất thường tự giám sát | NF-UNSW-NB15-v3 | F1 lớp tấn công **97,615%**, FAR **0,336%** |
| Phân loại 10 lớp | NF-UNSW-NB15-v3 | Macro F1 **55,961%** |
| Phân loại quy về hai lớp | NF-UNSW-NB15-v3 | F1 **99,700%**, Recall lớp tấn công **99,955%** |
| Đánh giá ngoài phân phối sau hiệu chuẩn bằng nhãn của nửa tập đích | NF-ToN-IoT-v3 | F1 **74,073%**, ROC-AUC **82,583%** |
| Cắt bỏ định danh node (`random_nodes`) | NF-UNSW-NB15-v3 | Macro F1 giảm **2,827 điểm phần trăm** |

Ở bài toán phân loại 10 lớp, mô hình cao hơn TE-G-SAGE 6,904 điểm phần trăm và
thấp hơn XGBoost 0,821 điểm phần trăm trên cùng lát cắt kiểm thử. Thí nghiệm cắt
bỏ cho thấy định danh node có ích chủ yếu khi gọi tên loại tấn công, trong khi
F1 hai lớp vẫn đạt 99,903% sau khi xáo trộn định danh.

UNSW-v3 có lối tắt dữ liệu mạnh: toàn bộ luồng tấn công xuất phát từ bốn địa
chỉ nguồn và riêng `MAX_TTL` đã đạt ROC-AUC 0,9984. Cây quyết định sâu 3 không
dùng đồ thị vẫn đạt F1 hai lớp 99,79%. Vì vậy, kết quả hai lớp không đủ để kết
luận TGN tốt hơn mô hình dạng bảng. Báo cáo dựa thêm vào bài toán nhiều lớp,
đánh giá ngoài phân phối và thí nghiệm cắt bỏ để phân tích vai trò của đồ thị.

Điều kiện đo, kết quả theo lớp và các giới hạn thực nghiệm được trình bày trong
[`docs/`](docs/README.md). Dữ liệu gốc, checkpoint, báo cáo Word và toàn văn
tài liệu tham khảo được nhóm lưu trong gói Google Drive đi kèm, không nằm trong
kho mã công khai.

## Dashboard

Ứng dụng Streamlit trong `apps/dashboard/` hỗ trợ:

- phân tích dữ liệu CSV hoặc Parquet theo lô;
- mô phỏng luồng dữ liệu theo từng mini-batch;
- theo dõi cảnh báo, điểm bất thường và hàng đợi cần xem xét;
- so sánh chỉ số giữa các mô hình;
- xem đồ thị lân cận và độ quan trọng đặc trưng bằng phép ablation.

Dashboard xử lý tệp có sẵn và mô phỏng phát lại theo khối. Hệ thống chưa nhận
NetFlow trực tiếp từ hạ tầng mạng và chưa tích hợp với Kafka hoặc SIEM. Các chỉ
số trên ba tệp trình diễn chỉ kiểm tra chức năng giao diện, không phải kết quả
đánh giá khoa học của mô hình.

Kho mã công khai không kèm dữ liệu thô hoặc checkpoint. Khi không có trọng số,
dashboard chuyển sang chế độ demo; kết quả khi đó chỉ dùng để kiểm tra luồng xử
lý và giao diện, không đại diện cho hiệu năng của mô hình đã huấn luyện. Muốn
chạy suy luận bằng trọng số đã chốt, cần có gói Drive đi kèm.

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS hoặc Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e .
```

Chạy dashboard:

```bash
streamlit run apps/dashboard/app.py
```

Chạy kiểm thử:

```bash
pytest
```

## Cấu trúc kho mã công khai

```text
TGN-NIDS/
├── .vscode/                    # Thiết lập workspace dùng chung
├── apps/                       # Dashboard Streamlit
├── docs/                       # Tài liệu kỹ thuật và số liệu chi tiết
├── notebooks/
│   └── executed/               # Bốn notebook đã chạy kèm đầu ra
├── references/
│   └── TAI_LIEU_THAM_KHAO.md   # Danh mục trích dẫn của báo cáo
├── scripts/                    # Công cụ chuẩn bị dữ liệu và phân tích
├── src/                        # Mã nguồn thư viện tgn_nids
├── tests/                      # Bộ kiểm thử
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

| Thành phần | Nội dung |
| --- | --- |
| [`apps/`](apps/README.md) | Dashboard và tài nguyên giao diện |
| [`docs/`](docs/README.md) | Số liệu đầu ra, kiến trúc, giao thức và tài liệu chuyên đề |
| [`notebooks/executed/`](notebooks/executed/README.md) | Notebook đã thực thi dùng để đối chiếu quá trình thực nghiệm |
| [`references/TAI_LIEU_THAM_KHAO.md`](references/TAI_LIEU_THAM_KHAO.md) | 23 tài liệu được trích dẫn trong báo cáo |
| [`scripts/`](scripts/README.md) | Script tải, chuẩn bị, kiểm tra và tổng hợp dữ liệu |
| [`src/`](src/README.md) | Mã nguồn xử lý dữ liệu, mô hình, thực nghiệm và XAI |
| [`tests/`](tests/README.md) | Kiểm thử các điều kiện ảnh hưởng đến tính đúng của kết quả |

Dữ liệu thô, checkpoint, kết quả thực nghiệm gốc, gói phân phối, báo cáo Word, hồ sơ chính thức của
trường và toàn văn tài liệu tham khảo không nằm trong kho công khai. Chúng được
tổ chức riêng trong gói Google Drive của đồ án.

## Nhóm thực hiện

- **Phạm Văn Cường** — 25410027
- **Đặng Thiên Phước** — 25410111
- **Giảng viên hướng dẫn:** TS. Phan Thế Duy

Đồ án tốt nghiệp ngành Trí tuệ nhân tạo, Trường Đại học Công nghệ Thông tin,
Đại học Quốc gia Thành phố Hồ Chí Minh, năm 2026.
