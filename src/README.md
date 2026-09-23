# Thư mục `src`

Thư mục này chứa gói Python `tgn_nids`, tức phần mã nguồn lõi của hệ thống.
Các tệp trong đây xử lý dữ liệu NetFlow, dựng đồ thị thời gian, định nghĩa mô
hình, điều phối thí nghiệm, giải thích dự đoán và chuẩn hóa đầu ra báo cáo.

## Cấu trúc thư mục

```text
src/
├── README.md
└── tgn_nids/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── graph_builder.py
    │   ├── labels.py
    │   ├── preprocess.py
    │   └── protocols.py
    ├── experiments/
    │   ├── __init__.py
    │   ├── ood_eval.py
    │   └── runner.py
    ├── explainers/
    │   ├── __init__.py
    │   ├── gnn_explainer.py
    │   └── subgraph_extractor.py
    ├── models/
    │   ├── __init__.py
    │   ├── anomaly_detector.py
    │   ├── edge_decoder.py
    │   ├── embedding.py
    │   └── tgn_memory.py
    ├── simulation/
    │   ├── __init__.py
    │   └── generator.py
    └── utils/
        ├── __init__.py
        ├── figures.py
        ├── format.py
        └── paper_metrics.py
```

## Mô tả từng tệp

### Cấp gói

| Tệp | Chức năng |
| --- | --- |
| `README.md` | Mô tả cấu trúc và trách nhiệm của các tệp trong thư mục `src`. |
| `tgn_nids/__init__.py` | Khai báo phiên bản mã nguồn qua hằng `SOURCE_VERSION`. |

### `data/` — dữ liệu NetFlow và đồ thị thời gian

| Tệp | Chức năng |
| --- | --- |
| `data/__init__.py` | Công khai các thành phần tiền xử lý, dựng đồ thị và quy đổi nhãn. |
| `data/graph_builder.py` | Ánh xạ địa chỉ IP thành đỉnh, chuyển từng luồng thành cạnh có hướng và tạo `TemporalData`. |
| `data/labels.py` | Quy đổi các cách ghi nhãn về `0` cho Benign và `1` cho Attack; báo lỗi khi gặp nhãn lạ ở chế độ nghiêm ngặt. |
| `data/preprocess.py` | Chọn đặc trưng số, xử lý giá trị thiếu/vô hạn, biến đổi log, chuẩn hóa và mã hóa nhãn đa lớp. |
| `data/protocols.py` | Cài đặt các cách chia tập TE-G-SAGE, GraphIDS, Anomal-E và chia ngẫu nhiên phân tầng. |

### `models/` — các khối của mô hình

| Tệp | Chức năng |
| --- | --- |
| `models/__init__.py` | Công khai `EdgeFeatureDecoder` từ gói mô hình. |
| `models/anomaly_detector.py` | Định nghĩa bộ phân loại cạnh nhận biểu diễn hai đỉnh và, khi cần, đặc trưng của chính cạnh. |
| `models/edge_decoder.py` | Che một phần đặc trưng cạnh, tái thiết phần bị che và tính điểm bất thường từ sai số tái thiết. |
| `models/embedding.py` | Tổng hợp ngữ cảnh lân cận bằng `TransformerConv`, `LayerNorm` và kết nối tắt. |
| `models/tgn_memory.py` | Bọc `TGNMemory` để lưu trạng thái của từng đỉnh theo thời gian. |

### `experiments/` — huấn luyện và đánh giá

| Tệp | Chức năng |
| --- | --- |
| `experiments/__init__.py` | Đánh dấu gói chứa các thành phần thí nghiệm. |
| `experiments/ood_eval.py` | Nạp checkpoint an toàn và đánh giá mô hình trên dữ liệu NetFlow ngoài phân phối theo từng khối. |
| `experiments/runner.py` | Khai báo cấu hình, chia dữ liệu, huấn luyện hai giai đoạn, dừng sớm, hiệu chỉnh ngưỡng và lưu đầu ra. |

### `explainers/` — giải thích dự đoán

| Tệp | Chức năng |
| --- | --- |
| `explainers/__init__.py` | Công khai các lớp giải thích và chỉ nạp chúng khi được truy cập. |
| `explainers/gnn_explainer.py` | Tính độ quan trọng của đặc trưng bằng ablation và học mặt nạ cạnh/đặc trưng cho một đồ thị con. |
| `explainers/subgraph_extractor.py` | Trích Ego-Network trong phạm vi `k` bước quanh một đỉnh IP để hiển thị ngữ cảnh cảnh báo. |

### `simulation/` — dữ liệu mô phỏng

| Tệp | Chức năng |
| --- | --- |
| `simulation/__init__.py` | Công khai lớp phát lại dữ liệu mẫu `LogGenerator`. |
| `simulation/generator.py` | Đọc CSV NetFlow mẫu và phát tuần tự từng bản ghi theo nhịp cấu hình. |

### `utils/` — tiện ích dùng chung

| Tệp | Chức năng |
| --- | --- |
| `utils/__init__.py` | Đánh dấu gói tiện ích dùng chung. |
| `utils/figures.py` | Thiết lập phong cách hình, vẽ các biểu đồ đánh giá và xuất bảng hoặc bản kê kết quả. |
| `utils/format.py` | Định dạng số đếm, số thực và phần trăm; tránh làm tròn khiến giá trị bị hiểu sai. |
| `utils/paper_metrics.py` | Tính chỉ số nhị phân, đa lớp và ánh xạ tên chỉ số theo từng bài báo đối sánh. |
