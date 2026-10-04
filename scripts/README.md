# Thư mục `scripts/`

Toàn bộ kịch bản phụ trợ của dự án: chuẩn bị dữ liệu, chạy thực nghiệm, phân
tích kết quả, kiểm toán chất lượng và vận hành.

Bảng điều khiển Streamlit không nằm ở đây mà nằm độc lập tại `apps/dashboard/`.

Nhiều script cần dữ liệu, checkpoint hoặc gói bàn giao không được công bố trên
GitHub. Các tài sản đó nằm trong gói Google Drive đi kèm; đọc mô tả tham số và
đường dẫn trước khi chạy để tránh ghi vào kết quả đã chốt.

Liên kết trong tệp này là **đường dẫn tương đối**, để tệp đọc được như nhau trên
Windows, macOS và trên trang kho mã.

---

## 1. Bố cục

```
scripts/
├── pipeline/   chuẩn bị dữ liệu và chạy thực nghiệm
├── analysis/   phân tích kết quả, lập bảng, kiểm toán
└── README.md   tệp này
```

---

## 2. `pipeline/` (dữ liệu và thực nghiệm)

| Tệp | Việc nó làm |
|---|---|
| [`build_demo_samples.py`](pipeline/build_demo_samples.py) | Rút 10.000 luồng phân tầng theo lớp từ Parquet gốc, ghi ra ba tệp `data/samples/demo_*.csv` cho bảng điều khiển |
| [`download_nids_datasets.py`](pipeline/download_nids_datasets.py) | Tải các bộ NetFlow thô về `data/raw/` |
| [`rebuild_parquet.py`](pipeline/rebuild_parquet.py) | Chuyển CSV gốc sang Parquet và đối chiếu toàn vẹn từng ô |

Thêm vào đây: script về chuẩn bị dữ liệu, tải dữ liệu, tiền xử lý, chạy pipeline
mô hình, hoặc đồng bộ kết quả thực nghiệm.

---

## 3. `analysis/` (phân tích, lập bảng, kiểm toán)

Các script tổng hợp và kiểm toán chỉ đọc kết quả đã chốt. Hai script đo lường
có phạm vi riêng: `supplementary_measurements.py` huấn luyện các baseline dạng
bảng nhỏ và ghi vào `models/supplementary/`, còn `do_tap_demo.py` chạy phép thử
giao diện. Không chạy lại `do_tap_demo.py` hoặc notebook thực nghiệm trên bản
dữ liệu hoàn thiện vì chúng có thể ghi đè kết quả đã dùng trong báo cáo.

| Tệp | Việc nó làm |
|---|---|
| [`tong_hop_ket_qua.py`](analysis/tong_hop_ket_qua.py) | Đọc mọi `result_*.json` trong `models/saved/`, sinh `docs/KET_QUA_DO_LUONG.md` |
| [`sinh_so_lieu_dau_ra.py`](analysis/sinh_so_lieu_dau_ra.py) | Sinh `docs/SO_LIEU_DAU_RA_CHI_TIET.md`, ghi kèm điều kiện đo của từng chỉ số: bộ dữ liệu, bài toán, tập đo, khoá JSON |
| [`supplementary_measurements.py`](analysis/supplementary_measurements.py) | Đo lối tắt dữ liệu và huấn luyện các baseline Decision Tree, Logistic Regression trên UNSW-v3; ghi kết quả vào `models/supplementary/`. |
| [`do_tap_demo.py`](analysis/do_tap_demo.py) | Đo ba tệp demo ở hai trạng thái bộ nhớ của mô hình; chỉ dùng cho kiểm thử chức năng giao diện. |
| [`profile_datasets.py`](analysis/profile_datasets.py) | In bảng quy mô các bộ dữ liệu thô và bảng đối chiếu dung lượng CSV với Parquet |

Thêm vào đây: công cụ phân tích, tính chỉ số, kiểm chất lượng mã, lập bảng đối
sánh.

---

## 4. Quy tắc bắt buộc khi thêm script

1. **Đặt đúng thư mục con.** Không thêm tệp `.py` mới trực tiếp ở thư mục gốc
   `scripts/`. Xác định mục đích trước rồi đặt vào `pipeline/` hoặc
   `analysis/`.

2. **Dựng đường dẫn gốc từ vị trí tệp.** Script nằm ở thư mục con cấp một nên
   gốc dự án là `parents[2]`:

   ```python
   from pathlib import Path
   PROJECT_ROOT = Path(__file__).resolve().parents[2]
   ```

   Không ghi cứng `C:\...` hay `/Users/...`: dự án được mở luân phiên trên
   Windows và macOS.

3. **Chỉ đọc đối với `models/saved/`.** Script mới không ghi đè, không xoá,
   không sửa tệp trọng số hay tệp `result_*.json` đã lưu. Đó là nguồn sự thật
   về số liệu của đồ án; sửa một byte là số trong báo cáo không còn tái lập
   được. Đầu ra bổ trợ mới phải ghi sang thư mục riêng như
   `models/supplementary/`.
