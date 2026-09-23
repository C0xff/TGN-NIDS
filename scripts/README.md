# Thư mục `scripts/`

Toàn bộ kịch bản phụ trợ của dự án: chuẩn bị dữ liệu, chạy thực nghiệm, phân
tích kết quả, kiểm toán chất lượng và vận hành.

Bảng điều khiển Streamlit không nằm ở đây mà nằm độc lập tại `apps/dashboard/`.

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

## 2. `pipeline/` — dữ liệu và thực nghiệm

| Tệp | Việc nó làm |
|---|---|
| [`build_demo_samples.py`](pipeline/build_demo_samples.py) | Rút 10.000 luồng phân tầng theo lớp từ Parquet gốc, ghi ra ba tệp `data/samples/demo_*.csv` cho bảng điều khiển |
| [`download_nids_datasets.py`](pipeline/download_nids_datasets.py) | Tải các bộ NetFlow thô về `data/raw/` |
| [`rebuild_parquet.py`](pipeline/rebuild_parquet.py) | Chuyển CSV gốc sang Parquet và đối chiếu toàn vẹn từng ô |

Thêm vào đây: script về chuẩn bị dữ liệu, tải dữ liệu, tiền xử lý, chạy pipeline
mô hình, hoặc đồng bộ kết quả thực nghiệm.

---

## 3. `analysis/` — phân tích, lập bảng, kiểm toán

Mọi script trong nhóm này **chỉ đọc** `models/saved/`, trừ `do_tap_demo.py` chỉ
ghi vào riêng cây `models/saved/twoDTS_demo/`.

| Tệp | Việc nó làm |
|---|---|
| [`tong_hop_ket_qua.py`](analysis/tong_hop_ket_qua.py) | Đọc mọi `result_*.json` trong `models/saved/`, sinh `docs/KET_QUA_DO_LUONG.md` |
| [`sinh_so_lieu_dau_ra.py`](analysis/sinh_so_lieu_dau_ra.py) | Sinh `docs/SO_LIEU_DAU_RA_CHI_TIET.md`, ghi kèm điều kiện đo của từng chỉ số: bộ dữ liệu, bài toán, tập đo, khoá JSON |
| [`do_tap_demo.py`](analysis/do_tap_demo.py) | Đo ba tệp demo ở hai trạng thái bộ nhớ của mô hình, ghi vào `models/saved/twoDTS_demo/` |
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
   được. Ngoại lệ duy nhất là `do_tap_demo.py`, chỉ ghi vào riêng cây
   `models/saved/twoDTS_demo/`.
