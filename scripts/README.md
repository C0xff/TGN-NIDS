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
├── ops/        vận hành, sao lưu, đóng gói
└── README.md   tệp này
```

---

## 2. `pipeline/` — dữ liệu và thực nghiệm

| Tệp | Việc nó làm |
|---|---|
| [`build_demo_samples.py`](pipeline/build_demo_samples.py) | Rút 10.000 luồng phân tầng theo lớp từ Parquet gốc, ghi ra ba tệp `data/samples/demo_*.csv` cho bảng điều khiển |
| [`chay_thu_notebook.py`](pipeline/chay_thu_notebook.py) | Chạy thử notebook tại máy với cấu hình rút gọn, để lỗi lộ ra trước khi tải lên Kaggle |
| [`collect_kaggle_results.py`](pipeline/collect_kaggle_results.py) | Tải checkpoint `.pt`, tệp chỉ số `.json`, bảng và hình từ Kaggle về `models/saved/` |
| [`download_nids_datasets.py`](pipeline/download_nids_datasets.py) | Tải các bộ NetFlow thô về `data/raw/` |
| [`rebuild_parquet.py`](pipeline/rebuild_parquet.py) | Chuyển CSV gốc sang Parquet và đối chiếu toàn vẹn từng ô |

Thêm vào đây: script về chuẩn bị dữ liệu, tải dữ liệu, tiền xử lý, chạy pipeline
mô hình, hoặc đồng bộ kết quả thực nghiệm.

---

## 3. `analysis/` — phân tích, lập bảng, kiểm toán

Mọi script trong nhóm này **chỉ đọc** kết quả đã chốt. Riêng `do_tap_demo.py`
ghi vào một cây đầu ra kiểm tra tách biệt.

| Tệp | Việc nó làm |
|---|---|
| [`tong_hop_ket_qua.py`](analysis/tong_hop_ket_qua.py) | Đọc mọi `result_*.json` trong `models/saved/`, sinh `docs/KET_QUA_DO_LUONG.md` |
| [`sinh_so_lieu_dau_ra.py`](analysis/sinh_so_lieu_dau_ra.py) | Sinh `docs/SO_LIEU_DAU_RA_CHI_TIET.md`, ghi kèm điều kiện đo của từng chỉ số: bộ dữ liệu, bài toán, tập đo, khoá JSON |
| [`do_tap_demo.py`](analysis/do_tap_demo.py) | Đo ba tệp demo ở hai trạng thái bộ nhớ của mô hình và ghi kết quả vào cây kiểm tra riêng. |
| [`kiem_toan_du_an.py`](analysis/kiem_toan_du_an.py) | Kiểm toán toàn dự án: chú thích không dấu, emoji trong mã, đường dẫn ghi cứng, dấu vết sinh dữ liệu không thật, khối bắt lỗi nuốt ngoại lệ, `torch.load` thiếu chế độ an toàn, mã chết, tài liệu nhắc thứ đã xoá, sơ đồ kiến trúc có khớp mã hay không |
| [`profile_datasets.py`](analysis/profile_datasets.py) | In bảng quy mô các bộ dữ liệu thô và bảng đối chiếu dung lượng CSV với Parquet |
| [`check_notebooks.py`](analysis/check_notebooks.py) | Kiểm cú pháp Python và lược đồ nbformat của các tệp `.ipynb` |

Thêm vào đây: công cụ phân tích, tính chỉ số, kiểm chất lượng mã, lập bảng đối
sánh.

---

## 4. `ops/` — vận hành, sao lưu, đóng gói

| Tệp | Việc nó làm |
|---|---|
| [`package_for_kaggle.py`](ops/package_for_kaggle.py) | Nén `src/` thành tệp zip để tải lên làm Kaggle Dataset, và dừng hẳn nếu một tệp đọc ra ít byte hơn dung lượng khai báo |
| [`backup_ban_hoan_thien.py`](ops/backup_ban_hoan_thien.py) | Đóng gói bản sao lưu trạng thái hoàn thiện của dự án |
| [`kaggle_account.py`](ops/kaggle_account.py) | Chuyển đổi thông tin xác thực giữa các tài khoản Kaggle |

Thêm vào đây: công cụ bảo trì kho mã, sao lưu, quản trị token và môi trường.

---

## 5. Quy tắc bắt buộc khi thêm script

1. **Đặt đúng thư mục con.** Không thêm tệp `.py` mới trực tiếp ở thư mục gốc
   `scripts/`. Xác định mục đích trước rồi đặt vào `pipeline/`, `analysis/`
   hay `ops/`.

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
   được. Ngoại lệ duy nhất là `do_tap_demo.py`, chỉ ghi vào cây đầu ra kiểm tra
   tách biệt.
