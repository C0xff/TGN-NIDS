"""In bảng tóm tắt quy mô của các bộ dữ liệu thô, dạng Markdown dán thẳng vào tài liệu.

Chỉ đọc siêu dữ liệu Parquet nên chạy trong vài giây.

Chạy:  python scripts/analysis/profile_datasets.py
"""

import os
from pathlib import Path

import pyarrow.parquet as pq

# Gốc dự án suy từ vị trí tệp này, không ghi cứng đường dẫn của một hệ điều
# hành. Dự án được mở luân phiên trên cả Windows lẫn macOS.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Tên tệp giữ đúng hoa thường như bản phát hành, vì Linux phân biệt hoa thường.
DATASETS = [
    ("NF-UNSW-NB15-v3", RAW_DIR / "nf-unsw-nb15-v3" / "NF-UNSW-NB15-v3.parquet"),
    ("NF-UNSW-NB15-v2", RAW_DIR / "nf-unsw-nb15-v2" / "NF-UNSW-NB15-v2.parquet"),
    ("NF-CSE-CIC-IDS2018-v3", RAW_DIR / "nf-cse-cic-ids2018-v3" / "NF-CSE-CIC-IDS2018-v3.parquet"),
    ("NF-CSE-CIC-IDS2018-v2", RAW_DIR / "nf-cse-cic-ids2018-v2" / "NF-CSE-CIC-IDS2018-V2.parquet"),
    ("NF-ToN-IoT-v3", RAW_DIR / "nf-ton-iot-v3" / "NF-ToN-IoT-v3.parquet"),
]


def main():
    """In bảng tổng quan dung lượng và cấu trúc các bộ dữ liệu."""
    print("| Bộ dữ liệu | Số luồng | Số cột | Dung lượng | Tệp |")
    print("|---|---:|---:|---:|---|")

    for name, path in DATASETS:
        # Bộ nào chưa tải về thì ghi rõ là không có, không bỏ qua im lặng: một
        # bảng thiếu dòng khiến người đọc tưởng bộ đó không nằm trong phạm vi.
        if not path.exists():
            print(f"| **{name}** | chưa tải về | — | — | — |")
            continue

        # read_metadata chỉ đọc phần chân tệp Parquet, không nạp dữ liệu.
        meta = pq.read_metadata(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        so_dong = "{:,}".format(meta.num_rows).replace(",", ".")
        dung_luong = "{:,.2f}".format(size_mb).replace(",", "@").replace(".", ",").replace("@", ".")
        print(f"| **{name}** | {so_dong} | {meta.num_columns} | "
              f"{dung_luong} MB | `{path.name}` |")


def bang_nen():
    """In bảng đối chiếu dung lượng CSV gốc với Parquet nén Snappy.

    """
    print()
    print("| Bộ dữ liệu | CSV gốc | Parquet | Mức giảm |")
    print("|---|---:|---:|---:|")

    for name, path in DATASETS:
        if not path.exists():
            print(f"| {name} | — | *chưa tải về* | — |")
            continue
        parquet_bytes = os.path.getsize(path)
        parquet_gb = so_gb(parquet_bytes)
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            # Không có CSV thì mức giảm không tính được, và chỗ đó phải hiện ra
            # là không tính được chứ không rơi về một con số trông như số đo.
            print(f"| {name} | *không có trong dự án* | {parquet_gb} GB | "
                  f"*không tính được* |")
            continue
        csv_bytes = os.path.getsize(csv_path)
        giam = round(100 * (1 - parquet_bytes / csv_bytes))
        print(f"| {name} | {so_gb(csv_bytes)} GB | {parquet_gb} GB | {giam}% |")


def so_gb(so_byte):
    """Đổi byte sang GB, hai chữ số thập phân, dấu phẩy thập phân tiếng Việt."""
    return "{:.2f}".format(so_byte / (1024 ** 3)).replace(".", ",")


if __name__ == "__main__":
    main()
    bang_nen()
