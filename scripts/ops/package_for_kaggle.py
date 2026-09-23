#!/usr/bin/env python3
"""
Đóng gói mã nguồn và dữ liệu thành hai tệp ZIP
riêng biệt, mỗi tệp tương ứng một Kaggle Dataset.

Mã nguồn và dữ liệu được tách rời vì hai thứ này thay đổi với nhịp độ hoàn
toàn khác nhau. Mã nguồn nhẹ vài trăm KB nhưng sửa liên tục; dữ liệu nặng hơn
200 MB nhưng gần như không đổi. Gộp chung thành một gói nghĩa là mỗi lần sửa
một dòng mã lại phải tải lên lại toàn bộ dữ liệu. Chính ma sát đó đã khiến lần
chạy đầu tiên diễn ra trên một bản mã nguồn cũ, và toàn bộ kết quả phải bỏ đi.

Notebook không nằm trong gói nào. Chúng được tải thẳng lên Kaggle Code từ thư
mục notebooks/, vì cần sửa và chạy lại nhiều lần.

    python3 scripts/package_for_kaggle.py           # dựng cả bốn gói
    python3 scripts/package_for_kaggle.py --code    # chỉ dựng gói mã nguồn
    python3 scripts/package_for_kaggle.py --data    # chỉ dựng gói dữ liệu
"""

import argparse
import os
import re
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sản phẩm dựng ra nằm riêng trong dist/, không lẫn với mã nguồn
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
CODE_ZIP = "kaggle_tgn_nids_code.zip"
DATA_ZIP = "kaggle_tgn_nids_data.zip"

# Gói mã nguồn: toàn bộ package src cùng vài tệp mô tả đi kèm
CODE_DIRS = ["src"]
CODE_FILES = ["requirements.txt", "notebooks/HUONG_DAN_KAGGLE.md"]

# Gói dữ liệu chính: hai bản NF-UNSW-NB15 kèm bảng mô tả thuộc tính. Đây là
# phạm vi dữ liệu huấn luyện mà đề cương chi tiết quy định, và mọi notebook
# huấn luyện đều cần tới.
DATA_FILES = [
    "data/raw/nf-unsw-nb15-v2/NF-UNSW-NB15-v2.parquet",
    "data/raw/nf-unsw-nb15-v2/NetFlow_v2_Features.csv",
    "data/raw/nf-unsw-nb15-v3/NF-UNSW-NB15-v3.parquet",
    "data/raw/nf-unsw-nb15-v3/NetFlow_v3_Features.csv",
]

# Bộ dữ liệu bổ sung của bản thực nghiệm mở rộng, đóng gói riêng vì chỉ notebook
# thứ hai cần tới; gộp vào gói chính sẽ buộc notebook thứ nhất phải tải thừa
# hơn một gigabyte mỗi lần được cấp máy.
#
# Chỉ bản đặc trưng thứ ba nằm trong gói. Bản NF-CSE-CIC-IDS2018-v2 do
# University of Queensland phát hành có 43 cột và không kèm cột địa chỉ IPv4,
# nên không dựng được đồ thị máy chủ và không notebook nào đọc tới. Đóng nó vào
# gói chỉ làm tệp tải lên nặng thêm khoảng sáu trăm megabyte.
EXTENDED_PACKAGES = {
    "kaggle_tgn_nids_cic_ids2018.zip": [
        "data/raw/nf-cse-cic-ids2018-v3/NF-CSE-CIC-IDS2018-v3.parquet",
    ],
}

# Bộ dùng riêng cho phần đánh giá khả năng khái quát ngoài phân phối.
# NF-ToN-IoT không tham gia huấn luyện ở bất kỳ cấu hình nào; nó chỉ được đưa
# qua mô hình đã huấn luyện để đo mức suy giảm khi rời khỏi mạng đã học.
#
# Cũng như NF-CSE-CIC-IDS2018, chỉ bản đặc trưng thứ ba dùng được: bản thứ hai
# không có cột địa chỉ, mà evaluate_ood_binary dừng với lỗi khi thiếu cột đó
# thay vì sinh địa chỉ thay thế.
OOD_PACKAGES = {
    "kaggle_tgn_nids_ood_ton_iot.zip": [
        "data/raw/nf-ton-iot-v3/NF-ToN-IoT-v3.parquet",
    ],
}

# Tệp rác không bao giờ đưa vào gói
SKIP_DIRS = ("__pycache__", ".ipynb_checkpoints")
SKIP_FILES = (".pyc", ".DS_Store")


def read_source_version() -> str:
    """Đọc số hiệu phiên bản mã nguồn để ghi vào tài liệu hướng dẫn."""
    path = os.path.join(PROJECT_ROOT, "src", "tgn_nids", "__init__.py")
    with open(path, encoding="utf-8") as handle:
        match = re.search(r'SOURCE_VERSION\s*=\s*"([^"]+)"', handle.read())
    if not match:
        raise RuntimeError(f"Không tìm thấy SOURCE_VERSION trong {path}")
    return match.group(1)


def write_readme(source_version: str) -> None:
    """Soạn lại tài liệu hướng dẫn, gắn số hiệu phiên bản đang đóng gói."""
    content = f"""# Hướng dẫn tải lên và chạy trên Kaggle (TGN-NIDS)

> **Phiên bản mã nguồn: {source_version}.**
> Số hiệu phiên bản được ghi kèm mọi tệp kết quả qua trường `source_version`,
> nhờ đó mỗi con số trong báo cáo truy được về đúng bản mã đã sinh ra nó.

Bốn gói được tách rời, mỗi thứ cập nhật theo nhịp riêng. Chỉ hai gói đầu là bắt
buộc; hai gói sau chỉ cần khi chạy nhánh mở rộng và nhánh đánh giá ngoài phân
phối.

| Gói | Nội dung | Nơi đặt trên Kaggle | Khi nào cần |
|---|---|---|---|
| `kaggle_tgn_nids_code.zip` | mã nguồn `src/tgn_nids/` | Dataset mã nguồn | luôn luôn, tải lại mỗi lần sửa mã |
| `kaggle_tgn_nids_data.zip` | NF-UNSW-NB15 v2 và v3 | Dataset dữ liệu | luôn luôn, gần như không đổi |
| `kaggle_tgn_nids_cic_ids2018.zip` | NF-CSE-CIC-IDS2018 v3 | Dataset dữ liệu | khi chạy NB2 |
| `kaggle_tgn_nids_ood_ton_iot.zip` | NF-ToN-IoT v3 | Dataset dữ liệu | khi chạy NB3, đánh giá ngoài phân phối |
| `notebooks/twoDTS/*.ipynb`, `notebooks/ablation/*.ipynb` | bốn notebook | Tải thẳng lên Kaggle Code | mỗi lần sửa notebook |

Nhờ cách tách này, sửa một dòng mã chỉ cần tải lên vài trăm KB thay vì hơn
200 MB dữ liệu.

---

## 1. Cấu trúc bốn gói

Dung lượng ghi bên cạnh là kích thước tệp Parquet nguồn trước khi nén.

```text
kaggle_tgn_nids_code.zip
├── src/tgn_nids/            # Mã nguồn cốt lõi: TGN, dựng đồ thị, giao thức, chỉ số, XAI
├── requirements.txt
└── HUONG_DAN_KAGGLE.md

kaggle_tgn_nids_data.zip
└── data/raw/
    ├── nf-unsw-nb15-v2/     # Parquet v2 kèm bảng mô tả thuộc tính, 67 MB
    └── nf-unsw-nb15-v3/     # Parquet v3 kèm bảng mô tả thuộc tính, 136 MB

kaggle_tgn_nids_cic_ids2018.zip
└── data/raw/
    └── nf-cse-cic-ids2018-v3/                                       # 1.052 MB

kaggle_tgn_nids_ood_ton_iot.zip
└── data/raw/
    └── nf-ton-iot-v3/                                                 # 611 MB
```

Bản Parquet là bản chuyển đổi nội bộ từ tệp CSV gốc đã xác minh checksum, đối
chiếu từng ô không lệch. Nguồn chính thức chỉ phát hành CSV.

---

## 2. Hướng dẫn từng bước

### Bước 1: Dataset dữ liệu, chỉ làm một lần

[Kaggle Datasets](https://www.kaggle.com/datasets) -> **New Dataset** -> đặt
tiêu đề (ví dụ `nf-unsw-nb15-parquet`) -> tải `kaggle_tgn_nids_data.zip` lên
-> **Create**.

Sau đó gần như không phải đụng tới nữa, trừ khi bản Parquet được dựng lại.

### Bước 2: Dataset mã nguồn, cập nhật mỗi lần sửa mã

Lần đầu: **New Dataset** -> đặt tiêu đề (ví dụ `tgn-nids-source`) -> tải
`kaggle_tgn_nids_code.zip` lên -> **Create**.

Những lần sau: mở đúng dataset đó -> **New Version** -> tải tệp zip mới lên.

**Cập nhật đúng dataset cũ, đừng tạo dataset mới.** Ô thiết lập dừng lại nếu
tìm thấy nhiều hơn một thư mục `src/tgn_nids` trong `/kaggle/input`, và liệt kê ra
để bạn gỡ bớt, thay vì tự chọn bừa một bản.

### Bước 3: Notebook

1. [Kaggle Code](https://www.kaggle.com/code) -> **New Notebook** -> **File**
   -> **Import Notebook**, chọn tệp `.ipynb` trong `notebooks/twoDTS/` hoặc
   `notebooks/ablation/` của kho mã trên máy.
2. Bảng **Notebook Options** bên phải:
   - **Accelerator**: **GPU T4 x2**. **Đừng chọn P100**: image mặc định
     của Kaggle nay dùng bản PyTorch cu128 không còn kernel cho kiến trúc
     Pascal (sm_60), nên P100 vẫn được cấp và `torch.cuda.is_available()`
     vẫn trả về True, nhưng phép tính CUDA đầu tiên chết với
     `cudaErrorNoKernelImageForDevice`. Kaggle đang loại dần P100.
   - **Internet**: **On**, để cài `torch_geometric`.
3. Mục **Input** -> **+ Add Data** -> tab **Your Datasets** -> thêm dataset mã nguồn,
   dataset dữ liệu, và các gói mở rộng mà notebook đang chạy cần tới.

### Bước 4: Chạy

Nhấn **Run All**. Bốn ô đầu đều dừng ngay khi phát hiện sai lệch, thay vì để lộ
ra sau nhiều giờ chạy:

- **Ô môi trường** hỏi thẳng PyTorch danh sách kiến trúc nó hỗ trợ rồi đối chiếu
  với bộ tăng tốc được cấp, và ném lỗi nếu không tương thích. Hỏi danh sách thay
  vì so với một ngưỡng ghi sẵn, vì ngưỡng ghi sẵn sai ngay khi thư viện đổi bản.
- **Ô nạp mã nguồn** dò đệ quy `/kaggle/input/**/src` và dừng nếu không tìm thấy,
  hoặc nếu tìm thấy nhiều hơn một bản mã nguồn.
- **Ô bố cục kết quả** dựng `/kaggle/working/outputs/<NOTEBOOK_ID>` với bốn thư
  mục con `models`, `metrics`, `tables`, `figures`.
- **Ô nạp dữ liệu** tự tìm tệp Parquet theo mảnh tên và dừng nếu thiếu.

---

## 3. Bốn notebook

| Thứ tự chạy | Notebook | Nội dung | Bộ dữ liệu |
|:---:|---|---|---|
| 1 | `twoDTS/NB1_train_v2.ipynb` | Hai giai đoạn trên bộ đặc trưng v2 | NF-UNSW-NB15-v2 |
| 2 | `twoDTS/NB2_train_v3.ipynb` | Hai giai đoạn trên bộ đặc trưng v3 | NF-UNSW-NB15-v3, NF-CSE-CIC-IDS2018-v3 |
| 3 | `twoDTS/NB3_phan_tich.ipynb` | Đánh giá ngoài phân phối và giải thích mô hình | ba mô hình trên, đối chiếu với NF-ToN-IoT-v3 |
| 4 | `ablation/NB4_ablation.ipynb` | Huấn luyện lại bốn cấu hình để cắt bỏ thành phần | NF-UNSW-NB15-v3 |

NF-CSE-CIC-IDS2018-v2 đã loại khỏi notebook thứ nhất ngày 03/09/2026: bản do
University of Queensland phát hành cho bộ đó không có cột địa chỉ nên không dựng
được đồ thị máy chủ.

Notebook thứ hai huấn luyện hai bộ dữ liệu thành **hai mô hình hoàn toàn độc lập**:
không gộp bảng, không truyền trọng số giữa hai bộ. Thứ duy nhất truyền lại là bộ
mã hoá của giai đoạn một sang giai đoạn hai **trong cùng một bộ dữ liệu**.

Thời gian tham khảo của NB1 đến NB3 lấy từ kết quả đã lưu. NB4 dùng toàn bộ
NF-UNSW-NB15-v3 và dự kiến cần khoảng 4,7 giờ huấn luyện cho bốn cấu hình.
Notebook này chỉ cần dataset mã nguồn và dataset dữ liệu chính.

Hai notebook huấn luyện dùng chung một bộ module, nên notebook nào chạy thông ô
thiết lập thì notebook kia gần như chắc chắn thông.
"""
    path = os.path.join(PROJECT_ROOT, "notebooks", "HUONG_DAN_KAGGLE.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def kiem_doc_du(full_path: str, relative_path: str) -> int:
    """Đọc hết một tệp và dừng hẳn nếu số byte đọc được ít hơn dung lượng khai báo.

    Dự án nằm trên Google Drive dạng phát trực tuyến. Một tệp chưa được tải đủ
    vẫn khai báo dung lượng thật trong mục lục thư mục, `os.path.exists` vẫn trả
    về True, và `os.path.getsize` vẫn trả về con số đầy đủ; nhưng đọc nội dung
    thì hết sớm mà **không** báo lỗi. `zipfile.write` đọc tới EOF nên nó nén đúng
    phần đã tải, còn bảng tổng kết lại in dung lượng khai báo. Hệ quả là gói tải
    lên Kaggle chứa dữ liệu thiếu mà không chỗ nào lộ ra.

    Phép kiểm này phải chạy **trước** khi ghi vào gói, và phải dừng hẳn chứ không
    cảnh báo rồi đi tiếp: một lần chạy huấn luyện trên dữ liệu thiếu cho ra chỉ
    số trông bình thường nhưng vô nghĩa.
    """
    khai_bao = os.path.getsize(full_path)
    doc_duoc = 0
    with open(full_path, "rb") as handle:
        while True:
            khoi = handle.read(1 << 22)
            if not khoi:
                break
            doc_duoc += len(khoi)
    if doc_duoc != khai_bao:
        raise RuntimeError(
            f"Tệp chưa tải đủ, không được đóng gói: {relative_path}\n"
            f"  dung lượng khai báo: {khai_bao:,} byte\n"
            f"  đọc được thật      : {doc_duoc:,} byte\n"
            f"  Bật chế độ tải sẵn cho tệp này trên Google Drive rồi chạy lại."
        )
    return khai_bao


def add_file(archive: zipfile.ZipFile, relative_path: str) -> int:
    """Ghi một tệp vào gói, trả về dung lượng gốc. Bỏ qua nếu tệp không tồn tại."""
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.exists(full_path):
        print(f"  bỏ qua, không tìm thấy: {relative_path}")
        return 0
    kich_thuoc = kiem_doc_du(full_path, relative_path)
    archive.write(full_path, arcname=relative_path)
    return kich_thuoc


def build_archive(zip_name: str, directories, files) -> None:
    """Dựng một tệp ZIP từ danh sách thư mục và danh sách tệp lẻ."""
    os.makedirs(DIST_DIR, exist_ok=True)
    output = os.path.join(DIST_DIR, zip_name)
    if os.path.exists(output):
        os.remove(output)

    count, raw_size = 0, 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path in files:
            size = add_file(archive, relative_path)
            if size:
                count, raw_size = count + 1, raw_size + size

        for directory in directories:
            for root, _, filenames in os.walk(os.path.join(PROJECT_ROOT, directory)):
                if any(skip in root for skip in SKIP_DIRS):
                    continue
                for filename in sorted(filenames):
                    if filename.endswith(SKIP_FILES) or filename in SKIP_FILES:
                        continue
                    full_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(full_path, PROJECT_ROOT)
                    archive.write(full_path, arcname=relative_path)
                    count, raw_size = count + 1, raw_size + os.path.getsize(full_path)

    packed = os.path.getsize(output)
    print(f"  {zip_name:<32} {count:>3} tệp  "
          f"{raw_size / (1024 * 1024):>7.2f} MB  ->  {packed / (1024 * 1024):.2f} MB")


def main() -> None:
    """Đóng gói các tệp cần thiết để chạy dự án trên Kaggle."""
    parser = argparse.ArgumentParser(
        description="Đóng gói mã nguồn và dữ liệu cho Kaggle Dataset")
    parser.add_argument("--ood", action="store_true",
                        help="chỉ dựng gói dữ liệu ngoài phân phối")
    parser.add_argument("--extended", action="store_true",
                        help="chỉ dựng gói dữ liệu của bản thực nghiệm mở rộng")
    parser.add_argument("--code", action="store_true",
                        help="chỉ dựng gói mã nguồn")
    parser.add_argument("--data", action="store_true",
                        help="chỉ dựng gói dữ liệu chính")
    arguments = parser.parse_args()

    # Không chỉ định gì thì dựng toàn bộ.
    explicit = (arguments.code or arguments.data or arguments.ood
                or arguments.extended)
    build_code = arguments.code or not explicit
    build_data = arguments.data or not explicit
    build_ood = arguments.ood or not explicit
    build_extended = arguments.extended or not explicit

    source_version = read_source_version()
    print(f"Đóng gói cho Kaggle, mã nguồn phiên bản {source_version}:")

    if build_code:
        write_readme(source_version)
        build_archive(CODE_ZIP, CODE_DIRS, CODE_FILES)
    if build_data:
        build_archive(DATA_ZIP, [], DATA_FILES)
    def build_group(packages) -> None:
        """Dựng từng gói trong một nhóm, bỏ qua gói nào thiếu tệp nguồn."""
        for zip_name, files in packages.items():
            missing = [f for f in files
                       if not os.path.exists(os.path.join(PROJECT_ROOT, f))]
            if missing:
                print(f"  {zip_name:<32} bỏ qua, thiếu: {', '.join(missing)}")
                continue
            build_archive(zip_name, [], files)

    if build_extended:
        build_group(EXTENDED_PACKAGES)
    if build_ood:
        build_group(OOD_PACKAGES)

    if build_code and not build_data:
        print("\nChỉ gói mã nguồn được dựng lại. Dataset dữ liệu trên Kaggle "
              "giữ nguyên, không cần tải lại.")


if __name__ == "__main__":
    main()
