#!/usr/bin/env python3
"""Dựng lại tệp Parquet từ CSV gốc, kèm bước đối chiếu từng ô.

Không ép kiểu, để Parquet giữ đúng giá trị của CSV. Sau khi ghi, đối chiếu số
dòng, tên cột, số mẫu tấn công với số công bố, và từng ô của bảng.

Dùng:
    python scripts/pipeline/rebuild_parquet.py                 # dựng lại cả v2 và v3
    python scripts/pipeline/rebuild_parquet.py --only v3
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# Gốc dự án suy ra từ vị trí của chính tệp này, để script chạy được trên cả
# Windows lẫn macOS mà không cần sửa đường dẫn.
GOC = Path(__file__).resolve().parents[2]

# Số dòng và số mẫu tấn công do University of Queensland công bố, dùng để đối chiếu.
BO_DU_LIEU = {
    "v2": {
        "csv": "data/raw/nf-unsw-nb15-v2/NF-UNSW-NB15-v2.csv",
        "parquet": "data/raw/nf-unsw-nb15-v2/NF-UNSW-NB15-v2.parquet",
        "cong_bo": {"tong": 2_390_275, "tan_cong": 95_053},
    },
    "v3": {
        "csv": "data/raw/nf-unsw-nb15-v3/NF-UNSW-NB15-v3.csv",
        "parquet": "data/raw/nf-unsw-nb15-v3/NF-UNSW-NB15-v3.parquet",
        "cong_bo": {"tong": 2_365_424, "tan_cong": 127_693},
    },
}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    """Mã băm của tệp nguồn, đọc theo khối để không nạp cả tệp vào bộ nhớ."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def chuyen_doi(ten: str, cau_hinh: dict) -> bool:
    """Chuyển một bộ dữ liệu sang Parquet rồi đối chiếu, trả về True nếu khớp."""
    csv_path = GOC / cau_hinh["csv"]
    pq_path = GOC / cau_hinh["parquet"]

    print(f"\n{'=' * 70}")
    print(f"  {ten.upper()}")
    print("=" * 70)

    if not csv_path.exists():
        print(f"  Không tìm thấy tệp CSV: {csv_path}")
        return False

    print(f"  Nguồn CSV : {csv_path.name} ({csv_path.stat().st_size:,} byte)")
    print(f"  SHA-256   : {sha256_file(csv_path)}")

    t0 = time.time()
    print("  Đang đọc CSV...", end="", flush=True)
    bang = pacsv.read_csv(csv_path)
    print(f" xong ({time.time() - t0:.1f}s)")

    # Chuẩn hoá tên cột: bỏ khoảng trắng thừa ở hai đầu. Bản đặc trưng thứ ba
    # có vài cột IAT bị dính khoảng trắng trong tệp gốc.
    ten_cot_moi = [c.strip() for c in bang.column_names]
    if ten_cot_moi != bang.column_names:
        so_sua = sum(1 for a, b in zip(ten_cot_moi, bang.column_names, strict=True)
                     if a != b)
        bang = bang.rename_columns(ten_cot_moi)
        print(f"  Đã cắt khoảng trắng thừa ở {so_sua} tên cột")

    print(f"  Kích thước: {bang.num_rows:,} dòng x {bang.num_columns} cột")

    if pq_path.exists():
        pq_path.unlink()
    pq.write_table(bang, pq_path, compression="snappy")
    ti_le_nen = pq_path.stat().st_size / csv_path.stat().st_size * 100
    print(f"  Đã ghi    : {pq_path.name} "
          f"({pq_path.stat().st_size:,} byte, nén còn {ti_le_nen:.1f}%)")

    # --- Đối chiếu ---
    print("\n  Đối chiếu tệp Parquet vừa ghi với CSV gốc:")
    lai = pq.read_table(pq_path)
    kt = []

    kt.append(("số dòng", lai.num_rows == bang.num_rows, f"{lai.num_rows:,}"))
    kt.append(("tên cột", lai.column_names == bang.column_names,
               f"{lai.num_columns} cột"))

    cot_nhan = next((c for c in lai.column_names if c.lower() == "label"), None)
    if cot_nhan:
        n_tan_cong = pc.sum(pc.equal(lai[cot_nhan], 1)).as_py()
        cong_bo = cau_hinh["cong_bo"]
        kt.append(("tổng số dòng khớp công bố", lai.num_rows == cong_bo["tong"],
                   f"{lai.num_rows:,} / công bố {cong_bo['tong']:,}"))
        kt.append(("số mẫu tấn công khớp công bố",
                   n_tan_cong == cong_bo["tan_cong"],
                   f"{n_tan_cong:,} / công bố {cong_bo['tan_cong']:,}"))

    # So từng ô; cộng dồn số thực theo thứ tự khác nhau sẽ lệch do làm tròn.
    lech = []
    for c in bang.column_names:
        cot_csv, cot_pq = bang[c].combine_chunks(), lai[c].combine_chunks()
        if cot_csv.type != cot_pq.type:
            lech.append((c, str(cot_csv.type), str(cot_pq.type)))
            continue
        so_o_khac = pc.sum(
            pc.cast(pc.invert(pc.equal(cot_csv, cot_pq)), "int64")).as_py()
        null_csv = pc.sum(pc.cast(pc.is_null(cot_csv), "int64")).as_py()
        null_pq = pc.sum(pc.cast(pc.is_null(cot_pq), "int64")).as_py()
        if so_o_khac or null_csv != null_pq:
            lech.append((c, f"{so_o_khac} ô khác",
                         f"rỗng {null_csv} so với {null_pq}"))
    tong_o = bang.num_rows * bang.num_columns
    kt.append((f"đối chiếu từng ô ({tong_o:,} ô)", not lech,
               "trùng khít tuyệt đối" if not lech else f"{len(lech)} cột lệch"))

    tat_ca_dat = True
    for ten_kt, dat, chi_tiet in kt:
        print(f"    {'ĐẠT ' if dat else 'HỎNG'} {ten_kt:34s} {chi_tiet}")
        tat_ca_dat &= dat
    for c, a, b in lech[:5]:
        print(f"         cột {c}: csv={a} parquet={b}")

    print(f"\n  Kết quả: "
          f"{'TRUNG THỰC' if tat_ca_dat else 'CÓ SAI LỆCH, KHÔNG DÙNG'}")
    return tat_ca_dat


def main() -> int:
    """Dựng lại tệp Parquet theo bộ dữ liệu được chọn."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=list(BO_DU_LIEU),
                    help="Chỉ dựng lại một bộ dữ liệu")
    doi_so = ap.parse_args()

    muc = ({doi_so.only: BO_DU_LIEU[doi_so.only]} if doi_so.only
           else BO_DU_LIEU)
    ket = {ten: chuyen_doi(ten, ch) for ten, ch in muc.items()}

    print(f"\n{'=' * 70}")
    for ten, ok in ket.items():
        print(f"  {ten}: {'TRUNG THỰC' if ok else 'CÓ VẤN ĐỀ'}")
    return 0 if all(ket.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
