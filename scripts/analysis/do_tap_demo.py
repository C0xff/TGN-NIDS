"""Đo ba tệp demo bằng chính động cơ suy luận của bảng điều khiển.

Kết quả ghi vào `models/saved/twoDTS_demo/`; trọng số trong
`models/saved/twoDTS_train_*` chỉ được đọc.

    PYTHONPATH="src:apps/dashboard" python scripts/analysis/do_tap_demo.py

Mỗi tệp đo ở hai trạng thái bộ nhớ:

- `checkpoint`: `engine.reset_memory()` khôi phục bảng nhớ đã lưu trong tệp trọng số.
- `xoa_trang`: `memory_module.reset_state()` đặt bảng nhớ về không.

Checkpoint được lưu sau khi đã đi qua tập kiểm thử, nên số đo ở trạng thái
`checkpoint` lạc quan hơn khi triển khai thật.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "dashboard"))

from inference_engine import NIDSInferenceEngine  # noqa: E402

MAU = PROJECT_ROOT / "data" / "samples"
DICH = PROJECT_ROOT / "models" / "saved" / "twoDTS_demo"

# Mô tả từng tệp, để bảng kết quả tự nói tệp đó trả lời câu hỏi gì.
KICH_BAN = {
    "demo_1_cung_phan_phoi": "cùng phân phối với dữ liệu mô hình đã học",
    "demo_2_bo_du_lieu_la": "bộ dữ liệu lạ, mô hình chưa từng huấn luyện trên đó",
    "demo_3_thieu_cot_dia_chi": "thiếu cột địa chỉ, không dựng được đồ thị máy chủ",
}


def ty_le(tu: int, mau: int):
    """Tỷ lệ dạng số thực, trả về None khi mẫu số bằng không thay vì bịa số 0."""
    if mau <= 0:
        return None
    return float(Decimal(tu) / Decimal(mau))


def ho_so_dinh(data: "pd.DataFrame", dinh_da_biet: set, so_o: int) -> dict:
    """Đếm địa chỉ của một tệp và xem bao nhiêu địa chỉ có ô nhớ riêng.

    Địa chỉ lạ được băm vào ô có sẵn bằng `zlib.crc32(địa_chỉ) % số_ô`, nên có
    thể dùng chung vector nhớ với máy khác.
    """
    if "IPV4_SRC_ADDR" not in data.columns or "IPV4_DST_ADDR" not in data.columns:
        return {"so_dinh": None, "dinh_co_o_rieng": None,
                "ghi_chu_dinh": "tệp không có cột địa chỉ, định danh đỉnh do giao diện tổng hợp"}
    dinh = set(data["IPV4_SRC_ADDR"].astype(str)) | set(data["IPV4_DST_ADDR"].astype(str))
    rieng = len(dinh & dinh_da_biet)
    return {
        "so_dinh": len(dinh),
        "dinh_co_o_rieng": rieng,
        "so_o_bang_nho": so_o,
        "ghi_chu_dinh": ("mọi địa chỉ đều có ô nhớ riêng" if rieng == len(dinh)
                         else "%d địa chỉ phải dùng chung ô với máy khác" % (len(dinh) - rieng)),
    }


def do_mot_tep(engine: NIDSInferenceEngine, duong_dan: Path, trang_thai: str,
               dinh_da_biet: set, so_o: int) -> dict:
    """Chấm điểm một tệp demo ở một trạng thái bộ nhớ, trả về mọi chỉ số đếm được."""
    if trang_thai == "checkpoint":
        engine.reset_memory()
    elif trang_thai == "xoa_trang":
        engine.memory_module.reset_state()
        engine._t_counter = 0.0
    else:
        raise ValueError("Trạng thái bộ nhớ không hợp lệ: " + trang_thai)
    data = pd.read_csv(duong_dan)
    ket_qua = engine.predict(data)

    that = data["Label"].astype(int).to_numpy()
    doan = ket_qua["prediction"].astype(int).to_numpy()
    tp = int(((that == 1) & (doan == 1)).sum())
    fn = int(((that == 1) & (doan == 0)).sum())
    fp = int(((that == 0) & (doan == 1)).sum())
    tn = int(((that == 0) & (doan == 0)).sum())

    ty_le_duong = ty_le(tp + fn, len(that))
    moc = None
    if ty_le_duong is not None and ty_le_duong > 0:
        r = Decimal(repr(ty_le_duong))
        moc = float(2 * r / (1 + r))
    ra = {
        "tep": duong_dan.name,
        "kich_ban": KICH_BAN.get(duong_dan.stem, ""),
        "so_cot": int(data.shape[1]),
        "trang_thai_bo_nho": trang_thai,
        "che_do_suy_luan": str(ket_qua["inference_mode"].iloc[0]),
        "nguon_thu_tu": engine.last_sequence_source,
        "so_cot_thieu": len(engine.last_missing_features),
        "cot_thieu": list(engine.last_missing_features),
        "so_luong": int(len(that)),
        "so_tan_cong_that": tp + fn,
        "ty_le_tan_cong_that": ty_le_duong,
        "moc_phan_loai_tam_thuong": moc,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "do_chinh_xac": ty_le(tp + tn, len(that)),
        "recall_lop_tan_cong": ty_le(tp, tp + fn),
        "ty_le_bao_dong_gia": ty_le(fp, fp + tn),
    }
    ra.update(ho_so_dinh(data, dinh_da_biet, so_o))
    return ra


def main() -> int:
    tep = sorted(MAU.glob("demo_*.csv"))
    if not tep:
        print("Không tìm thấy tệp demo nào trong", MAU)
        return 1

    engine = NIDSInferenceEngine(device="cpu")
    if not engine.is_model_loaded:
        print("Không nạp được mô hình. Kiểm tra models/saved trước khi đo.")
        return 1

    # Chụp ánh xạ địa chỉ trước khi mở tệp nào: sau lần predict đầu tiên, động cơ
    # sẽ thêm cả địa chỉ lạ vào ánh xạ, và phép đếm mất ý nghĩa.
    dinh_da_biet = set(engine.ip_to_id)
    so_o = engine.n_nodes

    hang = []
    for p in tep:
        for trang_thai in ("checkpoint", "xoa_trang"):
            hang.append(do_mot_tep(engine, p, trang_thai, dinh_da_biet, so_o))
            print("đã đo %-30s trạng thái %s" % (p.name, trang_thai))

    (DICH / "metrics").mkdir(parents=True, exist_ok=True)
    (DICH / "tables").mkdir(parents=True, exist_ok=True)
    goi = {
        "mo_ta": "Phép đo chức năng trên ba tệp demo, không phải kết quả đánh giá mô hình.",
        "mo_hinh": engine.version,
        "thiet_bi": "cpu",
        "han_che": ("Checkpoint lưu bảng nhớ sau khi đã chạy qua tập kiểm thử, nên chỉ số ở "
                    "trạng thái checkpoint lạc quan hơn tình huống triển khai thật."),
        "ket_qua": hang,
    }
    goi["so_o_bang_nho"] = so_o
    goi["so_dia_chi_co_o_rieng_tu_checkpoint"] = len(dinh_da_biet)
    (DICH / "metrics" / "do_tap_demo.json").write_text(
        json.dumps(goi, ensure_ascii=False, indent=1), encoding="utf-8")
    cot = [c for c in hang[0] if c != "cot_thieu"]
    pd.DataFrame(hang)[cot].to_csv(DICH / "tables" / "do_tap_demo.csv",
                                   index=False, encoding="utf-8-sig")
    print("\nĐã ghi:", (DICH / "metrics" / "do_tap_demo.json").relative_to(PROJECT_ROOT))
    print("Đã ghi:", (DICH / "tables" / "do_tap_demo.csv").relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
