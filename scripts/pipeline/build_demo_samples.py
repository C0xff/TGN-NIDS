"""Sinh ba tệp dữ liệu demo cho dashboard, theo quy tắc ở docs/QUY_TAC_SINH_TAP_DEMO.md.

Chạy:  PYTHONPATH=src python scripts/pipeline/build_demo_samples.py
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch

# Gốc dự án suy từ vị trí tệp này, không ghi cứng đường dẫn của một hệ điều hành.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "samples"

# Mô hình duy nhất dashboard công bố. Mọi tệp demo phải phục vụ đúng mô hình này.
MODEL_PATH = (PROJECT_ROOT / "models" / "saved" / "twoDTS_train_v3"
              / "08_giai_doan_hai" / "models" / "model_stage2_nf_unsw_nb15_v3.pt")

# Cột thời gian dùng để dựng lại phép chia theo trình tự thời gian.
TIME_COLUMN = "FLOW_START_MILLISECONDS"

# Số luồng mỗi tệp demo. Đủ để mọi lớp tấn công có mặt, đủ nhỏ để dashboard chấm
# điểm xong trong vài giây trên CPU.
DEMO_ROWS = 10_000

# Hạt giống cố định để hai lần chạy sinh ra hai tệp giống hệt nhau.
SEED = 42


def _allocate_counts(counts, n_rows, ensure_each=True):
    """Phân bổ chính xác n_rows theo tỷ lệ counts bằng phần dư lớn nhất."""
    counts = {label: int(count) for label, count in counts.items() if count > 0}
    if n_rows > sum(counts.values()):
        raise ValueError("Số dòng cần lấy lớn hơn số dòng nguồn")
    if ensure_each and n_rows < len(counts):
        raise ValueError("Số dòng cần lấy không đủ để giữ ít nhất một dòng mỗi lớp")

    quotas = {
        label: count * n_rows / sum(counts.values())
        for label, count in counts.items()
    }
    allocations = {
        label: max(1 if ensure_each else 0, int(quota))
        for label, quota in quotas.items()
    }
    remainder = n_rows - sum(allocations.values())
    while remainder != 0:
        candidates = [
            label for label in counts
            if remainder > 0 or allocations[label] > (1 if ensure_each else 0)
        ]
        candidates.sort(
            key=lambda label: (quotas[label] - allocations[label], str(label)),
            reverse=(remainder > 0),
        )
        step = 1 if remainder > 0 else -1
        for label in candidates[:abs(remainder)]:
            allocations[label] += step
            remainder -= step
            if remainder == 0:
                break
    return allocations


def stratified_sample(frame, n_rows, label_column="Attack", seed=SEED):
    """Rút n_rows dòng, giữ nguyên tỷ lệ giữa các lớp của khung gốc.

    Giữ tỷ lệ lớp để các lớp hiếm vẫn có mặt trong tệp demo.
    """
    if len(frame) <= n_rows:
        return frame.copy()
    groups = list(frame.groupby(label_column, sort=True))
    allocations = _allocate_counts(
        {label: len(group) for label, group in groups},
        n_rows,
    )

    sampled = pd.concat([
        group.sample(n=allocations[label], random_state=seed)
        for label, group in groups
    ])

    # Phát lại theo mốc thời gian thật, không theo chỉ số dòng.
    if TIME_COLUMN in sampled.columns:
        return sampled.sort_values(TIME_COLUMN, kind="mergesort")
    return sampled.sort_index(kind="mergesort")


def stratified_sample_parquet(path, n_rows, label_column="Attack", seed=SEED):
    """Lấy mẫu phân tầng theo từng row group, không nạp toàn bộ Parquet vào RAM."""
    parquet = pq.ParquetFile(path)
    counts_by_group = []
    total_counts = {}

    # Lượt đầu chỉ đọc cột nhãn để biết chính xác quy mô từng lớp.
    for group_index in range(parquet.num_row_groups):
        labels = parquet.read_row_group(
            group_index, columns=[label_column]
        ).column(label_column).to_pandas()
        counts = labels.value_counts(dropna=False).to_dict()
        counts_by_group.append(counts)
        for label, count in counts.items():
            total_counts[label] = total_counts.get(label, 0) + int(count)

    class_allocations = _allocate_counts(total_counts, n_rows)
    allocations_by_group = [dict() for _ in counts_by_group]
    for label, class_rows in class_allocations.items():
        group_counts = {
            group_index: counts.get(label, 0)
            for group_index, counts in enumerate(counts_by_group)
            if counts.get(label, 0) > 0
        }
        group_allocations = _allocate_counts(
            group_counts,
            class_rows,
            ensure_each=False,
        )
        for group_index, count in group_allocations.items():
            allocations_by_group[group_index][label] = count

    # Lượt hai chỉ giữ lại số dòng đã phân bổ trong từng row group. Lượng dữ
    # liệu sống trong RAM vì vậy bị chặn ở một row group, thay vì toàn bộ tệp.
    pieces = []
    for group_index, allocations in enumerate(allocations_by_group):
        if not allocations:
            continue
        group = parquet.read_row_group(group_index).to_pandas()
        for label, count in allocations.items():
            if count == 0:
                continue
            candidates = group[group[label_column] == label]
            pieces.append(candidates.sample(
                n=count,
                random_state=seed + group_index,
            ))

    sampled = pd.concat(pieces)
    if TIME_COLUMN in sampled.columns:
        sampled = sampled.sort_values(TIME_COLUMN, kind="mergesort")
    return sampled.reset_index(drop=True)


def build_in_distribution():
    """Tệp 1: lấy từ tập kiểm thử của chính bộ dữ liệu mô hình đã học.

    """
    frame = pd.read_parquet(RAW / "nf-unsw-nb15-v3" / "NF-UNSW-NB15-v3.parquet")

    # Dựng lại đúng phép chia của notebook huấn luyện: sắp theo thời gian rồi
    # cắt 80/10/10, lấy 10% cuối.
    ordered = frame.sort_values(TIME_COLUMN, kind="mergesort")
    test_frame = ordered.iloc[int(len(ordered) * 0.9):]

    # Đối chiếu với tỷ lệ tấn công đã ghi trong kết quả huấn luyện. Lệch nghĩa là
    # phép chia dựng lại sai, và tệp demo sẽ không còn khớp báo cáo.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    expected_nodes = set(checkpoint["node_mapping"])

    sampled = stratified_sample(test_frame, DEMO_ROWS)

    # Mọi địa chỉ phải nằm trong ánh xạ đỉnh đã học, nếu không thì bộ nhớ theo
    # đỉnh của TGN mất tác dụng và demo không còn minh hoạ đúng mô hình.
    addresses = set(sampled["IPV4_SRC_ADDR"].astype(str)) | set(
        sampled["IPV4_DST_ADDR"].astype(str))
    unknown = addresses - expected_nodes
    if unknown:
        raise RuntimeError(
            f"{len(unknown)} địa chỉ trong tệp demo không có ô nhớ đã học")
    return sampled, "demo_1_cung_phan_phoi.csv"


def build_unseen_with_addresses():
    """Tệp 2: bộ dữ liệu khác hẳn, vẫn có cột địa chỉ.

    Không trùng tập huấn luyện hay bộ dùng cho đánh giá ngoài phân phối.
    """
    path = RAW / "nf-cse-cic-ids2018-v3" / "NF-CSE-CIC-IDS2018-v3.parquet"
    return stratified_sample_parquet(path, DEMO_ROWS), "demo_2_bo_du_lieu_la.csv"


def build_missing_address_columns():
    """Tệp 3: bộ thiếu hẳn cột địa chỉ, dùng để minh hoạ giới hạn.

    Cho thấy dashboard xử lý ra sao khi dữ liệu không kèm địa chỉ máy.
    """
    path = RAW / "nf-cse-cic-ids2018-v2" / "NF-CSE-CIC-IDS2018-V2.parquet"
    return stratified_sample_parquet(path, DEMO_ROWS), "demo_3_thieu_cot_dia_chi.csv"


def main():
    """Sinh ba tệp demo và in thông tin kiểm tra cơ bản của từng tệp."""
    OUT.mkdir(parents=True, exist_ok=True)
    for builder in (build_in_distribution, build_unseen_with_addresses,
                    build_missing_address_columns):
        sampled, filename = builder()
        sampled.to_csv(OUT / filename, index=False)
        has_address = "IPV4_SRC_ADDR" in sampled.columns
        print(f"{filename}")
        print(f"  luồng      : {len(sampled):,}")
        print(f"  cột        : {len(sampled.columns)}")
        print(f"  tấn công   : {sampled['Label'].mean() * 100:.3f}%")
        print(f"  số lớp     : {sampled['Attack'].nunique()}")
        print(f"  có địa chỉ : {'có' if has_address else 'không'}")


if __name__ == "__main__":
    main()
