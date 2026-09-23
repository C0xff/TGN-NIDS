"""Kiểm thử phép lấy mẫu dùng để tạo dữ liệu trình diễn."""

import pandas as pd

from scripts.pipeline.build_demo_samples import TIME_COLUMN, stratified_sample


def test_stratified_sample_has_exact_size_and_preserves_all_classes():
    """Mẫu đầu ra phải đúng kích thước và giữ đủ các lớp."""
    frame = pd.DataFrame({
        "Attack": ["Benign"] * 73 + ["DoS"] * 19 + ["Worms"] * 8,
        TIME_COLUMN: list(range(100)),
    })

    sampled = stratified_sample(frame, 31)

    assert len(sampled) == 31
    assert set(sampled["Attack"]) == {"Benign", "DoS", "Worms"}


def test_stratified_sample_orders_real_timestamps_not_source_indices():
    """Mẫu phải được sắp theo mốc thời gian thay vì chỉ số dòng gốc."""
    frame = pd.DataFrame({
        "Attack": ["Benign", "DoS", "Benign", "DoS", "Benign", "DoS"],
        TIME_COLUMN: [60, 10, 50, 20, 40, 30],
    })

    sampled = stratified_sample(frame, 4)

    assert sampled[TIME_COLUMN].is_monotonic_increasing
