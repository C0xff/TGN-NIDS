"""Kiểm thử nhãn đa lớp và chống rò rỉ nhãn cho TemporalGraphBuilder."""

import numpy as np
import pandas as pd

from tgn_nids.data.graph_builder import TemporalGraphBuilder


def _make_df():
    """Tạo sáu luồng đã mã hóa theo thứ tự Benign, DoS, Exploits."""
    return pd.DataFrame({
        'ipv4_src_addr': ['10.0.0.1', '10.0.0.2', '10.0.0.1', '10.0.0.3', '10.0.0.2', '10.0.0.1'],
        'ipv4_dst_addr': ['10.0.0.9', '10.0.0.9', '10.0.0.8', '10.0.0.9', '10.0.0.8', '10.0.0.9'],
        'timestamp':     [100, 200, 300, 400, 500, 600],
        'IN_BYTES':      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'OUT_BYTES':     [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        'Label':         [0, 1, 0, 1, 1, 0],
        'Attack':        ['Benign', 'Exploits', 'Benign', 'DoS', 'Exploits', 'Benign'],
        'Attack_encoded': [0, 2, 0, 1, 2, 0],
    })


def test_nhan_da_lop_duoc_tao_dung():
    """y_multi phải khớp chính xác cột Attack_encoded."""
    data = TemporalGraphBuilder().build_pyg_temporal_data(_make_df())

    assert data.y_multi is not None, "Thiếu nhãn đa lớp y_multi"
    assert data.y_multi.tolist() == [0, 2, 0, 1, 2, 0]
    assert data.y_multi.dtype.is_floating_point is False, "y_multi phải là kiểu nguyên"


def test_nhan_nhi_phan_van_hoat_dong_nhu_cu():
    """Nhãn nhị phân phải được giữ nguyên khi có thêm nhãn đa lớp."""
    data = TemporalGraphBuilder().build_pyg_temporal_data(_make_df())
    assert data.y.tolist() == [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]


def test_khong_ro_ri_nhan_vao_dac_trung_canh():
    """Vector cạnh không được chứa nhãn nhị phân hoặc đa lớp."""
    data = TemporalGraphBuilder().build_pyg_temporal_data(_make_df())
    msg = data.msg.numpy()

    # Chỉ IN_BYTES và OUT_BYTES là đặc trưng hợp lệ trong bảng giả lập.
    assert msg.shape == (6, 2), f"Kỳ vọng 2 đặc trưng, nhận được {msg.shape[1]}"

    for col_idx in range(msg.shape[1]):
        col = msg[:, col_idx]
        assert not np.array_equal(col, data.y.numpy()), \
            f"RÒ RỈ: cột {col_idx} trùng khớp nhãn nhị phân"
        assert not np.array_equal(col, data.y_multi.numpy().astype(np.float32)), \
            f"RÒ RỈ: cột {col_idx} trùng khớp nhãn đa lớp"


def test_loai_tru_moi_bien_the_ten_cot_nhan():
    """Các biến thể tên cột nhãn cũng phải bị loại khỏi đặc trưng cạnh."""
    df = _make_df()
    df['attack_cat'] = [0, 2, 0, 1, 2, 0]
    df['label_encoded'] = [0, 1, 0, 1, 1, 0]
    df['target'] = [0, 1, 0, 1, 1, 0]

    data = TemporalGraphBuilder().build_pyg_temporal_data(df)
    assert data.msg.shape == (6, 2), \
        f"Biến thể tên cột nhãn đã lọt vào đặc trưng: {data.msg.shape[1]} cột thay vì 2"


def test_khong_co_attack_encoded_thi_khong_co_y_multi():
    """Dữ liệu không có `Attack_encoded` vẫn phải dựng được đồ thị."""
    df = _make_df().drop(columns=['Attack_encoded'])
    data = TemporalGraphBuilder().build_pyg_temporal_data(df)

    assert getattr(data, 'y_multi', None) is None
    assert data.y.tolist() == [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert data.msg.shape == (6, 2)
