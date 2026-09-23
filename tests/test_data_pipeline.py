"""Kiểm thử đường đi từ tệp CSV đến đồ thị thời gian."""

import numpy as np

import os

from tgn_nids.simulation.generator import LogGenerator
from tgn_nids.data.preprocess import NetFlowPreprocessor
from tgn_nids.data.graph_builder import TemporalGraphBuilder


def test_full_data_pipeline():
    """Toàn bộ đường ống phải giữ đúng dữ liệu và cấu trúc đầu ra."""
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "samples", "demo_1_cung_phan_phoi.csv")
    generator = LogGenerator(file_path=csv_path)
    # Bộ dựng đồ thị giữ nguyên thứ tự đầu vào nên mẫu phải được sắp trước.
    df = generator.get_all_logs()
    time_col = next((c for c in df.columns
                     if c.strip().upper() == "FLOW_START_MILLISECONDS"), None)
    if time_col is not None:
        df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    df = df.head(3000)
    
    assert len(df) > 0, f"Không đọc được dữ liệu mẫu từ {os.path.basename(csv_path)}"
    
    # Tên cột được đối chiếu không phân biệt hoa thường trong toàn bộ pipeline.
    col_by_lower = {str(c).strip().lower(): c for c in df.columns}
    required_cols = ['ipv4_src_addr', 'ipv4_dst_addr', 'label', 'attack']
    for col in required_cols:
        assert col in col_by_lower, f"Thiếu cột bắt buộc: {col}"

    src_col = col_by_lower['ipv4_src_addr']
    dst_col = col_by_lower['ipv4_dst_addr']

    preprocessor = NetFlowPreprocessor()
    processed_df = preprocessor.fit_transform(df)
    
    assert 'Attack_encoded' in processed_df.columns
    assert processed_df['Attack_encoded'].isnull().sum() == 0, "Lỗi mã hóa nhãn Attack"
    
    for col in preprocessor.feature_names:
        if col in processed_df.columns:
            mean_val = processed_df[col].mean()
            assert np.abs(mean_val) < 1.0, f"Cột {col} chưa được chuẩn hóa thích hợp"

    builder = TemporalGraphBuilder()
    graph_data = builder.build_pyg_temporal_data(processed_df)
    
    unique_ips = set(df[src_col].unique()).union(set(df[dst_col].unique()))
    
    assert builder.get_node_count() == len(unique_ips), "Số lượng Node không trùng khớp với số IP duy nhất"
    
    assert graph_data.src.shape[0] == len(df), "Số lượng cạnh nguồn sai lệch"
    assert graph_data.dst.shape[0] == len(df), "Số lượng cạnh đích sai lệch"
    assert graph_data.t.shape[0] == len(df), "Số lượng mốc thời gian không khớp"
    assert graph_data.msg.shape[0] == len(df), "Số lượng thuộc tính cạnh không khớp"
    assert graph_data.y.shape[0] == len(df), "Số lượng nhãn cạnh không khớp"
    
    timestamps = graph_data.t.numpy()
    assert np.all(np.diff(timestamps) >= 0), "Mốc thời gian đồ thị động phải tăng dần tuần tự"

    first_src_ip = df[src_col].iloc[0]
    first_src_id = graph_data.src[0].item()
    assert builder.get_ip_from_id(first_src_id) == first_src_ip, "Tra cứu ngược địa chỉ IP bị lỗi"

    print("Hoàn tất kiểm thử đường ống dữ liệu thành công!")
