"""Chuyển dữ liệu NetFlow thành đồ thị thời gian dạng `TemporalData`."""

import pandas as pd
import numpy as np
import torch

from .labels import to_binary_label

try:
    from torch_geometric.data import TemporalData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    TemporalData = None


class TemporalGraphBuilder:
    """Dựng đồ thị có đỉnh là địa chỉ IP và cạnh là luồng mạng."""

    def __init__(self, node_mapping=None):
        self.node_mapping = node_mapping if node_mapping is not None else {}
        self.next_node_id = len(self.node_mapping)
        self.id_to_ip = {v: k for k, v in self.node_mapping.items()}

    def _get_or_create_node_id(self, ip_address: str) -> int:
        """Trả về mã đỉnh ổn định của một địa chỉ IP."""
        if ip_address not in self.node_mapping:
            self.node_mapping[ip_address] = self.next_node_id
            self.id_to_ip[self.next_node_id] = ip_address
            self.next_node_id += 1
        return self.node_mapping[ip_address]

    def build_pyg_temporal_data(self, df: pd.DataFrame):
        """Tạo `TemporalData` từ một bảng NetFlow đã tiền xử lý."""
        exclude_lower = {
            'ipv4_src_addr', 'ipv4_dst_addr', 'timestamp', 'timestamp_end',
            'flow_start_milliseconds', 'flow_end_milliseconds',
            'label', 'attack', 'attack_encoded', 'attack_cat', 'label_encoded',
            'class', 'target', 'y',
            'srcip', 'dstip', 'src_ip', 'dst_ip', 'source_ip', 'destination_ip',
            'stime', 'ts', 'start_time'
        }

        # Chỉ giữ các cột có thể dùng làm đặc trưng cạnh.
        avail_cols = [c for c in df.columns if str(c).strip().lower() not in exclude_lower]

        # Dừng sớm nếu bước tiền xử lý còn để lọt giá trị không hữu hạn.
        feature_mats = []
        degenerate = []
        for col in avail_cols:
            values = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
            invalid = int((~np.isfinite(values)).sum())
            if invalid:
                degenerate.append((col, invalid, len(values)))
            feature_mats.append(np.nan_to_num(values, nan=0.0,
                                              posinf=0.0, neginf=0.0))
        if degenerate:
            detail = "; ".join(f"{name}: {count:,}/{total:,} dòng"
                               for name, count, total in degenerate)
            raise ValueError(
                f"Ma trận đặc trưng còn giá trị không hữu hạn sau bước tiền xử "
                f"lý ({detail}). Điền 0 cho chúng sẽ biến đặc trưng thành hằng "
                f"số mà không ai biết, nên bước dựng đồ thị dừng tại đây.")

        edge_features = np.column_stack(feature_mats)

        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        src_col = cols_lower.get('ipv4_src_addr', cols_lower.get('srcip'))
        dst_col = cols_lower.get('ipv4_dst_addr', cols_lower.get('dstip'))
        if src_col is None or dst_col is None:
            raise ValueError(
                "Không tìm thấy cột địa chỉ nguồn và đích trong bảng đưa vào "
                "dựng đồ thị. Không thay bằng cột khác: đỉnh của đồ thị phải là "
                "máy chủ có thật, và một định danh đỉnh dựng từ cột bất kỳ sẽ "
                "tham gia vào mọi chỉ số được báo cáo. Bộ dữ liệu không có cột "
                "địa chỉ thì không dùng cho mô hình đồ thị này được.")

        src_ids = np.array([self._get_or_create_node_id(str(ip)) for ip in df[src_col]])
        dst_ids = np.array([self._get_or_create_node_id(str(ip)) for ip in df[dst_col]])

        # Dữ liệu không có mốc thời gian dùng vị trí dòng làm biến trình tự.
        time_col = cols_lower.get('timestamp', cols_lower.get('flow_start_milliseconds', None))
        self.has_real_timestamps = bool(time_col and time_col in df.columns)
        if self.has_real_timestamps:
            timestamps = pd.to_numeric(df[time_col], errors='coerce').fillna(0).values.astype(np.int64)
        else:
            timestamps = np.arange(len(df), dtype=np.int64)

        label_col = cols_lower.get('label', cols_lower.get('attack', None))
        if label_col and label_col in df.columns:
            labels = df[label_col].apply(to_binary_label).values.astype(np.float32)
        else:
            labels = np.zeros(len(df), dtype=np.float32)

        multi_col = cols_lower.get('attack_encoded', None)
        if multi_col and multi_col in df.columns:
            multi_labels = pd.to_numeric(df[multi_col], errors='coerce').fillna(0).values.astype(np.int64)
            y_multi_tensor = torch.tensor(multi_labels, dtype=torch.long)
        else:
            y_multi_tensor = None

        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        dst_tensor = torch.tensor(dst_ids, dtype=torch.long)
        t_tensor = torch.tensor(timestamps, dtype=torch.long)
        msg_tensor = torch.tensor(edge_features, dtype=torch.float)
        y_tensor = torch.tensor(labels, dtype=torch.float)

        if PYG_AVAILABLE:
            data = TemporalData(
                src=src_tensor,
                dst=dst_tensor,
                t=t_tensor,
                msg=msg_tensor,
                y=y_tensor
            )
            if y_multi_tensor is not None:
                data.y_multi = y_multi_tensor
            return data
        else:
            class FallbackTemporalData:
                def __init__(self, src, dst, t, msg, y, y_multi=None):
                    self.src, self.dst, self.t, self.msg, self.y = src, dst, t, msg, y
                    self.y_multi = y_multi
                    self.num_events = len(src)
                def to(self, device):
                    return FallbackTemporalData(
                        self.src.to(device), self.dst.to(device), self.t.to(device),
                        self.msg.to(device), self.y.to(device),
                        self.y_multi.to(device) if self.y_multi is not None else None,
                    )

            return FallbackTemporalData(src_tensor, dst_tensor, t_tensor, msg_tensor,
                                        y_tensor, y_multi_tensor)

    def get_node_count(self) -> int:
        """Trả về số địa chỉ IP đã được ánh xạ thành đỉnh."""
        return self.next_node_id

    def get_ip_from_id(self, node_id: int):
        """Tìm địa chỉ IP tương ứng với một mã đỉnh."""
        return self.id_to_ip.get(int(node_id))
