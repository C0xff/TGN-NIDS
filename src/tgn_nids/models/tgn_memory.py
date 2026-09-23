"""Bộ nhớ trạng thái theo đỉnh của mạng đồ thị thời gian (TGN Memory)."""

import torch
import torch.nn as nn

from torch_geometric.nn import TGNMemory as PyGTGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
)


class TGNMemoryModule(nn.Module):
    """Quản lý vector nhớ của toàn bộ đỉnh trong đồ thị máy chủ."""

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int = 64,
        time_dim: int = 64,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.raw_msg_dim = raw_msg_dim
        self.time_dim = time_dim

        msg_module = IdentityMessage(raw_msg_dim, memory_dim, time_dim)
        aggregator = LastAggregator()

        self.memory = PyGTGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=msg_module,
            aggregator_module=aggregator,
        )

    def forward(self, n_id: torch.Tensor) -> tuple:
        """Đọc vector nhớ và mốc thời gian cập nhật gần nhất của nhóm đỉnh."""
        return self.memory(n_id)

    def update_state(self, src: torch.Tensor, dst: torch.Tensor,
                     t: torch.Tensor, msg: torch.Tensor):
        """Cập nhật trạng thái bộ nhớ sau khi hoàn thành dự đoán."""
        self.memory.update_state(src, dst, t, msg)

    def detach_memory(self):
        """Tách bộ nhớ khỏi đồ thị tính toán để giải phóng bộ nhớ đồ họa."""
        self.memory.detach()

    def reset_state(self):
        """Khởi tạo lại trạng thái bộ nhớ về 0 trước mỗi epoch/đánh giá."""
        self.memory.reset_state()
