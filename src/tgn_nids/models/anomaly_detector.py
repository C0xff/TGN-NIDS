"""Khối phân loại cạnh: dự đoán nhãn luồng từ vector nhúng hai đỉnh và đặc trưng cạnh."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class EdgeClassifier(nn.Module):
    """Mạng nhiều lớp phân loại luồng vào các lớp hành vi."""

    def __init__(self, in_channels: int, num_classes: int = 10,
                 hidden_channels: int = 128, dropout: float = 0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(hidden_channels)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                edge_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Tính logits cho từng luồng trong lô."""
        if edge_feat is not None:
            x = torch.cat([z_src, z_dst, edge_feat], dim=-1)
        else:
            x = torch.cat([z_src, z_dst], dim=-1)

        # BatchNorm cần ít nhất hai mẫu khi huấn luyện.
        x = self.lin1(x)
        if x.size(0) > 1:
            x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = F.relu(self.lin2(x))
        x = self.dropout(x)

        return self.lin3(x)


class AnomalyDetector(nn.Module):
    """Lớp bọc thống nhất cấu trúc phân loại nhị phân và đa lớp."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128,
                 num_classes: int = 2, dropout: float = 0.3, edge_dim: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.edge_dim = edge_dim
        in_dim = embedding_dim * 2 + edge_dim
        self.detector = EdgeClassifier(in_dim, num_classes, hidden_dim, dropout)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                edge_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Tính logits cho từng cạnh trong lô."""
        return self.detector(z_src, z_dst, edge_feat)
