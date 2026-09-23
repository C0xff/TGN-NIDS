"""Khối nhúng không gian: tổng hợp thông tin lân cận bằng cơ chế tự chú ý (TransformerConv)."""

import torch
import torch.nn as nn

from torch_geometric.nn import TransformerConv


class GraphAttentionEmbedding(nn.Module):
    """Một tầng chú ý đồ thị tổng hợp vector lân cận vào đỉnh đích."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=num_heads,
            edge_dim=edge_dim,
            concat=False,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention_weights: bool = False
    ):
        """Lan truyền thông tin từ các đỉnh lân cận về từng đỉnh."""
        if return_attention_weights:
            return self.conv(x, edge_index, edge_attr,
                             return_attention_weights=True)
        return self.conv(x, edge_index, edge_attr)


class TemporalEmbeddingModule(nn.Module):
    """Mô-đun nhúng không gian đa tầng kết hợp chuẩn hóa LayerNorm và kết nối tắt."""

    def __init__(
        self,
        memory_dim: int,
        embedding_dim: int,
        edge_dim: int,
        time_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = memory_dim if i == 0 else embedding_dim
            self.gat_layers.append(
                GraphAttentionEmbedding(in_dim, embedding_dim, edge_dim,
                                        num_heads, dropout)
            )
            self.norms.append(nn.LayerNorm(embedding_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
        return_attention_weights: bool = False
    ):
        """Biến vector nhớ thành vector nhúng có ngữ cảnh đồ thị."""
        # Không có cạnh thì không có ngữ cảnh mới để tổng hợp.
        if edge_index is None or edge_index.numel() == 0:
            if return_attention_weights:
                return x, None
            return x

        h = x
        attention_weights_list = []

        for gat, norm in zip(self.gat_layers, self.norms):
            if return_attention_weights:
                h_attn, (edge_index_attn, alpha) = gat(
                    h, edge_index, edge_attr, return_attention_weights=True)
                attention_weights_list.append(alpha)
            else:
                h_attn = gat(h, edge_index, edge_attr)

            # Chỉ dùng kết nối tắt khi hai tensor có cùng số chiều.
            h = norm(h + h_attn) if h.shape == h_attn.shape else norm(h_attn)

        if return_attention_weights:
            return h, attention_weights_list
        return h
