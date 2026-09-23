"""Tái thiết đặc trưng cạnh bị che cho bài toán học tự giám sát."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class EdgeFeatureDecoder(nn.Module):
    """Tái thiết phần bị che từ hai đỉnh và các thuộc tính còn quan sát được."""

    def __init__(self, embedding_dim: int, edge_dim: int, hidden_dim: int = 128,
                 mask_ratio: float = 0.15, eval_mask_samples: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.edge_dim = edge_dim
        self.mask_ratio = mask_ratio
        self.eval_mask_samples = eval_mask_samples

        # Mỗi cạnh luôn có ít nhất một thuộc tính bị che.
        self.num_masked = max(1, int(round(mask_ratio * edge_dim)))

        input_dim = 2 * embedding_dim + 2 * edge_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

    def build_mask(self, edge_features: torch.Tensor,
                   seed: Optional[int] = None) -> torch.Tensor:
        """Tạo mặt nạ ngẫu nhiên với số vị trí che cố định trên mỗi cạnh."""
        n = edge_features.shape[0]
        device = edge_features.device

        if seed is None:
            noise = torch.rand(n, self.edge_dim, device=device)
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            noise = torch.rand(n, self.edge_dim, generator=generator).to(device)

        chosen = noise.topk(self.num_masked, dim=1, largest=False).indices
        mask = torch.zeros(n, self.edge_dim, dtype=torch.bool, device=device)
        return mask.scatter(1, chosen, True)

    @staticmethod
    def visible_input(edge_features: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """Đặt các thuộc tính bị che về 0 trước khi đưa vào bộ giải mã."""
        return edge_features.masked_fill(mask, 0.0)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                edge_features: torch.Tensor, mask: Optional[torch.Tensor] = None,
                seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tái thiết toàn bộ vector cạnh và trả kèm mặt nạ đã dùng."""
        if mask is None:
            mask = self.build_mask(edge_features, seed)

        visible = self.visible_input(edge_features, mask)
        combined = torch.cat([z_src, z_dst, visible, mask.to(visible.dtype)], dim=-1)
        return self.decoder(combined), mask

    def reconstruction_loss(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                            edge_features: torch.Tensor,
                            seed: Optional[int] = None) -> torch.Tensor:
        """Tính MSE chỉ trên các thuộc tính bị che."""
        reconstructed, mask = self.forward(z_src, z_dst, edge_features, seed=seed)
        squared_error = (reconstructed - edge_features) ** 2
        return (squared_error * mask).sum() / mask.sum().clamp(min=1)

    def anomaly_score(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                      edge_features: torch.Tensor,
                      seed: Optional[int] = None) -> torch.Tensor:
        """Tính sai số tái thiết trung bình của từng cạnh."""
        samples = 1 if self.training else max(1, self.eval_mask_samples)
        total = torch.zeros(edge_features.shape[0], device=edge_features.device)

        for step in range(samples):
            step_seed = None if seed is None else int(seed) + step
            reconstructed, mask = self.forward(z_src, z_dst, edge_features,
                                               seed=step_seed)
            squared_error = (reconstructed - edge_features) ** 2
            total = total + ((squared_error * mask).sum(dim=1)
                             / mask.sum(dim=1).clamp(min=1))

        return total / samples
