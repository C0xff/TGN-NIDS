"""Giải thích dự đoán bằng ablation đặc trưng và mặt nạ đồ thị con."""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class TGNExplainer:
    """Xếp hạng đặc trưng theo mức thay đổi xác suất khi triệt tiêu từng cột."""

    def __init__(self, memory_module: nn.Module, embedding_module: nn.Module,
                 detector_module: nn.Module, device: str = 'cpu',
                 benign_class_id: Optional[int] = None):
        self.memory = memory_module
        self.embedding = embedding_module
        self.detector = detector_module
        self.device = torch.device(device)
        self.benign_class_id = benign_class_id

        self.memory.eval()
        self.embedding.eval()
        self.detector.eval()

    def _attack_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """Quy đổi logits thành xác suất thuộc nhóm tấn công."""
        probabilities = torch.softmax(logits, dim=1)
        if self.benign_class_id is not None:
            return 1.0 - probabilities[:, self.benign_class_id]
        return probabilities[:, 1]

    def compute_feature_importance(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Tính độ quan trọng bằng cách lần lượt đặt từng đặc trưng về 0."""
        with torch.no_grad():
            base_probs = self._attack_probability(
                self.detector(z_src, z_dst, edge_attr))

            importance = {}
            num_features = edge_attr.shape[1]

            if len(feature_names) != num_features:
                feature_names = [f"Feature_{i}" for i in range(num_features)]

            for i in range(num_features):
                perturbed_edge_attr = edge_attr.clone()
                perturbed_edge_attr[:, i] = 0.0

                perturbed_probs = self._attack_probability(
                    self.detector(z_src, z_dst, perturbed_edge_attr))

                importance[feature_names[i]] = torch.abs(
                    base_probs - perturbed_probs).mean().item()

        total_diff = sum(importance.values())
        if total_diff > 0:
            for k in importance:
                importance[k] = (importance[k] / total_diff) * 100.0

        return dict(sorted(importance.items(), key=lambda item: item[1],
                           reverse=True))


class GNNExplainer:
    """Học mặt nạ cạnh và đặc trưng trên một đồ thị con cục bộ."""

    def __init__(self, memory_module: nn.Module, embedding_module: nn.Module,
                 detector_module: nn.Module, device: str = "cpu",
                 benign_class_id: Optional[int] = None,
                 epochs: int = 60, lr: float = 0.02,
                 lambda_edge_sparse: float = 0.005,
                 lambda_edge_entropy: float = 0.10,
                 lambda_feat_sparse: float = 0.01):
        self.memory = memory_module
        self.embedding = embedding_module
        self.detector = detector_module
        self.device = torch.device(device)
        self.benign_class_id = benign_class_id
        self.epochs = epochs
        self.lr = lr
        self.lambda_edge_sparse = lambda_edge_sparse
        self.lambda_edge_entropy = lambda_edge_entropy
        self.lambda_feat_sparse = lambda_feat_sparse

        # Chỉ tối ưu mặt nạ, không cập nhật mô hình đã huấn luyện.
        for module in (self.memory, self.embedding, self.detector):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _entropy(mask: torch.Tensor) -> torch.Tensor:
        """Tính entropy để khuyến khích mặt nạ tiến gần 0 hoặc 1."""
        eps = 1e-6
        m = torch.clamp(mask, eps, 1.0 - eps)
        return -torch.mean(m * torch.log(m) + (1.0 - m) * torch.log(1.0 - m))

    def _attack_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """Quy đổi logits thành xác suất thuộc nhóm tấn công."""
        probabilities = torch.softmax(logits, dim=1)
        if self.benign_class_id is not None:
            return 1.0 - probabilities[:, self.benign_class_id]
        return probabilities[:, 1]

    def explain_edge(self, target_index: int, node_memory: torch.Tensor,
                     edge_index: torch.Tensor, edge_attr: torch.Tensor,
                     feature_names: List[str],
                     seed: Optional[int] = None) -> Dict[str, Any]:
        """Tối ưu mặt nạ giải thích cho một cạnh mục tiêu."""
        num_edges = int(edge_index.shape[1])
        num_features = int(edge_attr.shape[1])
        if num_edges == 0:
            raise ValueError("Đồ thị con không có cạnh nào để giải thích")

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        def khoi_tao(n):
            noise = torch.randn(n, generator=generator) * 0.1
            return nn.Parameter(noise.to(self.device))

        edge_mask_param = khoi_tao(num_edges)
        feat_mask_param = khoi_tao(num_features)
        optimizer = torch.optim.Adam([edge_mask_param, feat_mask_param], lr=self.lr)

        target = torch.tensor([target_index], dtype=torch.long, device=self.device)
        src = edge_index[0, target]
        dst = edge_index[1, target]

        # Xác suất gốc là mục tiêu cần bảo toàn khi rút gọn lời giải thích.
        with torch.no_grad():
            z_goc = self.embedding(node_memory, edge_index, edge_attr)
            p_goc = float(self._attack_probability(
                self.detector(z_goc[src], z_goc[dst], edge_attr[target])).item())

        for _ in range(self.epochs):
            optimizer.zero_grad()

            edge_mask = torch.sigmoid(edge_mask_param)
            feat_mask = torch.sigmoid(feat_mask_param)

            edge_attr_masked = edge_attr * edge_mask.unsqueeze(1)
            z = self.embedding(node_memory, edge_index, edge_attr_masked)

            attr_target = edge_attr[target] * feat_mask.unsqueeze(0)
            p_attack = self._attack_probability(
                self.detector(z[src], z[dst], attr_target))

            loss = (p_attack - p_goc).pow(2).mean()
            loss = loss + self.lambda_edge_sparse * edge_mask.mean()
            loss = loss + self.lambda_feat_sparse * feat_mask.mean()
            loss = loss + self.lambda_edge_entropy * self._entropy(edge_mask)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            edge_mask = torch.sigmoid(edge_mask_param)
            feat_mask = torch.sigmoid(feat_mask_param)
            z = self.embedding(node_memory, edge_index,
                               edge_attr * edge_mask.unsqueeze(1))
            p_sau = float(self._attack_probability(
                self.detector(z[src], z[dst],
                              edge_attr[target] * feat_mask.unsqueeze(0))).item())

        edge_np = edge_mask.detach().cpu().numpy()
        feat_np = feat_mask.detach().cpu().numpy()

        if len(feature_names) != num_features:
            feature_names = [f"Feature_{i}" for i in range(num_features)]

        tong = float(feat_np.sum())
        feature_importance = (
            {n: float(v) / tong * 100.0 for n, v in zip(feature_names, feat_np)}
            if tong > 0 else {n: 0.0 for n in feature_names})

        return {
            "edge_mask": edge_np,
            "target_index": int(target_index),
            "feature_importance": dict(sorted(feature_importance.items(),
                                              key=lambda kv: kv[1], reverse=True)),
            "attack_probability_before": p_goc,
            "attack_probability_after": p_sau,
            "epochs": self.epochs,
        }
