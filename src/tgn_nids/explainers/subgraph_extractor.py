"""Trích đồ thị con `k` bước quanh một địa chỉ IP mục tiêu."""

from typing import Dict, Any
import torch
from torch_geometric.utils import k_hop_subgraph


class SubgraphExtractor:
    """Tạo Ego-Network để hiển thị ngữ cảnh của một cảnh báo."""

    def __init__(self, k_hops: int = 1):
        self.k_hops = k_hops

    def extract_ego_network(
        self,
        target_node_id: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        node_labels: torch.Tensor = None,
        edge_predictions: torch.Tensor = None
    ) -> Dict[str, Any]:
        """Trả về các đỉnh và cạnh nằm trong phạm vi `k` bước."""
        undirected_edge_index = torch.cat(
            [edge_index, edge_index.flip(0)], dim=1)
        subset, _, mapping, _ = k_hop_subgraph(
            node_idx=target_node_id,
            num_hops=self.k_hops,
            edge_index=undirected_edge_index,
            relabel_nodes=True,
            num_nodes=None
        )

        # Dùng cạnh vô hướng để tìm lân cận nhưng giữ hướng của cạnh đầu ra.
        in_subset = torch.zeros(
            int(max(int(edge_index.max()) + 1, int(subset.max()) + 1)),
            dtype=torch.bool)
        in_subset[subset] = True
        edge_mask = in_subset[edge_index[0]] & in_subset[edge_index[1]]

        relabel = torch.full((in_subset.numel(),), -1, dtype=torch.long)
        relabel[subset] = torch.arange(subset.numel(), dtype=torch.long)
        sub_edge_index = relabel[edge_index[:, edge_mask]]

        if isinstance(mapping, torch.Tensor) and mapping.numel() > 0:
            center_node_idx = mapping[0].item()
        else:
            center_node_idx = mapping

        sub_edge_preds = edge_predictions[edge_mask] if edge_predictions is not None else None

        nodes_data = []
        subset_list = subset.tolist()

        for idx, original_id in enumerate(subset_list):
            is_center = bool(idx == center_node_idx)
            if node_labels is not None and original_id < len(node_labels):
                current_label = node_labels[original_id].item()
            else:
                current_label = "Unknown"

            nodes_data.append({
                "id": idx,
                "original_id": original_id,
                "is_center": is_center,
                "label": current_label
            })

        edges_data = []
        src_nodes = sub_edge_index[0].tolist()
        dst_nodes = sub_edge_index[1].tolist()
        num_sub_edges = len(src_nodes)

        for i in range(num_sub_edges):
            if sub_edge_preds is not None and i < len(sub_edge_preds):
                is_anomaly = bool(sub_edge_preds[i].item() == 1)
            else:
                is_anomaly = False

            edges_data.append({
                "source": src_nodes[i],
                "target": dst_nodes[i],
                "is_anomaly": is_anomaly
            })

        return {
            "center_node": center_node_idx,
            "original_center_node": target_node_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "num_nodes": len(nodes_data),
            "num_edges": len(edges_data)
        }
