"""Đánh giá nhị phân trên một bộ NetFlow ngoài phân phối huấn luyện."""

from __future__ import annotations

import gc
import zlib
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from tgn_nids.data.preprocess import NetFlowPreprocessor
from tgn_nids.models.tgn_memory import TGNMemoryModule
from tgn_nids.models.embedding import TemporalEmbeddingModule
from tgn_nids.models.anomaly_detector import AnomalyDetector
from tgn_nids.utils.paper_metrics import evaluate_binary


def load_checkpoint(model_path: str, device: str = "cuda"):
    """Nạp checkpoint và khôi phục ba khối của mô hình ở chế độ an toàn."""
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=device_t, weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f"Không nạp được checkpoint {model_path} ở chế độ an toàn. Nếu tệp "
            f"được sinh ra bởi bản mã cũ, nó chứa đối tượng đã pickle và phải "
            f"chạy lại notebook để sinh bản mới."
        ) from error

    cfg = checkpoint["config"]
    memory = TGNMemoryModule(
        num_nodes=checkpoint["n_nodes"],
        raw_msg_dim=checkpoint["edge_dim"],
        memory_dim=cfg["memory_dim"],
        time_dim=cfg["time_dim"],
    ).to(device_t)
    memory.load_state_dict(checkpoint["memory_state"])

    embedding = TemporalEmbeddingModule(
        memory_dim=cfg["memory_dim"],
        embedding_dim=cfg["embedding_dim"],
        edge_dim=checkpoint["edge_dim"],
        time_dim=cfg["time_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device_t)
    embedding.load_state_dict(checkpoint["embedding_state"])

    use_edge_feats = cfg.get("classifier_edge_features", True)
    detector = AnomalyDetector(
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=checkpoint["n_classes"],
        dropout=0.3,
        edge_dim=checkpoint["edge_dim"] if use_edge_feats else 0,
    ).to(device_t)
    detector.load_state_dict(checkpoint["head_state"])

    memory.eval()
    embedding.eval()
    detector.eval()
    return checkpoint, memory, embedding, detector, device_t


def evaluate_ood_binary(
    model_path: str,
    df_external: pd.DataFrame,
    device: str = "cuda",
    batch_size: int = 4096,
    chunk_rows: int = 200_000,
    return_scores: bool = False,
) -> Dict[str, object]:
    """Chạy suy luận ngoài phân phối theo từng khối để giới hạn bộ nhớ."""
    checkpoint, memory, embedding, detector, device_t = load_checkpoint(model_path, device)
    preprocessor = NetFlowPreprocessor.from_state_dict(checkpoint["preprocessor_state"])
    node_mapping: Dict[str, int] = dict(checkpoint["node_mapping"])
    n_nodes = checkpoint["n_nodes"]
    benign_class_id = checkpoint["benign_class_id"]
    n_classes = checkpoint["n_classes"]
    classifier_edge_features = checkpoint["config"].get("classifier_edge_features", True)

    df_external.columns = df_external.columns.str.strip()
    cols_lower = {str(c).strip().lower(): c for c in df_external.columns}
    src_col = cols_lower.get("ipv4_src_addr", cols_lower.get("srcip"))
    dst_col = cols_lower.get("ipv4_dst_addr", cols_lower.get("dstip"))

    if src_col is None or dst_col is None:
        raise ValueError(
            "Bộ dữ liệu ngoài không có cột địa chỉ IPv4 nguồn và đích nên "
            "không dựng được đồ thị máy chủ. Không sinh địa chỉ thay thế, vì "
            "định danh đỉnh quyết định toàn bộ cấu trúc đồ thị và sẽ tham gia "
            "vào mọi chỉ số được báo cáo. Dùng bộ thuộc phiên bản đặc trưng "
            "thứ ba, hoặc loại bộ dữ liệu này khỏi phần đánh giá ngoài phân "
            "phối.")
    label_col = cols_lower.get("label")
    if label_col is None:
        raise ValueError("Dữ liệu ngoài thiếu cột Label để đối chiếu.")

    missing_before: List[str] = [
        c for c in preprocessor.feature_names if c not in df_external.columns
    ]

    n = len(df_external)
    all_probs_attack = np.zeros(n, dtype=np.float32)
    all_preds = np.zeros(n, dtype=np.int64)
    hashed_new = 0
    seen_ips: set = set(node_mapping.keys())

    with torch.no_grad():
        for chunk_start in range(0, n, chunk_rows):
            chunk_end = min(chunk_start + chunk_rows, n)
            chunk = df_external.iloc[chunk_start:chunk_end].copy()

            chunk_src_col, chunk_dst_col = src_col, dst_col

            processed = preprocessor.transform(chunk)
            features_chunk = processed[preprocessor.feature_names].values.astype(np.float32)
            # Chặn giá trị ngoài miền biểu diễn an toàn của float32.
            features_chunk = np.nan_to_num(features_chunk, nan=0.0, posinf=1e4, neginf=-1e4)
            del processed

            src_ips = chunk[chunk_src_col].astype(str)
            dst_ips = chunk[chunk_dst_col].astype(str)
            for ip in pd.unique(pd.concat([src_ips, dst_ips], ignore_index=True)):
                if ip not in seen_ips:
                    # Băm địa chỉ chưa gặp để ánh xạ lặp lại được giữa các lần chạy.
                    node_mapping[ip] = zlib.crc32(ip.encode("utf-8")) % n_nodes
                    seen_ips.add(ip)
                    hashed_new += 1

            src_ids_chunk = src_ips.map(node_mapping).astype(np.int64).values
            dst_ids_chunk = dst_ips.map(node_mapping).astype(np.int64).values

            for start in range(0, len(chunk), batch_size):
                end = min(start + batch_size, len(chunk))
                src_t = torch.tensor(src_ids_chunk[start:end], dtype=torch.long, device=device_t)
                dst_t = torch.tensor(dst_ids_chunk[start:end], dtype=torch.long, device=device_t)
                edge_attr = torch.tensor(features_chunk[start:end], dtype=torch.float32, device=device_t)

                n_id = torch.unique(torch.cat([src_t, dst_t]))
                assoc = torch.full((int(n_id.max().item()) + 1,), -1, dtype=torch.long, device=device_t)
                assoc[n_id] = torch.arange(n_id.size(0), device=device_t)

                mem, _last_update = memory(n_id)
                local_edge_index = torch.stack([assoc[src_t], assoc[dst_t]], dim=0)
                z = embedding(mem, local_edge_index, edge_attr)
                z_src, z_dst = z[assoc[src_t]], z[assoc[dst_t]]

                logits = detector(
                    z_src, z_dst,
                    edge_feat=edge_attr if classifier_edge_features else None,
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

                out_start, out_end = chunk_start + start, chunk_start + end
                if n_classes > 2:
                    pred_idx = probs.argmax(axis=1)
                    all_preds[out_start:out_end] = (pred_idx != benign_class_id).astype(int)
                    all_probs_attack[out_start:out_end] = 1.0 - probs[:, benign_class_id]
                else:
                    all_preds[out_start:out_end] = probs.argmax(axis=1)
                    all_probs_attack[out_start:out_end] = probs[:, 1]

                # Cập nhật sau dự đoán để không làm lộ thông tin của cạnh hiện tại.
                t = torch.arange(out_start, out_end, dtype=torch.long, device=device_t)
                memory.update_state(src_t, dst_t, t, edge_attr)
                memory.detach_memory()

            del chunk, features_chunk, src_ids_chunk, dst_ids_chunk, src_ips, dst_ips
            gc.collect()

    all_probs_attack = np.nan_to_num(all_probs_attack, nan=0.0, posinf=1.0, neginf=0.0)
    labels = df_external[cols_lower["label"]].astype(int).values
    metrics = evaluate_binary(labels, all_preds, all_probs_attack)
    metrics["missing_features"] = missing_before
    metrics["missing_features_count"] = len(missing_before)
    metrics["hashed_unseen_ip_count"] = int(hashed_new)
    metrics["total_unique_ip_count"] = int(len(seen_ips))
    metrics["n_flows"] = int(n)

    if return_scores:
        metrics["scores"] = all_probs_attack
        metrics["predictions"] = all_preds
        metrics["labels"] = labels
    return metrics


def reservoir_sample_parquet(path: str, target_rows: int = 250_000,
                             seed: int = 42, label_col: str = "Label"):
    """Lấy mẫu phân tầng theo nhãn từ một tệp Parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    actual_label_col = next((c for c in table.column_names if c.lower() == label_col.lower()), label_col)
    labels = table.column(actual_label_col).to_numpy()

    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    rng = np.random.default_rng(seed)

    quotas = {k: int(round(c / total * target_rows)) for k, c in zip(unique_labels, counts)}
    largest = max(quotas, key=quotas.get)
    quotas[largest] += target_rows - sum(quotas.values())

    chosen_indices = []
    for lbl, q in quotas.items():
        if q <= 0:
            continue
        lbl_indices = np.where(labels == lbl)[0]
        if len(lbl_indices) <= q:
            chosen_indices.append(lbl_indices)
        else:
            chosen_indices.append(rng.choice(lbl_indices, size=q, replace=False))

    all_indices = np.concatenate(chosen_indices)
    rng.shuffle(all_indices)

    sampled_table = table.take(all_indices)
    return sampled_table.to_pandas().reset_index(drop=True)
