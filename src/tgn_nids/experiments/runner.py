"""Điều phối toàn bộ quy trình huấn luyện và đánh giá TGN-NIDS."""

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tgn_nids import SOURCE_VERSION
from tgn_nids.data.graph_builder import TemporalGraphBuilder
from tgn_nids.data.protocols import (split_te_g_sage, split_graphids, split_anomal_e,
                                split_random_stratified,
                                PROTOCOLS_WITH_ATTACKS_IN_TRAIN)
from tgn_nids.models.tgn_memory import TGNMemoryModule
from tgn_nids.models.embedding import TemporalEmbeddingModule
from tgn_nids.models.anomaly_detector import AnomalyDetector
from tgn_nids.models.edge_decoder import EdgeFeatureDecoder
from tgn_nids.utils.paper_metrics import evaluate_binary, evaluate_multiclass


@dataclass
class ExperimentConfig:
    """Tập hợp tham số cần thiết cho một lần chạy thí nghiệm."""

    name: str                          # Tên ngắn dùng đặt tên file kết quả
    description: str                   # Mô tả mục đích thí nghiệm
    protocol: str                      # te_g_sage | graphids | anomal_e | random_stratified
    task: str = "binary"               # binary | multiclass
    mode: str = "supervised"           # supervised | self_supervised

    # Kiến trúc
    memory_dim: int = 128
    time_dim: int = 64
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 1
    hidden_dim: int = 128
    dropout: float = 0.1               # Dropout của encoder
    classifier_dropout: float = 0.3    # Dropout của classification head giai đoạn 2

    # Huấn luyện
    batch_size: int = 512
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"  # val_loss | pr_auc | f1_macro
    use_class_weights: bool = True
    class_weight_scheme: str = "sqrt_inverse"  # sqrt_inverse | inverse | none
    class_weight_cap: float = 20.0
    threshold_strategy: Optional[str] = None   # 'f1' | 'far' | None
    target_far: float = 0.01

    # Kiến trúc hai giai đoạn
    pretrained_encoder: Optional[str] = None
    freeze_encoder: bool = True
    far_targets: Tuple[float, ...] = (0.0009, 0.01)
    mask_ratio: float = 0.15
    classifier_edge_features: bool = True
    sample_ratio: float = 1.0
    log_every_epochs: int = 1

    # Ablation
    ablate_node_identity: bool = False
    ablate_features: Tuple[str, ...] = ()

    seed: int = 42
    parameter_sources: Dict[str, str] = field(default_factory=dict)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_dataset(df: pd.DataFrame, protocol: str, seed: int) -> Dict[str, pd.DataFrame]:
    if protocol == "te_g_sage":
        return split_te_g_sage(df)
    if protocol == "graphids":
        return split_graphids(df, seed=seed)
    if protocol == "anomal_e":
        return split_anomal_e(df, seed=seed)
    if protocol == "random_stratified":
        return split_random_stratified(df, seed=seed)
    raise ValueError(f"Giao thức không hợp lệ: {protocol}")


def _validate_config(config: ExperimentConfig,
                     splits_provided: bool = False) -> None:
    """Chặn các tổ hợp cấu hình gây rò rỉ hoặc không thể học."""
    if (not splits_provided and config.mode == "supervised"
            and config.protocol not in PROTOCOLS_WITH_ATTACKS_IN_TRAIN):
        raise ValueError(
            f"Cấu hình '{config.name}' học có giám sát nhưng dùng giao thức "
            f"'{config.protocol}', vốn loại sạch mẫu tấn công khỏi tập huấn "
            f"luyện. Hàm mất mát sẽ không có mẫu dương nào để học. Dùng giao "
            f"thức 'random_stratified' nếu muốn chia ngẫu nhiên 80/10/10 mà "
            f"vẫn giữ mẫu tấn công."
        )
    needs_threshold = (config.mode == "self_supervised"
                       or config.threshold_strategy is not None)
    if needs_threshold and not splits_provided and config.protocol == "anomal_e":
        raise ValueError(
            f"Cấu hình '{config.name}' cần hiệu chỉnh ngưỡng quyết định, "
            f"nhưng giao thức 'anomal_e' chỉ chia thành hai tập train và "
            f"test, không có tập kiểm định. Hiệu chỉnh ngưỡng trên chính tập "
            f"kiểm thử là rò rỉ thông tin đánh giá, nên tổ hợp này bị chặn."
        )


def _validate_splits(splits: Dict[str, pd.DataFrame],
                     config: ExperimentConfig) -> None:
    """Kiểm tra các tập đã chia có đủ dữ liệu cho chế độ huấn luyện."""
    for part in ("train", "val", "test"):
        if part not in splits:
            raise KeyError(f"Tập đã phân chia thiếu khoá '{part}'")
        if len(splits[part]) == 0:
            raise ValueError(f"Tập '{part}' rỗng")

    label_column = next((column for column in splits["train"].columns
                         if column.strip().lower() == "label"), None)
    if label_column is None:
        raise KeyError("Không tìm thấy cột nhãn trong tập đã phân chia")

    train_attacks = int((splits["train"][label_column].astype(int) == 1).sum())
    if config.mode == "supervised" and train_attacks == 0:
        raise ValueError(
            f"Cấu hình '{config.name}' học có giám sát nhưng tập huấn luyện "
            f"không có mẫu tấn công nào. Hàm mất mát sẽ không có mẫu dương để "
            f"học.")
    if config.mode == "self_supervised" and train_attacks > 0:
        raise ValueError(
            f"Cấu hình '{config.name}' học tự giám sát trên lưu lượng lành "
            f"tính, nhưng tập huấn luyện còn {train_attacks:,} mẫu tấn công. "
            f"Điểm bất thường tái thiết sẽ mất khả năng phân biệt vì mô hình "
            f"đã học coi lưu lượng tấn công là bình thường.")

    needs_threshold = (config.mode == "self_supervised"
                       or config.threshold_strategy is not None)
    if needs_threshold:
        val_attacks = int((splits["val"][label_column].astype(int) == 1).sum())
        if val_attacks == 0:
            raise ValueError(
                f"Cấu hình '{config.name}' hiệu chỉnh ngưỡng quyết định trên "
                f"tập kiểm định, nhưng tập đó không có mẫu tấn công nào nên "
                f"không tính được F1 để cực đại hoá.")


def _get_labels(data, task: str) -> torch.Tensor:
    """Lấy nhãn nhị phân hoặc đa lớp theo cấu hình bài toán."""
    if task == "multiclass":
        labels = getattr(data, "y_multi", None)
        if labels is None:
            raise RuntimeError(
                "Thiếu nhãn đa lớp. Dữ liệu cần đi qua NetFlowPreprocessor "
                "để sinh cột Attack_encoded trước khi dựng đồ thị."
            )
        return labels.long()
    return data.y.long()


def _class_weights(labels: torch.Tensor, n_classes: int, cap: float,
                   device, scheme: str = "sqrt_inverse") -> torch.Tensor:
    """Tính trọng số lớp và giới hạn giá trị cực đại."""
    if scheme == "none":
        return torch.ones(n_classes, dtype=torch.float32, device=device)

    counts = Counter(labels.cpu().numpy().tolist())
    total = len(labels)
    raw = np.array([total / (n_classes * max(1, counts.get(c, 0)))
                    for c in range(n_classes)], dtype=np.float64)

    if scheme == "sqrt_inverse":
        raw = np.sqrt(raw)

    raw = raw / raw.mean()
    raw = np.minimum(raw, cap)
    return torch.tensor(raw, dtype=torch.float32, device=device)


def _calibrate_threshold(scores: np.ndarray, truths: np.ndarray,
                         strategy: str, target_far: float) -> float:
    """Chọn ngưỡng trên tập kiểm định theo Macro F1 hoặc FAR."""
    from sklearn.metrics import f1_score

    candidates = np.unique(np.quantile(scores, np.concatenate([
        np.linspace(0.05, 0.50, 100),
        np.linspace(0.50, 0.99, 200),
        np.linspace(0.99, 1.0, 300),
    ])))

    if strategy == "far":
        benign_scores = scores[truths == 0]
        for threshold in candidates:
            far = float((benign_scores > threshold).mean())
            if far <= target_far:
                return float(threshold)
        return float(candidates[-1])

    values = [f1_score(truths, (scores > c).astype(int),
                       average="macro", zero_division=0)
              for c in candidates]
    return float(candidates[int(np.argmax(values))])


_HIGHER_IS_BETTER = {"pr_auc", "f1_macro"}


def _monitored_value(metric: str, loss: float, truths, predictions, scores,
                     benign_class_id: int, task: str):
    """Tính đại lượng mà cơ chế dừng sớm đang theo dõi."""
    if metric == "val_loss":
        return float(loss)
    if truths is None or len(truths) == 0:
        return None
    if metric == "pr_auc":
        if scores is None:
            return None
        from sklearn.metrics import average_precision_score
        binary = (truths != benign_class_id).astype(int) if task == "multiclass" else truths
        if len(np.unique(binary)) < 2:
            return None
        return float(average_precision_score(binary, scores))
    if metric == "f1_macro":
        if predictions is None:
            return None
        from sklearn.metrics import f1_score
        return float(f1_score(truths, predictions, average="macro", zero_division=0))
    raise ValueError(f"Đại lượng theo dõi không hợp lệ: {metric!r}")


def run_experiment(df: pd.DataFrame, config: ExperimentConfig,
                    preprocessor, output_dir: str,
                    max_batches: Optional[int] = None,
                    verbose: bool = True,
                    splits: Optional[Dict[str, pd.DataFrame]] = None) -> dict:
    """Chạy một thí nghiệm hoàn chỉnh và trả về toàn bộ kết quả."""
    def log(*args):
        if verbose:
            print(*args)

    def rule():
        if verbose:
            print("------")

    _validate_config(config, splits_provided=splits is not None)
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started_at = time.time()

    log("")
    rule()
    log(f"Thí nghiệm: {config.name}")
    log(f"Mô tả: {config.description}")
    log(f"Protocol: {config.protocol} | Task: {config.task} "
        f"| Mode: {config.mode} | Device: {device}")
    rule()

    if config.sample_ratio < 1.0 and splits is not None:
        raise ValueError(
            f"Cấu hình '{config.name}' vừa khai báo sample_ratio="
            f"{config.sample_ratio} vừa truyền tập đã phân chia sẵn. Khi phía "
            f"gọi tự phân chia thì việc lấy mẫu cũng phải thực hiện ở phía gọi, "
            f"trước khi chia, và đặt sample_ratio bằng 1.0.")

    if config.sample_ratio < 1.0:
        stratify_column = ("Attack" if "Attack" in df.columns else "Label")
        before = len(df)
        df = (df.groupby(stratify_column, group_keys=False)
                .sample(frac=config.sample_ratio, random_state=config.seed))
        df = df.sort_index()
        log(f"Lấy mẫu phân tầng {config.sample_ratio:.0%}: "
            f"{before:,} -> {len(df):,} flow")

    if config.ablate_features:
        missing = [c for c in config.ablate_features if c not in df.columns]
        if missing:
            raise ValueError(
                f"Cấu hình '{config.name}' yêu cầu loại các cột không tồn tại "
                f"trong dữ liệu: {', '.join(missing)}"
            )
        df = df.drop(columns=list(config.ablate_features))
        log(f"Ablation, đã loại khỏi dữ liệu: "
            f"{', '.join(config.ablate_features)}")

    # Nhận tập chia sẵn hoặc áp dụng giao thức trong cấu hình.
    if splits is None:
        splits = _split_dataset(df, config.protocol, config.seed)
    else:
        splits = dict(splits)
        _validate_splits(splits, config)
        log(f"Dùng tập đã phân chia sẵn do phía gọi cung cấp, "
            f"bỏ qua giao thức '{config.protocol}'")
    removed = splits.pop("_removed_from_train", None)
    if removed is not None:
        log(f"Protocol yêu cầu tập train chỉ chứa lưu lượng benign: "
            f"đã loại {removed:,} flow tấn công")

    split_stats = {k: {"count": len(v), "attacks": int((v["Label"] == 1).sum())}
                   for k, v in splits.items()}
    for name, stat in split_stats.items():
        log(f"{name.capitalize() + ':':<7}{stat['count']:>11,} flow "
            f"| {stat['attacks']:>8,} attack")
    rule()

    # Chỉ fit bộ tiền xử lý trên tập huấn luyện.
    attack_column = next((c for c in df.columns
                          if c.upper() in ("ATTACK", "ATTACK_CAT")), None)
    if attack_column is not None:
        preprocessor.class_vocabulary = sorted(df[attack_column].astype(str).unique())

    splits["train"] = preprocessor.fit_transform(splits["train"])
    for name in splits:
        if name != "train":
            splits[name] = preprocessor.transform(splits[name])
    if getattr(preprocessor, "unknown_label_count", 0):
        raise RuntimeError(
            f"{preprocessor.unknown_label_count:,} nhãn nằm ngoài danh mục đã "
            f"chốt và bị quy về Benign. Kết quả đa lớp sẽ sai."
        )

    class_names = [str(name) for name in preprocessor.label_encoder.classes_]
    benign_class_id = class_names.index("Benign") if "Benign" in class_names else 0
    n_classes = len(class_names) if config.task == "multiclass" else 2
    log(f"Danh mục nhãn: {len(class_names)} lớp "
        f"| Benign ở vị trí {benign_class_id}")
    if config.task == "multiclass" and len(class_names) < 2:
        raise RuntimeError(
            "Bài toán đa lớp nhưng danh mục nhãn chỉ có một lớp. Kiểm tra lại "
            "cột Attack của dữ liệu đầu vào."
        )

    builder = TemporalGraphBuilder()
    graphs = {k: builder.build_pyg_temporal_data(v) for k, v in splits.items()}
    edge_dim = graphs["train"].msg.shape[1]
    n_nodes = builder.get_node_count()

    if config.ablate_node_identity:
        generator = np.random.default_rng(config.seed)
        for graph in graphs.values():
            count = len(graph.src)
            graph.src = torch.tensor(generator.integers(0, n_nodes, count),
                                     dtype=graph.src.dtype)
            graph.dst = torch.tensor(generator.integers(0, n_nodes, count),
                                     dtype=graph.dst.dtype)
        log("Ablation, đã gán lại node nguồn và node đích một cách ngẫu nhiên")

    log(f"Edge features: {edge_dim} | Số node: {n_nodes:,}")

    memory = TGNMemoryModule(num_nodes=n_nodes, raw_msg_dim=edge_dim,
                             memory_dim=config.memory_dim,
                             time_dim=config.time_dim).to(device)
    embedding = TemporalEmbeddingModule(memory_dim=config.memory_dim,
                                        embedding_dim=config.embedding_dim,
                                        edge_dim=edge_dim,
                                        time_dim=config.time_dim,
                                        num_heads=config.num_heads,
                                        num_layers=config.num_layers,
                                        dropout=config.dropout).to(device)

    # Giai đoạn hai có thể dùng lại và đóng băng bộ mã hóa của giai đoạn một.
    encoder_source = None
    if config.pretrained_encoder:
        if not os.path.exists(config.pretrained_encoder):
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint của giai đoạn thứ nhất: "
                f"{config.pretrained_encoder}")
        try:
            stage_one = torch.load(config.pretrained_encoder, map_location=device,
                                   weights_only=True)
        except Exception as error:
            raise RuntimeError(
                f"Không nạp được checkpoint {config.pretrained_encoder} ở chế độ "
                f"an toàn. Nếu tệp được sinh ra bởi bản mã cũ, nó chứa đối tượng "
                f"đã pickle và phải chạy lại notebook để sinh bản mới."
            ) from error
        for required in ("memory_state", "embedding_state"):
            if required not in stage_one:
                raise KeyError(
                    f"Checkpoint {config.pretrained_encoder} thiếu khoá "
                    f"'{required}', không dùng làm bộ mã hoá được")
        saved_config = stage_one.get("config", {})
        for field_name in ("memory_dim", "embedding_dim", "time_dim"):
            saved_value = saved_config.get(field_name)
            current_value = getattr(config, field_name)
            if saved_value is not None and saved_value != current_value:
                raise ValueError(
                    f"Giai đoạn thứ nhất dùng {field_name}={saved_value} còn "
                    f"giai đoạn này khai báo {current_value}. Hai giai đoạn "
                    f"phải dùng cùng kích thước biểu diễn.")

        node_indexed_buffers = {"memory.memory", "memory.last_update",
                                "memory._assoc"}
        learned_names = {name for name, _ in memory.named_parameters()}
        saved_learned = {name: tensor
                         for name, tensor in stage_one["memory_state"].items()
                         if name not in node_indexed_buffers}
        if set(saved_learned) != learned_names:
            raise ValueError(
                f"Checkpoint của giai đoạn thứ nhất có tham số học được "
                f"{sorted(saved_learned)} còn mô hình hiện tại cần "
                f"{sorted(learned_names)}. Hai giai đoạn không cùng kiến trúc.")
        memory.load_state_dict(saved_learned, strict=False)
        embedding.load_state_dict(stage_one["embedding_state"])
        encoder_source = config.pretrained_encoder
        log(f"Đã nạp bộ mã hoá của giai đoạn thứ nhất từ "
            f"{os.path.basename(config.pretrained_encoder)}")
        if config.freeze_encoder:
            for module in (memory, embedding):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                module.eval()
            log("Đã đóng băng khối bộ nhớ thời gian và khối nhúng. "
                "Giai đoạn này chỉ huấn luyện đầu phân loại.")

    self_supervised = config.mode == "self_supervised"
    if self_supervised:
        head = EdgeFeatureDecoder(config.embedding_dim, edge_dim,
                                  config.hidden_dim,
                                  mask_ratio=config.mask_ratio).to(device)
        criterion = None
        log(f"Masked reconstruction: che {head.num_masked}/{edge_dim} thuộc tính "
            f"mỗi cạnh (mask_ratio {config.mask_ratio})")
    else:
        head = AnomalyDetector(embedding_dim=config.embedding_dim,
                               hidden_dim=config.hidden_dim,
                               num_classes=n_classes,
                               dropout=config.classifier_dropout,
                               edge_dim=edge_dim if config.classifier_edge_features
                               else 0).to(device)
        log(f"Bộ phân loại nhìn đặc trưng cạnh: "
            f"{'có' if config.classifier_edge_features else 'không'}")
        weights = None
        if config.use_class_weights:
            weights = _class_weights(_get_labels(graphs["train"], config.task),
                                     n_classes, config.class_weight_cap, device,
                                     config.class_weight_scheme)
            log("Class weights: "
                + " ".join(f"{name}={value:.3f}" for name, value
                           in zip(class_names if config.task == "multiclass"
                                  else ["Benign", "Attack"], weights.tolist())))
        criterion = nn.CrossEntropyLoss(weight=weights)

    class_weights_used = (None if self_supervised or weights is None
                          else [round(w, 4) for w in weights.tolist()])

    parameters = [parameter for module in (memory, embedding, head)
                  for parameter in module.parameters()
                  if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    n_parameters = sum(p.numel() for p in parameters)
    log(f"Tham số: {n_parameters:,}")
    rule()

    def run_pass(data, training: bool, need_scores: bool = True,
                 score_samples: Optional[int] = None):
        """Chạy một lượt dữ liệu và thu loss, nhãn, dự đoán cùng điểm số."""
        for module in (memory, embedding, head):
            module.train() if training else module.eval()

        all_labels = _get_labels(data, config.task)
        n = len(all_labels)
        n_batches = (n + config.batch_size - 1) // config.batch_size
        if max_batches:
            n_batches = min(n_batches, max_batches)

        total_loss = 0.0
        predictions, scores, truths = [], [], []

        for i in range(n_batches):
            start = i * config.batch_size
            end = min(start + config.batch_size, n)
            src = data.src[start:end].to(device)
            dst = data.dst[start:end].to(device)
            timestamps = data.t[start:end].to(device)
            messages = data.msg[start:end].to(device)
            labels = all_labels[start:end].to(device)

            node_ids, inverse = torch.cat([src, dst]).unique(return_inverse=True)
            src_idx, dst_idx = inverse[:len(src)], inverse[len(src):]

            with torch.set_grad_enabled(training):
                state, _ = memory(node_ids)
                embedded = embedding(state, torch.stack([src_idx, dst_idx]),
                                     messages)

            if self_supervised:
                mask_seed = None if training else config.seed + i
                with torch.set_grad_enabled(training):
                    loss = head.reconstruction_loss(embedded[src_idx],
                                                    embedded[dst_idx], messages,
                                                    seed=mask_seed)
                if need_scores:
                    with torch.no_grad():
                        goc = head.eval_mask_samples
                        if score_samples is not None:
                            head.eval_mask_samples = score_samples
                        try:
                            scores.append(head.anomaly_score(
                                embedded[src_idx], embedded[dst_idx], messages,
                                seed=mask_seed).cpu().numpy())
                        finally:
                            head.eval_mask_samples = goc
            else:
                with torch.set_grad_enabled(training):
                    logits = head(embedded[src_idx], embedded[dst_idx],
                                  messages if config.classifier_edge_features
                                  else None)
                    loss = criterion(logits, labels)
                with torch.no_grad():
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions.append(probabilities.argmax(-1).cpu().numpy())
                    scores.append(
                        (1.0 - probabilities[:, benign_class_id]).cpu().numpy()
                        if config.task == "multiclass"
                        else probabilities[:, 1].cpu().numpy())

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

            with torch.no_grad():
                memory.update_state(src, dst, timestamps, messages)
                memory.detach_memory()

            total_loss += float(loss.item())
            truths.append(labels.cpu().numpy())

        return (total_loss / max(1, n_batches),
                np.concatenate(truths) if truths else np.array([]),
                np.concatenate(predictions) if predictions else None,
                np.concatenate(scores) if scores else None)

    # Lưu trạng thái tốt nhất để đánh giá đúng mô hình được chọn trên tập val.
    monitor = config.early_stopping_metric
    higher_better = monitor in _HIGHER_IS_BETTER
    monitor_needs_scores = monitor != "val_loss"
    history, best_score, patience, best_epoch = [], -float("inf"), 0, 0
    best_loss = float("inf")
    validation_split = "val" if "val" in graphs else "test"
    modules = {"memory": memory, "embedding": embedding, "head": head}
    training_started = time.time()

    def checkpoint_path(label: str) -> str:
        return os.path.join(output_dir, f"{label}_{config.name}.pth")

    for epoch in range(config.num_epochs):
        memory.reset_state()
        train_loss, *_ = run_pass(graphs["train"], True, need_scores=False)
        with torch.no_grad():
            val_loss, val_truths, val_predictions, val_scores = run_pass(
                graphs[validation_split], False,
                need_scores=monitor_needs_scores,
                score_samples=None)

        monitored = _monitored_value(monitor, val_loss, val_truths,
                                     val_predictions, val_scores,
                                     benign_class_id, config.task)
        if monitored is None:
            if epoch == 0:
                log(f"Không tính được '{monitor}' trên tập kiểm định, cơ chế "
                    f"dừng sớm lùi về val_loss cho toàn bộ lần chạy này")
            monitor, higher_better = "val_loss", False
            monitored = float(val_loss)
        score = monitored if higher_better else -monitored

        record = {"epoch": epoch + 1, "train_loss": train_loss,
                  "val_loss": val_loss}
        if monitor != "val_loss":
            record[f"val_{monitor}"] = monitored
        history.append(record)
        is_best = score > best_score + 1e-5
        step = config.log_every_epochs
        if epoch == 0 or step <= 1 or (epoch + 1) % step == 0:
            extra = ("" if monitor == "val_loss"
                     else f" | Val {monitor}: {monitored:>9.5f}")
            log(f"Epoch {epoch + 1:>3} | Train loss: {train_loss:>10.5f} "
                f"| Val loss: {val_loss:>10.5f}{extra}")

        if is_best:
            best_score, patience, best_epoch = score, 0, epoch + 1
            best_loss = val_loss
            os.makedirs(output_dir, exist_ok=True)
            for label, module in modules.items():
                torch.save(module.state_dict(), checkpoint_path(label))
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                log(f"Early stopping tại epoch {epoch + 1}")
                break
    else:
        if best_epoch > config.num_epochs - config.early_stopping_patience:
            log(f"CẢNH BÁO: hết {config.num_epochs} epoch mà chưa dừng sớm, "
                f"vòng tốt nhất là {best_epoch}. Mô hình có thể còn tiến bộ "
                f"nếu tăng num_epochs.")

    training_time = time.time() - training_started

    # Tập test chỉ được dùng sau khi đã chọn xong epoch tốt nhất.
    if best_epoch:
        for label, module in modules.items():
            module.load_state_dict(torch.load(checkpoint_path(label),
                                              map_location=device,
                                              weights_only=True))
        rule()
        chon_theo = ("val_loss" if monitor == "val_loss"
                     else f"{monitor} = {best_score:.5f}")
        log(f"Nạp lại trọng số tốt nhất: epoch {best_epoch} "
            f"| Val loss: {best_loss:.5f} | chọn theo {chon_theo}")

    memory.reset_state()
    run_pass(graphs["train"], False, need_scores=False)
    val_true = val_scores = None
    if "val" in graphs:
        _, val_true, _, val_scores = run_pass(graphs["val"], False)

    inference_started = time.time()
    _, y_true, y_pred, y_score = run_pass(graphs["test"], False)
    inference_time = time.time() - inference_started

    needs_threshold = self_supervised or (
        config.threshold_strategy is not None and config.task == "binary")

    def to_binary(values):
        return ((values != benign_class_id).astype(int)
                if config.task == "multiclass" else values)

    test_binary_true = to_binary(y_true)
    argmax_binary_pred = None if y_pred is None else to_binary(y_pred)

    threshold = None
    if needs_threshold:
        strategy = config.threshold_strategy or "f1"
        if val_scores is None:
            raise RuntimeError(
                f"Cấu hình '{config.name}' cần hiệu chỉnh ngưỡng nhưng giao "
                f"thức '{config.protocol}' không sinh ra tập kiểm định."
            )
        threshold = _calibrate_threshold(val_scores, to_binary(val_true),
                                         strategy, config.target_far)
        y_pred = (y_score > threshold).astype(int)
        y_true = test_binary_true
        log(f"Threshold hiệu chỉnh trên tập val theo chiến lược "
            f"'{strategy}': {threshold:.6f}")

    if config.task == "multiclass" and not needs_threshold:
        metrics = evaluate_multiclass(y_true, y_pred, class_names, benign_class_id)
    else:
        metrics = evaluate_binary(y_true, y_pred, y_score)

    # Báo thêm các ngưỡng vận hành nhưng không thay ngưỡng chính đã chọn.
    operating_points = {}
    if val_scores is not None and y_score is not None:
        val_binary = to_binary(val_true)
        if argmax_binary_pred is not None:
            operating_points["argmax"] = {
                "threshold": None,
                "val_far": None,
                "dat_rang_buoc": None,
                "mo_ta": "quy tắc mặc định, lấy lớp có xác suất lớn nhất",
                **evaluate_binary(test_binary_true, argmax_binary_pred, y_score),
            }
        for label, strategy, target in (
                [("f1", "f1", None)]
                + [(f"far_{t:g}", "far", t) for t in config.far_targets]):
            value = _calibrate_threshold(val_scores, val_binary, strategy,
                                         target if target is not None else 0.01)
            achieved = float((val_scores[val_binary == 0] > value).mean())
            operating_points[label] = {
                "threshold": value,
                "val_far": achieved,
                "dat_rang_buoc": None if target is None else bool(achieved <= target),
                "mo_ta": ("ngưỡng cực đại hóa F1 trên tập kiểm định"
                          if target is None
                          else f"ngưỡng thấp nhất giữ tỉ lệ cảnh báo sai "
                               f"trên tập kiểm định dưới {target * 100:.2f}%"),
                **evaluate_binary(test_binary_true,
                                  (y_score > value).astype(int), y_score),
            }
        rule()
        log("Operating point:")
        for label, point in operating_points.items():
            log(f"  {label:<12} | F1: {point['f1'] * 100:>6.2f}% "
                f"| Recall: {point['recall'] * 100:>6.2f}% "
                f"| FAR: {point['far'] * 100:>7.3f}%")

    result = {
        "source_version": SOURCE_VERSION,
        "run_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_parameters": int(n_parameters),
        "environment": {
            "python": ".".join(str(v) for v in __import__("sys").version_info[:3]),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
        },
        "dataset": {
            "rows_input": int(len(df)),
            "attacks_input": int((df["Label"] == 1).sum()),
        },
        "config": asdict(config),
        "split_stats": split_stats,
        "class_names": class_names,
        "benign_class_id": benign_class_id,
        "class_weights": class_weights_used,
        "ablation": {
            "node_identity": bool(config.ablate_node_identity),
            "features": list(config.ablate_features),
        },
        "edge_dim": int(edge_dim),
        "n_nodes": int(n_nodes),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss if best_epoch else None,
        "early_stopping_metric": monitor,
        "best_monitored_value": (
            (best_score if higher_better else -best_score) if best_epoch else None),
        "threshold": threshold,
        "metrics": metrics,
        "operating_points": operating_points,
        "training_time_seconds": round(training_time, 1),
        "inference_time_seconds": round(inference_time, 3),
        "latency_ms_per_100_flows": round(
            inference_time / max(1, len(y_true)) * 100 * 1000, 3),
        "total_time_seconds": round(time.time() - started_at, 1),
        "output_dir": os.path.abspath(output_dir),
    }

    # Checkpoint đủ thông tin để khôi phục mà không pickle đối tượng Python.
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_{config.name}.pt")
    torch.save({
        "pretrained_encoder": encoder_source,
        "encoder_frozen": bool(config.pretrained_encoder and config.freeze_encoder),
        "source_version": SOURCE_VERSION,
        "config": asdict(config),
        "memory_state": memory.state_dict(),
        "embedding_state": embedding.state_dict(),
        "head_state": head.state_dict(),
        "preprocessor_state": preprocessor.state_dict(),
        "node_mapping": builder.node_mapping,
        "class_names": class_names,
        "benign_class_id": benign_class_id,
        "threshold": threshold,
        "edge_dim": int(edge_dim),
        "n_nodes": int(n_nodes),
        "n_classes": int(n_classes),
        "n_parameters": int(n_parameters),
    }, model_path)
    log(f"Đã ghi mô hình: {model_path} "
        f"({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")

    predictions_path = os.path.join(output_dir, f"predictions_{config.name}.npz")
    np.savez_compressed(
        predictions_path,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred) if y_pred is not None else np.array([]),
        y_score=np.asarray(y_score) if y_score is not None else np.array([]),
        src=graphs["test"].src.cpu().numpy(),
        dst=graphs["test"].dst.cpu().numpy(),
        t=graphs["test"].t.cpu().numpy(),
    )
    log(f"Đã ghi dự đoán tập test: {predictions_path}")

    path = os.path.join(output_dir, f"result_{config.name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=float)
    rule()
    log(f"Đã ghi kết quả: {path}")

    return result
