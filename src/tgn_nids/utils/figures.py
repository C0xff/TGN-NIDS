"""Vẽ hình và xuất bảng kết quả theo quy cách chung của báo cáo."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tgn_nids.utils import format as fmt

# Bảng màu dùng chung để các hình giữ cùng phong cách.
ACCENT = "#b3261e"      # Màu nhấn chính
NEUTRAL = "#4a4a4a"     # Xám đậm cho đường chính
LIGHT = "#9e9e9e"       # Xám nhạt cho đường tham chiếu
GRID = "#d9d9d9"

FIG_DPI = 300
FONT_FAMILY = ["Times New Roman", "DejaVu Serif", "serif"]


def apply_style() -> None:
    """Áp dụng font, màu và kích thước mặc định cho Matplotlib."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": FONT_FAMILY,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": NEUTRAL,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def save_figure(fig, out_dir: Optional[str], name: str) -> Dict[str, str]:
    """Lưu một hình ở cả định dạng PNG và PDF."""
    if out_dir is None:
        return {}

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for extension in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{extension}")
        fig.savefig(path)
        paths[extension] = path
    return paths


def trivial_f1(positive_rate: float) -> float:
    """Tính F1 của bộ phân loại luôn dự đoán lớp dương."""
    if positive_rate <= 0:
        return 0.0
    return 2.0 * positive_rate / (1.0 + positive_rate)


def plot_learning_curve(history: List[dict], out_dir: str,
                        name: str = "duong_hoc",
                        best_epoch: Optional[int] = None):
    """Vẽ loss huấn luyện và kiểm định theo epoch."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(epochs, train, color=NEUTRAL, linewidth=1.4, label="Train loss")
    ax.plot(epochs, val, color=ACCENT, linewidth=1.4, linestyle="--",
            label="Validation loss")
    if best_epoch:
        ax.axvline(best_epoch, color=LIGHT, linewidth=1.0, linestyle=":")
        ax.annotate(f"epoch {best_epoch}", xy=(best_epoch, max(val)),
                    xytext=(3, -8), textcoords="offset points",
                    fontsize=8, color=NEUTRAL)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig, save_figure(fig, out_dir, name)


def plot_confusion_matrix(cm, class_names: Sequence[str], out_dir: str,
                          name: str = "ma_tran_nham_lan",
                          normalize: bool = True):
    """Vẽ ma trận nhầm lẫn theo số lượng hoặc tỷ lệ trên từng lớp thật."""
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    display = cm.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display),
                            where=row_sums > 0)

    n = len(class_names)
    size = max(3.6, 0.52 * n + 1.8)
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    shown = display * 100 if normalize else display
    im = ax.imshow(shown, cmap="Greys", vmin=0,
                   vmax=shown.max() if shown.max() > 0 else 1)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Lớp dự đoán")
    ax.set_ylabel("Lớp thực tế")
    ax.grid(False)

    threshold = shown.max() * 0.55 if shown.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            value = display[i, j]
            if normalize:
                text = "0" if value == 0 else fmt.percent(value, 1, unit="")
            else:
                text = fmt.count(cm[i, j])
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8 if n > 4 else 10,
                    color="white" if shown[i, j] > threshold else NEUTRAL)

    label = "Tỷ lệ theo hàng (%)" if normalize else "Số lượng flow"
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    return fig, save_figure(fig, out_dir, name)


def plot_roc_pr(y_true, y_score, out_dir: str, name: str = "roc_pr",
                positive_label: int = 1, title: Optional[str] = None):
    """Vẽ đường cong ROC và Precision–Recall cho bài toán nhị phân."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import (auc, average_precision_score,
                                 precision_recall_curve, roc_curve)

    y_true = np.asarray(y_true)
    y_binary = (y_true == positive_label).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, _ = roc_curve(y_binary, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_binary, y_score)
    ap = average_precision_score(y_binary, y_score)
    base_rate = float(y_binary.mean())

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))

    axes[0].plot(fpr, tpr, color=ACCENT, linewidth=1.5,
                 label=f"ROC (AUC = {fmt.percent(roc_auc)})")
    axes[0].plot([0, 1], [0, 1], color=LIGHT, linewidth=1.0, linestyle="--",
                 label="Đoán ngẫu nhiên")
    axes[0].set_xlabel("Tỷ lệ dương tính giả (FPR)")
    axes[0].set_ylabel("Tỷ lệ dương tính thật (TPR)")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color=ACCENT, linewidth=1.5,
                 label=f"PR (AP = {fmt.percent(ap)})")
    axes[1].axhline(base_rate, color=LIGHT, linewidth=1.0, linestyle="--",
                    label=f"Tỷ lệ nền = {fmt.percent(base_rate)}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    if title:
        fig.suptitle(title, fontsize=10, color=NEUTRAL)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    return fig, save_figure(fig, out_dir, name), {
        "roc_auc": float(roc_auc), "average_precision": float(ap),
        "base_rate": base_rate}


def normalise_per_class(per_class) -> List[dict]:
    """Chuẩn hóa bảng chỉ số từng lớp về dạng `list[dict]`."""
    if isinstance(per_class, dict):
        return [{"name": key, **value} for key, value in per_class.items()]
    return [dict(record) for record in (per_class or [])]


def plot_per_class(per_class, out_dir: str, name: str = "chi_so_theo_lop"):
    """Vẽ Precision, Recall và F1 của từng lớp kèm số mẫu."""
    import matplotlib.pyplot as plt

    names, precision, recall, f1, support = [], [], [], [], []
    for record in normalise_per_class(per_class):
        names.append(str(record.get("name", "?")))
        precision.append(record.get("precision", np.nan))
        recall.append(record.get("recall", np.nan))
        f1.append(record.get("f1", record.get("f1-score", np.nan)))
        support.append(record.get("support") or 0)

    order = np.argsort(support)[::-1]
    names = [f"{names[i]}  (n={fmt.count(support[i])})" for i in order]
    precision = [precision[i] for i in order]
    recall = [recall[i] for i in order]
    f1 = [f1[i] for i in order]

    y = np.arange(len(names))
    height = 0.26
    fig, ax = plt.subplots(figsize=(6.6, max(3.0, 0.46 * len(names) + 1.2)))
    to_pct = lambda seq: [v * 100 if v is not None else np.nan for v in seq]
    ax.barh(y + height, to_pct(precision), height, color=LIGHT, label="Precision")
    ax.barh(y, to_pct(recall), height, color=NEUTRAL, label="Recall")
    ax.barh(y - height, to_pct(f1), height, color=ACCENT, label="F1")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100.0)
    ax.set_xlabel("Giá trị chỉ số (%)")
    ax.legend(loc="lower right")
    return fig, save_figure(fig, out_dir, name)


def plot_score_distribution(y_true, y_score, out_dir: str,
                            name: str = "phan_bo_diem", positive_label: int = 1,
                            threshold: Optional[float] = None):
    """So sánh phân bố điểm của hai nhóm Benign và Attack."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    benign = y_score[y_true != positive_label]
    attack = y_score[y_true == positive_label]

    bins = np.linspace(0, 1, 61)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.hist(benign, bins=bins, color=LIGHT, label=f"Lành tính (n={fmt.count(len(benign))})")
    ax.hist(attack, bins=bins, color=ACCENT, alpha=0.75,
            label=f"Tấn công (n={fmt.count(len(attack))})")
    if threshold is not None:
        ax.axvline(threshold, color=NEUTRAL, linewidth=1.2, linestyle="--",
                   label=f"Ngưỡng = {fmt.number(threshold, decimals=4)}")
    ax.set_yscale("log")
    ax.set_xlabel("Điểm bất thường")
    ax.set_ylabel("Số flow (thang log)")
    ax.legend()
    return fig, save_figure(fig, out_dir, name)


def plot_threshold_sweep(y_true, y_score, out_dir: str,
                         name: str = "quet_nguong", positive_label: int = 1,
                         far_targets: Sequence[float] = (0.0009, 0.01)):
    """Vẽ các chỉ số nhị phân trên toàn dải ngưỡng quyết định."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_binary = (y_true == positive_label).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.linspace(0.001, 0.999, 300)
    precision, recall, f1, fpr = [], [], [], []
    n_positive = max(1, int(y_binary.sum()))
    n_negative = max(1, int((1 - y_binary).sum()))
    for t in thresholds:
        predicted = (y_score >= t).astype(int)
        tp = int(((predicted == 1) & (y_binary == 1)).sum())
        fp = int(((predicted == 1) & (y_binary == 0)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / n_positive
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) else 0.0)
        fpr.append(fp / n_negative)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(thresholds, precision, color=LIGHT, linewidth=1.3, label="Precision")
    ax.plot(thresholds, recall, color=NEUTRAL, linewidth=1.3, label="Recall")
    ax.plot(thresholds, f1, color=ACCENT, linewidth=1.6, label="F1")
    ax.plot(thresholds, fpr, color=NEUTRAL, linewidth=1.1, linestyle=":",
            label="FPR")
    ax.set_xlabel("Ngưỡng quyết định")
    ax.set_ylabel("Giá trị chỉ số")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right")

    operating = {}
    for target in far_targets:
        feasible = [i for i, v in enumerate(fpr) if v <= target]
        if feasible:
            best = max(feasible, key=lambda i: recall[i])
            operating[f"far_{target}"] = {
                "threshold": float(thresholds[best]),
                "precision": float(precision[best]),
                "recall": float(recall[best]),
                "f1": float(f1[best]), "fpr": float(fpr[best])}
    return fig, save_figure(fig, out_dir, name), operating


def plot_baseline_comparison(ours: Dict[str, float], published: Dict[str, Dict[str, float]],
                             metric: str, out_dir: str,
                             name: str = "doi_sanh_cong_bo",
                             trivial: Optional[float] = None,
                             ylabel: Optional[str] = None):
    """Vẽ biểu đồ cột đối sánh với các kết quả đã công bố."""
    import matplotlib.pyplot as plt

    labels, values, colors = [], [], []
    for key, metrics in published.items():
        if metric in metrics and metrics[metric] is not None:
            labels.append(key); values.append(metrics[metric]); colors.append(LIGHT)
    for key, value in ours.items():
        if value is not None:
            labels.append(key); values.append(value); colors.append(ACCENT)

    percents = [v * 100 for v in values]

    fig, ax = plt.subplots(figsize=(max(5.4, 1.15 * len(labels) + 2.0), 3.8))
    bars = ax.bar(range(len(labels)), percents, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 100 + 1.2,
                fmt.percent(value, unit=""), ha="center", fontsize=8.5,
                color=NEUTRAL)
    if trivial is not None:
        ax.axhline(trivial * 100, color=NEUTRAL, linewidth=1.1, linestyle="--")
        ax.text(-0.45, trivial * 100 + 1.8,
                f"Mốc tầm thường = {fmt.percent(trivial, unit='')}",
                ha="left", fontsize=8, color=NEUTRAL)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(f"{ylabel or metric} (%)")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 108)
    return fig, save_figure(fig, out_dir, name)


def plot_calibration(y_true, y_score, out_dir: str, name: str = "hieu_chuan",
                     positive_label: int = 1, n_bins: int = 12):
    """Vẽ biểu đồ độ tin cậy và tính sai số hiệu chuẩn kỳ vọng."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_binary = (y_true == positive_label).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, observed, counts = [], [], []
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_score >= edges[i]) & (y_score < edges[i + 1] + (1e-9 if i == n_bins - 1 else 0))
        n = int(in_bin.sum())
        if n == 0:
            continue
        confidence = float(y_score[in_bin].mean())
        accuracy = float(y_binary[in_bin].mean())
        centres.append(confidence); observed.append(accuracy); counts.append(n)
        ece += (n / len(y_score)) * abs(accuracy - confidence)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color=LIGHT, linewidth=1.0, linestyle="--",
            label="Hiệu chuẩn hoàn hảo")
    ax.plot(centres, observed, color=ACCENT, linewidth=1.5, marker="o",
            markersize=4, label=f"Thực đo (ECE = {fmt.percent(ece)})")
    ax.set_xlabel("Xác suất mô hình đưa ra")
    ax.set_ylabel("Tần suất tấn công quan sát được")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    return fig, save_figure(fig, out_dir, name), {"ece": float(ece)}


def plot_class_distribution(counts: Dict[str, int], out_dir: str,
                            name: str = "phan_bo_lop"):
    """Vẽ số lượng mẫu của từng lớp trên thang log."""
    import matplotlib.pyplot as plt

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(max(5.2, 0.62 * len(labels) + 1.6), 3.4))
    bars = ax.bar(range(len(labels)), values, color=LIGHT, width=0.62)
    bars[0].set_color(NEUTRAL)
    ax.set_yscale("log")
    digits = len(fmt.count(max(values)))
    ax.set_ylim(top=max(values) * 10 ** (0.16 * digits + 0.35))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 10 ** 0.13,
                fmt.count(value), ha="center", va="bottom",
                fontsize=8, color=NEUTRAL, rotation=90)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Số flow (thang log)")
    return fig, save_figure(fig, out_dir, name)


def plot_embedding_projection(embeddings, labels, class_names: Sequence[str],
                              out_dir: str, name: str = "chieu_embedding",
                              max_points: int = 4000, seed: int = 42):
    """Chiếu vector nhúng xuống hai chiều bằng t-SNE."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    if len(embeddings) > max_points:
        index = rng.choice(len(embeddings), max_points, replace=False)
        embeddings, labels = embeddings[index], labels[index]

    projected = TSNE(n_components=2, init="pca", random_state=seed,
                     perplexity=min(30, max(5, len(embeddings) // 100))
                     ).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    unique = np.unique(labels)
    greys = np.linspace(0.15, 0.78, len(unique))
    for i, value in enumerate(unique):
        mask = labels == value
        label_name = class_names[value] if value < len(class_names) else str(value)
        colour = ACCENT if i == 0 else str(greys[i])
        ax.scatter(projected[mask, 0], projected[mask, 1], s=7, alpha=0.65,
                   c=colour, marker=markers[i % len(markers)], label=label_name,
                   linewidths=0)
    ax.set_xlabel("Thành phần t-SNE thứ nhất")
    ax.set_ylabel("Thành phần t-SNE thứ hai")
    ax.legend(markerscale=1.8, ncol=2, fontsize=8)
    ax.grid(False)
    return fig, save_figure(fig, out_dir, name)


def plot_time_distribution(timestamps, split_sizes: Sequence[int], out_dir: str,
                           name: str = "phan_bo_thoi_gian", bins: int = 160):
    """Vẽ mật độ luồng theo thời gian và ranh giới giữa các tập."""
    import matplotlib.pyplot as plt

    ordered = np.sort(np.asarray(timestamps, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("Không có mốc thời gian nào để vẽ.")

    hours = (ordered - ordered[0]) / 3_600_000.0

    train_end = int(split_sizes[0])
    val_end = train_end + int(split_sizes[1])
    boundaries = [hours[min(train_end, len(hours) - 1)],
                  hours[min(val_end, len(hours) - 1)]]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(hours, bins=bins, color=LIGHT, edgecolor="none")

    span = float(hours[-1]) if hours[-1] > 0 else 1.0
    top = ax.get_ylim()[1]
    for value in boundaries:
        ax.axvline(value, color=ACCENT, linewidth=1.3, linestyle="--")
    if (boundaries[1] - boundaries[0]) / span < 0.04:
        ax.annotate("hai ranh giới train/val và val/test",
                    xy=(boundaries[0], top * 0.96),
                    xytext=(-6, 0), textcoords="offset points",
                    color=ACCENT, fontsize=8, ha="right", va="top", rotation=90)
    else:
        for value, label in zip(boundaries,
                                ("ranh giới train/val", "ranh giới val/test")):
            ax.text(value, top * 0.96, f" {label}", color=ACCENT,
                    fontsize=8, ha="left", va="top", rotation=90)

    total_hours = hours[-1]
    test_hours = total_hours - boundaries[1]
    ax.set_xlabel("Giờ tính từ luồng sớm nhất "
                  f"(toàn bộ {fmt.number(total_hours, decimals=1)} giờ)")
    ax.set_ylabel("Số flow mỗi khoảng")
    ax.set_title(f"Tập kiểm thử phủ {fmt.number(test_hours, decimals=2)} giờ "
                 f"({fmt.percent(test_hours / total_hours, 2)} trục thời gian)",
                 fontsize=10, color=NEUTRAL)
    return fig, save_figure(fig, out_dir, name)


def plot_feature_importance(importance: Dict[str, float], out_dir: str,
                            name: str = "do_quan_trong_dac_trung",
                            top_n: int = 15):
    """Vẽ các đặc trưng có độ quan trọng cao nhất."""
    import matplotlib.pyplot as plt

    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=(6.2, max(3.0, 0.34 * len(labels) + 1.2)))
    bars = ax.barh(range(len(labels)), values, color=LIGHT, height=0.68)
    bars[-1].set_color(ACCENT)
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * 0.012, bar.get_y() + bar.get_height() / 2,
                fmt.number(value, decimals=3), va="center", fontsize=8,
                color=NEUTRAL)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Mức đóng góp do triệt tiêu thuộc tính")
    ax.set_xlim(0, max(values) * 1.16)
    return fig, save_figure(fig, out_dir, name)


def plot_ego_network(subgraph: Dict[str, Any], out_dir: str,
                     name: str = "do_thi_con_giai_thich",
                     center_ip: Optional[str] = None,
                     seed: int = 42,
                     edge_weights: Optional[Sequence[float]] = None):
    """Vẽ đồ thị con quanh đỉnh tâm và làm nổi bật các cạnh cảnh báo."""
    import matplotlib.pyplot as plt
    import networkx as nx

    nodes = subgraph["nodes"]
    edges = subgraph["edges"]
    if not nodes:
        raise ValueError("Đồ thị con không có đỉnh nào, không vẽ được")
    if not edges:
        raise ValueError(
            "Đồ thị con không có cạnh nào, không vẽ được. Đỉnh trung tâm không "
            "có luồng nào nối tới đỉnh khác trong tập dữ liệu đang xét.")

    pair_flows: Dict[tuple, int] = {}
    pair_alerts: Dict[tuple, int] = {}
    pair_weight: Dict[tuple, float] = {}
    for i, edge in enumerate(edges):
        pair = tuple(sorted((edge["source"], edge["target"])))
        pair_flows[pair] = pair_flows.get(pair, 0) + 1
        if edge["is_anomaly"]:
            pair_alerts[pair] = pair_alerts.get(pair, 0) + 1
        if edge_weights is not None and i < len(edge_weights):
            pair_weight[pair] = max(pair_weight.get(pair, 0.0),
                                    float(edge_weights[i]))

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["id"])
    for pair in pair_flows:
        graph.add_edge(*pair)

    positions = nx.spring_layout(graph, seed=seed, k=0.9)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    max_flows = max(pair_flows.values())
    dung_trong_so = bool(pair_weight)
    max_weight = max(pair_weight.values()) if dung_trong_so else 1.0

    def width_of(pair, lo, hi):
        if dung_trong_so:
            scale = pair_weight.get(pair, 0.0) / max_weight if max_weight else 0.0
        else:
            scale = math.log1p(pair_flows[pair]) / math.log1p(max_flows)
        return lo + (hi - lo) * scale

    alert_pairs = [p for p in pair_flows if p in pair_alerts]
    quiet_pairs = [p for p in pair_flows if p not in pair_alerts]

    if quiet_pairs:
        nx.draw_networkx_edges(
            graph, positions, edgelist=quiet_pairs, ax=ax, edge_color=LIGHT,
            width=[width_of(p, 0.4, 1.6) for p in quiet_pairs],
            style="dashed", alpha=0.8)
    if alert_pairs:
        nx.draw_networkx_edges(
            graph, positions, edgelist=alert_pairs, ax=ax, edge_color=ACCENT,
            width=[width_of(p, 0.9, 4.2) for p in alert_pairs], style="solid")

    center_ids = [n["id"] for n in nodes if n["is_center"]]
    other_ids = [n["id"] for n in nodes if not n["is_center"]]
    nx.draw_networkx_nodes(graph, positions, nodelist=other_ids, ax=ax,
                           node_color="white", edgecolors=NEUTRAL,
                           linewidths=0.9, node_size=300)
    nx.draw_networkx_nodes(graph, positions, nodelist=center_ids, ax=ax,
                           node_color=ACCENT, edgecolors=ACCENT,
                           linewidths=1.2, node_size=480, node_shape="s")

    labels = {n["id"]: str(n["original_id"]) for n in nodes}
    nx.draw_networkx_labels(graph, positions, labels, ax=ax, font_size=7.5,
                            font_family=FONT_FAMILY[0])

    total_alert_flows = sum(pair_alerts.values())
    total_flows = sum(pair_flows.values())
    handles = [
        plt.Line2D([], [], color=ACCENT, marker="s", linestyle="none",
                   markersize=9, label="Đỉnh trung tâm"
                   + (f" ({center_ip})" if center_ip else "")),
        plt.Line2D([], [], color="white", marker="o", linestyle="none",
                   markersize=9, markeredgecolor=NEUTRAL, label="Đỉnh lân cận"),
        plt.Line2D([], [], color=ACCENT, linewidth=2.6,
                   label=f"Cặp máy có luồng bị cảnh báo: {len(alert_pairs)} cặp, "
                         f"{total_alert_flows:,} luồng".replace(",", ".")),
        plt.Line2D([], [], color=NEUTRAL, linewidth=0, label=(
            "Bề rộng theo trọng số giải thích" if dung_trong_so
            else "Bề rộng theo số luồng")),
        plt.Line2D([], [], color=LIGHT, linewidth=1.2, linestyle="dashed",
                   label=f"Cặp máy còn lại: {len(quiet_pairs)} cặp, "
                         f"{total_flows - total_alert_flows:,} luồng".replace(",", ".")),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.01), fontsize=7.5, frameon=False,
              ncol=1, handletextpad=0.8)
    ax.set_axis_off()
    ax.margins(0.10)
    fig.subplots_adjust(bottom=0.20)
    return fig, save_figure(fig, out_dir, name)


def export_table(rows: List[dict], out_dir: str, name: str,
                 columns: Optional[Sequence[str]] = None) -> str:
    """Xuất danh sách bản ghi thành tệp CSV."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    if not rows:
        return path
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    else:
        columns = list(columns)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_manifest(out_dir: str, payload: dict, name: str = "manifest") -> str:
    """Ghi bản kê kết quả thành tệp JSON."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path
