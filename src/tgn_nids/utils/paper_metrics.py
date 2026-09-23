"""Tính các chỉ số đánh giá theo định nghĩa của các bài báo đối sánh."""

from typing import Dict, List, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Các chỉ số có thể đối chiếu trực tiếp với từng bài báo.
METRICS_BY_PAPER: Dict[str, set] = {
    "GraphIDS": {"pr_auc", "f1_macro"},
    "TE-G-SAGE": {"accuracy", "precision", "recall", "f1", "far", "fpr",
                  "tnr", "tpr", "f1_macro", "precision_macro", "recall_macro",
                  "far_macro"},
    "Anomal-E": {"accuracy", "f1_macro", "detection_rate", "recall"},
}


def _safe_divide(numerator, denominator) -> float:
    """Chia an toàn và trả về 0 khi mẫu số bằng 0."""
    return float(numerator) / float(denominator) if denominator else 0.0


def _precision_or_none(true_positive: int, false_positive: int):
    """Trả về `None` khi mô hình chưa dự đoán mẫu nào thuộc lớp."""
    if true_positive + false_positive == 0:
        return None
    return float(true_positive) / float(true_positive + false_positive)


def evaluate_binary(y_true, y_pred, y_score=None) -> dict:
    """Tính bộ chỉ số cho bài toán phân loại nhị phân."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    recall = _safe_divide(tp, tp + fn)
    tnr = _safe_divide(tn, tn + fp)
    fpr = _safe_divide(fp, fp + tn)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (recall + tnr) / 2.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": recall,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "tnr": tnr,
        "fnr": _safe_divide(fn, fn + tp),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": cm.tolist(),
        "support_attack": int(tp + fn),
        "support_benign": int(tn + fp),
    }

    # Giữ các bí danh để đối chiếu đúng thuật ngữ của từng bài báo.
    metrics["tpr"] = recall
    metrics["detection_rate"] = recall
    metrics["fpr"] = fpr
    metrics["far"] = fpr

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) > 1:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        else:
            metrics["pr_auc"] = None
            metrics["roc_auc"] = None
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None

    return metrics


def evaluate_multiclass(y_true, y_pred, class_names: Sequence[str],
                        benign_class_id: int, y_score=None) -> dict:
    """Tính chỉ số đa lớp, từng lớp và bản quy đổi nhị phân."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels,
                                           average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels,
                                                    average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels,
                                              average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "benign_class_id": int(benign_class_id),
    }

    per_class: List[dict] = []
    far_values: List[float] = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        far = _safe_divide(fp, fp + tn)
        far_values.append(far)
        per_class.append({
            "name": name,
            "precision": _precision_or_none(tp, fp),
            "recall": _safe_divide(tp, tp + fn),
            "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
            "far": far,
            "support": int(cm[i, :].sum()),
            "predicted_count": int(tp + fp),
        })
    metrics["per_class"] = per_class
    metrics["far_macro"] = float(np.mean(far_values)) if far_values else 0.0

    # Gộp mọi lớp không phải Benign thành nhóm tấn công.
    binary_true = (y_true != benign_class_id).astype(int)
    binary_pred = (y_pred != benign_class_id).astype(int)

    binary_score = None
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if y_score.ndim == 2 and y_score.shape[1] == n_classes:
            binary_score = 1.0 - y_score[:, benign_class_id]

    metrics["binary_equivalent"] = evaluate_binary(binary_true, binary_pred, binary_score)

    return metrics
