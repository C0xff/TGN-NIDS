"""Kiểm thử bộ chỉ số bao trùm cả ba bài báo đối sánh."""

import numpy as np
import pytest

from tgn_nids.utils.paper_metrics import (
    evaluate_binary,
    evaluate_multiclass,
    METRICS_BY_PAPER,
)


def test_nhi_phan_du_moi_chi_so_cac_bai_bao_dung():
    """Phải có đủ mọi chỉ số mà ba bài báo công bố, không thiếu cái nào."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.9, 0.8, 0.6, 0.4])

    m = evaluate_binary(y_true, y_pred, y_score)

    bat_buoc = {
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "f1_macro", "f1_weighted", "tpr", "tnr", "fpr", "far",
        "detection_rate", "pr_auc", "roc_auc",
        "tp", "tn", "fp", "fn", "confusion_matrix",
    }
    thieu = bat_buoc - set(m)
    assert not thieu, f"Thiếu chỉ số: {sorted(thieu)}"


def test_nhi_phan_tinh_dung_theo_dinh_nghia():
    """Đối chiếu với giá trị tính tay."""
    #            TN TN TN FP TP TP TP FN
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    m = evaluate_binary(y_true, y_pred)

    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["tnr"] == pytest.approx(3 / 4)
    assert m["fpr"] == pytest.approx(1 / 4)
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 0.75) / 2)


def test_far_bang_fpr_va_detection_rate_bang_recall():
    """Các bí danh trong bài báo phải trỏ đến đúng cùng đại lượng."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    m = evaluate_binary(y_true, y_pred)

    assert m["far"] == m["fpr"]
    assert m["detection_rate"] == m["recall"]
    assert m["tpr"] == m["recall"]


def test_da_lop_du_chi_so_va_bang_tung_lop():
    """Đa lớp phải có macro, weighted, và bảng từng lớp kèm FAR riêng."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0])
    class_names = ["Benign", "DoS", "Exploits"]

    m = evaluate_multiclass(y_true, y_pred, class_names, benign_class_id=0)

    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
              "far_macro", "per_class", "confusion_matrix", "binary_equivalent"):
        assert k in m, f"Thiếu {k}"

    assert len(m["per_class"]) == 3
    for lop in m["per_class"]:
        for k in ("name", "precision", "recall", "f1", "far", "support"):
            assert k in lop


def test_far_tung_lop_tinh_theo_kieu_mot_chong_tat_ca():
    """FAR từng lớp phải được tính theo cách một-chống-tất-cả."""
    # Lớp 1: có 2 mẫu thật. Trong 5 mẫu không thuộc lớp 1, có 1 bị gán nhầm.
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0])
    m = evaluate_multiclass(y_true, y_pred, ["A", "B", "C"], benign_class_id=0)

    lop_b = next(l for l in m["per_class"] if l["name"] == "B")
    assert lop_b["far"] == pytest.approx(1 / 5)


def test_quy_doi_nhi_phan_gop_moi_lop_khac_benign():
    """Quy đổi nhị phân: mọi lớp khác Benign đều tính là tấn công."""
    y_true = np.array([0, 0, 1, 2, 2])   # Benign=0
    y_pred = np.array([0, 1, 2, 2, 0])
    m = evaluate_multiclass(y_true, y_pred, ["Benign", "X", "Y"], benign_class_id=0)

    be = m["binary_equivalent"]
    assert (be["tn"], be["fp"], be["fn"], be["tp"]) == (1, 1, 1, 2)


def test_benign_khong_phai_lop_0():
    """LabelEncoder sắp theo chữ cái nên Benign thường không phải lớp 0."""
    y_true = np.array([2, 2, 0, 1, 2])   # Benign = 2
    y_pred = np.array([2, 0, 0, 1, 1])
    m = evaluate_multiclass(y_true, y_pred, ["Analysis", "DoS", "Benign"],
                        benign_class_id=2)

    be = m["binary_equivalent"]
    assert (be["tn"], be["fp"], be["fn"], be["tp"]) == (1, 2, 0, 2)


def test_bang_tra_cuu_chi_so_theo_bai_bao():
    """Phải biết bài nào công bố chỉ số nào, để bảng đối sánh ghi đúng."""
    assert "pr_auc" in METRICS_BY_PAPER["GraphIDS"]
    assert "far" in METRICS_BY_PAPER["TE-G-SAGE"]
    assert "detection_rate" in METRICS_BY_PAPER["Anomal-E"]
    assert "accuracy" not in METRICS_BY_PAPER["GraphIDS"]


def test_khong_vo_khi_chi_co_mot_lop():
    """Tập chỉ có một lớp không được làm chương trình chết."""
    y = np.zeros(5, dtype=int)
    m = evaluate_binary(y, y)
    assert m["tp"] == 0 and m["tn"] == 5
    assert m["recall"] == 0.0
