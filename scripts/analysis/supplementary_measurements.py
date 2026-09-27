#!/usr/bin/env python3
"""
Các phép đo bổ trợ trên NF-UNSW-NB15-v3 mà báo cáo có công bố (mục 4.9.1).

Script chỉ đọc: nạp dữ liệu gốc và dùng lại đúng giao thức chia tập cùng bộ
tiền xử lý của dự án, không huấn luyện lại mô hình TGN, không ghi vào
`models/saved/`. Kết quả ghi ra `models/supplementary/supplementary_measurements.json`, rồi
`scripts/analysis/tong_hop_ket_qua.py` đọc tệp đó để dựng mục *Các phép đo bổ
trợ* của `docs/KET_QUA_DO_LUONG.md`.

Hai nhóm phép đo:

1. Lối tắt trong dữ liệu: số địa chỉ nguồn theo loại lưu lượng, giá trị TTL
   phổ biến của hai lớp và ROC-AUC của riêng cột MAX_TTL, không huấn luyện gì.
2. Mốc tham chiếu dạng bảng, không dùng đồ thị: cây quyết định sâu 3 (đủ 49
   thuộc tính và bỏ hai cột TTL) và hồi quy logistic, theo giao thức chia tập
   TE-G-SAGE để đặt cạnh số liệu của bài báo đó.

    python scripts/analysis/supplementary_measurements.py

Chạy trên CPU, khoảng nửa phút.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tgn_nids.data.preprocess import NetFlowPreprocessor  # noqa: E402
from tgn_nids.data.protocols import split_te_g_sage  # noqa: E402

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "nf-unsw-nb15-v3", "NF-UNSW-NB15-v3.parquet")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "supplementary", "supplementary_measurements.json")
SEED = 42
TTL_COLUMNS = ("MIN_TTL", "MAX_TTL")


def binary_scores(model, x_train, y_train, x_test, y_test):
    """Huấn luyện trên tập train, đo F1, FAR và PR-AUC của lớp tấn công trên tập test."""
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]
    false_positive = int(((predicted == 1) & (y_test == 0)).sum())
    true_negative = int(((predicted == 0) & (y_test == 0)).sum())
    return {
        "attack_f1": float(f1_score(y_test, predicted)),
        "far": false_positive / (false_positive + true_negative),
        "pr_auc": float(average_precision_score(y_test, score)),
    }


def main():
    started = time.time()
    frame = pd.read_parquet(DATA_PATH)
    labels = frame["Label"].astype(int)

    # ---- 1. lối tắt trong dữ liệu ----
    by_source = frame.groupby("IPV4_SRC_ADDR")["Label"].agg(["min", "max"])
    attack_only = sorted(by_source.index[by_source["min"] == 1].tolist())
    splits = split_te_g_sage(frame)
    shortcuts = {
        "condition": "Toàn bộ 2.365.424 luồng, không chia tập, không huấn luyện gì. TTL dùng giá trị thô "
                     "chưa chuẩn hoá, ROC-AUC lấy chính giá trị MAX_TTL làm điểm của lớp tấn công.",
        "n_flows": int(len(frame)),
        "attack_only_sources": attack_only,
        "n_benign_only_sources": int((by_source["max"] == 0).sum()),
        "n_mixed_sources": int(((by_source["min"] == 0) & (by_source["max"] == 1)).sum()),
        "max_ttl_mode_benign": int(frame.loc[labels == 0, "MAX_TTL"].mode()[0]),
        "max_ttl_mode_attack": int(frame.loc[labels == 1, "MAX_TTL"].mode()[0]),
        "max_ttl_roc_auc": float(roc_auc_score(labels, frame["MAX_TTL"])),
    }

    # ---- 2. mốc tham chiếu dạng bảng ----
    vocabulary = sorted(frame["Attack"].unique().tolist())
    preprocessor = NetFlowPreprocessor(class_vocabulary=vocabulary).fit(splits["train"])
    features = list(preprocessor.feature_names)
    matrix = {name: preprocessor.transform(part)[features].values.astype(np.float32)
              for name, part in splits.items()}
    target = {name: part["Label"].values.astype(int) for name, part in splits.items()}
    without_ttl = [i for i, c in enumerate(features) if c not in TTL_COLUMNS]

    def run(model, columns=None):
        cols = slice(None) if columns is None else columns
        return binary_scores(model, matrix["train"][:, cols], target["train"],
                             matrix["test"][:, cols], target["test"])

    baselines = {
        "condition": "Giao thức TE-G-SAGE của dự án (split_te_g_sage): sắp theo FLOW_START_MILLISECONDS, "
                     "chia liên tục 60/30/10, huấn luyện trên tập train, đo trên tập test. Thuộc tính qua "
                     "NetFlowPreprocessor khớp trên tập train, không có địa chỉ IP và mốc thời gian. Nhãn là "
                     "cột Label hai lớp. Tập validation không dùng. Ngưỡng mặc định của predict().",
        "splits": {name: {"n_flows": int(len(part)), "n_attacks": int(part["Label"].sum())}
                     for name, part in splits.items()},
        "n_features": len(features),
        "decision_tree_depth3_all_features": {
            "model": f"DecisionTreeClassifier(max_depth=3, random_state={SEED})",
            **run(DecisionTreeClassifier(max_depth=3, random_state=SEED))},
        "decision_tree_depth3_without_ttl": {
            "model": f"DecisionTreeClassifier(max_depth=3, random_state={SEED}), bỏ MIN_TTL và MAX_TTL",
            **run(DecisionTreeClassifier(max_depth=3, random_state=SEED), without_ttl)},
        "logistic_regression_all_features": {
            "model": "LogisticRegression(max_iter=1000), tham số còn lại mặc định",
            **run(LogisticRegression(max_iter=1000))},
    }

    result = {
        "description": "Phép đo bổ trợ mà báo cáo công bố ở mục 4.9.1, bảng 4.24.",
        "dataset": os.path.relpath(DATA_PATH, PROJECT_ROOT).replace(os.sep, "/"),
        "scikit_learn": sklearn.__version__,
        "data_shortcuts": shortcuts,
        "tabular_baselines": baselines,
        "runtime_seconds": round(time.time() - started, 1),
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Đã ghi: {OUTPUT_PATH}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
