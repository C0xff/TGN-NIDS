"""Suy luận TGN và XAI cho dashboard."""

import os
import sys
import json
import zlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Đường dẫn dự án.
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(DASHBOARD_DIR))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
SAVED_MODELS_DIR = os.path.join(PROJECT_DIR, "models", "saved")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Ngưỡng hàng đợi xem xét.
DEFAULT_REVIEW_BUDGET_PCT = 5.0
UNCERTAIN_MARGIN_THRESHOLD = 0.20
MISSING_FEATURE_RATIO_THRESHOLD = 0.30

# Hàm định dạng số dùng chung.
from tgn_nids.utils.format import (
    number as _number,
    percent as _percent,
)


def format_exact_percent(value) -> str:
    # Định dạng tỷ lệ cho giao diện.
    return _percent(value, vietnamese=False)


def format_exact_number(value, unit: str = "") -> str:
    # Định dạng số đo.
    return _number(value, unit=unit, vietnamese=False)


# Cấu hình mô hình dashboard.
MODEL_CONFIGS = {
    "TGN-NIDS Multiclass": {
        "model_file": os.path.join(SAVED_MODELS_DIR, "twoDTS_train_v3", "08_giai_doan_hai", "models",
                                   "model_stage2_nf_unsw_nb15_v3.pt"),
        "metrics_file": os.path.join(SAVED_MODELS_DIR, "twoDTS_train_v3", "08_giai_doan_hai", "models",
                                     "result_stage2_nf_unsw_nb15_v3.json"),
        "task": "multiclass",
        "dataset_version": "v3",
        "role": "Mô hình chủ đạo, phân loại mười lớp tấn công",
    },
}

COMPARISON_MODEL_KEYS = list(MODEL_CONFIGS)

# Các mốc đối sánh đã công bố.
PUBLISHED_BASELINES = {
    # NF-UNSW-NB15-v3.
    "GraphIDS (NeurIPS 2025)\nNF-UNSW-NB15-v3": {
        "dataset": "NF-UNSW-NB15-v3",
        "f1_macro": 0.9961,
        "pr_auc": 0.9998,
        "note": "trung bình nhiều hạt giống, độ lệch chuẩn 0,0084",
    },
    "Anomal-E (KBS 2022)\nNF-UNSW-NB15-v3": {
        "dataset": "NF-UNSW-NB15-v3",
        "f1_macro": 0.9459,
        "pr_auc": 0.9032,
        "note": "do nhóm GraphIDS chạy lại, độ lệch chuẩn 0,0009",
    },
    "TE-G-SAGE (Modelling 2025)\nNF-UNSW-NB15-v3 đa lớp": {
        "dataset": "NF-UNSW-NB15-v3",
        "accuracy": 0.95586,
        "precision": 0.49419,
        "precision_macro": 0.49419,
        "recall": 0.62738,
        "recall_macro": 0.62738,
        "f1": 0.49057,
        "f1_macro": 0.49057,
        "fpr": 0.00446,
        "far": 0.00446,
        "note": "phân tách trình tự thời gian",
    },
    "TE-G-SAGE (Modelling 2025)\nNF-UNSW-NB15-v3 nhị phân": {
        "dataset": "NF-UNSW-NB15-v3",
        "precision": 0.9906,
        "recall": 0.9999,
        "f1": 0.9952,
        "f1_macro": 0.9952,
        "fpr": 0.0009,
        "far": 0.0009,
        "note": "phân tách trình tự thời gian",
    },
    # Mốc cùng điều kiện thí nghiệm với TE-G-SAGE.
    "XGBoost (Modelling 2025)\nNF-UNSW-NB15-v3 đa lớp": {
        "dataset": "NF-UNSW-NB15-v3",
        "accuracy": 0.97338,
        "precision": 0.68957,
        "precision_macro": 0.68957,
        "recall": 0.55742,
        "recall_macro": 0.55742,
        "f1": 0.56782,
        "f1_macro": 0.56782,
        "fpr": 0.00273,
        "far": 0.00273,
        "note": "mốc cơ sở dạng bảng, không dùng đồ thị",
    },
    "GCN (Modelling 2025)\nNF-UNSW-NB15-v3 đa lớp": {
        "dataset": "NF-UNSW-NB15-v3",
        "accuracy": 0.97257,
        "precision": 0.39587,
        "precision_macro": 0.39587,
        "recall": 0.39477,
        "recall_macro": 0.39477,
        "f1": 0.38781,
        "f1_macro": 0.38781,
        "fpr": 0.00301,
        "far": 0.00301,
        "note": "mốc đồ thị tĩnh, không có thuộc tính cạnh",
    },

    # NF-UNSW-NB15-v2.
    "GraphIDS (NeurIPS 2025)\nNF-UNSW-NB15-v2": {
        "dataset": "NF-UNSW-NB15-v2",
        "f1_macro": 0.9264,
        "pr_auc": 0.8116,
        "note": "trung bình nhiều hạt giống, độ lệch chuẩn 0,0217",
    },
    "Anomal-E (KBS 2022)\nNF-UNSW-NB15-v2": {
        "dataset": "NF-UNSW-NB15-v2",
        "accuracy": 0.9866,
        "f1_macro": 0.9235,
        "recall": 0.9877,
        "detection_rate": 0.9877,
        "note": "Bảng 4 của bài gốc, nhiễm bẩn 4%",
    },

    # NF-CSE-CIC-IDS2018 cho thí nghiệm mở rộng.
    "GraphIDS (NeurIPS 2025)\nNF-CSE-CIC-IDS2018-v3": {
        "dataset": "NF-CSE-CIC-IDS2018-v3",
        "f1_macro": 0.9447,
        "pr_auc": 0.8819,
        "note": "trung bình nhiều hạt giống, độ lệch chuẩn 0,0213",
    },
    "Anomal-E (KBS 2022)\nNF-CSE-CIC-IDS2018-v3": {
        "dataset": "NF-CSE-CIC-IDS2018-v3",
        "f1_macro": 0.6709,
        "pr_auc": 0.2555,
        "note": "do nhóm GraphIDS chạy lại, độ lệch chuẩn 0,0394",
    },
    "GraphIDS (NeurIPS 2025)\nNF-CSE-CIC-IDS2018-v2": {
        "dataset": "NF-CSE-CIC-IDS2018-v2",
        "f1_macro": 0.9431,
        "pr_auc": 0.9201,
        "note": "trung bình nhiều hạt giống, độ lệch chuẩn 0,0131",
    },
    "Anomal-E (KBS 2022)\nNF-CSE-CIC-IDS2018-v2": {
        "dataset": "NF-CSE-CIC-IDS2018-v2",
        "accuracy": 0.8979,
        "f1_macro": 0.8111,
        "recall": 0.9184,
        "detection_rate": 0.9184,
        "note": "Bảng 6 của bài gốc, nhiễm bẩn 4%",
    },
}


def load_saved_metrics(version: str) -> dict:
    # Đọc chỉ số từ tệp kết quả.
    config = MODEL_CONFIGS.get(version, {})
    metrics_path = config.get("metrics_file", "")
    if not os.path.exists(metrics_path):
        return {}

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[InferenceEngine] Không đọc được {metrics_path}: {error}")
        return {}

    m = data.get("metrics", {})
    task = config.get("task", "multiclass")
    res: dict = {}

    if task == "multiclass":
        op_argmax = data.get("operating_points", {}).get("argmax", {})
        res["accuracy"] = m.get("accuracy")
        res["precision"] = m.get("precision_macro")
        res["precision_macro"] = m.get("precision_macro")
        res["recall"] = m.get("recall_macro")
        res["recall_macro"] = m.get("recall_macro")
        res["f1"] = m.get("f1_macro")
        res["f1_macro"] = m.get("f1_macro")
        res["fpr"] = m.get("far_macro")
        res["far"] = m.get("far_macro")
        res["auc_roc"] = None
        res["pr_auc"] = None
        res["auc_roc_binary_equivalent"] = op_argmax.get("roc_auc")
        res["pr_auc_binary_equivalent"] = op_argmax.get("pr_auc")
        res["attack_recall"] = m.get("binary_equivalent", {}).get("recall")
    else:
        res["accuracy"] = m.get("accuracy")
        res["precision"] = m.get("precision")
        res["precision_macro"] = m.get("precision")
        res["recall"] = m.get("recall")
        res["recall_macro"] = m.get("recall")
        res["f1"] = m.get("f1")
        res["f1_macro"] = m.get("f1_macro", m.get("f1"))
        res["fpr"] = m.get("far", m.get("fpr"))
        res["far"] = m.get("far", m.get("fpr"))
        res["auc_roc"] = m.get("roc_auc")
        res["pr_auc"] = m.get("pr_auc")

    res["latency_ms_per_100_flows"] = data.get("latency_ms_per_100_flows")
    res["source_version"] = data.get("source_version")
    res["run_finished_at"] = data.get("run_finished_at")

    if "class_names" in data:
        res["class_names"] = data["class_names"]
    if "confusion_matrix" in m:
        res["confusion_matrix"] = m["confusion_matrix"]

    return {k: v for k, v in res.items() if v is not None}


def get_all_metrics_for_comparison() -> dict:
    # Gộp chỉ số mô hình và mốc công bố.
    result = {}
    for version in COMPARISON_MODEL_KEYS:
        m = load_saved_metrics(version)
        if m:
            result[version] = m
    for name, m in PUBLISHED_BASELINES.items():
        result[name] = m
    return result


class NIDSInferenceEngine:
    # Suy luận TGN và XAI cho dashboard.

    def __init__(self, version: str = "TGN-NIDS Multiclass", device: str = "cpu"):
        self.version = version
        self.device_str = device
        self.config = MODEL_CONFIGS.get(version, list(MODEL_CONFIGS.values())[0])
        self.is_model_loaded = False

        self.preprocessor = None
        self.class_names: List[str] = [
            'Analysis', 'Backdoor', 'Benign', 'DoS', 'Exploits',
            'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
        ]
        self.benign_class_id = 2
        self.node_mapping: Dict[str, int] = {}
        self.id_to_ip: Dict[int, str] = {}
        self.ip_to_id: Dict[str, int] = {}
        self.task = self.config.get("task", "multiclass")

        self.last_predict_used_real_model: Optional[bool] = None
        self.last_predict_fallback_reason: Optional[str] = None
        self.last_missing_features: List[str] = []

        self.last_needs_review_df: Optional[pd.DataFrame] = None
        self.last_needs_review_count: int = 0
        self.n_nodes: Optional[int] = None
        self._t_counter: float = 0.0
        self.last_sequence_source: Optional[str] = None

        self.feature_names = [
            'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
            'FLOW_DURATION_MILLISECONDS', 'L7_PROTO',
            'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
            'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL',
            'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
            'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
            'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
            'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
            'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
            'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
            'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
            'NUM_PKTS_1024_TO_1514_BYTES',
            'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
            'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE',
            'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE',
            'SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX', 'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
            'DST_TO_SRC_IAT_MIN', 'DST_TO_SRC_IAT_MAX', 'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV',
        ]

        self._try_load_model()

    def _try_load_model(self):
        # Nạp gói mô hình PyTorch.
        model_path = self.config.get("model_file", "")
        if not os.path.exists(model_path):
            print(f"[InferenceEngine] Model file not found: {model_path}. Using Demo Mode.")
            return

        try:
            import torch
            from tgn_nids.models.tgn_memory import TGNMemoryModule
            from tgn_nids.models.embedding import TemporalEmbeddingModule
            from tgn_nids.models.anomaly_detector import AnomalyDetector

            self.device = torch.device(self.device_str)
            try:
                checkpoint = torch.load(model_path, map_location=self.device,
                                        weights_only=True)
            except Exception as error:
                raise RuntimeError(
                    f"Không nạp được checkpoint {model_path} ở chế độ an toàn. "
                    f"Nếu tệp do bản mã cũ sinh ra thì nó chứa đối tượng đã "
                    f"pickle, phải chạy lại notebook để sinh bản mới."
                ) from error

            from tgn_nids.data.preprocess import NetFlowPreprocessor
            preprocessor_state = checkpoint.get("preprocessor_state")
            self.preprocessor = (NetFlowPreprocessor.from_state_dict(preprocessor_state)
                                 if preprocessor_state else None)
            if checkpoint.get("class_names") is not None:
                self.class_names = [str(c) for c in checkpoint["class_names"]]
            self.benign_class_id = checkpoint.get("benign_class_id", 2)
            self.node_mapping = checkpoint.get("node_mapping", {})
            self.id_to_ip = {idx: ip for ip, idx in self.node_mapping.items()}
            self.ip_to_id = self.node_mapping.copy()

            n_nodes = checkpoint.get("n_nodes", len(self.node_mapping) if self.node_mapping else 50)
            self.n_nodes = n_nodes
            edge_dim = checkpoint.get("edge_dim", 49)
            n_classes = checkpoint.get("n_classes", len(self.class_names))
            cfg = checkpoint.get("config", {})

            memory_dim = cfg.get("memory_dim", 128)
            time_dim = cfg.get("time_dim", 64)
            embedding_dim = cfg.get("embedding_dim", 128)
            hidden_dim = cfg.get("hidden_dim", 128)
            num_heads = cfg.get("num_heads", 4)
            num_layers = cfg.get("num_layers", 1)
            dropout = cfg.get("dropout", 0.3)
            use_edge_feats = cfg.get("classifier_edge_features", True)

            self.memory_module = TGNMemoryModule(
                num_nodes=n_nodes,
                raw_msg_dim=edge_dim,
                memory_dim=memory_dim,
                time_dim=time_dim,
            ).to(self.device)
            self.memory_module.load_state_dict(checkpoint["memory_state"])
            self._memory_state_goc = {
                k: v.clone() if hasattr(v, "clone") else v
                for k, v in checkpoint["memory_state"].items()
            }

            self.embedding_module = TemporalEmbeddingModule(
                memory_dim=memory_dim,
                embedding_dim=embedding_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            ).to(self.device)
            self.embedding_module.load_state_dict(checkpoint["embedding_state"])

            self.detector_module = AnomalyDetector(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_classes=n_classes,
                dropout=dropout,
                edge_dim=edge_dim if use_edge_feats else 0,
            ).to(self.device)
            self.detector_module.load_state_dict(checkpoint["head_state"])

            self.memory_module.eval()
            self.embedding_module.eval()
            self.detector_module.eval()
            self.is_model_loaded = True
            print(f"[InferenceEngine] Model {self.version} ({n_classes} classes) loaded successfully!")

        except Exception as e:
            print(f"[InferenceEngine] Cannot load PyTorch model ({e}). Using Demo Mode.")
            self.is_model_loaded = False

    def _build_ip_mapping(self, df: pd.DataFrame):
        # Tạo ánh xạ endpoint và node.
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        src_col = cols_lower.get("ipv4_src_addr", cols_lower.get("srcip", cols_lower.get("src_ip", None)))
        dst_col = cols_lower.get("ipv4_dst_addr", cols_lower.get("dstip", cols_lower.get("dst_ip", None)))

        if not src_col or not dst_col:
            if "L4_SRC_PORT" in df.columns and "L4_DST_PORT" in df.columns:
                df["IPV4_SRC_ADDR"] = "ANONYMOUS_SRC_PORT_" + df["L4_SRC_PORT"].astype(str)
                df["IPV4_DST_ADDR"] = "ANONYMOUS_DST_PORT_" + df["L4_DST_PORT"].astype(str)
            else:
                df["IPV4_SRC_ADDR"] = "ANONYMOUS_SRC_FLOW_" + (df.index % 40).astype(str)
                df["IPV4_DST_ADDR"] = "ANONYMOUS_DST_FLOW_" + ((df.index + 1) % 40).astype(str)
            df["endpoint_identity_source"] = "synthetic"
            src_col, dst_col = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
        elif "endpoint_identity_source" not in df.columns:
            df["endpoint_identity_source"] = "observed"

        all_ips = set(df[src_col].astype(str).unique()) | set(df[dst_col].astype(str).unique())
        for ip in sorted(all_ips):
            if ip in self.ip_to_id:
                continue
            if self.n_nodes:
                new_id = zlib.crc32(ip.encode("utf-8")) % self.n_nodes
            else:
                new_id = len(self.ip_to_id)
            self.ip_to_id[ip] = new_id
            if new_id not in self.id_to_ip:
                self.id_to_ip[new_id] = ip

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Suy luận trên dữ liệu luồng.
        result_df = df.copy().reset_index(drop=True)
        cols_lower = {str(c).strip().lower(): c for c in result_df.columns}
        src_col = cols_lower.get("ipv4_src_addr", cols_lower.get("srcip", cols_lower.get("src_ip", None)))
        dst_col = cols_lower.get("ipv4_dst_addr", cols_lower.get("dstip", cols_lower.get("dst_ip", None)))

        if not src_col or not dst_col:
            if "L4_SRC_PORT" in result_df.columns and "L4_DST_PORT" in result_df.columns:
                result_df["IPV4_SRC_ADDR"] = "ANONYMOUS_SRC_PORT_" + result_df["L4_SRC_PORT"].astype(str)
                result_df["IPV4_DST_ADDR"] = "ANONYMOUS_DST_PORT_" + result_df["L4_DST_PORT"].astype(str)
            else:
                result_df["IPV4_SRC_ADDR"] = "ANONYMOUS_SRC_FLOW_" + (result_df.index % 40).astype(str)
                result_df["IPV4_DST_ADDR"] = "ANONYMOUS_DST_FLOW_" + ((result_df.index + 1) % 40).astype(str)
            result_df["endpoint_identity_source"] = "synthetic"
        else:
            result_df["endpoint_identity_source"] = "observed"

        self._build_ip_mapping(result_df)

        if self.is_model_loaded:
            try:
                out = self._predict_with_model(result_df)
                self.last_predict_used_real_model = True
                self.last_predict_fallback_reason = None
                self.last_missing_features = list(self.preprocessor.last_missing_features) if self.preprocessor is not None else []
                out["inference_mode"] = "real_model"
                self._update_needs_review_queue(out)
                return out
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                print(f"[InferenceEngine] Error in _predict_with_model: {reason}. Fallback Demo Mode.")
                self.last_predict_used_real_model = False
                self.last_predict_fallback_reason = reason
                fallback = self._predict_demo_mode(result_df)
                fallback["inference_mode"] = "demo_fallback"
                self._update_needs_review_queue(fallback)
                return fallback
        else:
            self.last_predict_used_real_model = False
            self.last_predict_fallback_reason = "Mô hình TGN không nạp được (is_model_loaded=False)"
            fallback = self._predict_demo_mode(result_df)
            fallback["inference_mode"] = "demo_fallback"
            self._update_needs_review_queue(fallback)
            return fallback

    def _update_needs_review_queue(self, out_df: pd.DataFrame, max_rows: int = 200) -> None:
        # Cập nhật hàng đợi cần xem xét.
        if "needs_review" not in out_df.columns:
            self.last_needs_review_df = None
            self.last_needs_review_count = 0
            return
        flagged = out_df[out_df["needs_review"]]
        self.last_needs_review_count = int(len(flagged))
        sort_col = "uncertainty_margin" if "uncertainty_margin" in flagged.columns else None
        if sort_col is not None:
            flagged = flagged.sort_values(sort_col, ascending=True)
        self.last_needs_review_df = flagged.head(max_rows).copy()

    def predict_batch(self, batch_df: pd.DataFrame, accumulated_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Suy luận một lô và gộp kết quả.
        self._build_ip_mapping(batch_df)
        batch_result = self.predict(batch_df)

        if accumulated_df is None or len(accumulated_df) == 0:
            accumulated_result = batch_result.copy()
        else:
            accumulated_result = pd.concat([accumulated_df, batch_result], ignore_index=True)

        return batch_result, accumulated_result

    def reset_memory(self) -> None:
        # Khôi phục bộ nhớ từ checkpoint.
        if self.is_model_loaded and hasattr(self, "memory_module"):
            goc = getattr(self, "_memory_state_goc", None)
            if goc is not None:
                self.memory_module.load_state_dict(goc)
            else:
                self.memory_module.reset_state()
        self._t_counter = 0.0

    def _predict_with_model(self, df: pd.DataFrame) -> pd.DataFrame:
        # Suy luận bằng mô hình TGN.
        import torch

        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        src_col = cols_lower.get("ipv4_src_addr", cols_lower.get("srcip", "IPV4_SRC_ADDR"))
        dst_col = cols_lower.get("ipv4_dst_addr", cols_lower.get("dstip", "IPV4_DST_ADDR"))

        if self.preprocessor is not None:
            processed_df = self.preprocessor.transform(df)
            feature_cols = [c for c in self.preprocessor.feature_names if c in processed_df.columns]
            features = processed_df[feature_cols].values.astype(np.float32)
            total_expected = len(self.preprocessor.feature_names)
            n_missing = len(getattr(self.preprocessor, "last_missing_features", []))
            missing_ratio = (n_missing / total_expected) if total_expected else 0.0
        else:
            features = np.zeros((len(df), 49), dtype=np.float32)
            missing_ratio = 1.0

        src_ids = df[src_col].astype(str).map(self.ip_to_id).fillna(0).astype(int).values
        dst_ids = df[dst_col].astype(str).map(self.ip_to_id).fillna(0).astype(int).values

        with torch.no_grad():
            src_t = torch.tensor(src_ids, dtype=torch.long, device=self.device)
            dst_t = torch.tensor(dst_ids, dtype=torch.long, device=self.device)
            edge_attr = torch.tensor(features, dtype=torch.float32, device=self.device)

            n_id = torch.unique(torch.cat([src_t, dst_t]))
            assoc = torch.full((int(n_id.max().item()) + 1,), -1, dtype=torch.long, device=self.device)
            assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

            memory, _last_update = self.memory_module(n_id)
            local_edge_index = torch.stack([assoc[src_t], assoc[dst_t]], dim=0)

            z = self.embedding_module(memory, local_edge_index, edge_attr)
            z_src = z[assoc[src_t]]
            z_dst = z[assoc[dst_t]]

            t_np = None
            time_col = cols_lower.get("flow_start_milliseconds", None)
            if time_col is not None:
                parsed = pd.to_numeric(df[time_col], errors="coerce")
                if parsed.notna().all():
                    t_np = parsed.values.astype(np.int64)
                    self.last_sequence_source = "FLOW_START_MILLISECONDS"
            if t_np is None:
                t_np = np.arange(int(self._t_counter),
                                 int(self._t_counter) + len(df), dtype=np.int64)
                self._t_counter += len(df)
                self.last_sequence_source = "chỉ số thứ tự sự kiện (không phải thời gian)"
            t = torch.tensor(t_np, dtype=torch.long, device=self.device)
            self.memory_module.update_state(src_t, dst_t, t, edge_attr)
            self.memory_module.detach_memory()

            logits = self.detector_module(z_src, z_dst, edge_feat=edge_attr)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            if self.task == "multiclass" and len(self.class_names) == probs.shape[1]:
                pred_indices = np.argmax(probs, axis=1)
                df["prediction_label"] = [self.class_names[idx] for idx in pred_indices]
                df["prediction"] = (pred_indices != self.benign_class_id).astype(int)
                df["confidence"] = np.max(probs, axis=1)
                df["anomaly_score"] = 1.0 - probs[:, self.benign_class_id]
            else:
                if probs.shape[1] >= 2:
                    df["prediction"] = np.argmax(probs, axis=1)
                    df["confidence"] = np.max(probs, axis=1)
                    df["anomaly_score"] = probs[:, 1]
                else:
                    df["prediction"] = (probs[:, 0] > 0.5).astype(int)
                    df["confidence"] = probs[:, 0]
                    df["anomaly_score"] = probs[:, 0]
                df["prediction_label"] = np.where(df["prediction"] == 1, "Attack", "Benign")

            self._add_uncertainty_columns(df, probs, missing_ratio=missing_ratio,
                                          review_budget_pct=getattr(self, 'review_budget_pct', DEFAULT_REVIEW_BUDGET_PCT))

        return df

    @staticmethod
    def _add_uncertainty_columns(df: pd.DataFrame, probs: np.ndarray, missing_ratio: float = 0.0,
                                 review_budget_pct: float = DEFAULT_REVIEW_BUDGET_PCT) -> None:
        # Bổ sung độ tin cậy và cờ xem xét.
        n_classes = probs.shape[1]
        missing_ratio = float(missing_ratio)
        missing_flag = missing_ratio > MISSING_FEATURE_RATIO_THRESHOLD

        if n_classes < 2:
            df["uncertainty_margin"] = 1.0
            df["uncertainty_entropy"] = 0.0
            df["missing_features_ratio"] = round(missing_ratio, 4)
            df["needs_review"] = np.full(len(df), bool(missing_flag))
            return

        sorted_probs = np.sort(probs, axis=1)
        top1 = sorted_probs[:, -1]
        top2 = sorted_probs[:, -2]
        margin = (top1 - top2).round(4)

        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        normalized_entropy = (entropy / np.log(n_classes)).round(4)

        df["uncertainty_margin"] = margin
        df["uncertainty_entropy"] = normalized_entropy
        df["missing_features_ratio"] = round(missing_ratio, 4)

        n = len(margin)
        uncertain = margin < UNCERTAIN_MARGIN_THRESHOLD
        n_uncertain = int(uncertain.sum())

        budget = float(review_budget_pct)
        capacity = max(0, int(round(n * budget / 100.0)))

        flag = np.zeros(n, dtype=bool)
        if n_uncertain > 0:
            idx = np.flatnonzero(uncertain)
            if len(idx) > capacity:
                order = idx[np.lexsort((-normalized_entropy[idx], margin[idx]))]
                idx = order[:capacity]
            flag[idx] = True

        df["needs_review"] = flag | missing_flag
        df.attrs["review_n_uncertain"] = n_uncertain
        df.attrs["review_capacity"] = capacity
        df.attrs["review_threshold"] = float(UNCERTAIN_MARGIN_THRESHOLD)
        df.attrs["review_overflow"] = max(0, n_uncertain - capacity)

    def _predict_demo_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        # Trả khung rỗng khi suy luận thất bại.
        n = len(df)
        reason = getattr(self, "last_predict_fallback_reason", None) or (
            "mô hình không nạp được")
        df["prediction"] = pd.Series([pd.NA] * n, dtype="Int64")
        df["confidence"] = np.nan
        df["anomaly_score"] = np.nan
        df["prediction_label"] = None
        df["needs_review"] = False
        df["inference_unavailable_reason"] = reason
        self.last_needs_review_df = None
        self.last_needs_review_count = 0
        self.last_sequence_source = None
        return df

    def get_xai_results(
        self,
        df: pd.DataFrame,
        target_node_id: int,
        k_hops: int = 1,
        target_ip: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Trích xuất đồ thị con và độ quan trọng đặc trưng.
        self._build_ip_mapping(df)

        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        src_col = cols_lower.get("ipv4_src_addr", cols_lower.get("srcip", "IPV4_SRC_ADDR"))
        dst_col = cols_lower.get("ipv4_dst_addr", cols_lower.get("dstip", "IPV4_DST_ADDR"))

        all_ips = sorted(
            set(df[src_col].astype(str).unique()) | set(df[dst_col].astype(str).unique())
        ) if src_col in df.columns and dst_col in df.columns else []
        display_ip_to_id = {ip: idx for idx, ip in enumerate(all_ips)}
        display_id_to_ip = {idx: ip for ip, idx in display_ip_to_id.items()}
        if target_ip is None:
            target_ip = self.id_to_ip.get(target_node_id, f"Node-{target_node_id}")
        target_ip = str(target_ip)
        display_target_id = display_ip_to_id.get(target_ip, target_node_id)

        connected_nodes = set([display_target_id])
        edge_list = []
        hop_of: Dict[str, int] = {target_ip: 0}

        if src_col in df.columns and dst_col in df.columns:
            adjacency: Dict[str, set] = {}
            for s_ip, d_ip in zip(df[src_col].astype(str), df[dst_col].astype(str)):
                adjacency.setdefault(s_ip, set()).add(d_ip)
                adjacency.setdefault(d_ip, set()).add(s_ip)

            frontier = {target_ip}
            for step in range(1, max(1, int(k_hops)) + 1):
                nxt = set()
                for ip in frontier:
                    nxt |= adjacency.get(ip, set())
                nxt -= hop_of.keys()
                if not nxt:
                    break
                for ip in nxt:
                    hop_of[ip] = step
                frontier = nxt
            self.last_xai_hop_nodes = len(hop_of)

            for _, row in df.iterrows():
                s_ip, d_ip = str(row[src_col]), str(row[dst_col])
                s_id = display_ip_to_id.get(s_ip, -1)
                d_id = display_ip_to_id.get(d_ip, -1)

                s_hop = hop_of.get(s_ip)
                d_hop = hop_of.get(d_ip)
                nearest = min([h for h in (s_hop, d_hop) if h is not None], default=None)
                if nearest is not None and nearest <= int(k_hops) - 1:
                    if s_id != -1:
                        connected_nodes.add(s_id)
                    if d_id != -1:
                        connected_nodes.add(d_id)

                    is_anomaly = bool(row.get("prediction", 0) == 1)

                    _nhan = row.get("Label", None)
                    try:
                        nhan_that = (None if _nhan is None or pd.isna(_nhan)
                                     else bool(int(_nhan) == 1))
                    except (TypeError, ValueError):
                        nhan_that = None

                    attack_name = str(row.get("prediction_label", "Normal"))
                    if attack_name.lower() in ["benign", "normal", "0"]:
                        attack_name = "Normal Traffic"

                    loai_that = str(row.get("Attack", "")) or None
                    if loai_that and loai_that.lower() in ["benign", "normal", "0"]:
                        loai_that = None

                    _sc = row.get("anomaly_score", None)
                    try:
                        attn = float(_sc) if _sc is not None and not pd.isna(_sc) else None
                    except (TypeError, ValueError):
                        attn = None

                    edge_list.append({
                        "source": int(s_id),
                        "target": int(d_id),
                        "src_ip": s_ip,
                        "dst_ip": d_ip,
                        "is_anomaly": is_anomaly,
                        "anomaly_score": float(attn) if attn is not None else None,
                        "attack_type": attack_name,
                        "nhan_that": nhan_that,
                        "loai_that": loai_that,
                    })

        sorted_edges = sorted(edge_list, key=lambda x: (not x.get("is_anomaly", False), -(x.get("anomaly_score") or 0.0)))

        nodes_data = []
        for nid in sorted(connected_nodes):
            ip_str = display_id_to_ip.get(nid, f"Node-{nid}")
            is_target = (nid == display_target_id or ip_str == target_ip)

            sends_attack = any((e["source"] == nid or e["src_ip"] == ip_str) and e["is_anomaly"] for e in edge_list)
            receives_attack = any((e["target"] == nid or e["dst_ip"] == ip_str) and e["is_anomaly"] for e in edge_list)

            if is_target:
                role = "TARGET"
            elif sends_attack:
                role = "ATTACKER"
            elif receives_attack:
                role = "VICTIM"
            else:
                role = "SAFE"
            label = f"{ip_str}\n[{role}]"

            nodes_data.append({
                "id": int(nid),
                "ip": str(ip_str),
                "is_center": is_target,
                "role": role,
                "label": label,
                "hop": int(hop_of.get(ip_str, 99)),
            })

        KEY_NEIGHBOURS = 10
        RING_PER_NODE = 4

        def _peer(e):
            return e["dst_ip"] if e["src_ip"] == target_ip else e["src_ip"]

        direct_edges = [e for e in sorted_edges
                        if e["src_ip"] == target_ip or e["dst_ip"] == target_ip]

        rank = {}
        for e in direct_edges:
            ip = _peer(e)
            score = (1 if e.get("is_anomaly") else 0, float(e.get("anomaly_score") or 0.0))
            if ip not in rank or score > rank[ip]:
                rank[ip] = score
        key_ips = [ip for ip, _ in sorted(rank.items(), key=lambda kv: kv[1], reverse=True)][:KEY_NEIGHBOURS]
        key_set = set(key_ips)

        shown_edges = [e for e in direct_edges if _peer(e) in key_set]

        ring_of = dict.fromkeys(key_ips, 1)
        da_them = {id(e) for e in shown_edges}
        vanh_truoc = list(key_ips)
        for vanh in range(2, max(1, int(k_hops)) + 1):
            vanh_moi = []
            for ip in vanh_truoc:
                ung_vien = [
                    e for e in sorted_edges
                    if ip in (e["src_ip"], e["dst_ip"])
                    and target_ip not in (e["src_ip"], e["dst_ip"])
                    and id(e) not in da_them
                ]

                def _da_biet(e, _ip=ip):
                    other = e["dst_ip"] if e["src_ip"] == _ip else e["src_ip"]
                    return other in ring_of

                for e in sorted(ung_vien, key=_da_biet)[:RING_PER_NODE]:
                    other = e["dst_ip"] if e["src_ip"] == ip else e["src_ip"]
                    shown_edges.append(e)
                    da_them.add(id(e))
                    if other not in ring_of:
                        ring_of[other] = vanh
                        vanh_moi.append(other)
            if not vanh_moi:
                break
            vanh_truoc = vanh_moi

        top_attacks = [
            {
                "src_ip": e["src_ip"], "dst_ip": e["dst_ip"],
                "attack_type": e.get("attack_type", "Unknown"),
                "anomaly_score": (float(e["anomaly_score"]) if e.get("anomaly_score") is not None else None),
                "peer": _peer(e),
            }
            for e in direct_edges if e.get("is_anomaly")
        ][:5]

        visible_ids = {display_target_id}
        for e in shown_edges:
            visible_ids.add(e["source"]); visible_ids.add(e["target"])
        nodes_data = [n for n in nodes_data if n["id"] in visible_ids]

        for n in nodes_data:
            if n["id"] == display_target_id:
                n["hop"] = 0
            else:
                n["hop"] = ring_of.get(n["ip"], max(1, int(k_hops)))

        subgraph_data = {
            "center_node": int(display_target_id),
            "center_ip": target_ip,
            "nodes": nodes_data,
            "edges": shown_edges,
            "num_nodes": len(nodes_data),
            "num_edges": len(shown_edges),
            "num_pairs": len({(e["src_ip"], e["dst_ip"]) for e in shown_edges}),
            "so_bo_sot": sum(1 for e in shown_edges
                             if e.get("nhan_that") is True and not e["is_anomaly"]),
            "so_bao_nham": sum(1 for e in shown_edges
                               if e.get("nhan_that") is False and e["is_anomaly"]),
            "co_nhan_that": any(e.get("nhan_that") is not None for e in shown_edges),
            "max_hop": max([n["hop"] for n in nodes_data], default=0),
            "num_edges_total": len(sorted_edges),
            "edges_truncated": len(sorted_edges) > len(shown_edges),
            "k_hops": int(k_hops),
            "num_direct_total": len(direct_edges),
            "key_neighbours": key_ips,
            "top_attacks": top_attacks,
        }

        ego_ips = set(hop_of.keys()) | {target_ip}
        if src_col in df.columns and dst_col in df.columns:
            mask = df[src_col].astype(str).isin(ego_ips) | df[dst_col].astype(str).isin(ego_ips)
            df_ego = df[mask]
        else:
            df_ego = df.iloc[0:0]
        importance, imp_meta = self.compute_feature_importance(df_ego)

        return {
            "subgraph": subgraph_data,
            "feature_importance": importance,
            "feature_importance_meta": imp_meta,
            "ip_map": display_id_to_ip,
        }

    EXPLAIN_MAX_EDGES = 4000

    def compute_feature_importance(self, df_sub: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        # Đo độ quan trọng bằng cách triệt tiêu đặc trưng.
        meta: Dict[str, Any] = {"method": "zero-out perturbation", "available": False}

        if not self.is_model_loaded or self.preprocessor is None:
            meta["reason"] = (
                "Chế độ Demo Mode không nạp trọng số mô hình nên không thể đo "
                "độ quan trọng đặc trưng. Cần chạy Full Mode với gói .pt."
            )
            return {}, meta
        if df_sub is None or len(df_sub) == 0:
            meta["reason"] = "Đồ thị con không chứa luồng nào để đo."
            return {}, meta

        try:
            import torch
            from tgn_nids.explainers.gnn_explainer import TGNExplainer

            df_ex = df_sub if len(df_sub) <= self.EXPLAIN_MAX_EDGES else df_sub.iloc[: self.EXPLAIN_MAX_EDGES]

            cols_lower = {str(c).strip().lower(): c for c in df_ex.columns}
            src_col = cols_lower.get("ipv4_src_addr", cols_lower.get("srcip", "IPV4_SRC_ADDR"))
            dst_col = cols_lower.get("ipv4_dst_addr", cols_lower.get("dstip", "IPV4_DST_ADDR"))

            processed = self.preprocessor.transform(df_ex)
            feature_cols = [c for c in self.preprocessor.feature_names if c in processed.columns]
            features = processed[feature_cols].values.astype(np.float32)

            src_ids = df_ex[src_col].astype(str).map(self.ip_to_id).fillna(0).astype(int).values
            dst_ids = df_ex[dst_col].astype(str).map(self.ip_to_id).fillna(0).astype(int).values

            with torch.no_grad():
                src_t = torch.tensor(src_ids, dtype=torch.long, device=self.device)
                dst_t = torch.tensor(dst_ids, dtype=torch.long, device=self.device)
                edge_attr = torch.tensor(features, dtype=torch.float32, device=self.device)

                n_id = torch.unique(torch.cat([src_t, dst_t]))
                assoc = torch.full((int(n_id.max().item()) + 1,), -1, dtype=torch.long, device=self.device)
                assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

                memory, _ = self.memory_module(n_id)
                local_edge_index = torch.stack([assoc[src_t], assoc[dst_t]], dim=0)
                z = self.embedding_module(memory, local_edge_index, edge_attr)
                z_src = z[assoc[src_t]]
                z_dst = z[assoc[dst_t]]

            explainer = TGNExplainer(
                self.memory_module, self.embedding_module, self.detector_module,
                device=str(self.device),
                benign_class_id=self.benign_class_id,
            )
            raw = explainer.compute_feature_importance(z_src, z_dst, edge_attr, feature_cols)
        except Exception as exc:
            meta["reason"] = f"Không đo được độ quan trọng đặc trưng: {exc}"
            return {}, meta

        missing = set(getattr(self.preprocessor, "last_missing_features", []) or [])
        importance = {k: round(float(v), 4) for k, v in raw.items() if float(v) > 0.0}
        pct_filled = round(sum(v for k, v in importance.items() if k in missing), 2)

        meta.update({
            "available": True,
            "n_edges_measured": int(len(df_ex)),
            "n_features_probed": int(len(feature_cols)),
            "n_features_nonzero": len(importance),
            "filled_features": sorted(k for k in importance if k in missing),
            "pct_from_filled": pct_filled,
        })
        return importance, meta

    def get_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        # Trích xuất bảng cảnh báo cho giao diện.
        if "prediction" not in df.columns:
            return pd.DataFrame()

        attacks = df[df["prediction"] == 1].copy()
        if len(attacks) == 0:
            return pd.DataFrame()

        cols_lower = {str(c).strip().lower(): c for c in attacks.columns}
        src_col = cols_lower.get("ipv4_src_addr", cols_lower.get("srcip", "IPV4_SRC_ADDR"))
        dst_col = cols_lower.get("ipv4_dst_addr", cols_lower.get("dstip", "IPV4_DST_ADDR"))
        port_col = cols_lower.get("l4_dst_port", "L4_DST_PORT")

        _unknown = pd.Series(["không có trong dữ liệu"] * len(attacks), index=attacks.index)
        src_series = attacks[src_col] if src_col in attacks.columns else attacks.get("IPV4_SRC_ADDR", _unknown)
        dst_series = attacks[dst_col] if dst_col in attacks.columns else attacks.get("IPV4_DST_ADDR", _unknown)
        port_series = (
            attacks[port_col] if port_col in attacks.columns
            else pd.Series([None] * len(attacks), index=attacks.index)
        )

        synthetic = str(attacks.get("endpoint_identity_source", pd.Series(["observed"])).iloc[0]) == "synthetic"
        src_label = "Source (anonymous)" if synthetic else "Source IP"
        dst_label = "Destination (anonymous)" if synthetic else "Destination IP"

        alert_data = {
            src_label: src_series.values,
            dst_label: dst_series.values,
            "Port": port_series.values,
            "Attack Type": attacks["prediction_label"].values if "prediction_label" in attacks.columns else attacks.get("Attack", ["Attack"] * len(attacks)),
            "Confidence": [format_exact_percent(c) for c in attacks["confidence"].values] if "confidence" in attacks.columns else ["—"] * len(attacks),
            "Anomaly Score": [format_exact_number(v) for v in attacks["anomaly_score"].values] if "anomaly_score" in attacks.columns else ["—"] * len(attacks),
        }
        return pd.DataFrame(alert_data)
