"""Kiểm thử các ràng buộc bảo vệ của bộ điều phối thí nghiệm."""

import numpy as np
import pandas as pd
import pytest

from tgn_nids.data.preprocess import NetFlowPreprocessor
from tgn_nids.data.protocols import (split_graphids, split_random_stratified,
                                PROTOCOLS, PROTOCOLS_WITH_ATTACKS_IN_TRAIN)
from tgn_nids.experiments.runner import ExperimentConfig, _validate_config


ATTACK_TYPES = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
                "Generic", "Reconnaissance", "Shellcode", "Worms"]


def make_frame(n_benign: int = 900, n_per_attack: int = 20) -> pd.DataFrame:
    """Dựng bảng dữ liệu tối giản có đủ mười nhãn của NF-UNSW-NB15."""
    rng = np.random.default_rng(0)
    attacks = [t for t in ATTACK_TYPES for _ in range(n_per_attack)]
    labels = ["Benign"] * n_benign + attacks
    n = len(labels)
    return pd.DataFrame({
        "IPV4_SRC_ADDR": [f"10.0.0.{i % 20}" for i in range(n)],
        "IPV4_DST_ADDR": [f"10.0.1.{i % 15}" for i in range(n)],
        "L4_SRC_PORT": rng.integers(1024, 65535, n),
        "L4_DST_PORT": rng.integers(1, 1024, n),
        "IN_BYTES": rng.integers(40, 90000, n),
        "OUT_BYTES": rng.integers(40, 90000, n),
        "FLOW_START_MILLISECONDS": np.arange(n) * 1000,
        "Attack": labels,
        "Label": [0] * n_benign + [1] * len(attacks),
    })


class TestProtocolGuard:
    """Giao thức loại bỏ mẫu tấn công không dùng được cho học có giám sát."""

    def test_giao_thuc_graphids_bi_chan_khi_hoc_co_giam_sat(self):
        config = ExperimentConfig(name="t", description="t",
                                  protocol="graphids", mode="supervised")
        with pytest.raises(ValueError, match="loại sạch mẫu tấn công"):
            _validate_config(config)

    def test_giao_thuc_graphids_van_dung_duoc_cho_tu_giam_sat(self):
        config = ExperimentConfig(name="t", description="t",
                                  protocol="graphids", mode="self_supervised")
        _validate_config(config)

    def test_giao_thuc_anomal_e_bi_chan_khi_can_hieu_chinh_nguong(self):
        # Hiệu chỉnh trên tập test sẽ làm rò rỉ thông tin đánh giá.
        config = ExperimentConfig(name="t", description="t",
                                  protocol="anomal_e", threshold_strategy="far")
        with pytest.raises(ValueError, match="rò rỉ thông tin đánh giá"):
            _validate_config(config)

    def test_moi_giao_thuc_co_giam_sat_deu_duoc_khai_bao(self):
        assert PROTOCOLS_WITH_ATTACKS_IN_TRAIN <= set(PROTOCOLS)
        assert "graphids" not in PROTOCOLS_WITH_ATTACKS_IN_TRAIN


class TestRandomStratified:
    """Giao thức chia ngẫu nhiên giữ nguyên mẫu tấn công trong tập huấn luyện."""

    def test_giu_lai_mau_tan_cong_trong_tap_huan_luyen(self):
        splits = split_random_stratified(make_frame(), seed=7)
        assert int((splits["train"]["Label"] == 1).sum()) > 0
        assert "_removed_from_train" not in splits

    def test_giu_du_muoi_lop_trong_tap_huan_luyen(self):
        splits = split_random_stratified(make_frame(n_per_attack=40), seed=7)
        assert splits["train"]["Attack"].nunique() == 10

    def test_cung_ranh_gioi_voi_giao_thuc_graphids(self):
        # Hai giao thức chỉ khác nhau ở bước lọc mẫu tấn công khỏi tập train.
        frame = make_frame()
        a = split_graphids(frame, seed=7)
        b = split_random_stratified(frame, seed=7)
        assert len(a["val"]) == len(b["val"])
        assert len(a["test"]) == len(b["test"])
        assert len(b["train"]) == len(a["train"]) + a["_removed_from_train"]


class TestClassVocabulary:
    """Danh mục nhãn được chốt độc lập với tập huấn luyện."""

    def test_tap_huan_luyen_thieu_lop_van_giu_du_danh_muc(self):
        frame = make_frame()
        train_only_benign = frame[frame["Label"] == 0]
        preprocessor = NetFlowPreprocessor(
            class_vocabulary=sorted(frame["Attack"].unique()))
        preprocessor.fit(train_only_benign)
        assert len(preprocessor.label_encoder.classes_) == 10
        assert preprocessor.benign_class_id == 2

    def test_khong_chot_danh_muc_thi_nhan_that_bi_sua(self):
        # Danh mục học từ tập chỉ có Benign sẽ làm mất mọi nhãn tấn công.
        frame = make_frame()
        preprocessor = NetFlowPreprocessor()
        preprocessor.fit(frame[frame["Label"] == 0])
        transformed = preprocessor.transform(frame)
        assert len(preprocessor.label_encoder.classes_) == 1
        assert preprocessor.unknown_label_count == 180
        assert transformed["Attack_encoded"].nunique() == 1

    def test_chot_danh_muc_thi_khong_con_nhan_bi_sua(self):
        frame = make_frame()
        preprocessor = NetFlowPreprocessor(
            class_vocabulary=sorted(frame["Attack"].unique()))
        preprocessor.fit(frame[frame["Label"] == 0])
        transformed = preprocessor.transform(frame)
        assert preprocessor.unknown_label_count == 0
        assert transformed["Attack_encoded"].nunique() == 10


class TestAblation:
    """Cơ chế ablation phải thật sự cắt được nguồn thông tin tương ứng."""

    def test_loai_cot_khoi_edge_features(self):
        frame = make_frame()
        preprocessor = NetFlowPreprocessor()
        before = len(preprocessor.fit(frame).feature_names)

        ablated = NetFlowPreprocessor()
        ablated.features_to_drop = (list(ablated.features_to_drop)
                                    + ["MIN_TTL", "MAX_TTL"])
        # Thêm hai cột TTL để kiểm tra chúng bị loại đúng theo cấu hình.
        frame["MIN_TTL"] = 32
        frame["MAX_TTL"] = 64
        assert len(NetFlowPreprocessor().fit(frame).feature_names) == before + 2
        assert len(ablated.fit(frame).feature_names) == before

    def test_khai_bao_ablation_duoc_ghi_vao_cau_hinh(self):
        config = ExperimentConfig(
            name="t", description="t", protocol="te_g_sage",
            ablate_node_identity=True, ablate_features=("MIN_TTL", "MAX_TTL"))
        _validate_config(config)
        assert config.ablate_node_identity is True
        assert config.ablate_features == ("MIN_TTL", "MAX_TTL")

    def test_gan_lai_node_ngau_nhien_cat_duoc_lien_he_voi_nhan(self):
        # Gán lại đỉnh phải phá liên hệ giữa địa chỉ nguồn và nhãn.
        import numpy as np
        frame = make_frame(n_benign=500, n_per_attack=50)
        labels = frame["Label"].to_numpy()
        generator = np.random.default_rng(42)
        shuffled = generator.integers(0, 20, len(labels))
        by_node = {}
        for node, label in zip(shuffled, labels):
            by_node.setdefault(node, []).append(label)
        pure = sum(1 for v in by_node.values() if len(set(v)) == 1 and len(v) > 5)
        assert pure == 0, "Sau khi gán lại ngẫu nhiên không node nào được thuần nhãn"


class TestColumnSetsAgree:
    """Hai danh sách cột loại trừ độc lập phải luôn cho ra cùng một tập đặc trưng."""

    def test_preprocessor_va_graph_builder_chon_cung_mot_tap_cot(self):
        # Mọi cột đưa vào đồ thị phải được bộ tiền xử lý chuẩn hóa trước.
        import inspect
        import re
        from tgn_nids.data.graph_builder import TemporalGraphBuilder

        frame = make_frame()
        preprocessor = NetFlowPreprocessor()
        transformed = preprocessor.fit_transform(frame)

        source = inspect.getsource(TemporalGraphBuilder.build_pyg_temporal_data)
        block = re.search(r"exclude_lower = \{(.*?)\}", source, re.S).group(1)
        excluded = set(re.findall(r"'([^']+)'", block))
        builder_columns = {c for c in transformed.columns
                           if str(c).strip().lower() not in excluded}

        assert builder_columns == set(preprocessor.feature_names), (
            "Chỉ có ở builder: "
            f"{sorted(builder_columns - set(preprocessor.feature_names))} | "
            "Chỉ có ở preprocessor: "
            f"{sorted(set(preprocessor.feature_names) - builder_columns)}")


class TestSampleRatio:
    """Lấy mẫu phân tầng phải giữ đủ mọi lớp và tôn trọng tỉ lệ yêu cầu."""

    def test_giu_du_moi_lop_sau_khi_lay_mau(self):
        frame = make_frame(n_benign=1000, n_per_attack=100)
        reduced = (frame.groupby("Attack", group_keys=False)
                        .sample(frac=0.30, random_state=42))
        assert reduced["Attack"].nunique() == frame["Attack"].nunique()

    def test_ti_le_tung_lop_duoc_bao_toan(self):
        frame = make_frame(n_benign=1000, n_per_attack=100)
        reduced = (frame.groupby("Attack", group_keys=False)
                        .sample(frac=0.30, random_state=42))
        goc = frame["Attack"].value_counts(normalize=True).sort_index()
        moi = reduced["Attack"].value_counts(normalize=True).sort_index()
        assert (goc - moi).abs().max() < 0.01

    def test_mac_dinh_giu_nguyen_toan_bo_du_lieu(self):
        # Chỉ lấy mẫu khi cấu hình khai báo rõ tỷ lệ nhỏ hơn 1.
        config = ExperimentConfig(name="t", description="t", protocol="te_g_sage")
        assert config.sample_ratio == 1.0


class TestTemporalOrdering:
    """Mọi tập sau khi chia phải giữ trình tự thời gian, không xếp theo lớp."""

    @staticmethod
    def frame_dan_xen(n_benign=2000, n_per_attack=200):
        """Tạo bảng có các lớp đan xen theo thứ tự thời gian."""
        import numpy as np
        frame = make_frame(n_benign=n_benign, n_per_attack=n_per_attack)
        shuffled = np.random.default_rng(0).permutation(len(frame))
        frame["FLOW_START_MILLISECONDS"] = shuffled * 1000
        return frame

    def test_chia_phan_tang_khong_xep_theo_lop(self):
        # Mỗi lô phải còn nhiều lớp sau khi khôi phục thứ tự thời gian.
        frame = self.frame_dan_xen()
        for name, part in split_random_stratified(frame, seed=5).items():
            labels = part["Attack"].to_numpy()
            per_batch = [len(set(labels[i:i + 64]))
                         for i in range(0, len(labels) - 64, 64)]
            assert sum(per_batch) / len(per_batch) > 1.5, (
                f"Tập {name} xếp theo lớp, trung bình "
                f"{sum(per_batch) / len(per_batch):.2f} lớp mỗi lô")

    def test_moi_tap_tang_dan_theo_moc_thoi_gian(self):
        frame = make_frame()
        for splitter in (split_random_stratified, split_graphids):
            splits = splitter(frame, seed=5)
            splits.pop("_removed_from_train", None)
            for name, part in splits.items():
                times = part["FLOW_START_MILLISECONDS"].to_numpy()
                assert (times[1:] >= times[:-1]).all(), f"{name} không tăng dần"

    def test_sap_lai_khong_lam_doi_thanh_phan_cac_tap(self):
        # Sắp lại chỉ thay đổi thứ tự, không thay đổi thành phần các tập.
        frame = make_frame()
        splits = split_random_stratified(frame, seed=5)
        total = sum(len(v) for v in splits.values())
        assert total == len(frame)
        ids = set()
        for part in splits.values():
            ids |= set(map(tuple, part[["L4_SRC_PORT", "IN_BYTES"]].to_numpy()))
        assert len(ids) > 0


class TestInferenceArtifacts:
    """Trọng số mô hình một mình không đủ để chấm điểm luồng mới."""

    def test_artifacts_chua_du_thanh_phan_can_thiet(self, tmp_path):
        # Gói mô hình phải đủ thông tin để khôi phục toàn bộ pipeline suy luận.
        import torch
        from tgn_nids.experiments.runner import run_experiment
        from tgn_nids.data.preprocess import NetFlowPreprocessor

        frame = make_frame(n_benign=600, n_per_attack=60)
        config = ExperimentConfig(
            name="artifact_probe", description="t", protocol="te_g_sage",
            task="multiclass", memory_dim=16, embedding_dim=16, time_dim=8,
            num_heads=2, hidden_dim=16, batch_size=128, num_epochs=1,
            early_stopping_patience=1)
        run_experiment(frame, config, NetFlowPreprocessor(), str(tmp_path),
                       verbose=False)

        bundle = torch.load(tmp_path / "model_artifact_probe.pt",
                            map_location="cpu", weights_only=True)
        assert {"memory_state", "embedding_state", "head_state",
                "preprocessor_state", "node_mapping", "class_names",
                "benign_class_id", "edge_dim", "n_nodes", "n_classes",
                "threshold", "source_version"} <= set(bundle)
        assert bundle["preprocessor_state"]["fitted"]
        assert len(bundle["node_mapping"]) > 0
        assert len(bundle["class_names"]) == len(
            bundle["preprocessor_state"]["label_encoder_state"]["classes_"]["__array__"])

        # Lưu dự đoán để có thể vẽ lại hình mà không huấn luyện lại.
        import numpy as np
        predictions = np.load(tmp_path / "predictions_artifact_probe.npz")
        assert {"y_true", "y_pred", "y_score", "src", "dst", "t"} <= set(
            predictions.files)
        assert len(predictions["y_true"]) > 0



class TestPreprocessorStateDict:
    """Trạng thái đã lưu phải khôi phục đúng cùng một phép biến đổi."""

    @staticmethod
    def _frame():
        rng = np.random.default_rng(0)
        rows = 400
        return pd.DataFrame({
            "IPV4_SRC_ADDR": [f"10.0.0.{i % 7}" for i in range(rows)],
            "IPV4_DST_ADDR": [f"10.0.1.{i % 5}" for i in range(rows)],
            "L4_SRC_PORT": rng.integers(1024, 65535, rows),
            "L4_DST_PORT": rng.integers(1, 1024, rows),
            "PROTOCOL": rng.integers(1, 18, rows),
            "IN_BYTES": rng.integers(40, 10 ** 6, rows),
            "OUT_BYTES": rng.integers(40, 10 ** 6, rows),
            "IN_PKTS": rng.integers(1, 5000, rows),
            "OUT_PKTS": rng.integers(1, 5000, rows),
            "FLOW_DURATION_MILLISECONDS": rng.integers(0, 10 ** 5, rows),
            "Label": rng.integers(0, 2, rows),
            "Attack": rng.choice(["Benign", "DoS", "Exploits"], rows),
        })

    def test_dung_lai_cho_ra_cung_ket_qua(self):
        frame = self._frame()
        goc = NetFlowPreprocessor(class_vocabulary=["Benign", "DoS", "Exploits"])
        goc.fit(frame)
        lai = NetFlowPreprocessor.from_state_dict(goc.state_dict())

        a = goc.transform(frame.copy())
        b = lai.transform(frame.copy())
        assert list(a.columns) == list(b.columns)
        numeric = a.select_dtypes(include=[np.number]).columns
        np.testing.assert_array_equal(a[numeric].to_numpy(), b[numeric].to_numpy())
        assert goc.benign_class_id == lai.benign_class_id
        assert list(goc.feature_names) == list(lai.feature_names)

    def test_ban_ke_nap_duoc_o_che_do_an_toan(self, tmp_path):
        """Bản kê chỉ được chứa kiểu dữ liệu mà `weights_only` chấp nhận."""
        import torch

        goc = NetFlowPreprocessor(class_vocabulary=["Benign", "DoS", "Exploits"])
        goc.fit(self._frame())
        duong_dan = tmp_path / "state.pt"
        torch.save({"preprocessor_state": goc.state_dict()}, duong_dan)

        doc_lai = torch.load(duong_dan, map_location="cpu", weights_only=True)
        lai = NetFlowPreprocessor.from_state_dict(doc_lai["preprocessor_state"])
        np.testing.assert_array_equal(goc.scaler.scale_, lai.scaler.scale_)

    def test_tu_choi_ban_ke_sai_phien_ban(self):
        with pytest.raises(ValueError):
            NetFlowPreprocessor.from_state_dict({"format_version": 99})
