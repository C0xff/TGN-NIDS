"""Làm sạch, chuẩn hóa đặc trưng và mã hóa nhãn NetFlow."""

from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class NetFlowPreprocessor:
    """Học phép biến đổi trên tập huấn luyện và áp dụng lại cho tập khác."""

    def __init__(
        self,
        features_to_drop: Optional[List[str]] = None,
        use_log_transform: bool = True,
        scaler_type: str = 'standard',
        class_vocabulary: Optional[List[str]] = None,
    ):
        # Cổng mạng vẫn là đặc trưng số; chỉ loại định danh và mốc thời gian.
        self.features_to_drop = features_to_drop or [
            'IPV4_SRC_ADDR', 'IPV4_DST_ADDR',
            'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
            'timestamp', 'timestamp_end', 'srcip', 'dstip',
            'stime', 'ts', 'start_time'
        ]
        self.use_log_transform = use_log_transform
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.class_vocabulary = list(class_vocabulary) if class_vocabulary else None
        self.feature_names = []
        self.log_transform_cols = []
        self.fitted = False
        self.unknown_label_count = 0
        self.last_missing_features: List[str] = []

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Chọn các cột số không chứa định danh, thời gian hoặc nhãn."""
        drop_set = set(col.upper() for col in self.features_to_drop)
        drop_set.update(['LABEL', 'ATTACK', 'ATTACK_CAT', 'ATTACK_CATEGORY', 'ATTACK_ENCODED'])
        feature_cols = [col for col in df.columns if col.upper() not in drop_set]
        return df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    def _clean_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Thay giá trị thiếu hoặc vô hạn bằng 0."""
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def _robust_skew(column: pd.Series) -> float:
        """Ước lượng độ lệch trên thang log để hạn chế tràn số."""
        transformed = np.log1p(np.maximum(0, column.astype(float)))
        skewness = transformed.skew()
        return 0.0 if not np.isfinite(skewness) else float(skewness)

    def fit(self, df: pd.DataFrame) -> 'NetFlowPreprocessor':
        """Học danh sách đặc trưng, phép biến đổi và bộ mã hóa nhãn."""
        df_work = df.copy()
        self.feature_names = self._get_feature_cols(df_work)
        X = self._clean_numeric(df_work[self.feature_names].copy())

        # Chỉ biến đổi log các cột có độ lệch lớn.
        if self.use_log_transform:
            skewness = {col: self._robust_skew(X[col]) for col in X.columns}
            self.log_transform_cols = [col for col, value in skewness.items()
                                       if value > 1.5]
            for col in self.log_transform_cols:
                X[col] = np.log1p(np.maximum(0, X[col]))
        else:
            self.log_transform_cols = []

        unusable = [col for col in X.columns
                    if not np.isfinite(X[col].astype(float).var())]
        if unusable:
            raise ValueError(
                f"Không tính được phương sai hữu hạn cho các cột {unusable}. "
                f"Giá trị lớn nhất của cột đầu tiên là "
                f"{X[unusable[0]].max():.6e}. Chuẩn hoá thang đo trên cột như "
                f"vậy sẽ triệt tiêu nó thành hằng số không.")

        self.scaler.fit(X)

        attack_col = next((c for c in df_work.columns if c.upper() in ['ATTACK', 'ATTACK_CAT']), None)
        if attack_col:
            if self.class_vocabulary:
                self.label_encoder.fit(np.array(self.class_vocabulary, dtype=object).astype(str))
            else:
                self.label_encoder.fit(df_work[attack_col].astype(str))

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng đúng các tham số đã học lên một bảng NetFlow mới."""
        if not self.fitted:
            raise RuntimeError("Preprocessor chưa được fit! Vui lòng gọi fit() trước khi transform.")

        df_out = df.copy()

        # Giữ nguyên thứ tự và số chiều đặc trưng đã chốt khi fit.
        missing = [c for c in self.feature_names if c not in df_out.columns]
        self.last_missing_features = missing
        for col in missing:
            df_out[col] = 0.0

        X = self._clean_numeric(df_out[self.feature_names].copy())

        for col in self.log_transform_cols:
            X[col] = np.log1p(np.maximum(0, X[col]))

        X_scaled = self.scaler.transform(X)
        df_out[self.feature_names] = X_scaled

        # Nhãn chưa biết được đưa về Benign và được đếm để bên gọi kiểm tra.
        attack_col = next((c for c in df_out.columns if c.upper() in ['ATTACK', 'ATTACK_CAT']), None)
        if attack_col:
            known_classes = set(self.label_encoder.classes_)
            raw_attacks = df_out[attack_col].astype(str)
            unknown_mask = ~raw_attacks.isin(known_classes)
            self.unknown_label_count = int(unknown_mask.sum())
            safe_attacks = raw_attacks.where(~unknown_mask, 'Benign')
            df_out['Attack_encoded'] = self.label_encoder.transform(safe_attacks)

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Học tham số rồi biến đổi cùng một bảng dữ liệu."""
        return self.fit(df).transform(df)

    @staticmethod
    def _fitted_attributes(estimator) -> Dict:
        """Lấy các thuộc tính đã học theo quy ước của scikit-learn."""
        return {name: value for name, value in vars(estimator).items()
                if name.endswith("_") and not name.startswith("_")}

    def state_dict(self) -> Dict:
        """Xuất trạng thái đã học thành cấu trúc chỉ gồm kiểu dữ liệu an toàn."""
        def plain(value):
            if isinstance(value, np.ndarray):
                return {"__array__": value.tolist(), "dtype": str(value.dtype)}
            if isinstance(value, np.generic):
                return value.item()
            return value

        return {
            "format_version": 1,
            "features_to_drop": list(self.features_to_drop),
            "use_log_transform": bool(self.use_log_transform),
            "scaler_type": str(self.scaler_type),
            "class_vocabulary": (list(self.class_vocabulary)
                                 if self.class_vocabulary else None),
            "feature_names": list(self.feature_names),
            "log_transform_cols": list(self.log_transform_cols),
            "fitted": bool(self.fitted),
            "scaler_state": {k: plain(v)
                             for k, v in self._fitted_attributes(self.scaler).items()},
            "label_encoder_state": {
                k: plain(v)
                for k, v in self._fitted_attributes(self.label_encoder).items()},
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> 'NetFlowPreprocessor':
        """Khôi phục bộ tiền xử lý từ trạng thái đã xuất."""
        def restore(value):
            if isinstance(value, dict) and "__array__" in value:
                return np.asarray(value["__array__"], dtype=value["dtype"])
            return value

        version = state.get("format_version")
        if version != 1:
            raise ValueError(
                f"Bản kê tham số tiền xử lý ở phiên bản {version!r}, mã này chỉ "
                f"đọc được phiên bản 1.")

        instance = cls(features_to_drop=list(state["features_to_drop"]),
                       use_log_transform=state["use_log_transform"],
                       scaler_type=state["scaler_type"],
                       class_vocabulary=state.get("class_vocabulary"))
        instance.feature_names = list(state["feature_names"])
        instance.log_transform_cols = list(state["log_transform_cols"])
        instance.fitted = state["fitted"]
        for name, value in state["scaler_state"].items():
            setattr(instance.scaler, name, restore(value))
        for name, value in state["label_encoder_state"].items():
            setattr(instance.label_encoder, name, restore(value))
        return instance

    @property
    def benign_class_id(self) -> int:
        """Trả về vị trí của lớp Benign trong bộ mã hóa nhãn."""
        classes = list(getattr(self.label_encoder, 'classes_', []))
        return classes.index('Benign') if 'Benign' in classes else 0
