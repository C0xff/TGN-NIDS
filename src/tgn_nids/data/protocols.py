"""Các giao thức chia tập dùng trong thí nghiệm TGN-NIDS."""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Cột thời gian chuẩn của các bộ NetFlow v3.
TIME_COLUMN = "FLOW_START_MILLISECONDS"


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    """Tìm cột biểu diễn thứ tự thời gian, nếu có."""
    for c in df.columns:
        if c.strip().upper() == TIME_COLUMN:
            return c
    for c in df.columns:
        if c.strip().lower() in ("timestamp", "stime", "ts"):
            return c
    return None


def _find_label_column(df: pd.DataFrame) -> str:
    """Tìm cột nhãn nhị phân mà không phân biệt hoa thường."""
    for c in df.columns:
        if c.strip().lower() == "label":
            return c
    raise KeyError("Không tìm thấy cột Label trong dữ liệu")


def _restore_temporal_order(part: pd.DataFrame) -> pd.DataFrame:
    """Khôi phục thứ tự thời gian sau khi lấy mẫu phân tầng."""
    time_col = _find_time_column(part)
    if time_col is not None:
        return part.sort_values(time_col, kind="mergesort").copy()
    return part.sort_index(kind="mergesort").copy()


def split_te_g_sage(df: pd.DataFrame,
                    ratios: Tuple[float, float, float] = (0.60, 0.30, 0.10)
                    ) -> Dict[str, pd.DataFrame]:
    """Chia liên tục theo thời gian như giao thức TE-G-SAGE."""
    time_col = _find_time_column(df)
    if time_col is None:
        raise KeyError(
            "Dữ liệu không có cột mốc thời gian tuyệt đối. Giao thức này "
            "yêu cầu chia theo thời gian nên không áp dụng được cho phiên bản v2."
        )
    ordered = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    n = len(ordered)

    cut_1 = int(round(n * ratios[0]))
    cut_2 = int(round(n * (ratios[0] + ratios[1])))

    return {
        "train": ordered.iloc[:cut_1].copy(),
        "val": ordered.iloc[cut_1:cut_2].copy(),
        "test": ordered.iloc[cut_2:].copy()
    }


def split_graphids(df: pd.DataFrame,
                   ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10),
                   seed: int = 42,
                   stratify_column: str = "Attack",
                   benign_only_train: bool = True) -> Dict[str, pd.DataFrame]:
    """Chia phân tầng theo GraphIDS, có thể chỉ giữ Benign ở tập huấn luyện."""
    rng = np.random.default_rng(seed)
    column = (stratify_column if stratify_column in df.columns
              else _find_label_column(df))

    parts = {"train": [], "val": [], "test": []}

    for _, group in df.groupby(column, sort=True):
        indices = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(indices)

        n_group = len(indices)
        cut_1 = int(round(n_group * ratios[0]))
        cut_2 = int(round(n_group * (ratios[0] + ratios[1])))

        parts["train"].append(indices[:cut_1])
        parts["val"].append(indices[cut_1:cut_2])
        parts["test"].append(indices[cut_2:])

    result = {k: _restore_temporal_order(df.loc[np.concatenate(v)])
              for k, v in parts.items()}

    if benign_only_train:
        label_col = _find_label_column(result["train"])
        before = len(result["train"])
        result["train"] = result["train"][result["train"][label_col] == 0].copy()
        result["_removed_from_train"] = before - len(result["train"])

    return result


def split_random_stratified(df: pd.DataFrame,
                            ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10),
                            seed: int = 42,
                            stratify_column: str = "Attack"
                            ) -> Dict[str, pd.DataFrame]:
    """Chia phân tầng nhưng giữ nguyên mẫu tấn công trong tập huấn luyện."""
    return split_graphids(df, ratios=ratios, seed=seed,
                          stratify_column=stratify_column,
                          benign_only_train=False)


def split_anomal_e(df: pd.DataFrame,
                   sample_ratio: float = 0.10,
                   train_ratio: float = 0.70,
                   seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Lấy mẫu rồi chia Train/Test theo giao thức Anomal-E."""
    rng = np.random.default_rng(seed)
    indices = np.array(df.index.to_numpy(), copy=True)
    rng.shuffle(indices)

    indices = indices[:int(len(indices) * sample_ratio)]
    cut = int(len(indices) * train_ratio)
    return {"train": _restore_temporal_order(df.loc[indices[:cut]]),
              "test": _restore_temporal_order(df.loc[indices[cut:]])}


# Bảng tra dùng bởi bộ điều phối thí nghiệm.
PROTOCOLS = {
    "te_g_sage": split_te_g_sage,
    "graphids": split_graphids,
    "anomal_e": split_anomal_e,
    "random_stratified": split_random_stratified,
}

# Các giao thức phù hợp với bài toán học có giám sát.
PROTOCOLS_WITH_ATTACKS_IN_TRAIN = {"te_g_sage", "anomal_e", "random_stratified"}


def summarize(splits: Dict[str, pd.DataFrame]) -> str:
    """Tạo bảng văn bản tóm tắt phân bố nhãn của từng tập."""
    lines = [f"{'Tập':<8}{'Số luồng':>12}{'Tấn công':>11}{'Tỉ lệ':>9}"]
    for name, part in splits.items():
        if not isinstance(part, pd.DataFrame):
            continue
        try:
            n_attack = int((part[_find_label_column(part)] == 1).sum())
        except KeyError:
            n_attack = -1
        lines.append(f"{name:<8}{len(part):>12,}{n_attack:>11,}"
                     f"{n_attack / max(1, len(part)) * 100:>8.2f}%")
    return "\n".join(lines)
