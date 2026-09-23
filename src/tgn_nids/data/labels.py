"""Quy đổi nhãn luồng mạng về dạng nhị phân 0 (lành tính) hoặc 1 (tấn công)."""

from typing import Any

import pandas as pd

# Các cách ghi nhãn được chấp nhận trong dữ liệu đầu vào.
BENIGN_TOKENS = frozenset({"0", "0.0", "benign", "normal", ""})
ATTACK_TOKENS = frozenset({"1", "1.0", "attack", "anomaly", "malicious"})


class UnknownLabelError(ValueError):
    """Báo nhãn không thuộc quy ước của dự án."""
    pass


def to_binary_label(value: Any, strict: bool = True) -> int:
    """Quy đổi một giá trị nhãn về 0 (lành tính) hoặc 1 (tấn công)."""
    if pd.isna(value):
        return 0

    token = str(value).strip().lower()

    if token in BENIGN_TOKENS:
        return 0

    if token in ATTACK_TOKENS:
        return 1

    try:
        number = float(token)
    except (ValueError, TypeError):
        message = (
            f"Giá trị nhãn không nhận dạng được: {value!r}. Bổ sung giá trị vào "
            f"BENIGN_TOKENS hoặc ATTACK_TOKENS trong labels.py.")

        if strict:
            raise UnknownLabelError(message) from None

        print(f"[labels] CẢNH BÁO: {message}")
        return 1

    return 1 if number > 0 else 0
