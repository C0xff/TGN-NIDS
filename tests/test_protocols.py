"""Kiểm thử ba giao thức chia tập tái lập từ các bài báo."""

import numpy as np
import pandas as pd
import pytest

from tgn_nids.data.protocols import (
    split_te_g_sage,
    split_graphids,
    split_anomal_e,
    summarize,
)


def _df(n=1000, seed=0, has_time=True):
    """Tạo bảng NetFlow nhỏ với nhãn và mốc thời gian giả lập."""
    rng = np.random.default_rng(seed)
    d = {
        "IN_BYTES": rng.integers(0, 10000, n),
        "Label": (rng.random(n) < 0.05).astype(int),
    }
    d["Attack"] = np.where(d["Label"] == 1,
                           rng.choice(["DoS", "Exploits", "Worms"], n),
                           "Benign")
    if has_time:
        # Xáo trộn đầu vào để kiểm tra bước khôi phục thứ tự thời gian.
        d["FLOW_START_MILLISECONDS"] = rng.permutation(np.arange(n) * 1000)
    return pd.DataFrame(d)


def test_te_g_sage_dung_ty_le_60_30_10():
    tap = split_te_g_sage(_df(1000))
    assert len(tap["train"]) == 600
    assert len(tap["val"]) == 300
    assert len(tap["test"]) == 100


def test_te_g_sage_bao_toan_thu_tu_thoi_gian():
    """Mọi mốc thời gian trong Train phải nhỏ hơn Val, Val nhỏ hơn Test."""
    tap = split_te_g_sage(_df(1000))
    c = "FLOW_START_MILLISECONDS"
    assert tap["train"][c].max() <= tap["val"][c].min()
    assert tap["val"][c].max() <= tap["test"][c].min()


def test_te_g_sage_khong_mat_dong_nao():
    df = _df(997)
    tap = split_te_g_sage(df)
    assert sum(len(t) for t in tap.values()) == len(df)


def test_te_g_sage_bao_loi_ro_rang_khi_thieu_cot_thoi_gian():
    """Bộ v2 không có mốc thời gian, phải báo lỗi dễ hiểu chứ không vỡ."""
    with pytest.raises(KeyError, match="thời gian"):
        split_te_g_sage(_df(100, has_time=False))


def test_graphids_train_chi_con_benign():
    tap = split_graphids(_df(2000))
    assert (tap["train"]["Label"] == 1).sum() == 0
    assert len(tap["train"]) > 0


def test_graphids_val_va_test_van_giu_mau_tan_cong():
    """Lọc benign chỉ áp cho Train. Val và Test phải còn tấn công để đánh giá."""
    tap = split_graphids(_df(4000))
    assert (tap["val"]["Label"] == 1).sum() > 0
    assert (tap["test"]["Label"] == 1).sum() > 0


def test_graphids_phan_tang_giu_ty_le_lop():
    """Tỉ lệ tấn công ở Val và Test phải xấp xỉ tỉ lệ toàn tập."""
    df = _df(6000, seed=3)
    goc = (df["Label"] == 1).mean()
    tap = split_graphids(df)
    assert (tap["test"]["Label"] == 1).mean() == pytest.approx(goc, abs=0.02)


def test_graphids_lap_lai_duoc_voi_cung_seed():
    a = split_graphids(_df(1500), seed=7)
    b = split_graphids(_df(1500), seed=7)
    assert list(a["test"].index) == list(b["test"].index)


def test_graphids_khong_chong_lan_giua_cac_tap():
    tap = split_graphids(_df(2000), benign_only_train=False)
    idx = [set(tap[k].index) for k in ("train", "val", "test")]
    assert not (idx[0] & idx[1]) and not (idx[1] & idx[2]) and not (idx[0] & idx[2])


def test_anomal_e_ha_quy_mo_con_10_phan_tram():
    tap = split_anomal_e(_df(10000))
    tong = len(tap["train"]) + len(tap["test"])
    assert tong == pytest.approx(1000, abs=2)


def test_anomal_e_chia_70_30_va_khong_co_val():
    tap = split_anomal_e(_df(10000))
    tong = len(tap["train"]) + len(tap["test"])
    assert len(tap["train"]) == pytest.approx(tong * 0.7, abs=2)
    assert "val" not in tap


def test_anomal_e_khong_chong_lan():
    tap = split_anomal_e(_df(5000))
    assert not (set(tap["train"].index) & set(tap["test"].index))


def test_khop_so_luong_bai_bao_cong_bo():
    """Tỷ lệ 60/30/10 phải khớp số luồng trong Bảng 4 của TE-G-SAGE."""
    n = 2_365_424
    df = pd.DataFrame({
        "FLOW_START_MILLISECONDS": np.arange(n),
        "Label": np.zeros(n, dtype=int),
    })
    tap = split_te_g_sage(df)
    assert abs(len(tap["train"]) - 1_419_254) <= 1
    assert abs(len(tap["val"]) - 709_628) <= 1
    assert abs(len(tap["test"]) - 236_542) <= 1


def test_summarize_chay_duoc():
    assert "Tấn công" in summarize(split_te_g_sage(_df(500)))
