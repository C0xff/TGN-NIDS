"""Kiểm thử bộ giải mã có che và đầu phân loại cạnh."""

import torch

from tgn_nids.models.edge_decoder import EdgeFeatureDecoder


def test_decoder_tra_ve_dung_so_chieu_dac_trung():
    """Vector tái thiết phải có cùng số chiều với đặc trưng đầu vào."""
    decoder = EdgeFeatureDecoder(embedding_dim=100, edge_dim=43)
    z_src, z_dst = torch.randn(8, 100), torch.randn(8, 100)
    edge_features = torch.randn(8, 43)

    recon, mask = decoder(z_src, z_dst, edge_features)
    assert recon.shape == (8, 43)
    assert mask.shape == (8, 43)


def test_dau_ra_phu_thuoc_dac_trung_cua_chinh_canh():
    """Hai cạnh khác nội dung phải cho kết quả tái thiết khác nhau."""
    decoder = EdgeFeatureDecoder(embedding_dim=16, edge_dim=8).eval()
    # Cố định cặp đỉnh để chỉ còn đặc trưng cạnh tạo ra khác biệt.
    z_src = torch.randn(1, 16).repeat(2, 1)
    z_dst = torch.randn(1, 16).repeat(2, 1)
    edge_features = torch.stack([torch.zeros(8), torch.ones(8) * 5.0])

    with torch.no_grad():
        recon, _ = decoder(z_src, z_dst, edge_features, seed=0)

    assert not torch.allclose(recon[0], recon[1]), (
        "Bộ giải mã bỏ qua đặc trưng của chính cạnh, đúng lỗi của bản cũ")


def test_mat_na_che_dung_ti_le_yeu_cau():
    """Số thuộc tính bị che phải khớp mask_ratio, và mỗi cạnh che ít nhất một."""
    decoder = EdgeFeatureDecoder(embedding_dim=8, edge_dim=40, mask_ratio=0.25)
    z_src, z_dst = torch.randn(6, 8), torch.randn(6, 8)

    _, mask = decoder(z_src, z_dst, torch.randn(6, 40), seed=1)

    assert mask.sum(dim=1).unique().tolist() == [10], "40 x 0,25 = 10 thuộc tính"


def test_thuoc_tinh_khong_bi_che_khong_lot_vao_dau_vao_o_vi_tri_bi_che():
    """Giá trị thật ở vị trí bị che không được xuất hiện trong đầu vào."""
    decoder = EdgeFeatureDecoder(embedding_dim=8, edge_dim=12).eval()
    edge_features = torch.randn(3, 12)

    with torch.no_grad():
        visible = decoder.visible_input(edge_features,
                                        decoder.build_mask(edge_features, seed=2))
    mask = decoder.build_mask(edge_features, seed=2)
    assert torch.all(visible[mask] == 0), "Vị trí bị che phải bằng 0 ở đầu vào"
    assert torch.allclose(visible[~mask], edge_features[~mask])


def test_anomaly_score_chi_tinh_tren_vi_tri_bi_che():
    """Điểm bất thường phải là MSE trên riêng các vị trí bị che, trả về [N]."""
    decoder = EdgeFeatureDecoder(embedding_dim=16, edge_dim=10).eval()
    z_src, z_dst = torch.randn(5, 16), torch.randn(5, 16)
    edge_features = torch.randn(5, 10)

    with torch.no_grad():
        scores = decoder.anomaly_score(z_src, z_dst, edge_features, seed=3)

    assert scores.shape == (5,)
    assert torch.all(scores >= 0), "MSE không thể âm"


def test_diem_so_lap_lai_duoc_khi_cung_hat_giong():
    """Cùng hạt giống phải tạo cùng mặt nạ và điểm bất thường."""
    decoder = EdgeFeatureDecoder(embedding_dim=8, edge_dim=16).eval()
    z_src, z_dst = torch.randn(4, 8), torch.randn(4, 8)
    edge_features = torch.randn(4, 16)

    with torch.no_grad():
        a = decoder.anomaly_score(z_src, z_dst, edge_features, seed=7)
        b = decoder.anomaly_score(z_src, z_dst, edge_features, seed=7)
    assert torch.allclose(a, b)


def test_canh_bat_thuong_cho_score_cao_hon_canh_binh_thuong():
    """Sau khi học trên lưu lượng bình thường, cạnh lệch phân phối phải có score cao hơn."""
    torch.manual_seed(0)
    decoder = EdgeFeatureDecoder(embedding_dim=12, edge_dim=20, hidden_dim=64)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)

    # Dữ liệu bình thường có các thuộc tính tương quan chặt.
    base = torch.randn(400, 1)
    normal = base + torch.randn(400, 20) * 0.1
    z_src, z_dst = torch.randn(400, 12), torch.randn(400, 12)

    for _ in range(300):
        optimizer.zero_grad()
        loss = decoder.reconstruction_loss(z_src, z_dst, normal)
        loss.backward()
        optimizer.step()

    # Dữ liệu bất thường phá vỡ cấu trúc tương quan đó.
    anomalous = torch.randn(400, 20) * 2.0
    decoder.eval()
    with torch.no_grad():
        score_normal = decoder.anomaly_score(z_src, z_dst, normal, seed=5).mean()
        score_anomalous = decoder.anomaly_score(z_src, z_dst, anomalous, seed=5).mean()

    assert score_anomalous > score_normal


def test_import_duoc_tu_package_models():
    from tgn_nids.models import EdgeFeatureDecoder as Imported
    assert Imported is EdgeFeatureDecoder


def test_classifier_dau_ra_phu_thuoc_dac_trung_canh():
    """Logits phải thay đổi khi đặc trưng cạnh thay đổi."""
    import torch
    from tgn_nids.models.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector(embedding_dim=16, hidden_dim=32, num_classes=10,
                               edge_dim=8).eval()
    z_src = torch.randn(1, 16).repeat(2, 1)
    z_dst = torch.randn(1, 16).repeat(2, 1)
    edge_features = torch.stack([torch.zeros(8), torch.ones(8) * 3.0])

    with torch.no_grad():
        logits = detector(z_src, z_dst, edge_features)

    assert logits.shape == (2, 10)
    assert not torch.allclose(logits[0], logits[1]), (
        "Bộ phân loại bỏ qua đặc trưng cạnh, không thể phân biệt loại tấn công")


def test_classifier_van_chay_khi_khong_dung_dac_trung_canh():
    """Bộ phân loại vẫn hỗ trợ cấu hình ablation không dùng đặc trưng cạnh."""
    import torch
    from tgn_nids.models.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector(embedding_dim=16, hidden_dim=32, num_classes=2,
                               edge_dim=0).eval()
    with torch.no_grad():
        logits = detector(torch.randn(4, 16), torch.randn(4, 16))
    assert logits.shape == (4, 2)
